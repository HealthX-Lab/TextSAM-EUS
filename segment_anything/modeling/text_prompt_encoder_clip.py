import torch
import torch.nn as nn
from .prompt_encoder import PromptEncoder
from transformers import CLIPTextModel
# from open_clip.src.open_clip import create_model_from_pretrained, get_tokenizer
from transformers import AutoModel, AutoProcessor, AutoTokenizer
from clip import clip
from collections import OrderedDict
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
import random
import os
import numpy as np

_tokenizer = _Tokenizer()

# Text Prompt Encoder class
class TextPromptEncoderCLIP(PromptEncoder):
    def __init__(
        self,
        cfg,
        embed_dim = 256,
        image_embedding_size = (64, 64),
        input_image_size = (1024, 1024),
        mask_in_chans = 1,
        activation = nn.GELU,
        classnames = None
        ) -> None:
        super().__init__(embed_dim, image_embedding_size, input_image_size, mask_in_chans, activation)

        clip_model = load_clip_to_cpu(cfg)
        clip_model.float()

        self.prompt_learner = PromptLearner(cfg,classnames, clip_model)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model, self.prompt_learner.deep_prompts)

        self.text_head = nn.Sequential(
            nn.Linear(512, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )

        self.cfg = cfg
        self.n_cls = len(classnames)

    def forward(
        self, points,
        boxes,
        masks,
        labels,
        image_embeddings,
        clip_image
    ):
        """
        Embeds different types of prompts, returning both sparse and dense
        embeddings.

        Arguments:
          points (tuple(torch.Tensor, torch.Tensor) or none): point coordinates
            and labels to embed.
          boxes (torch.Tensor or none): boxes to embed
          tokens (torch.Tensor or none): text tokens to embed
          masks (torch.Tensor or none): masks to embed

        Returns:
          torch.Tensor: sparse embeddings for the points and boxes, with shape
            BxNx(embed_dim), where N is determined by the number of input points
            and boxes.
          torch.Tensor: dense embeddings for the masks, in the shape
            Bx(embed_dim)x(embed_H)x(embed_W)
        """
        # bs = self._get_batch_size(points, boxes, masks, labels)
        bs = self.cfg.TRAIN.BATCH_SIZE
        sparse_embeddings = torch.empty(
            (bs, 0, self.embed_dim), device=self._get_device()
        )
        
        if labels is not None:

            prompts = self.prompt_learner()
            text_features = self.text_encoder(prompts, self.tokenized_prompts).to(torch.float32)
            
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            image_features = self.image_encoder(clip_image.unsqueeze(0).to(self._get_device()))
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)


            text_features = self.text_head(text_features)
            labels = [label.item() for label in labels]
            text_features = text_features[labels]

            sparse_embeddings_all = []

            for text_embeddings in text_features:
                
                text_embeddings = text_embeddings.unsqueeze(0).unsqueeze(0) 

                sparse_embeddings_all.append(torch.cat([sparse_embeddings, text_embeddings], dim=1))

            sparse_embeddings = torch.cat(sparse_embeddings_all, dim=0)

        if masks is not None:
            dense_embeddings = self._embed_masks(masks)
        else:
            dense_embeddings = self.no_mask_embed.weight.reshape(1, -1, 1, 1).expand(
                bs, -1, self.image_embedding_size[0], self.image_embedding_size[1]
            )

        return sparse_embeddings, dense_embeddings
    
    def _get_batch_size(self, points, boxes, masks, labels):
        """
        Returns the batch size of the inputs.
        """
        if points is not None:
            return points[0].shape[0]
        elif boxes is not None:
            return boxes.shape[0]
        elif masks is not None:
            return masks.shape[0]
        elif labels is not None:
            return labels.shape[0]
        else:
            return 1

def load_clip_to_cpu(cfg):
    backbone_name = cfg.PROMPT_LEARNER.BACKBONE
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)

    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")
    design_details = {"trainer": 'IVLP',
                      "vision_depth": cfg.PROMPT_LEARNER.PROMPT_DEPTH_VISION,
                      "language_depth": cfg.PROMPT_LEARNER.PROMPT_DEPTH_TEXT, "vision_ctx": cfg.PROMPT_LEARNER.N_CTX_VISION,
                      "language_ctx": cfg.PROMPT_LEARNER.N_CTX_TEXT}
    model = clip.build_model(state_dict or model.state_dict(), design_details)

    return model


class TextEncoder(nn.Module):
    def __init__(self, clip_model, deep_prompts):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype
        self.deep_prompts = deep_prompts

    def forward(self, prompts, tokenized_prompts):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        for i in range(self.transformer.layers):
            if(i != 0 and i < len(self.deep_prompts)):
                x = self.transformer.resblocks[i](x, self.deep_prompts[i-1])
            else:
                x = self.transformer.resblocks[i](x)
        # x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection

        return x

class PromptLearner(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        n_ctx = cfg.PROMPT_LEARNER.N_CTX_TEXT
        ctx_init = cfg.PROMPT_LEARNER.CTX_INIT
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]
        vis_dim = clip_model.visual.output_dim
        clip_imsize = clip_model.visual.input_resolution
        cfg_imsize = 224
        n_cls = len(classnames)
        assert cfg_imsize == clip_imsize, f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"
        assert cfg.PROMPT_LEARNER.PROMPT_DEPTH_TEXT >= 1, "In Independent VL prompting, Language prompt depth should be >=1" \
                                                        "\nPlease use VPT trainer if you want to learn only vision " \
                                                        "branch  "
        if ctx_init:
            # use given words to initialize context vectors
            ctx_init = ctx_init.replace("_", " ")
            n_ctx = len(ctx_init.split(" "))
            prompt = clip.tokenize(ctx_init)
            with torch.no_grad():
                embedding = clip_model.token_embedding(prompt).type(dtype)
            ctx_vectors = embedding[0, 1 : 1 + n_ctx, :]
            prompt_prefix = ctx_init

        else:
            # random initialization
            if cfg.PROMPT_LEARNER.CSC:
                print("Initializing class-specific contexts")
                ctx_vectors = torch.empty(n_cls, n_ctx, ctx_dim, dtype=dtype)
            else:
                print("Initializing a generic context")
                ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
                # neg_ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)

            nn.init.normal_(ctx_vectors, std=0.02)
            # nn.init.normal_(neg_ctx_vectors, std=0.02)
            prompt_prefix = " ".join(["X"] * n_ctx)
            # prompt_prefix_neg = " ".join(["X"] * n_ctx)

        print(f"Independent V-L design")
        print(f'Initial text context: "{prompt_prefix}"')
        print(f"Number of context words (tokens) for Language prompting: {n_ctx}")
        print(f"Number of context words (tokens) for Vision prompting: {cfg.PROMPT_LEARNER.N_CTX_VISION}")

        self.ctx = nn.Parameter(ctx_vectors)  # to be optimized
        # self.neg_ctx = nn.Parameter(neg_ctx_vectors)  # to be optimized

        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        prompts = [prompt_prefix + " " + name + "." for name in classnames]
        # prompts_neg = [prompt_prefix_neg + " " + name + "." for name in classnames]

        # tokenized_prompts_pos = []
        # tokenized_prompts_neg = []
        # for p_pos, p_neg in zip(prompts, prompts_neg):
        #     tokenized_prompts_pos.append(clip.tokenize(p_pos))
        #     tokenized_prompts_neg.append(clip.tokenize(p_neg))

        # tokenized_prompts_pos = torch.cat(tokenized_prompts_pos)
        # tokenized_prompts_neg = torch.cat(tokenized_prompts_neg)

        # with torch.no_grad():
        #     embedding_pos = clip_model.token_embedding(tokenized_prompts_pos).type(dtype)
        #     embedding_neg = clip_model.token_embedding(tokenized_prompts_neg).type(dtype)

        # # These token vectors will be saved when in save_model(),
        # # but they should be ignored in load_model() as we want to use
        # # those computed using the current class names
        # self.register_buffer("token_prefix_pos", embedding_pos[:, :1, :] )
        # self.register_buffer("token_suffix_pos", embedding_pos[:, 1 + n_ctx:, :])
        # self.register_buffer("token_prefix_neg", embedding_neg[:, :1, :])
        # self.register_buffer("token_suffix_neg", embedding_neg[:, 1 + n_ctx:, :])

        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)

        # These token vectors will be saved when in save_model(),
        # but they should be ignored in load_model() as we want to use
        # those computed using the current class names
        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx :, :])  # CLS, EOS

        self.n_cls = n_cls
        self.n_ctx = n_ctx
        # self.tokenized_prompts = torch.cat([tokenized_prompts_pos, tokenized_prompts_neg], dim=0)
        self.tokenized_prompts = tokenized_prompts
        self.name_lens = name_lens
        self.class_token_position = cfg.PROMPT_LEARNER.CLASS_TOKEN_POSITION

        self.deep_prompts = nn.ParameterList([nn.Parameter(nn.init.normal_(torch.empty(n_ctx, ctx_dim, dtype=dtype))) \
                                                for i in range(cfg.PROMPT_LEARNER.PROMPT_DEPTH_TEXT)])

        # self.neg_deep_prompts = nn.ParameterList([nn.Parameter(nn.init.normal_(torch.empty(n_ctx, ctx_dim, dtype=dtype))) \
        #                                         for i in range(cfg.PROMPT_LEARNER.PROMPT_DEPTH_TEXT)])

    def forward(self):
        ctx = self.ctx
        # neg_ctx = self.neg_ctx
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)
        # if neg_ctx.dim() == 2:
        #     neg_ctx = neg_ctx.unsqueeze(0).expand(self.n_cls, -1, -1)

        prefix = self.token_prefix
        suffix = self.token_suffix

        # prefix_pos = self.token_prefix_pos
        # prefix_neg = self.token_prefix_neg
        # suffix_pos = self.token_suffix_pos
        # suffix_neg = self.token_suffix_neg

        if self.class_token_position == "end":
            prompts = torch.cat(
                [
                    prefix,  # (n_cls, 1, dim)
                    ctx,     # (n_cls, n_ctx, dim)
                    suffix,  # (n_cls, *, dim)
                ],
                dim=1,
            )

        # if self.class_token_position == "end":

        #     prompts_pos = torch.cat(
        #         [
        #             prefix_pos,  # (n_cls, 1, dim)
        #             ctx,  # (n_cls, n_ctx, dim)
        #             suffix_pos,  # (n_cls, *, dim)
        #         ],
        #         dim=1,
        #     )

        #     prompts_neg = torch.cat(
        #         [
        #             prefix_neg,  # (n_cls, 1, dim)
        #             neg_ctx,  # (n_cls, n_ctx, dim)
        #             suffix_neg,  # (n_cls, *, dim)
        #         ],
        #         dim=1,
        #     )

        #     prompts = torch.cat([prompts_pos, prompts_neg], dim=0)

        # elif self.class_token_position == "middle":
        #     half_n_ctx = self.n_ctx // 2
        #     prompts = []
        #     for i in range(self.n_cls):
        #         name_len = self.name_lens[i]
        #         prefix_i = prefix[i : i + 1, :, :]
        #         class_i = suffix[i : i + 1, :name_len, :]
        #         suffix_i = suffix[i : i + 1, name_len:, :]
        #         ctx_i_half1 = ctx[i : i + 1, :half_n_ctx, :]
        #         ctx_i_half2 = ctx[i : i + 1, half_n_ctx:, :]
        #         prompt = torch.cat(
        #             [
        #                 prefix_i,     # (1, 1, dim)
        #                 ctx_i_half1,  # (1, n_ctx//2, dim)
        #                 class_i,      # (1, name_len, dim)
        #                 ctx_i_half2,  # (1, n_ctx//2, dim)
        #                 suffix_i,     # (1, *, dim)
        #             ],
        #             dim=1,
        #         )
        #         prompts.append(prompt)
        #     prompts = torch.cat(prompts, dim=0)

        # elif self.class_token_position == "front":
        #     prompts = []
        #     for i in range(self.n_cls):
        #         name_len = self.name_lens[i]
        #         prefix_i = prefix[i : i + 1, :, :]
        #         class_i = suffix[i : i + 1, :name_len, :]
        #         suffix_i = suffix[i : i + 1, name_len:, :]
        #         ctx_i = ctx[i : i + 1, :, :]
        #         prompt = torch.cat(
        #             [
        #                 prefix_i,  # (1, 1, dim)
        #                 class_i,   # (1, name_len, dim)
        #                 ctx_i,     # (1, n_ctx, dim)
        #                 suffix_i,  # (1, *, dim)
        #             ],
        #             dim=1,
        #         )
        #         prompts.append(prompt)
        #     prompts = torch.cat(prompts, dim=0)

        else:
            raise ValueError

        return prompts
        

