import torch
import torch.nn as nn
from .prompt_encoder import PromptEncoder
from transformers import CLIPTextModel
# from open_clip.src.open_clip import create_model_from_pretrained, get_tokenizer
from transformers import AutoModel, AutoProcessor, AutoTokenizer
from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
import torch.nn.functional as F
from collections import OrderedDict
import copy

_tokenizer = _Tokenizer()

def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

# Text Prompt Encoder class
class MultimodalPromptEncoderCLIP(PromptEncoder):
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
        # text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch16")
        # self.clip_model,_ = create_model_from_pretrained('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')
        # self.tokenizer = get_tokenizer('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')
        # clip_model = AutoModel.from_pretrained("chuhac/BiomedCLIP-vit-bert-hf", trust_remote_code=True)
        clip_model = load_clip_to_cpu(cfg)
        clip_model.float()
        # print(clip_model.text_model)
        # for i,p in model.named_parameters():
        #     print(i, p.shape)
        # text_encoder = clip_model.text.transformer
        # text_encoder = clip_model.text_model
        # text_proj = clip_model.text_projection
        self.text_encoder = TextEncoder(clip_model)
        self.image_encoder = clip_model.visual
        self.logit_scale = clip_model.logit_scale
        # self.dtype = clip_model.dtype
        # text_encoder.requires_grad_(False)
        # text_proj.requires_grad_(False)
        # self.text_encoder = text_encoder
        # self.text_proj = text_proj
        # self.text_encoder_head = nn.Linear(512, embed_dim)
        self.text_head = nn.Sequential(
            nn.Linear(512, embed_dim),
            nn.ReLU(),
            # nn.BatchNorm1d(embed_dim)
        )
        self.image_head = nn.Sequential(
            nn.Linear(512, embed_dim),
            nn.ReLU(),
            # nn.BatchNorm1d(embed_dim)
        )
        if(cfg.PROMPT_LEARNER.FUSE_SAM):
            sam_input_dim = 64*64
            self.sam_head = nn.Sequential(OrderedDict([
                ("linear1", nn.Linear(sam_input_dim, sam_input_dim // 16)),
                ("relu", nn.ReLU(inplace=True)),
                ("linear2", nn.Linear(sam_input_dim // 16, 16))
            ]))
        if(cfg.PROMPT_LEARNER.FUSE_TYPE == "attention"):
            self.cross_attention = nn.MultiheadAttention(embed_dim, num_heads=8, batch_first=True)
        self.prompt_learner = PromptLearner(cfg,classnames, clip_model)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts

        self.cfg = cfg

        print(sum(p.numel() for p in self.prompt_learner.parameters() if p.requires_grad))

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
        bs = self._get_batch_size(points, boxes, masks, labels)
        sparse_embeddings = torch.empty(
            (bs, 0, self.embed_dim), device=self._get_device()
        )
        if(self.cfg.PROMPT_LEARNER.FUSE_SAM):
            reshaped_image_embeddings = image_embeddings.view(bs, 256, -1) # (B,  256, 64*64)
            reshaped_image_embeddings = self.sam_head(reshaped_image_embeddings).view(bs, -1, 256) # (B, 16, 256)
        # input_image_embeddings = F.interpolate(
        #     reshaped_image_embeddings,
        #     (224, 224),
        #     mode="bilinear",
        #     align_corners=False,
        # )
        # if points is not None:
        #     coords, labels = points
        #     point_embeddings = self._embed_points(coords, labels, pad=(boxes is None))
        #     sparse_embeddings = torch.cat([sparse_embeddings, point_embeddings], dim=1)
        # if boxes is not None:
        #     box_embeddings = self._embed_boxes(boxes)
        #     sparse_embeddings = torch.cat([sparse_embeddings, box_embeddings], dim=1)
        if labels is not None:

            logit_scale = self.logit_scale.exp()
            tokenized_prompts = self.tokenized_prompts
        
            prompts, shared_ctx, deep_compound_prompts_text, deep_compound_prompts_vision = self.prompt_learner()
            text_features = self.text_encoder(prompts, tokenized_prompts, deep_compound_prompts_text)
            image_features = self.image_encoder(clip_image.unsqueeze(0).to(self._get_device()), shared_ctx, deep_compound_prompts_vision)

            # logits = logit_scale * image_features @ text_features.t()

            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

            image_features = self.image_head(image_features).unsqueeze(1)

            text_features = self.text_head(text_features)
            labels = [label.item()-1 for label in labels]
            text_features = text_features[labels]

            sparse_embeddings_all = []

            for text_feature in text_features:
                
                text_feature = text_feature.unsqueeze(0).unsqueeze(0)

                if(self.cfg.PROMPT_LEARNER.FUSE_TYPE == "attention"):
                # Attention-based fusion
                    if(not self.cfg.PROMPT_LEARNER.FUSE_SAM):
                        fused_embeddings, _ = self.cross_attention(
                            text_feature,  # Query (B, 1, embed_dim)
                            image_features,  # Key (B, 16, embed_dim)
                            image_features  # Value (B, 16, embed_dim)
                        )
                        sparse_embeddings_all.append(torch.cat([sparse_embeddings, fused_embeddings], dim=1))
                    else:
                        fused_embeddings, _ = self.cross_attention(
                            text_features,  # Query (B, 1, embed_dim)
                            reshaped_image_embeddings,  # Key (B, 16, embed_dim)
                            reshaped_image_embeddings  # Value (B, 16, embed_dim)
                        )
                        sparse_embeddings_all.append(torch.cat([sparse_embeddings, fused_embeddings], dim=1))

                elif(self.cfg.PROMPT_LEARNER.FUSE_TYPE == "add"):
                    if(not self.cfg.PROMPT_LEARNER.FUSE_SAM):
                        fused_embeddings = text_feature + image_features
                        # print(text_features.shape,image_features.shape)
                    else:
                        fused_embeddings = text_feature + image_features + reshaped_image_embeddings
                    sparse_embeddings_all.append(torch.cat([sparse_embeddings, fused_embeddings], dim=1))

                elif(self.cfg.PROMPT_LEARNER.FUSE_TYPE == "concat"):
                    if(not self.cfg.PROMPT_LEARNER.FUSE_SAM):
                        sparse_embeddings_all.append(torch.cat([sparse_embeddings, text_feature, image_features], dim=1))
                    else:
                        sparse_embeddings_all.append(torch.cat([sparse_embeddings, text_features, image_features, reshaped_image_embeddings], dim=1))
            # else:
            #     sparse_embeddings = torch.cat([sparse_embeddings, text_embeddings], dim=1)

            # print(logits.shape)

            sparse_embeddings = torch.cat(sparse_embeddings_all, dim=0)

            


            # if(self.cfg.PROMPT_LEARNER.FUSE_TYPE == "attention"):
            # # Attention-based fusion
            #     if(not self.cfg.PROMPT_LEARNER.FUSE_SAM):
            #         fused_embeddings, _ = self.cross_attention(
            #             text_features,  # Query (B, 1, embed_dim)
            #             image_features,  # Key (B, 16, embed_dim)
            #             image_features  # Value (B, 16, embed_dim)
            #         )
            #         sparse_embeddings = torch.cat([sparse_embeddings, fused_embeddings], dim=1)
            #     else:
            #         fused_embeddings, _ = self.cross_attention(
            #             text_features,  # Query (B, 1, embed_dim)
            #             reshaped_image_embeddings,  # Key (B, 16, embed_dim)
            #             reshaped_image_embeddings  # Value (B, 16, embed_dim)
            #         )
            #         sparse_embeddings = torch.cat([sparse_embeddings, fused_embeddings], dim=1)

            # elif(self.cfg.PROMPT_LEARNER.FUSE_TYPE == "add"):
            #     if(not self.cfg.PROMPT_LEARNER.FUSE_SAM):
            #         fused_embeddings = text_features + image_features
            #     else:
            #         fused_embeddings = text_features + image_features + reshaped_image_embeddings
            #     sparse_embeddings = torch.cat([sparse_embeddings,fused_embeddings], dim=1)

            # elif(self.cfg.PROMPT_LEARNER.FUSE_TYPE == "concat"):
            #     if(not self.cfg.PROMPT_LEARNER.FUSE_SAM):
            #         sparse_embeddings = torch.cat([sparse_embeddings, text_features, image_features], dim=1)
            #     else:
            #         sparse_embeddings = torch.cat([sparse_embeddings, text_features, image_features, reshaped_image_embeddings], dim=1)
            # else:
            #     sparse_embeddings = torch.cat([sparse_embeddings, text_embeddings], dim=1)

            # print(logits.shape)



            # text_embeddings = text_features.view(-1, 2, 256)
            # sparse_embeddings = torch.cat([sparse_embeddings, logits], dim=1)
            # return sparse_embeddings


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
    design_details = {"trainer": 'MaPLe',
                      "vision_depth": 0,
                      "language_depth": 0, "vision_ctx": 0,
                      "language_ctx": 0,
                      "maple_length": cfg.PROMPT_LEARNER.N_CTX_TEXT}
    model = clip.build_model(state_dict or model.state_dict(), design_details)

    return model


class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts, compound_prompts_deeper_text):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        # Pass as the list, as nn.sequential cannot process multiple arguments in the forward pass
        combined = [x, compound_prompts_deeper_text, 0]  # third argument is the counter which denotes depth of prompt
        outputs = self.transformer(combined)
        x = outputs[0]  # extract the x back from here
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection

        return x

class PromptLearner(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        n_cls = len(classnames)
        n_ctx = cfg.PROMPT_LEARNER.N_CTX_TEXT
        ctx_init = cfg.PROMPT_LEARNER.CTX_INIT
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]
        clip_imsize = clip_model.visual.input_resolution
        cfg_imsize = 224
        # Default is 1, which is compound shallow prompting
        assert cfg.PROMPT_LEARNER.PROMPT_DEPTH_TEXT >= 1, "For MaPLe, PROMPT_DEPTH should be >= 1"
        self.compound_prompts_depth = cfg.PROMPT_LEARNER.PROMPT_DEPTH_TEXT  # max=12, but will create 11 such shared prompts
        assert cfg_imsize == clip_imsize, f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"

        if ctx_init and (n_ctx) <= 4:
            # use given words to initialize context vectors
            ctx_init = ctx_init.replace("_", " ")
            n_ctx = n_ctx
            prompt = clip.tokenize(ctx_init)
            with torch.no_grad():
                embedding = clip_model.token_embedding(prompt).type(dtype)
            ctx_vectors = embedding[0, 1: 1 + n_ctx, :]
            prompt_prefix = ctx_init
        else:
            # random initialization
            ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
            nn.init.normal_(ctx_vectors, std=0.02)
            prompt_prefix = " ".join(["X"] * n_ctx)
        print('MaPLe design: Multi-modal Prompt Learning')
        print(f'Initial context: "{prompt_prefix}"')
        print(f"Number of MaPLe context words (tokens): {n_ctx}")
        # These below, related to the shallow prompts
        # Linear layer so that the tokens will project to 512 and will be initialized from 768
        self.proj = nn.Linear(ctx_dim, 768)
        self.proj.float()
        self.ctx = nn.Parameter(ctx_vectors)
        # These below parameters related to the shared prompts
        # Define the compound prompts for the deeper layers

        # Minimum can be 1, which defaults to shallow MaPLe
        # compound prompts
        self.compound_prompts_text = nn.ParameterList([nn.Parameter(torch.empty(n_ctx, 512))
                                                      for _ in range(self.compound_prompts_depth - 1)])
        for single_para in self.compound_prompts_text:
            nn.init.normal_(single_para, std=0.02)
        # Also make corresponding projection layers, for each prompt
        single_layer = nn.Linear(ctx_dim, 768)
        self.compound_prompt_projections = _get_clones(single_layer, self.compound_prompts_depth - 1)

        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        prompts = [prompt_prefix + " " + name + "." for name in classnames]

        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])  # (n_cls, n_tkn)
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)

        # These token vectors will be saved when in save_model(),
        # but they should be ignored in load_model() as we want to use
        # those computed using the current class names
        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx:, :])  # CLS, EOS

        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor
        self.name_lens = name_lens
        self.class_token_position = cfg.PROMPT_LEARNER.CLASS_TOKEN_POSITION

    def construct_prompts(self, ctx, prefix, suffix, label=None):
        if self.class_token_position == "end":
                prompts = torch.cat(
                    [
                        prefix,  # (n_cls, 1, dim)
                        ctx,     # (n_cls, n_ctx, dim)
                        suffix,  # (n_cls, *, dim)
                    ],
                    dim=1,
                )

        elif self.class_token_position == "middle":
            half_n_ctx = self.n_ctx // 2
            prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i : i + 1, :, :]
                class_i = suffix[i : i + 1, :name_len, :]
                suffix_i = suffix[i : i + 1, name_len:, :]
                ctx_i_half1 = ctx[i : i + 1, :half_n_ctx, :]
                ctx_i_half2 = ctx[i : i + 1, half_n_ctx:, :]
                prompt = torch.cat(
                    [
                        prefix_i,     # (1, 1, dim)
                        ctx_i_half1,  # (1, n_ctx//2, dim)
                        class_i,      # (1, name_len, dim)
                        ctx_i_half2,  # (1, n_ctx//2, dim)
                        suffix_i,     # (1, *, dim)
                    ],
                    dim=1,
                )
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)

        elif self.class_token_position == "front":
            prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i : i + 1, :, :]
                class_i = suffix[i : i + 1, :name_len, :]
                suffix_i = suffix[i : i + 1, name_len:, :]
                ctx_i = ctx[i : i + 1, :, :]
                prompt = torch.cat(
                    [
                        prefix_i,  # (1, 1, dim)
                        class_i,   # (1, name_len, dim)
                        ctx_i,     # (1, n_ctx, dim)
                        suffix_i,  # (1, *, dim)
                    ],
                    dim=1,
                )
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)

        else:
            raise ValueError

        return prompts


    def forward(self):
        ctx = self.ctx

        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)

        prefix = self.token_prefix
        suffix = self.token_suffix
        prompts = self.construct_prompts(ctx, prefix, suffix)

        # Before returning, need to transform
        # prompts to 768 for the visual side
        visual_deep_prompts = []
        for index, layer in enumerate(self.compound_prompt_projections):
            visual_deep_prompts.append(layer(self.compound_prompts_text[index]))
        # Now the other way around
        # We will project the textual prompts from 512 to 768
        return prompts, self.proj(self.ctx), self.compound_prompts_text, visual_deep_prompts   # pass here original, as for visual 768 is required
        

