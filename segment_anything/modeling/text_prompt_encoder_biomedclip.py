import os.path as osp
from collections import OrderedDict
import math
import copy
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast
from open_clip import create_model_from_pretrained, get_tokenizer
from biomedclip.vision_modules import Block
from biomedclip.text_modules import BertLayer
from typing import Optional
from .prompt_encoder import PromptEncoder

# Text Prompt Encoder class
class TextPromptEncoderBiomedCLIP(PromptEncoder):
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
        clip_model = load_biomedclip_to_cpu()
        clip_model.float()

        self.prompt_learner = MultiModalPromptLearner(cfg,classnames, clip_model)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.clip_model = CustomCLIP(cfg, classnames, clip_model)
        self.image_encoder = clip_model.visual
        self.text_encoder = clip_model.text
        self.logit_scale = clip_model.logit_scale

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
            (1, 0, self.embed_dim), device=self._get_device()
        )

        if labels is not None:

            prompts, ctx, deep_prompts = self.prompt_learner()

            text_features = self.clip_model.encode_text(self.tokenized_prompts, prompts, deep_prompts)

            labels = [label.item() for label in labels]
            text_features = text_features[labels]
            
            text_features = self.text_head(text_features)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True) # (N, 512)

            sparse_embeddings_all = []
            
            for text_embeddings in text_features:
                text_embeddings = text_embeddings.expand(1, 1, -1)
                sparse_embeddings_all.append(torch.cat([sparse_embeddings, text_embeddings], dim=1))

            sparse_embeddings = torch.cat(sparse_embeddings_all, dim=0)

        if points is not None:
            coords, labels = points
            point_embeddings = self._embed_points(coords, labels, pad=(boxes is None))
            sparse_embeddings = torch.cat([sparse_embeddings, point_embeddings], dim=1)
        
        if boxes is not None:
            box_embeddings = self._embed_boxes(boxes)
            sparse_embeddings = torch.cat([sparse_embeddings, box_embeddings], dim=1)

        if masks is not None:
            dense_embeddings = self._embed_masks(masks)
        else:
            dense_embeddings = self.no_mask_embed.weight.reshape(1, -1, 1, 1).expand(
                1, -1, self.image_embedding_size[0], self.image_embedding_size[1]
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


def load_biomedclip_to_cpu():
    
    model, _ = create_model_from_pretrained('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')
    
    vision_target_network = nn.Sequential(*[Block(768,12) for i in range(12)]).to('cuda')
    vision_network = model.visual.trunk.blocks.to('cuda')

    text_target_network = nn.ModuleList([BertLayer() for i in range(12)]).to('cuda')
    text_network = model.text.transformer.encoder.layer.to('cuda')

    for target_param, param in zip(vision_target_network.parameters(), vision_network.parameters()):
            target_param.data.copy_(param.data)

    for target_param, param in zip(text_target_network.parameters(), text_network.parameters()):
            target_param.data.copy_(param.data)

    model.visual.trunk.blocks = vision_target_network.to('cuda')
    model.text.transformer.encoder.layer  = text_target_network.to('cuda')

    return model.to('cuda').eval()

class MultiModalPromptLearner(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        n_cls = len(classnames)
        n_ctx = cfg.PROMPT_LEARNER.N_CTX_TEXT
        ctx_init = cfg.PROMPT_LEARNER.CTX_INIT
        dtype = clip_model.text.transformer.dtype
        ctx_dim = 768
        self.tokenizer = get_tokenizer('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')
        # Default is 1, which is compound shallow prompting
        assert cfg.PROMPT_LEARNER.PROMPT_DEPTH_TEXT >= 1, "For MaPLe, PROMPT_DEPTH should be >= 1"
        self.prompts_depth = cfg.PROMPT_LEARNER.PROMPT_DEPTH_TEXT  # max=12, but will create 11 such shared prompts

        if ctx_init:
            # use given words to initialize context vectors
            ctx_init = ctx_init.replace("_", " ")
            n_ctx = n_ctx
            prompt = self.tokenizer(ctx_init).to("cuda")
            with torch.no_grad():
                embedding = clip_model.text.transformer.embeddings.word_embeddings(prompt).type(dtype)
            ctx_vectors = embedding[0, 1: 1 + n_ctx, :]
            prompt_prefix = ctx_init
        else:
            # random initialization
            ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
            nn.init.normal_(ctx_vectors, std=0.02)
            prompt_prefix = " ".join(["X"] * n_ctx)

        print(f'Initial context: "{prompt_prefix}"')
        print(f"Number of context words (tokens): {n_ctx}")

        self.ctx = nn.Parameter(ctx_vectors)

        self.deep_prompts = nn.ParameterList([nn.Parameter(torch.empty(n_ctx, 768)) for _ in range(self.prompts_depth - 1)])

        for single_prompt in self.deep_prompts:
            nn.init.normal_(single_prompt, std=0.02)

        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(self.tokenizer(name)) for name in classnames]
        prompts = [prompt_prefix + " " + name + "." for name in classnames]

        tokenized_prompts = torch.cat([self.tokenizer(p) for p in prompts]).to("cuda")  # (n_cls, n_tkn)
        with torch.no_grad():
            embedding = clip_model.text.transformer.embeddings.word_embeddings(tokenized_prompts).type(dtype)

        # These token vectors will be saved when in save_model(),
        # but they should be ignored in load_model() as we want to use
        # those computed using the current class names
        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx:, :])  # CLS, EOS

        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor
        self.name_lens = name_lens

    def construct_prompts(self, ctx, prefix, suffix, label=None):
        # dim0 is either batch_size (during training) or n_cls (during testing)
        # ctx: context tokens, with shape of (dim0, n_ctx, ctx_dim)
        # prefix: the sos token, with shape of (n_cls, 1, ctx_dim)
        # suffix: remaining tokens, with shape of (n_cls, *, ctx_dim)

        if label is not None:
            prefix = prefix[label]
            suffix = suffix[label]

        prompts = torch.cat(
            [
                prefix,  # (dim0, 1, dim)
                ctx,  # (dim0, n_ctx, dim)
                suffix,  # (dim0, *, dim)
            ],
            dim=1,
        )

        return prompts

    def forward(self):
        ctx = self.ctx

        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)

        prefix = self.token_prefix
        suffix = self.token_suffix
        prompts = self.construct_prompts(ctx, prefix, suffix)

        return prompts, self.ctx, self.deep_prompts

class CustomCLIP(nn.Module):
    def __init__(self, cfg, classnames, clip_model, output_hidden_states=False):
        super(CustomCLIP, self).__init__()
        self.vision_model = clip_model.visual
        self.text_model = clip_model.text
        self.logit_scale = clip_model.logit_scale
        self.prompt_depth = cfg.PROMPT_LEARNER.PROMPT_DEPTH_TEXT
        self.prompt_length = cfg.PROMPT_LEARNER.N_CTX_TEXT
        self.output_hidden_states = output_hidden_states
        self.dtype = self.text_model.transformer.dtype
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def encode_image(self, x, ctx: torch.Tensor = None, prompt_tokens: torch.Tensor = None):
        trunk = self.vision_model.trunk
        x = trunk.patch_embed(x)
        x = trunk._pos_embed(x)
        ctx = ctx.expand(x.shape[0], -1, -1)
        x = torch.cat([x, ctx], dim=1)
        x = trunk.norm_pre(x)

        hidden_states = []

        for i, block in enumerate(trunk.blocks):
            if(i == 0):
                x = block(x)
            elif(i < self.prompt_depth - 1 and i > 0):
                x = block(x, prompt_tokens[i-1])
            else:
                x = block(x) 

            hidden_states.append(x)

        x = x[:, 0:x.shape[1] - self.prompt_length, :]

        x = trunk.norm(x)

        # Linear Projection: 768 -> 512
        x = self.vision_model.head(x)

        if self.output_hidden_states:
            return x, hidden_states
        else:
            return x
        
    def encode_text(self, tokenized_prompts, text_prompts, prompt_tokens : torch.Tensor = None, attention_mask: Optional[torch.LongTensor] = None):

        if attention_mask is None:
            attention_mask = (tokenized_prompts != self.text_model.config.pad_token_id).long()
        
        x = self.text_model.transformer.embeddings(
            inputs_embeds=text_prompts
        )

        extended_attention_mask = attention_mask[:, None, None, :]
        extended_attention_mask = extended_attention_mask.to(dtype=self.dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * torch.finfo(self.dtype).min

        for i, layer in enumerate(self.text_model.transformer.encoder.layer):
            if(i == 0):
                x = layer(x, attention_mask=extended_attention_mask)
            elif(i < self.prompt_depth - 1 and i > 0):
                x = layer(x, attention_mask=extended_attention_mask, prompt_tokens=prompt_tokens[i-1])
            else:
                x = layer(x, attention_mask=extended_attention_mask)
            x = x[0]

        pooled_out = x[:, 0, :]
        projected = self.text_model.proj(pooled_out)
        x = self.text_model.proj(x)

        # return projected
        return projected
    
    def forward(self, image):

        # prompts, ctx, text_deep_prompts, image_deep_prompts = self.prompt_learner()
        prompts, ctx, deep_prompts = self.prompt_learner()

        image_features = self.encode_image(image, ctx, deep_prompts)
        text_features = self.encode_text(self.tokenized_prompts, prompts, deep_prompts)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True) # (N, 512)

        cls_token = image_features[:, 0, :]
        seg_logits = image_features[:, 1:, :]
        cls_token = cls_token / cls_token.norm(dim=-1, keepdim=True) # (N, 512)
        seg_logits = seg_logits / seg_logits.norm(dim=-1, keepdim=True) # (N, 512)
       
        return seg_logits