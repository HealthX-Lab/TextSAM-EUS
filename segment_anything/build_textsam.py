# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import os
import requests
from tqdm import tqdm

from functools import partial

from .modeling import ImageEncoderViT, MaskDecoder, PromptEncoder, textsam, TwoWayTransformer
from .modeling import TextPromptEncoderBiomedCLIP

from .modeling import TextSam
# File URLs
files = {
    "sam_vit_b_01ec64.pth": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth",
    "sam_vit_l_0b3195.pth": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth",
    "sam_vit_h_4b8939.pth": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",
}

backbones = {
    "vit_l": "sam_vit_l_0b3195.pth",
    "vit_b": "sam_vit_b_01ec64.pth",
    "vit_h": "sam_vit_h_4b8939.pth",
    "default": "sam_vit_h_4b8939.pth"
}

# Function to download a file
def download_file(url, filepath):
    print(f"Downloading {filepath}...")
    response = requests.get(url, stream=True)
    if response.status_code == 200:
        total_size = int(response.headers.get('content-length', 0))
        with open(filepath, "wb") as file:
            # Use tqdm to show the progress bar
            with tqdm(total=total_size, unit='B', unit_scale=True, desc=filepath) as pbar:
                for chunk in response.iter_content(chunk_size=1024):
                    file.write(chunk)
                    pbar.update(len(chunk))  # Update progress bar by the chunk size
        print(f"{filepath} downloaded successfully.")
    else:
        print(f"Failed to download {filepath}. HTTP Status Code: {response.status_code}")


def build_textsam_vit_h(checkpoint=None):
    return _build_textsam(
        encoder_embed_dim=1280,
        encoder_depth=32,
        encoder_num_heads=16,
        encoder_global_attn_indexes=[7, 15, 23, 31],
        checkpoint=checkpoint,
    )


build_textsam = build_textsam_vit_h


def build_textsam_vit_l(checkpoint=None):
    return _build_textsam(
        encoder_embed_dim=1024,
        encoder_depth=24,
        encoder_num_heads=16,
        encoder_global_attn_indexes=[5, 11, 17, 23],
        checkpoint=checkpoint,
    )


def build_textsam_vit_b(cfg=None,checkpoint=None,classnames=None):
    return _build_textsam(
        cfg,
        encoder_embed_dim=768,
        encoder_depth=12,
        encoder_num_heads=12,
        encoder_global_attn_indexes=[2, 5, 8, 11],
        checkpoint=checkpoint,
        classnames=classnames
    )


sam_model_registry = {
    "default": build_textsam_vit_h,
    "vit_h": build_textsam_vit_h,
    "vit_l": build_textsam_vit_l,
    "vit_b": build_textsam_vit_b,
}


def _build_textsam(
    cfg,
    encoder_embed_dim,
    encoder_depth,
    encoder_num_heads,
    encoder_global_attn_indexes,
    checkpoint=None,
    classnames=None
):
    prompt_embed_dim = 256
    image_size = 1024
    vit_patch_size = 16
    image_embedding_size = image_size // vit_patch_size
    if(cfg.PROMPT_LEARNER.MODEL == "biomedclip" and cfg.PROMPT_LEARNER.MODALITY == "text"):
        clip_prompt_encoder = TextPromptEncoderBiomedCLIP(
            cfg=cfg,
            embed_dim=prompt_embed_dim,
            image_embedding_size=(image_embedding_size, image_embedding_size),
            input_image_size=(image_size, image_size),
            mask_in_chans=16,
            classnames=classnames
        )

    else:
        raise NotImplementedError
    
    sam = TextSam(
        image_encoder=ImageEncoderViT(
            depth=encoder_depth,
            embed_dim=encoder_embed_dim,
            img_size=image_size,
            mlp_ratio=4,
            norm_layer=partial(torch.nn.LayerNorm, eps=1e-6),
            num_heads=encoder_num_heads,
            patch_size=vit_patch_size,
            qkv_bias=True,
            use_rel_pos=True,
            global_attn_indexes=encoder_global_attn_indexes,
            window_size=14,
            out_chans=prompt_embed_dim,
        ),
        prompt_encoder=clip_prompt_encoder,
        mask_decoder=MaskDecoder(
            num_multimask_outputs=3,
            transformer=TwoWayTransformer(
                depth=2,
                embedding_dim=prompt_embed_dim,
                mlp_dim=2048,
                num_heads=8,
            ),
            transformer_dim=prompt_embed_dim,
            iou_head_depth=3,
            iou_head_hidden_dim=256,
        ),
        pixel_mean=[123.675, 116.28, 103.53],
        pixel_std=[58.395, 57.12, 57.375],
    )
    sam.train()
    filename = backbones[cfg.SAM.MODEL]
    url = files[filename]
    checkpoint = os.path.join(checkpoint, filename)
    if not os.path.exists(checkpoint):
        print(f"{filename} not found in {checkpoint}. Downloading...")
        download_file(url, checkpoint)
    if checkpoint is not None:
        with open(checkpoint, "rb") as f:
            state_dict = torch.load(f)
        sam.load_state_dict(state_dict,strict=False)

    params_to_update = ["prompt_learner", "text_head"]

    for name,param in sam.named_parameters():

        if(name.startswith("prompt_encoder")):
            
            param.requires_grad_(False)
            for p in params_to_update:
                if(p in name):
                    param.requires_grad_(True)
                    
    return sam