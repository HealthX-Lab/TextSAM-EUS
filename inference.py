import torch
import monai
from tqdm import tqdm
from statistics import mean
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from torch.optim import Adam
from torch.nn.functional import threshold, normalize
from torchvision.utils import save_image
import utils.utils as utils
from datasets.dataloader import DatasetSegmentation, collate_fn
from utils.processor import Samprocessor
from segment_anything import build_sam_vit_b, build_textsam_vit_b, build_textsam_vit_h, build_textsam_vit_l
from utils.lora import LoRA_Sam
import matplotlib.pyplot as plt
import yaml
import torch.nn.functional as F
import numpy as np
import cv2
import argparse
import os
import random
from utils.utils import load_cfg_from_cfg_file
import logging
from utils import utils

def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True

def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-file", required=True, type=str, help="Path to config file")
    parser.add_argument('--resume', action='store_true', help="Whether to resume training")
    parser.add_argument('--seed', type=int, default=1, help="Random seed for reproducibility.")
    parser.add_argument("--output-dir", type=str, default="", help="output directory")
    parser.add_argument("opts", default=None, nargs=argparse.REMAINDER, help="modify config options using the command-line")
    args = parser.parse_args()
    cfg = load_cfg_from_cfg_file(args.config_file)
    cfg.update({k: v for k, v in vars(args).items()})
    return cfg

def logger_config(log_path):
    loggerr = logging.getLogger()
    loggerr.setLevel(level=logging.INFO)
    handler = logging.FileHandler(log_path, encoding='UTF-8')
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(message)s')
    handler.setFormatter(formatter)
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    loggerr.addHandler(handler)
    loggerr.addHandler(console)
    return loggerr

device = "cuda" if torch.cuda.is_available() else "cpu"
cfg = get_arguments()

os.makedirs(os.path.join(cfg.output_dir, cfg.DATASET.NAME, "seg_results", f"seed{cfg.seed}"), exist_ok=True)
logger = logger_config(os.path.join(cfg.output_dir, cfg.DATASET.NAME, "trained_models", f"seed{cfg.seed}", "log.txt"))

logger.info("************")
logger.info("** Config **")
logger.info("************")
logger.info(cfg)

if cfg.seed >= 0:
    logger.info("Setting fixed seed: {}".format(cfg.seed))
    set_random_seed(cfg.seed)

results_name = (
    f"LORA{cfg.SAM.RANK}_"
    f"SHOTS{cfg.DATASET.NUM_SHOTS}_"
    f"NCTX{cfg.PROMPT_LEARNER.N_CTX_TEXT}_"
    f"CSC{cfg.PROMPT_LEARNER.CSC}_"
    f"CTP{cfg.PROMPT_LEARNER.CLASS_TOKEN_POSITION}"
)

checkpoint_type = "latest" if cfg.TEST.USE_LATEST else "best"

with torch.no_grad():
    checkpoint_path = os.path.join(
        cfg.output_dir,
        cfg.DATASET.NAME,
        "trained_models",
        f"seed{cfg.seed}",
        f"{results_name}_{checkpoint_type}.pth"
    )

    classnames = cfg.PROMPT_LEARNER.CLASSNAMES

    if cfg.SAM.MODEL == "vit_b":
        sam = build_textsam_vit_b(cfg=cfg, checkpoint=cfg.SAM.CHECKPOINT, classnames=classnames)
    elif cfg.SAM.MODEL == "vit_l":
        sam = build_textsam_vit_l(cfg=cfg, checkpoint=cfg.SAM.CHECKPOINT, classnames=classnames)
    else:
        sam = build_textsam_vit_h(cfg=cfg, checkpoint=cfg.SAM.CHECKPOINT, classnames=classnames)

    sam_lora = LoRA_Sam(sam, cfg.SAM.RANK)
    model = sam_lora.sam
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model"], strict=False)

    processor = Samprocessor(model)
    dice_scores = {}
    total_dice_values = []

    for text_label in classnames[1:]:
        dice_scores[text_label] = []
        os.makedirs(os.path.join(cfg.output_dir, cfg.DATASET.NAME, "seg_results", f"seed{cfg.seed}", text_label, results_name), exist_ok=True)
        os.makedirs(os.path.join(cfg.output_dir, cfg.DATASET.NAME, "gt_masks", f"seed{cfg.seed}", text_label, results_name), exist_ok=True)

    dataset = DatasetSegmentation(cfg, processor, mode="test")
    test_dataloader = DataLoader(dataset, batch_size=1, collate_fn=collate_fn)

    model.eval()
    model.to(device)

    progress_bar = tqdm(test_dataloader, desc="Evaluating", dynamic_ncols=True)

    for i, batch in enumerate(progress_bar):
        outputs = model(batched_input=batch, multimask_output=False)
        stk_gt = batch[0]["ground_truth_mask"]
        stk_out = torch.cat([out["masks"].squeeze(0) for out in outputs], dim=0)
        text_labels = batch[0]["text_labels"].squeeze(0)

        all_points = []
        all_labels = []
        all_boxes = []

        for b in range(stk_out.shape[0]):  # batch size
            mask = stk_out[b]              # shape: (H, W) or (C, H, W)

            pts, lbls = utils.get_centroid_points(mask.detach().cpu().numpy(), text_labels.detach().cpu().numpy())
            box = utils.get_bounding_box(mask.detach().cpu().numpy())
            all_boxes.append(box)
            all_points.append(pts)
            all_labels.append(lbls)

        # Stack all batch outputs
        point_coords = torch.stack(all_points)  # shape: (B, N, 2) if N same across batch
        point_labels = torch.stack(all_labels)  # shape: (B, N)
        points = point_coords, point_labels

        # Stack to shape (B, 1, 4) — [x_min, y_min, x_max, y_max] per sample
        bboxes = torch.cat(all_boxes, dim=0)  # (B, 4)

        batch[0]["points"] = points
        batch[0]["boxes"] = bboxes

        outputs = model(batched_input=batch, multimask_output=False)
        stk_out = torch.cat([out["masks"].squeeze(0) for out in outputs], dim=0)


        all_points = []
        all_labels = []
        all_boxes = []

        for b in range(stk_out.shape[0]):  # batch size
            mask = stk_out[b]              # shape: (H, W) or (C, H, W)

            pts, lbls = utils.get_centroid_points(mask.detach().cpu().numpy(), text_labels.detach().cpu().numpy())
            box = utils.get_bounding_box(mask.detach().cpu().numpy())
            all_boxes.append(box)
            all_points.append(pts)
            all_labels.append(lbls)

        # Stack all batch outputs
        point_coords = torch.stack(all_points)  # shape: (B, N, 2) if N same across batch
        point_labels = torch.stack(all_labels)  # shape: (B, N)
        points = point_coords, point_labels

        # Stack to shape (B, 1, 4) — [x_min, y_min, x_max, y_max] per sample
        bboxes = torch.cat(all_boxes, dim=0)  # (B, 4)

        batch[0]["points"] = points
        batch[0]["boxes"] = bboxes

        outputs = model(batched_input=batch, multimask_output=False)
        stk_out = torch.cat([out["masks"].squeeze(0) for out in outputs], dim=0)

        for j, label in enumerate(text_labels):
            label_j = int(label.detach().cpu())
            mask_pred = np.uint8(stk_out[j].detach().cpu())
            gt_mask = np.uint8(stk_gt[j].detach().cpu())

            cv2.imwrite(os.path.join(cfg.output_dir,
                                     cfg.DATASET.NAME,
                                     "seg_results",
                                     f"seed{cfg.seed}",
                                     classnames[label_j],
                                     results_name,
                                     batch[0]["mask_name"]), mask_pred * 255)

            cv2.imwrite(os.path.join(cfg.output_dir,
                                     cfg.DATASET.NAME,
                                     "gt_masks",
                                     f"seed{cfg.seed}",
                                     classnames[label_j],
                                     results_name,
                                     batch[0]["mask_name"]), gt_mask * 255)