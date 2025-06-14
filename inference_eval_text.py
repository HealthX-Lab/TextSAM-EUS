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
from segment_anything import build_sam_vit_b, SamPredictor, build_textsam_vit_b,  build_textsam_vit_h, build_textsam_vit_l
from utils.lora import LoRA_Sam
import torch.nn.functional as F
import matplotlib.pyplot as plt
import yaml
import torch.nn.functional as F
import monai
import numpy as np
import cv2
import argparse
import os
import random
from utils.utils import load_cfg_from_cfg_file
import logging

def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True

def get_arguments():

    parser = argparse.ArgumentParser()
    parser.add_argument(
    "--config-file",
    required=True,
    type=str,
    help="Path to config file",
    )
    parser.add_argument(
        '--resume',
        action='store_true',
        help="Whether to resume training"
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=1,
        help="Random seed for reproducibility."
    )
    parser.add_argument(
        "--output-dir", 
        type=str,
        default="", 
        help="output directory")
    parser.add_argument(
            "opts",
            default=None,
            nargs=argparse.REMAINDER,
            help="modify config options using the command-line",
        )

    args = parser.parse_args()

    cfg = load_cfg_from_cfg_file(args.config_file)

    cfg.update({k: v for k, v in vars(args).items()})

    return cfg

def compute_binary_dice(pred, target, eps=1e-6):
    pred = np.uint8(pred > 0)
    target = np.uint8(target > 0)
    intersection = (pred * target).sum()
    dice = (2.0 * intersection) / (pred.sum() + target.sum() + eps)
    return dice

def print_args(args, cfg):
    logging.info("***************")
    logging.info("** Arguments **")
    logging.info("***************")
    # optkeys = list(args.__dict__.keys())
    # optkeys.sort()
    # for key in optkeys:
    #     logging.info("{}: {}".format(key, args.__dict__[key]))
    logging.info("************")
    logging.info("** Config **")
    logging.info("************")
    logging.info(cfg)

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
os.makedirs(os.path.join(cfg.output_dir, cfg.DATASET.NAME, "seg_results", f"seed{cfg.seed}"),exist_ok=True)
logger = logger_config(os.path.join(cfg.output_dir, cfg.DATASET.NAME, "trained_models", f"seed{cfg.seed}", "log.txt"))
logger.info("************")
logger.info("** Config **")
logger.info("************")
logger.info(cfg)
if cfg.seed >= 0:
    logger.info("Setting fixed seed: {}".format(cfg.seed))
    set_random_seed(cfg.seed)

# Load SAM model
results_name = (
    f"LORA{cfg.SAM.RANK}_"
    f"SHOTS{cfg.DATASET.NUM_SHOTS}_"
    f"NCTX{cfg.PROMPT_LEARNER.N_CTX_TEXT}_"
    f"CSC{cfg.PROMPT_LEARNER.CSC}_"
    f"CTP{cfg.PROMPT_LEARNER.CLASS_TOKEN_POSITION}"
)

if(cfg.TEST.USE_LATEST):
    checkpoint_type = "latest"
else:
    checkpoint_type = "best"

with torch.no_grad():

    # Specify the path to the .pth checkpoint file
    checkpoint_path = os.path.join(
            cfg.output_dir,
            cfg.DATASET.NAME,
            "trained_models",
            f"seed{cfg.seed}",
            f"{results_name}_{checkpoint_type}.pth")
    
    # Load SAM model
    classnames = cfg.PROMPT_LEARNER.CLASSNAMES

    # Load SAM model
    if(cfg.SAM.MODEL == "vit_b"):
        sam = build_textsam_vit_b(cfg=cfg, checkpoint=cfg.SAM.CHECKPOINT, classnames=classnames)
    elif(cfg.SAM.MODEL == "vit_l"):
        sam = build_textsam_vit_l(cfg=cfg, checkpoint=cfg.SAM.CHECKPOINT, classnames=classnames)
    else:
        sam = build_textsam_vit_h(cfg=cfg, checkpoint=cfg.SAM.CHECKPOINT, classnames=classnames)
    sam_lora = LoRA_Sam(sam, cfg.SAM.RANK)
    model = sam_lora.sam
    # model = sam
    
    # Load the saved state_dict into the model
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model"])  # Assuming the key "model_state_dict"
    
    processor = Samprocessor(model)

    dice_scores = {}
    
    dataset = DatasetSegmentation(cfg, processor, mode="test")
    test_dataloader = DataLoader(dataset, batch_size=1, collate_fn=collate_fn)
    model.eval()
    model.to(device)

    # Output directory for predictions
    out_dir = os.path.join(cfg.output_dir, cfg.DATASET.NAME, "seg_results", f"seed{cfg.seed}", results_name)
    gt_dir = os.path.join(cfg.output_dir, cfg.DATASET.NAME, "gt_masks", f"seed{cfg.seed}", results_name)
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(gt_dir, exist_ok=True)

    dice_scores = []
    model.eval()
    model.to(device)

    for batch in tqdm(test_dataloader):
        outputs = model(batched_input=batch, multimask_output=False)
        image_name = batch[0]["image_name"]

        stk_gt = batch[0]["ground_truth_mask"]
        stk_out = torch.cat([out["masks"].squeeze(0) for out in outputs], dim=0).squeeze(0)

        stk_out = stk_out.cpu().numpy()
        # mask_probs = F.sigmoid(stk_out, dim=0)

        # mask_pred = torch.argmax(mask_probs, dim=0).detach().cpu().numpy()
        gt_mask = stk_gt[0].detach().cpu().numpy()
        dice = compute_binary_dice(stk_out, gt_mask)
        dice_scores.append(dice)

        cv2.imwrite(os.path.join(out_dir, batch[0]["mask_name"]), (stk_out * 255).astype(np.uint8))
        cv2.imwrite(os.path.join(gt_dir, batch[0]["mask_name"]), (gt_mask * 255).astype(np.uint8))

    overall_average = mean(dice_scores)
    print(f"\nOverall Binary Dice Score across all test images: {overall_average:.4f}")
