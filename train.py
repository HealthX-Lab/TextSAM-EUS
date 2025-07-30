import torch
import torch.nn as nn
import monai
from tqdm import tqdm
from statistics import mean
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import utils.utils as utils
from datasets.dataloader import DatasetSegmentation, collate_fn
from utils.processor import Samprocessor
from segment_anything import build_textsam_vit_b,  build_textsam_vit_h, build_textsam_vit_l
from utils.lora import LoRA_Sam
import os
from time import time
import argparse
import random
import numpy as np
import logging
from utils.utils import load_cfg_from_cfg_file

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


def print_args(cfg):
    logging.info("***************")
    logging.info("** Arguments **")
    logging.info("***************")
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

# Validation function
def evaluate_validation_loss(model, val_dataloader, device, seg_loss, ce_loss):
    model.eval()  # Set model to evaluation mode
    val_losses = []
    with torch.no_grad():
        for batch in tqdm(val_dataloader, desc="Validation"):
            outputs = model(batched_input=batch, multimask_output=False)
            stk_gt, stk_out = utils.stacking_batch(batch, outputs)
            stk_out = stk_out.squeeze(1)
            loss = seg_loss(stk_out, stk_gt.float().to(device)) + ce_loss(stk_out, stk_gt.float().to(device))
            val_losses.append(loss.item())
    model.train()
    return mean(val_losses)

cfg = get_arguments()
os.makedirs(os.path.join(cfg.output_dir, cfg.DATASET.NAME, "trained_models", f"seed{cfg.seed}"),exist_ok = True)
logger = logger_config(os.path.join(cfg.output_dir, cfg.DATASET.NAME, "trained_models", f"seed{cfg.seed}", "log.txt"))
logger.info("************")
logger.info("** Config **")
logger.info("************")
logger.info(cfg)
if cfg.seed >= 0:
    logger.info("Setting fixed seed: {}".format(cfg.seed))
    set_random_seed(cfg.seed)

# Take dataset path
train_dataset_path = cfg.DATASET.TRAIN_PATH

classnames = cfg.PROMPT_LEARNER.CLASSNAMES

# Load SAM model
if(cfg.SAM.MODEL == "vit_b"):
    sam = build_textsam_vit_b(cfg=cfg, checkpoint=cfg.SAM.CHECKPOINT, classnames=classnames)
elif(cfg.SAM.MODEL == "vit_l"):
    sam = build_textsam_vit_l(cfg=cfg, checkpoint=cfg.SAM.CHECKPOINT, classnames=classnames)
else:
    sam = build_textsam_vit_h(cfg=cfg, checkpoint=cfg.SAM.CHECKPOINT, classnames=classnames)
# Create SAM LoRA
sam_lora = LoRA_Sam(sam, cfg.SAM.RANK)
model = sam_lora.sam

# Process the dataset
processor = Samprocessor(model)
train_ds = DatasetSegmentation(cfg, processor, mode="train", num_shots=cfg.DATASET.NUM_SHOTS, seed= cfg.seed)
# Create a dataloader
train_dataloader = DataLoader(train_ds, batch_size=cfg.TRAIN.BATCH_SIZE, shuffle=True, collate_fn=collate_fn)

#Load validation dataset
val_ds = DatasetSegmentation(cfg, processor, mode="val", num_shots=cfg.DATASET.NUM_SHOTS, seed=cfg.seed*cfg.seed)
val_dataloader = DataLoader(val_ds, batch_size=cfg.TRAIN.BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

enabled = set()
for name, param in model.named_parameters():
    if param.requires_grad:
        enabled.add(name)

print(f"Parameters to be updated: {enabled}")
print("Number of trainable parameters: ", sum(p.numel() for p in model.parameters() if p.requires_grad))
# Initialize optimizer and Loss
optimizer = AdamW(model.parameters(), lr=cfg.TRAIN.LEARNING_RATE)
seg_loss = monai.losses.DiceLoss(sigmoid=True, squared_pred=True, reduction='mean')
ce_loss = nn.BCEWithLogitsLoss(reduction="mean")
num_epochs = cfg.TRAIN.NUM_EPOCHS
scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)

results_name = (
    f"LORA{cfg.SAM.RANK}_"
    f"SHOTS{cfg.DATASET.NUM_SHOTS}_"
    f"NCTX{cfg.PROMPT_LEARNER.N_CTX_TEXT}_"
    f"CSC{cfg.PROMPT_LEARNER.CSC}_"
    f"CTP{cfg.PROMPT_LEARNER.CLASS_TOKEN_POSITION}"
)

# Resume functionality
resume_path = os.path.join(
            cfg.output_dir,
            cfg.DATASET.NAME,
            "trained_models",
            f"{results_name}_latest.pth")

start_epoch = 0
best_loss = float("inf")

if cfg.resume and os.path.exists(resume_path):
    checkpoint = torch.load(resume_path)
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    start_epoch = checkpoint["epoch"] + 1
    best_loss = checkpoint["best_loss"]
    print(f"Loaded checkpoint from epoch {start_epoch}, best loss: {best_loss:.4f}")

# Set device
device = "cuda" if torch.cuda.is_available() else "cpu"
# Set model to train and into the device
model.train()
model.to(device)

mse_loss = nn.MSELoss(reduction="mean")

total_loss = []
epoch_time = []

for epoch in range(start_epoch, num_epochs):
    epoch_losses = []
    epoch_start_time = time()

    for i, batch in enumerate(tqdm(train_dataloader)):
        outputs = model(batched_input=batch, multimask_output=False)
        stk_gt, stk_out = utils.stacking_batch(batch, outputs)
        stk_out = stk_out.squeeze(1)
        loss = seg_loss(stk_out, stk_gt.float().to(device)) + ce_loss(stk_out, stk_gt.float().to(device))

        optimizer.zero_grad()
        loss.backward()
        # Optimize
        optimizer.step()
        epoch_losses.append(loss.item())

    # End of epoch operations
    epoch_end_time = time()
    epoch_time.append(epoch_end_time - epoch_start_time)
    mean_epoch_loss = mean(epoch_losses)
    # Validation phase
    mean_val_loss = evaluate_validation_loss(model, val_dataloader, device, seg_loss, ce_loss)
    print(f'EPOCH: {epoch+1} | Training Loss: {mean_epoch_loss:.4f} | Validation Loss: {mean_val_loss:.4f}')

    # Save the best model based on validation loss
    if mean_val_loss < best_loss:
        print(f"New best validation loss: {best_loss:.4f} -> {mean_val_loss:.4f}")
        best_loss = mean_val_loss
        torch.save({
            "model": model.state_dict(),
            "epoch": epoch,
            "optimizer": optimizer.state_dict(),
            "best_loss": best_loss
        }, os.path.join(
            cfg.output_dir,
            cfg.DATASET.NAME,
            "trained_models",
            f"seed{cfg.seed}",
            f"{results_name}_best.pth")
        )
    # Save the latest model
    torch.save({
        "model": model.state_dict(),
        "epoch": epoch,
        "optimizer": optimizer.state_dict(),
        "best_loss": best_loss
    }, 
    os.path.join(
    cfg.output_dir,
    cfg.DATASET.NAME,
    "trained_models",
    f"seed{cfg.seed}",
    f"{results_name}_latest.pth")
    )
