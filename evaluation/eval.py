import numpy as np
import cv2
import os
from collections import OrderedDict
import pandas as pd
from SurfaceDice import compute_surface_distances, compute_surface_dice_at_tolerance, compute_dice_coefficient
from tqdm import tqdm
import argparse
import yaml
from dassl.utils import setup_logger, set_random_seed, collect_env_info
from dassl.config import get_cfg_default
from dassl.engine import build_trainer

parser = argparse.ArgumentParser()
parser.add_argument(
    "--config-file",
    required=True,
    type=str,
    help="Path to config file",
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

def print_args(args, cfg):
    print("***************")
    print("** Arguments **")
    print("***************")
    optkeys = list(args.__dict__.keys())
    optkeys.sort()
    for key in optkeys:
        print("{}: {}".format(key, args.__dict__[key]))
    print("************")
    print("** Config **")
    print("************")
    print(cfg)


def reset_cfg(cfg, args):

    if args.output_dir:
        cfg.OUTPUT_DIR = args.output_dir

    if args.seed:
        cfg.SEED = args.seed


def extend_cfg(cfg):
    """
    Add new config variables.

    E.g.
        from yacs.config import CfgNode as CN
        cfg.TRAINER.MY_MODEL = CN()
        cfg.TRAINER.MY_MODEL.PARAM_A = 1.
        cfg.TRAINER.MY_MODEL.PARAM_B = 0.5
        cfg.TRAINER.MY_MODEL.PARAM_C = False
    """
    from yacs.config import CfgNode as CN

    cfg.DATASET = CN()
    cfg.DATASET.NAME = "Lung_Xray"
    cfg.DATASET.TRAIN_PATH = "./dataset/lung_Xray/train"
    cfg.DATASET.VAL_PATH = "./dataset/lung_Xray/val"
    cfg.DATASET.TEST_PATH = "./dataset/lung_Xray/test"
    cfg.DATASET.IMAGE_FORMAT = ".png"
    cfg.DATASET.MASK_FORMAT = ".png"
    cfg.DATASET.MASK_LABEL = ""
    cfg.DATASET.NUM_SHOTS = 16
    cfg.DATASET.TYPE = "binary"

    cfg.SAM = CN()
    cfg.SAM.MODEL = "vit_b"
    cfg.SAM.CHECKPOINT = "segment_anything/checkpoints"
    cfg.SAM.RANK = 64

    cfg.TRAIN = CN()
    cfg.TRAIN.BATCH_SIZE = 1
    cfg.TRAIN.NUM_EPOCHS = 1
    cfg.TRAIN.LEARNING_RATE = 0.0001

    cfg.TEST = CN()
    cfg.TEST.USE_LATEST = False

    cfg.PROMPT_LEARNER = CN()
    cfg.PROMPT_LEARNER.MODEL = "clip"  # number of context vectors
    cfg.PROMPT_LEARNER.MODALITY = "text"
    cfg.PROMPT_LEARNER.N_CTX_TEXT = 4
    cfg.PROMPT_LEARNER.N_CTX_VISION = 0
    cfg.PROMPT_LEARNER.PROMPT_DEPTH_VISION = 0
    cfg.PROMPT_LEARNER.PROMPT_DEPTH_TEXT = 12
    cfg.PROMPT_LEARNER.FUSE_SAM = True
    cfg.PROMPT_LEARNER.FUSE_CLIP = True
    cfg.PROMPT_LEARNER.FUSE_TYPE = "none" #["concat", "add", "attention"]
    cfg.PROMPT_LEARNER.CSC = False  # class-specific context
    cfg.PROMPT_LEARNER.CTX_INIT = ""  # initialization words
    cfg.PROMPT_LEARNER.PREC = "fp32"  # fp16, fp32, amp
    cfg.PROMPT_LEARNER.CLASS_TOKEN_POSITION = "end"  # 'middle' or 'end' or 'front'
    cfg.PROMPT_LEARNER.BACKBONE = "ViT-B/16"
    cfg.PROMPT_LEARNER.CLASSNAMES = ["lungs"]
    cfg.PROMPT_LEARNER.TEXT_LABEL = "lungs"


def setup_cfg(args):
    cfg = get_cfg_default()
    extend_cfg(cfg)

    if args.config_file:
        cfg.merge_from_file(args.config_file)

    reset_cfg(cfg, args)

    cfg.merge_from_list(args.opts)

    cfg.freeze()

    return cfg

cfg = setup_cfg(args)
# if cfg.SEED >= 0:
#     print("Setting fixed seed: {}".format(cfg.SEED))
#     set_random_seed(cfg.SEED)
setup_logger(os.path.join(cfg.OUTPUT_DIR, cfg.DATASET.NAME, "test_results", f"seed{cfg.SEED}"))

join = os.path.join
basename = os.path.basename

results_name = (
    f"LORA{cfg.SAM.RANK}_"
    f"SHOTS{cfg.DATASET.NUM_SHOTS}_"
    f"NCTX{cfg.PROMPT_LEARNER.N_CTX_TEXT}_"
    f"CSC{cfg.PROMPT_LEARNER.CSC}_"
    f"CTP{cfg.PROMPT_LEARNER.CLASS_TOKEN_POSITION}"
)

os.makedirs(os.path.join(cfg.OUTPUT_DIR, cfg.DATASET.NAME, "test_results", f"seed{cfg.SEED}"), exist_ok=True)

gt_path = os.path.join(cfg.DATASET.TEST_PATH, "masks")
seg_path = os.path.join(cfg.OUTPUT_DIR, cfg.DATASET.NAME, "seg_results", f"seed{cfg.SEED}")

classnames = cfg.PROMPT_LEARNER.CLASSNAMES

for segClass in classnames:

    class_seg_path = os.path.join(seg_path, segClass, results_name)

    # Get list of filenames
    filenames = os.listdir(class_seg_path)
    filenames = [x for x in filenames if x.endswith('.png') or x.endswith('.jpg') or x.endswith('.jpeg')]
    filenames = [x for x in filenames if os.path.exists(join(class_seg_path, x))]
    filenames.sort()

    save_path = os.path.join(cfg.OUTPUT_DIR, cfg.DATASET.NAME, "test_results", f"seed{cfg.SEED}", segClass + "_" + results_name + ".csv")

    # Initialize metrics dictionary
    seg_metrics = OrderedDict(
        Name = list(),
        DSC = list(),
        IoU = list(),
        NSD = list(),
    )

    # Compute metrics for each file
    for name in tqdm(filenames):
        seg_metrics['Name'].append(name)
        gt_mask = cv2.imread(join(gt_path, name), cv2.IMREAD_GRAYSCALE)
        seg_mask = cv2.imread(join(class_seg_path, name), cv2.IMREAD_GRAYSCALE)
        seg_mask = cv2.resize(seg_mask, (gt_mask.shape[1], gt_mask.shape[0]), interpolation=cv2.INTER_NEAREST)
        gt_mask = cv2.threshold(gt_mask, 200, 255, cv2.THRESH_BINARY)[1]
        seg_mask = cv2.threshold(seg_mask, 200, 255, cv2.THRESH_BINARY)[1]
        gt_data = np.uint8(gt_mask)
        seg_data = np.uint8(seg_mask)

        gt_labels = np.unique(gt_data)[1:]
        seg_labels = np.unique(seg_data)[1:]
        labels = np.union1d(gt_labels, seg_labels)

        assert len(labels) > 0, 'Ground truth mask max: {}'.format(gt_data.max())

        DSC_arr = []
        IoU_arr = []
        NSD_arr = []
        for i in labels:
            if np.sum(gt_data==i)==0 and np.sum(seg_data==i)==0:
                DSC_i = 1
                IoU_i = 1
                NSD_i = 1
            elif np.sum(gt_data==i)==0 and np.sum(seg_data==i)>0:
                DSC_i = 0
                IoU_i = 0
                NSD_i = 0
            else:
                i_gt, i_seg = gt_data == i, seg_data == i
                
                # Compute Dice coefficient
                DSC_i = compute_dice_coefficient(i_gt, i_seg)

                # Compute IoU
                intersection = np.logical_and(i_gt, i_seg)
                union = np.logical_or(i_gt, i_seg)
                IoU_i = np.sum(intersection) / np.sum(union)

                # Compute NSD
                case_spacing = [1, 1, 1]
                surface_distances = compute_surface_distances(i_gt[..., None], i_seg[..., None], case_spacing)
                NSD_i = compute_surface_dice_at_tolerance(surface_distances, 2)

            DSC_arr.append(DSC_i)
            IoU_arr.append(IoU_i)
            NSD_arr.append(NSD_i)

        DSC = np.mean(DSC_arr)
        IoU = np.mean(IoU_arr)
        NSD = np.mean(NSD_arr)
        seg_metrics['IoU'].append(round(IoU, 4))
        seg_metrics['DSC'].append(round(DSC, 4))
        seg_metrics['NSD'].append(round(NSD, 4))

    # Save metrics to CSV
    dataframe = pd.DataFrame(seg_metrics)
    dataframe.to_csv(save_path, index=False)

    # Calculate and print average and std deviation for metrics
    case_avg_DSC = dataframe['DSC'].mean()
    case_avg_IoU = dataframe['IoU'].mean()
    case_avg_NSD = dataframe['NSD'].mean()
    case_std_DSC = dataframe['DSC'].std()
    case_std_IoU = dataframe['IoU'].std()
    case_std_NSD = dataframe['NSD'].std()

    print(20 * '>')
    print(f'Average DSC for {basename(class_seg_path)}: {case_avg_DSC}')
    print(f'Standard deviation DSC for {basename(class_seg_path)}: {case_std_DSC}')
    print(f'Average IoU for {basename(class_seg_path)}: {case_avg_IoU}')
    print(f'Standard deviation IoU for {basename(class_seg_path)}: {case_std_IoU}')
    print(f'Average NSD for {basename(class_seg_path)}: {case_avg_NSD}')
    print(f'Standard deviation NSD for {basename(class_seg_path)}: {case_std_NSD}')
    print(20 * '<')
