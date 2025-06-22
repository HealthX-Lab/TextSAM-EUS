import torch
import glob
import os 
from PIL import Image
import numpy as np

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torchvision.transforms as transforms


from utils.processor import Samprocessor
import utils.utils as utils
from utils.transforms import Compose, HorizontalFlip, VerticalFlip, RandomCrop
import yaml
from os.path import join, basename
import random
import cv2
import clip

import collections

from .mapping import DATASET_LABEL_MAPPINGS


class DatasetSegmentation(Dataset):
    """
    Dataset to process the images and masks

    Arguments:
        folder_path (str): The path of the folder containing the images
        processor (obj): Samprocessor class that helps pre processing the image, and prompt 
    
    Return:
        (dict): Dictionnary with 4 keys (image, original_size, boxes, ground_truth_mask)
            image: image pre processed to 1024x1024 size
            original_size: Original size of the image before pre processing
            boxes: bouding box after adapting the coordinates of the pre processed image
            ground_truth_mask: Ground truth mask
    """

    def __init__(self, cfg: dict, processor: Samprocessor, mode: str, num_shots: int = -1, seed: int = None, device="cuda:0", caption=""):
        super().__init__()

        self.mode = mode
        self.type = cfg.DATASET.TYPE
        self.cfg = cfg
        # train_list_file = cfg.DATASET.TRAIN_LIST
        self.device = device


        if self.mode == "train":
            self.img_files = glob.glob(os.path.join(cfg.DATASET.TRAIN_PATH,'images',"*"+cfg.DATASET.IMAGE_FORMAT))
            self.mask_files = []
            for img_path in self.img_files:
                self.mask_files.append(os.path.join(cfg.DATASET.TRAIN_PATH,'masks', os.path.basename(img_path)[:-4] + cfg.DATASET.MASK_FORMAT)) 
        
        elif self.mode == "val":
            self.img_files = glob.glob(os.path.join(cfg.DATASET.VAL_PATH,'images',"*"+cfg.DATASET.IMAGE_FORMAT))
            self.mask_files = []
            for img_path in self.img_files:
                self.mask_files.append(os.path.join(cfg.DATASET.VAL_PATH,'masks', os.path.basename(img_path)[:-4] + cfg.DATASET.MASK_FORMAT))

        else:
            self.img_files = glob.glob(os.path.join(cfg.DATASET.TEST_PATH,'images',"*"+cfg.DATASET.IMAGE_FORMAT))
            self.mask_files = []
            for img_path in self.img_files:
                self.mask_files.append(os.path.join(cfg.DATASET.TEST_PATH,'masks', os.path.basename(img_path)[:-4] + cfg.DATASET.MASK_FORMAT))

        # # Filter images based on the provided .txt file
        # if train_list_file and os.path.exists(train_list_file) and self.mode == "train":
        #     with open(train_list_file, "r") as f:
        #         allowed_images = {line.strip() for line in f}

        #     # Keep only images that are listed in the .txt file
        #     filtered_img_files = []
        #     filtered_mask_files = []
        #     for img_path, mask_path in zip(self.img_files, self.mask_files):
        #         if os.path.basename(img_path)[:-4] in allowed_images:
        #             filtered_img_files.append(img_path)
        #             filtered_mask_files.append(mask_path)

        #     self.img_files = filtered_img_files
        #     self.mask_files = filtered_mask_files

        # Handle k-shot sampling
        if num_shots != -1:
            if seed is not None:
                random.seed(seed)  # Set the random seed for reproducibility
            indices = random.sample(range(len(self.img_files)), min(num_shots, len(self.img_files)))
            self.img_files = [self.img_files[i] for i in indices]
            self.mask_files = [self.mask_files[i] for i in indices]

        self.caption = caption.lower()

        # self.caption_mapping = DATASET_LABEL_MAPPINGS[cfg.DATASET.NAME]

        # _, self.clip_preprocess = clip.load("ViT-B/16", device="cpu")
        self.clip_preprocess = transforms.Compose([
                transforms.RandomResizedCrop(size=224, scale=(0.08, 1), interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))
            ])
        
        self.processor = processor

        if self.mode == "hello":
            self.transforms = Compose(transforms = [VerticalFlip(p=0.5), 
                                                    HorizontalFlip(p=0.5)])

    def __len__(self):
        return len(self.img_files)
    
    def __getitem__(self, index: int) -> list:

        img_path = self.img_files[index]
        mask_path = self.mask_files[index]
        # get image and mask in PIL format
        image =  Image.open(img_path).convert("RGB")
        clip_image = self.clip_preprocess(image)
        mask = Image.open(mask_path)
        mask = mask.convert('L')
        ground_truth_mask =  np.array(mask)
        
        original_size = tuple(image.size)[::-1]
        ground_truth_mask = cv2.resize(ground_truth_mask, (original_size[1], original_size[0]), interpolation=cv2.INTER_NEAREST)
        if(self.mode == 'hello'):
            transformed = self.transforms(image=np.array(image), mask=ground_truth_mask)
            image, gt_mask = transformed["image"], transformed['mask']
        
        # if(self.mode in ["train","val"]):
        if(self.type == 'binary'):
            ground_truth_mask = np.uint8(ground_truth_mask > 0)
        unique_labels = np.unique(ground_truth_mask)
        if(self.cfg.DATASET.IGNORE_BG and len(unique_labels)>= 2):
            unique_labels = unique_labels[1:].astype(np.uint8)  # Exclude background (0)
        # else:
        #     unique_labels = np.unique(ground_truth_mask).astype(np.uint8)  # Exclude background (0)
            # unique_labels = np.unique(ground_truth_mask)

            # unique_label = random.choice(unique_labels.tolist())

        # get bounding box prompt
        # box = utils.get_bounding_box(ground_truth_mask)
        # inputs = self.processor(image, original_size, box)
        # inputs.["ground_truth_mask"] = torch.from_numpy(ground_truth_mask)
        # inputs["image_name"] = basename(img_path)
        # inputs["text_labels"] = torch.from_numpy(unique_labels)[None,:]

            # return inputs
        
        # else:
        #     unique_label = self.caption_mapping[self.caption]

        # unique_label = np.array([unique_label])
        # ground_truth_mask = np.uint8(ground_truth_mask == unique_label)
        # if(np.max(ground_truth_mask) == 0):
        #     return None
        # get bounding box prompt
        
        # box = utils.get_bounding_box(ground_truth_mask, self.device)
        # points, labels = utils.get_centroid_points(ground_truth_mask, unique_labels, device=self.device)
        inputs = self.processor(image, original_size)
        # binary_masks = [np.uint8(ground_truth_mask == label) for label in unique_labels] + [np.uint8(ground_truth_mask != label) for label in unique_labels]
        binary_masks = [np.uint8(ground_truth_mask == label) for label in unique_labels]
        inputs["ground_truth_mask"] = torch.from_numpy(np.stack(binary_masks))
        # inputs["points"] = points,labels
        # inputs["boxes"] = box
        # inputs["ground_truth_mask"] = torch.tensor(ground_truth_mask)
        inputs["original_size"] = original_size
        inputs["image_name"] = basename(img_path)
        inputs["mask_name"] = basename(mask_path)
        inputs["text_labels"] = torch.from_numpy(unique_labels)[None,:]
        inputs["clip_image"] = clip_image
        return inputs


    
def collate_fn(batch: torch.utils.data) -> list:
    """
    Used to get a list of dict as output when using a dataloader

    Arguments:
        batch: The batched dataset
    
    Return:
        (list): list of batched dataset so a list(dict)
    """
    return list(batch)