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

    def __init__(self, cfg: dict, processor: Samprocessor, mode: str, num_shots: int = -1, seed: int = None, caption=""):
        super().__init__()

        self.mode = mode
        self.type = cfg.DATASET.TYPE
        self.cfg = cfg
        # train_list_file = cfg.DATASET.TRAIN_LIST


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

        # Handle k-shot sampling
        if num_shots != -1:
            if seed is not None:
                random.seed(seed)  # Set the random seed for reproducibility
            indices = random.sample(range(len(self.img_files)), min(num_shots, len(self.img_files)))
            self.img_files = [self.img_files[i] for i in indices]
            self.mask_files = [self.mask_files[i] for i in indices]

        self.caption = caption.lower()

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
        image = Image.open(img_path).convert("RGB")
        clip_image = self.clip_preprocess(image)
        mask = Image.open(mask_path)
        mask = mask.convert('L')
        ground_truth_mask = np.array(mask)
        
        original_size = tuple(image.size)[::-1]
        ground_truth_mask = cv2.resize(ground_truth_mask, (original_size[1], original_size[0]), interpolation=cv2.INTER_NEAREST)
        
        if self.mode == 'hello':
            transformed = self.transforms(image=np.array(image), mask=ground_truth_mask)
            image, ground_truth_mask = transformed["image"], transformed['mask']
        
        if self.type == 'binary':
            ground_truth_mask = np.uint8(ground_truth_mask > 0)

        unique_labels = np.array([1])

        inputs = self.processor(image, original_size)
        binary_masks = [np.uint8(ground_truth_mask == label) for label in unique_labels]
        inputs["ground_truth_mask"] = torch.from_numpy(np.stack(binary_masks))
        inputs["image_name"] = basename(img_path)
        inputs["mask_name"] = basename(mask_path)
        inputs["text_labels"] = torch.from_numpy(unique_labels)[None, :]
        inputs["clip_image"] = clip_image

        # Assign 0 if filename starts with 'H' (Healthy), 1 if 'C' (Condition/Unhealthy)
        filename = basename(img_path)
        if filename.startswith('H'):
            inputs["label"] = torch.tensor(0, dtype=torch.long)
        elif filename.startswith('C'):
            inputs["label"] = torch.tensor(1, dtype=torch.long)
        else:
            raise ValueError(f"Filename '{filename}' does not start with 'H' or 'C'.")

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