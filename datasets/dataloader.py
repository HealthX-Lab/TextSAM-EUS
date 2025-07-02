import torch
import glob
import os 
from PIL import Image
import numpy as np
from torch.utils.data import Dataset
from utils.processor import Samprocessor
from os.path import basename
import random
import cv2


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

        # Handle k-shot sampling
        if num_shots != -1:
            if seed is not None:
                random.seed(seed)  # Set the random seed for reproducibility
            indices = random.sample(range(len(self.img_files)), min(num_shots, len(self.img_files)))
            self.img_files = [self.img_files[i] for i in indices]
            self.mask_files = [self.mask_files[i] for i in indices]

        self.caption = caption.lower()
        
        self.processor = processor

    def __len__(self):
        return len(self.img_files)
    
    def __getitem__(self, index: int) -> list:

        img_path = self.img_files[index]
        mask_path = self.mask_files[index]
        # get image and mask in PIL format
        image =  Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path)
        mask = mask.convert('L')
        ground_truth_mask =  np.array(mask)
        
        original_size = tuple(image.size)[::-1]
        ground_truth_mask = cv2.resize(ground_truth_mask, (original_size[1], original_size[0]), interpolation=cv2.INTER_NEAREST)
       
        if(self.type == 'binary'):
            ground_truth_mask = np.uint8(ground_truth_mask > 0)
        unique_labels = np.unique(ground_truth_mask)
        if(self.cfg.DATASET.IGNORE_BG and len(unique_labels)>= 2):
            unique_labels = unique_labels[1:].astype(np.uint8)  # Exclude background (0)
    
        
        inputs = self.processor(image, original_size)
    
        binary_masks = [np.uint8(ground_truth_mask == label) for label in unique_labels]
        inputs["ground_truth_mask"] = torch.from_numpy(np.stack(binary_masks))
        inputs["image_name"] = basename(img_path)
        inputs["mask_name"] = basename(mask_path)
        inputs["text_labels"] = torch.from_numpy(unique_labels)[None,:]
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