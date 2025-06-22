import numpy as np
import matplotlib.pyplot as plt
import PIL
from PIL import Image
from torchvision import datasets, transforms
import torchvision.transforms.functional as F
import torch
from torch.nn.functional import pad
import yaml
import os
from scipy.ndimage import center_of_mass
import numpy as np
import torch


def show_mask(mask: np.array, ax, random_color=False):
    """
    Plot the mask

    Arguments:
        mask: Array of the binary mask (or float)
    """
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[:2]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def plot_image_mask(image: PIL.Image, mask: PIL.Image, filename: str):
    """
    Plot the image and the mask superposed

    Arguments:
        image: PIL original image
        mask: PIL original binary mask
    """
    fig, axes = plt.subplots()
    axes.imshow(np.array(image))
    ground_truth_seg = np.array(mask)
    show_mask(ground_truth_seg, axes)
    axes.title.set_text(f"{filename} predicted mask")
    axes.axis("off")
    plt.savefig("./plots/" + filename + ".jpg")
    plt.close()
    

def plot_image_mask_dataset(dataset: torch.utils.data.Dataset, idx: int):
    """
    Take an image from the dataset and plot it

    Arguments:
        dataset: Dataset class loaded with our images
        idx: Index of the data we want
    """
    image_path = dataset.img_files[idx]
    mask_path = dataset.mask_files[idx]
    image = Image.open(image_path)
    mask = Image.open(mask_path)
    mask = mask.convert('1')
    plot_image_mask(image, mask)


def get_bounding_box(ground_truth_map: np.array, device="cuda:0") -> list:
  """
  Get the bounding box of the image with the ground truth mask
  
    Arguments:
        ground_truth_map: Take ground truth mask in array format

    Return:
        bbox: Bounding box of the mask [X, Y, X, Y]

  """
  # get bounding box from mask
  ground_truth_map = ground_truth_map.detach().cpu().numpy()
  idx = np.where(ground_truth_map > 0)
  x_indices = idx[1]
  y_indices = idx[0]
  x_min, x_max = np.min(x_indices), np.max(x_indices)
  y_min, y_max = np.min(y_indices), np.max(y_indices)
  # add perturbation to bounding box coordinates
  H, W = ground_truth_map.shape
  x_min = max(0, x_min - np.random.randint(0, 20))
  x_max = min(W, x_max + np.random.randint(0, 20))
  y_min = max(0, y_min - np.random.randint(0, 20))
  y_max = min(H, y_max + np.random.randint(0, 20))
  bbox = torch.tensor([x_min, y_min, x_max, y_max], dtype=torch.int, device=device).unsqueeze(0)
  return bbox

def get_extreme_points(ground_truth_mask, unique_labels, device="cuda"):
    ground_truth_mask = ground_truth_mask.detach().cpu().numpy()
    unique_labels = unique_labels.detach().cpu().numpy()
    def compute_extremes(mask):
        indices = np.argwhere(mask)
        if len(indices) == 0:
            return []
        top = indices[np.argmin(indices[:, 0])]
        bottom = indices[np.argmax(indices[:, 0])]
        left = indices[np.argmin(indices[:, 1])]
        right = indices[np.argmax(indices[:, 1])]
        return [left[::-1], right[::-1], top[::-1], bottom[::-1]]  # x, y order

    points = []
    point_labels = []
    for label in unique_labels:
        mask = ground_truth_mask == label
        extremes = compute_extremes(mask)
        if extremes:
            points.extend(extremes)
            point_labels.extend([label] * len(extremes))
    return torch.tensor(points, dtype=torch.float, device=device), torch.tensor(point_labels, dtype=torch.long, device=device)

def get_centroid_points(ground_truth_mask, unique_labels, device='cuda'):
    points = []
    point_labels = []
    for label in unique_labels:
        binary_mask = ground_truth_mask == label
        binary_mask = binary_mask.detach().cpu().numpy()
        if np.any(binary_mask):
            cy, cx = center_of_mass(binary_mask)
            points.append([cx, cy])
            point_labels.append(label)
    return torch.tensor(points, dtype=torch.float, device=device), torch.tensor(point_labels, dtype=torch.long, device=device)

def get_random_points(ground_truth_mask, unique_labels, num_points_per_label=1, device="cuda"):
    points = []
    point_labels = []

    ground_truth_mask = ground_truth_mask.detach().cpu().numpy()
    unique_labels = unique_labels.detach().cpu().numpy()

    for label in unique_labels:
        indices = np.argwhere(ground_truth_mask == label)
        if len(indices) > 0:
            # Randomly choose without replacement if enough points, else sample with replacement
            if len(indices) >= num_points_per_label:
                chosen_indices = indices[np.random.choice(len(indices), size=num_points_per_label, replace=False)]
            else:
                chosen_indices = indices[np.random.choice(len(indices), size=num_points_per_label, replace=True)]
            
            for y, x in chosen_indices:
                points.append([x, y])
                point_labels.append(label)

    return torch.tensor(points, dtype=torch.float, device=device), torch.tensor(point_labels, dtype=torch.long, device=device)


def stacking_batch(batch, outputs):
    """
    Given the batch and outputs of SAM, stacks the tensors to compute the loss. We stack by adding another dimension.

    Arguments:
        batch(list(dict)): List of dict with the keys given in the dataset file
        outputs: list(dict): List of dict that are the outputs of SAM
    
    Return: 
        stk_gt: Stacked tensor of the ground truth masks in the batch. Shape: [batch_size, H, W] -> We will need to add the channels dimension (dim=1)
        stk_out: Stacked tensor of logits mask outputed by SAM. Shape: [batch_size, 1, 1, H, W] -> We will need to remove the extra dimension (dim=1) needed by SAM 
    """
    stk_gt = torch.stack([b["ground_truth_mask"] for b in batch], dim=0)
    stk_out = torch.stack([out["low_res_logits"] for out in outputs], dim=0)
        
    return stk_gt, stk_out



class CfgNode(dict):
    """
    CfgNode represents an internal node in the configuration tree. It's a simple
    dict-like container that allows for attribute-based access to keys.
    """

    def __init__(self, init_dict=None, key_list=None, new_allowed=False):
        # Recursively convert nested dictionaries in init_dict into CfgNodes
        init_dict = {} if init_dict is None else init_dict
        key_list = [] if key_list is None else key_list
        for k, v in init_dict.items():
            if type(v) is dict:
                # Convert dict to CfgNode
                init_dict[k] = CfgNode(v, key_list=key_list + [k])
        super(CfgNode, self).__init__(init_dict)

    def __getattr__(self, name):
        if name in self:
            return self[name]
        else:
            raise AttributeError(name)

    def __setattr__(self, name, value):
        self[name] = value

    def __str__(self):
        def _indent(s_, num_spaces):
            s = s_.split("\n")
            if len(s) == 1:
                return s_
            first = s.pop(0)
            s = [(num_spaces * " ") + line for line in s]
            s = "\n".join(s)
            s = first + "\n" + s
            return s

        r = ""
        s = []
        for k, v in sorted(self.items()):
            seperator = "\n" if isinstance(v, CfgNode) else " "
            attr_str = "{}:{}{}".format(str(k), seperator, str(v))
            attr_str = _indent(attr_str, 2)
            s.append(attr_str)
        r += "\n".join(s)
        return r

    def __repr__(self):
        return "{}({})".format(self.__class__.__name__, super(CfgNode, self).__repr__())


def load_cfg_from_cfg_file(file: str):
    cfg = {}
    assert os.path.isfile(file) and file.endswith('.yaml'), \
        '{} is not a yaml file'.format(file)

    with open(file, 'r') as f:
        cfg_from_file = yaml.safe_load(f)

    for key in cfg_from_file:

        # for k, v in cfg_from_file[key].items():
        cfg[key] = cfg_from_file[key]

    cfg = CfgNode(cfg)

    return cfg