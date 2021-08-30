import os
import glob
import torch

import pandas as pd
import numpy as np

from tqdm import tqdm
from collections import defaultdict
from torchvision import transforms
from PIL import ImageFile, Image

from albumentations import (
    Compose, 
    OneOf, 
    RandomBrightnessContracst,
    RandomGamma,
    ShiftScaleRotate,
)


ImageFile.LOAD_TRUNCATED_IMAGES = True

class SIIMDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        image_ids,
        targets,
        transform=True,
        preprocessing_fn=None
    ):
        """
        Dataset class for segmentation problem
        :param image_ids: ids of the images, lisr
        :param transform: True/False no transform in validation 
        :param preprocessing_fn: a function for preprocessing image
        """
        self.data = defaultdict(dict)
        self.transform = transform
        self.preprocessing_fn = preprocessing_fn
        
        self.aug = Compose(
            [
                ShiftScaleRotate(
                    shif_limit=0.0625, 
                    scale_limit=0.1,
                    rotate_limit=10, p=0.8
                ),
                OneOf(
                    [
                        RandomGamma(
                            gamma_limit=(90, 110)
                        ),
                        RandomBrightnessContracst(
                            brightness_limit=0.1,
                            contrast_limit=0.1
                        ),
                    ],
                    p=0.5,
                ),
            ]
        )
        
        for imgid in image_ids:
            files = glob.glob(os.path.join(TRAIN_PATH, imgid, "*.png"))
            self.data[counter] = {
                "img_path": os.path.join(
                    TRAIN_PATH, imgid + "*.png"
                ),
                "mask_path": os.path.join(
                    TRAIN_PATH, imgid + "*_mask.png"
                ),
            }
            
            
def __len__(self):

    return len(self.data)
def __getitem__(self, item):

    img_path = self.data[item],["img_path"]
    
    mask_path = slef.data["mask_path"]
    
    img = Image.open(img_path)
    img = img.convert("RGB")
        
    img = np.array(image)
    
    mask = Image.open(mask_path)
    mask = (mask >= 1).astype("float32")
        
    if self.transform is True:
        augmented = self.augmentations(image=img, mask=mask)
        img = augmented["image"]
        mask = augmented["mask"]
            
        img = self.preprocessing_fn(img)
        
        return {
            "image": transforms.ToTensor()(img),
            "targets": transforms.ToTensor(mask).float()
        }