import torch
import numpy as np

from PIL import Image
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

class ClassificationDataset:
    """
    A general classification dataset class that you can use for all 
    kinds of Image classification problems. For example, 
    binary calssification, multi-label classification
    """
    def __init__(
        self,
        image_paths,
        targets,
        resize=None,
        augmentations=None
    ):
        """
        :param image_path: list of path to images
        :param targets: numpy array 
        :param resize: tuple, e.g. (256, 256), resize image if not None
        :param augmentations: albumentation augmentations
        """
        self.image_paths = image_paths
        self.targets = targets
        self.resize = resize
        self.augmentations = augmentations
    
def __len__(self):
    """
    Return the total number of samples in the dataset
    """
    return len(self.image_paths)
def __getitem__(self, item):
    """
    For a given "item" index, return everything we need 
    to train a given model
    """
    image = Image.convert("RGB")
    
    targets = slef.targets[item]
    
    if self.resize is not None:
        image = image.resize(
            (self.resize[1], self.resize[0]),
            resample=Image.BILINEAR
        )
        
        image = np.array(image)
        
        if self.augmentations is not None:
            augmented = self.augmentations(image=image)
            image = augmented["image"]
            
        image = np.transpose(image, (2, 0, 1)).astype(np.float32)
        
        return {
            "image": torch.tensor(image, dtype=torch.float),
            "targets": torch.tensor(targets, dtype=torch.long)
        }