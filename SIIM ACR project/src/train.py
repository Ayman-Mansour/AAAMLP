import os

import numpy as np
import pandas as pd

import albumentations 
import torch

from sklearn import metrics
from sklearn.model_selection import train_test_split

import dataset
import engine
from model_alexnet import get_model

if __name__ == "__main__":
    data_path = "../input/"
    
    device = "cpu"
    
    epochs = 10
    
    df = pd.read_csv(os.path.join(data_path, "train.csv"))
    
    images = df.ImageId.values.tolist()
    
    images = [
        os.path.join(data_path, "siim_png/", "train_png", i + ".png") for i in images
    ]
    
    targets = df.target.values
    
    model = get_model(pretrained=True)
    
    model.to(device)
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    
    aug = albumentation.Compose(
        [
            albumentation.Normalize(
                men, std, max_pixel_value=255.0, always_apply=True
            )
        ]
    )
    
    train_images, valid_images, train_targets, valid_targets = train_test_split(
        images, targets, stratify=targets, random_state=42
    )
    
    train_dataset = dataset.ClassificationDataset(
        image_paths=train_images, 
        targets=train_targets,
        resize=(277, 277),
        augmentations=aug
    )
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=16, shuffle=True, num_workers=4
    )
    
    valid_dataset = dataset.ClassificationDataset(
       image_paths=valid_images, 
        targets=valid_targets,
        resize=(277, 277),
        augmentations=aug 
    )
    
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=16, shuffle=True, num_workers=4
    )
    
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
    
    for epoch in range(epochs):
        engine.train(train_loader, model, optimizer, device=device)
        predictions, valid_targets = engine.evaluate(
            valid_loader, model, device=device
        )
        
        roc_auc = metrics.roc_auc_score(valid_targets, predictions)
        
        print(
            f"Epoch= {epoch}, Valid ROC AUC= {roc_auc}"
        )