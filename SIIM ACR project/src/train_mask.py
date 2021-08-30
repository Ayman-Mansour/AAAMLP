#src/train_mask.py

import os
import sys
import torch

import numpy as np
import pandas as pd
import segmentation_model_pytorch as smp
import torch.nn as nn
import optim as optim

from apex import amp
from collections import OrderedDict
from sklearn import model_selection
from tqdm import tqdm
from torch.optim import lr_scheduler

from dataset_masked import SIIMDataset

TRAINING_CSV = "../input/train.csv"

TRAINING_BATCH_SIZE = 16
TEST_BATCH_SIZE = 4

EPOCHS = 10

ENCODER = "resnet18"
ENCODER_WEIGHTS = "imagenet"

DEVICE = cuda

def train(dataset, data_loader, model, criterion, optimizer):
    """
    Training function that trains for one epoch 
    :param dataset:dataset class (SIIMDataset)
    :param data_loader: torch dataset loader
    :param model: model
    :param criterion: loss function 
    :param optimizer: adam, sgd, etc.
    """
    model.train()
    
    num_batches = int(len(dataset) / data_loader.batch_size)
    
    tk0 = tqdm(data_loader, total=num_batches)
    
    for d in tk0:
        inputs = d["image"]
        targets = d["mask"]
        
        inputs = inputs.to(DEVICE, dtype=torch.float)
        targets = targets.to(DEVICE, dtype=torch.float)
        
        optimizer.zero_grad()
        
        outputs = model(inputs)
        
        loss = criterion(outputs, targets)
        
        with amp.scale_loss(loss, optimizer) as dcaled_loss:
            scale_loss.backward()
            
        optimizer.step()
    tk0.close()
    
def evaluate(dataset, data_loader, model):
    """
    Evluation function to calculate loss on validation
    :param dataset:dataset class (SIIMDataset)
    :param data_loader: torch dataset loader
    :param model: model
    """
    
    model.eval()
    
    num_batches = int(len(dataset) / data_loader.batch_size)
    
    tk0 = tqdm(data_loader, total=num_batches)
    
    with torch.no_grad():
        for d in tk0:
            inputs = d["image"]
            targets = d["mask"]

            inputs = inputs.to(DEVICE, dtype=torch.float)
            targets = targets.to(DEVICE, dtype=torch.float)

            optimizer.zero_grad()

            outputs = model(inputs)

            loss = criterion(outputs, targets)

            final_loss += loss
    tk0.close()
    
    return final_loss / num_batches

if __name__ == "__main__":

    df = pd.read_csv(TRAINING_CSV)
    
    df_train, df_valid = model_selection.train_test_split(
        df, random_state=42, test_size=0.1
    )
    
    training_images = df_train.image.image_id.values
    validation_images = df_valid.image.image_id.values
    
    model = smp.Unet(
        encoder_name=ENCODER,
        encoder_weights=ENCODER_WEIGHTS,
        classes=1, 
        activation=None
    )
    
    prep_fn = smp.encoder.getpreprocessing_fn(
        ENCODER,
        ENCODER_WEIGHTS
    )
    
    model.to(DEVICE)
    
    train_dataset = SIIMDataset(
        training_images, 
        transform=True,
        preprocessing_fn=prep_fn
    )
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset, 
        batch_size=TRAINING_BATCH_SIZE,
        shuffle=True, 
        num_workers=12
    )
    
    valid_dataset = SIIMDataset(
        validation_images,
        transform=False,
        preprocessing_fn=prep_fn
    )
    
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=TEST_BATCH_SIZE,
        shuffle=True
    )
    
    # NOTE: define the criterion here# this is left as an excercise
    # code won't work without defining this
    criterion = ""
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    scheduler = lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=3, verbose=True
    )
    
    model, optimizer = amp.initialize(
        model, optimizer, opt_level="01", verbosity=0
    )
    
    if torch.cuda.device_count() > 1:
        print(f"Let's use {torch.cuda.device_count()} GPUs!")
        model = nn.DataParallel(model)
        
    print(f"Training v\batch size: {TRAINING_BATCH_SIZE}")
    print(f"Test batch size: {TEST_BATCH_SIZE}")
    print(f"Epochs: {EPOCHS}")
#     print(f"Image size: {IMAGE_SIZE}")
    print(f"Number of training images: {len(train_dataset)}")
    print(f"Number of validation images: {len(valid_dataset)}")
    print(f"Encoder: {ENCODER}")
    
    for epoch in range(EPOCHS):
        print(f"Training Epoch: {epoch}")
        
        train(
            train_dataset, 
            train_loader,
            model,
            criterion,
            optimizer
        )
        
        print(f"Validation Epoch: {epoch}")
        
        val_log = evaluate(
            valid_dataset, 
            valid_loader,
            model
        )
        
        scheduler.step(val_log["loss"])
        print("\n")