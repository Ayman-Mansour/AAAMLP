# src/engine.py

import torch
import torch.nn as nn

def train(data_loader, model, optimizer, device):
    """
    This is the main training function that trains model 
    for one epoch
    :param data_loader: this is the torch dataloader
    :param model: model (lstm model)
    :param optimizer: torch optimizer, e.g. adam, sgd, etc.
    :param deviceL: this can be "cuda" or "cpu"
    """
    
    model.train()
    
    for data in data_loader:
        reviews = data["review"]
        targets = data["target"]
        
        reviews.to(device, dtype=torch.long)
        targets.to(device, dtype=torch.float)
        
        optimizer.zero_grad()
        
        predictions = model(reviews)
        
        loss = nn.BCEWithLogitsLoss(
            predictions,
            targets.view(-1, 1)
        )
        
        loss.backward()
        
        optimizer.step()
        

def evaluate(data_loader, model, device):
    final_predictions = []
    final_tragets = []
    
    model.eval()
    
    with torch.no_grad():
        reviews = data["review"]
        targets = data["target"]
        
        reviews.to(device, dtype=torch.long)
        targets.to(device, dtype=torch.float)
        
        predictions = model(reviews)
        
        predictions = predictions.cpu().numpy().tolist()
        targets = data["targets"].cpu().numpy().tolist()
        final_predictions.extend(predictions)
        final_tragets.extend(targets)
        
    return final_predictions, final_tragets