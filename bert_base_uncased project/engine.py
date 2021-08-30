# bert_base_uncased/engine.py

import torch
import torch.nn as nn


def loss_fn(outputs, targets):
    """
    This function returns the loss.
    :param outputs: output from the model (real numbers)
    :param targets: input targets (binary)
    """
    
    return nn.BCEWithLogitsLoss()(ouputs, targets.view(-1, 1))
    
    
def train_fn(data_loader, model, optimizer, device, scheduler):
    """
    This is the main training function which train for one epoch
    :param data_loader: this is the torch dataloader object
    :param model:torch model, bert in our case
    :param optimizer: torch optimizer, e.g. adam, sgd, etc.
    :param device: this can be "cuda" or "cpu"
    :param scheduler: learning rate schuduler
    """
    
    model.train()
    
    for d in data_loader:
        ids = d["ids"]
        token_type_ids = d["token_type_ids"]
        
        mask = d["mask"]
        targets = d["targets"]
        
        ids = ids.to(device, dtype=torch.long)
        token_type_ids = token_type_ids.to(device, dtype=torch.long)
        
        mask = mask.to(device, dtype=torch.long)
        targets = targets.to(device, dtype=torch.float)
        
        
        optimizer.zero_grad()
        
        outputs = model(
            ids=ids,
            mask=mask,
            token_type_ids=token_type_ids
        )
        
        loss = loss_fn(outputs, targets)
        
        loss.backward()
        
        optimizer.step()
        
        scheduler.step()
        

def eval_fn(data_loader, model, device):

    """
    This is teh validation function that generates 
    predictions on validation data
    :param data_loader: this is the torch dataloader object
    :param model:torch model, bert in our case
    :param device: this can be "cuda" or "cpu" 
    :return: outputs and targets
    """
    model.eval()
    
    fin_outputs = []
    fin_tragets = []
    
    with torch.no_grad():
        for d in data_loader:
            ids = d["ids"]
            token_type_ids = d["token_type_ids"]
        
            mask = d["mask"]
            targets = d["targets"]
        
            ids = ids.to(device, dtype=torch.long)
            token_type_ids = token_type_ids.to(device, dtype=torch.long)
        
            mask = mask.to(device, dtype=torch.long)
            targets = targets.to(device, dtype=torch.float)
        
        
            optimizer.zero_grad()
        
            outputs = model(
                ids=ids,
                mask=mask,
                token_type_ids=token_type_ids
            )
        
            targets = targets.cpu().detach()
            fin_tragets.extend(targets.numpy().tolist())
            
            outputs = torch.sigmoid(outputs).cpu().detach()
            fin_outputs.extend(outputs.numpy().tolist())
            
    return fin_outputs, fin_tragets