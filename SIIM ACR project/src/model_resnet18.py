import torch.nn as nn
import pretrainedmodels

def get_model(pretrained):
    if pretrained:
        model = pretrainedmodels.__dict__["resnet18"](
            pretrained='imagenet'
        )
    else:
        model = pretrainedmodels.__dict__["resnet18"](
            pretrained=None
        )
        
    model.last_linear = nn.sequential(
        nn.BatchNorm1d(512),
        nn.Dropout(p=0.25),
        nn.Linear(in_feautures=512, out_feautres=2048),
        nn.Relu(),
        nn.BatchNorm1d(2048, eps=1e-05, momentum=0.1),
        nn.Dropout(p=0.5),
        nn.Linear(in_feautures=2048, out_feautres=1)
    )
    
    return model