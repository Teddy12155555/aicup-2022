import torch
from torch import nn
from torchsummary import summary
from efficientnet_pytorch import EfficientNet

class EfNetModel(nn.Module):
    def __init__(self, num_classes=33, dropout=0.2, pretrained_path=None):
        super().__init__()
        self.model = EfficientNet.from_pretrained('efficientnet-b4', num_classes=num_classes)
        # set dropout
        self.model._dropout = nn.Dropout(dropout)
        
        for param in self.model.parameters():
            param.requires_grad = True
            
    def forward(self, x):
        x = self.model(x)
        return x

class TownMapping(nn.Module):
    def __init__(self):
        super().__init__()
