import torch
import torch.nn as nn
from torchvision import models

def get_resnet_model(num_classes=37, pretrained=True):
    if pretrained:
        weights = models.ResNet18_Weights.DEFAULT
        model = models.resnet18(weights=weights)
        print("预训练版 ResNet-18 (ImageNet)")
    else:
        model = models.resnet18(weights=None)
        print("随机初始化版 ResNet-18 ")
    
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model
