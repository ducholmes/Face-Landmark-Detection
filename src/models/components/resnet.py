import torch.nn as nn
from torchvision import models

class FaceLandmarkResNet(nn.Module):
    def __init__(self, num_landmarks=98):
        super(FaceLandmarkResNet, self).__init__()
        self.backbone = models.resnet18(pretrained=True)
        
        self.backbone.fc = nn.Linear(512, num_landmarks * 2)

    def forward(self, x):
        return self.backbone(x)