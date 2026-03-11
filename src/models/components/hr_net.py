import torch.nn as nn
import timm

class HRNet(nn.Module):
    def __init__(self, num_landmarks=76, pretrained=True, freeze_backbone=True):
        super().__init__()

        self.backbone=timm.create_model(
            model_name='hrnet_w18',
            pretrained=pretrained,
            num_classes = 0,
            global_pool = 'avg'
        )

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        in_features = self.backbone.num_features
        self.head = nn.Linear(in_features, num_landmarks*2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        features = self.backbone(x)
        raw_coords = self.head(features)
        rel_coords = self.sigmoid(raw_coords)

        return rel_coords