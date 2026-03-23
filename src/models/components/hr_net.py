import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

class HRNetLandmarks(nn.Module):
    def __init__(self, num_landmarks=76, pretrained=True):
        super().__init__()

        self.backbone = timm.create_model(
            model_name='hrnet_w18',
            pretrained=pretrained,
            num_classes=0,
            features_only=True,
            global_pool=''
        )

        in_channels = 270 
        
        self.head = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, num_landmarks, kernel_size=1)
        )

    def forward(self, x):
        features = self.backbone.forward_features(x)
        
        x0 = features[0]
        target_size = x0.shape[2:]

        x1 = F.interpolate(features[1], size=target_size, mode='bilinear', align_corners=False)
        x2 = F.interpolate(features[2], size=target_size, mode='bilinear', align_corners=False)
        x3 = F.interpolate(features[3], size=target_size, mode='bilinear', align_corners=False)

        combined_features = torch.cat([x0, x1, x2, x3], dim=1)

        heatmap = self.head(combined_features)

        return heatmap

    @staticmethod
    def decode_heatmaps(heatmaps, stride=4):
        B, C, H, W = heatmaps.shape
        heatmaps_reshaped = heatmaps.reshape(B, C, -1)
        
        maxvals, idx = torch.max(heatmaps_reshaped, dim=2)
        maxvals = maxvals.unsqueeze(-1)
        
        preds_x = (idx % W).float()
        preds_y = (idx // W).float()
        
        for b in range(B):
            for c in range(C):
                hm = heatmaps[b, c]
                px = int(preds_x[b, c].item())
                py = int(preds_y[b, c].item())
                
                if 0 < px < W - 1 and 0 < py < H - 1:
                    diff_x = hm[py, px + 1] - hm[py, px - 1]
                    diff_y = hm[py + 1, px] - hm[py - 1, px]
                    
                    preds_x[b, c] += torch.sign(diff_x) * 0.25
                    preds_y[b, c] += torch.sign(diff_y) * 0.25

        preds_x = preds_x / W
        preds_y = preds_y / H
        preds = torch.stack([preds_x, preds_y], dim=-1)
        
        return preds, maxvals