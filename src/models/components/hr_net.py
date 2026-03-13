import torch
import torch.nn as nn
import timm

class HRNet(nn.Module):
    def __init__(self, num_landmarks=76, pretrained=True, freeze_backbone=True):
        super().__init__()

        self.backbone=timm.create_model(
            model_name='hrnet_w18',
            pretrained=pretrained,
            features_only=True,
            out_indices=(0,)
        )

        # if freeze_backbone:
        #     for param in self.backbone.parameters():
        #         param.requires_grad = False

        in_features = self.backbone.num_features
        self.head = nn.Conv2d(kernel_size=1, in_channels=in_features, out_channels=num_landmarks)

    def forward(self, x):
        features = self.backbone(x)[0]

        heatmap = self.head(features)

        return heatmap

    def decode_heatmaps(heatmaps):
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

        preds = torch.stack([preds_x, preds_y], dim=-1)
        preds = preds / W
        
        return preds, maxvals