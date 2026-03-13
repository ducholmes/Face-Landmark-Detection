import torch
import torch.nn as nn
import timm

class HRNetLandmarks(nn.Module):
    def __init__(self, num_landmarks=76, pretrained=True):
        super().__init__()

        self.backbone = timm.create_model(
            model_name='hrnet_w18',
            pretrained=pretrained,
            num_classes=0 
        )

        in_channels = 18 
        
        self.head = nn.Conv2d(
            in_channels=in_channels, 
            out_channels=num_landmarks,
            kernel_size=1
        )

    def forward(self, x):
        high_res_features = []

        def hook(module, input, output):
            high_res_features.append(output[0])

        handle = self.backbone.stage4.register_forward_hook(hook)

        _ = self.backbone(x)

        handle.remove()

        heatmap = self.head(high_res_features[0])

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