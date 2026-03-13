import torch
import torch.nn as nn
import timm

class HRNet(nn.Module):
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
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.act1(x)
        x = self.backbone.conv2(x)
        x = self.backbone.bn2(x)
        x = self.backbone.act2(x)
        x = self.backbone.layer1(x)
        
        x = [x]
        
        x = self.backbone.stage2(x)
        x = self.backbone.stage3(x)
        x = self.backbone.stage4(x)
        
        high_res_features = x[0]
        
        heatmap = self.head(high_res_features)

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