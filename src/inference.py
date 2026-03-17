import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2

from src.models.facial_landmark_module import FacialLandmarkModule
from src.models.components.hr_net import HRNetLandmarks

torch.serialization.add_safe_globals([HRNetLandmarks])

def predict_and_draw(image_path, checkpoint_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    raw_image = Image.open(image_path).convert("RGB")
    image_np = np.array(raw_image)
    
    preprocess = A.Compose([
        A.LongestMaxSize(max_size=256),
        A.PadIfNeeded(min_height=256, min_width=256, border_mode=0, value=(0,0,0)),
        A.Normalize(),
        ToTensorV2()
    ])
    
    vis_transform = A.Compose([
        A.LongestMaxSize(max_size=256),
        A.PadIfNeeded(min_height=256, min_width=256, border_mode=0, value=(0,0,0)),
    ])

    input_tensor = preprocess(image=image_np)["image"].unsqueeze(0).to(device)
    display_image = vis_transform(image=image_np)["image"]
    
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = checkpoint["state_dict"]
    new_state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
    
    model = FacialLandmarkModule(HRNetLandmarks(num_landmarks=98))
    model.load_state_dict(new_state_dict)
    model.to(device)
    model.eval()

    with torch.no_grad():
        outputs = model(input_tensor)
        landmarks_norm, _  = model.decode_heatmaps(outputs)
        
    landmarks = landmarks_norm[0].cpu().numpy()
    landmarks = landmarks * 256 
    
    plt.imshow(display_image)
    plt.scatter(landmarks[:, 0], landmarks[:, 1], s=5, c='hotpink', edgecolors='white', marker='o')
    
    plt.title(f"Face Landmark Detection - {image_path.split('/')[-1]}")
    plt.axis('on')
    plt.savefig('images/Niggaa_plot.png')

TEST_IMAGE_PATH = "images/Nigga.jpg" 
CKPT_PATH = "logs/train/last.ckpt"

def decode_heatmaps(self, heatmaps):
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

if __name__ == "__main__":
    predict_and_draw(TEST_IMAGE_PATH, CKPT_PATH)