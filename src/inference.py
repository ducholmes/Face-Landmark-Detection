import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2

from models.facial_landmark_module import FacialLandmarkModule
from models.components.hr_net import HRNetLandmarks

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
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint["state_dict"]
    new_state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
    
    model = FacialLandmarkModule(HRNetLandmarks())
    model.load_state_dict(new_state_dict)
    model.to(device)
    model.eval()

    with torch.no_grad():
        outputs = model(input_tensor)
        
    landmarks = outputs.cpu().numpy().squeeze() 
    landmarks = landmarks * 256 
    
    plt.figure(figsize=(10, 10))
    plt.imshow(display_image)
    plt.scatter(landmarks[:, 0], landmarks[:, 1], s=5, c='hotpink', edgecolors='white', marker='o')
    
    plt.title(f"Face Landmark Detection - {image_path.split('/')[-1]}")
    plt.axis('on')
    plt.savefig('images/VHD_plot.png')

TEST_IMAGE_PATH = "images/VHD.png" 
CKPT_PATH = "logs/train/last.ckpt"

if __name__ == "__main__":
    predict_and_draw(TEST_IMAGE_PATH, CKPT_PATH)