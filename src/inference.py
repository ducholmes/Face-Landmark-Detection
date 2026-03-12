import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
import wandb

from models.components.hr_net import HRNet
from models.hrnet_module import HRNetLandmarkModule

# api = wandb.Api()
# artifact = api.artifact("ducholmes-vietnam-national-university-hanoi/Facial Landmark Detection/7j1k2oqg")
# save_dir = "logs/train"
# artifact_dir = artifact.download(root=save_dir)

def predict_and_draw(image_path, checkpoint_path):
    image = np.array(Image.open(image_path).convert("RGB"))
    
    transform = A.Compose([
            A.LongestMaxSize(max_size=256),
            A.PadIfNeeded(
                min_height=256,
                min_width=256,
                border_mode=0,
                value=(0,0,0),
            ),
            A.Normalize(),
            A.ToTensorV2()
        ])
    
    input_tensor = transform(image=image)["image"].unsqueeze(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = HRNetLandmarkModule().load_from_checkpoint(checkpoint_path)
    model.eval()
    model.to(device)
    input_tensor = input_tensor.to(device)

    with torch.no_grad():
        outputs = model(input_tensor)
        
    landmarks = outputs.cpu().numpy().squeeze()
    landmarks = landmarks.reshape(-1, 2)
    transformed_image = transform(input_tensor)
    
    plt.figure(figsize=(8, 8))
    plt.imshow(transformed_image)

    plt.scatter(landmarks[:, 0], landmarks[:, 1], s=15, c='hotpink', marker='o')
    
    plt.axis('off')
    plt.show()

TEST_IMAGE_PATH = "images/VHD.png" 
CKPT_PATH = "logs/train/last.ckpt"

predict_and_draw(TEST_IMAGE_PATH, CKPT_PATH)