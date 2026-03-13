from pathlib import Path
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image

class KeypointDataset(Dataset):
    def __init__(self,
        img_dir: str,
        csv_path: str,
        transform=None,
        img_size=(256, 256),    
        heatmap_size=(64, 64),  
        sigma=2.0,  
        num_keypoints=76
    ):
        self.img_dir = Path(img_dir)
        self.keypoints = pd.read_csv(csv_path).iloc[:3755]
        self.transform = transform
        self.width, self.height = img_size
        self.heatmap_width, self.heatmap_height = heatmap_size
        self.sigma = sigma
        self.num_keypoints = num_keypoints

        x_grid = np.arange(0, self.heatmap_width, 1, dtype=np.float32)
        y_grid = np.arange(0, self.heatmap_height, 1, dtype=np.float32)
        self.xx, self.yy = np.meshgrid(x_grid, y_grid)

    def _generate_heatmap(self, keypoints, mask):
        heatmaps = np.zeros(
            (self.num_keypoints, self.heatmap_height, self.heatmap_width), 
            dtype=np.float32
        )

        for i in range(self.num_keypoints):
            if not mask[i]:
                continue 

            x_hm = keypoints[i, 0] * (self.heatmap_width / self.width)
            y_hm = keypoints[i, 1] * (self.heatmap_height / self.height)

            heatmaps[i] = np.exp(
                -((self.xx - x_hm)**2 + (self.yy - y_hm)**2) / (2 * self.sigma**2)
            )

        return heatmaps

    def __len__(self):
        return len(self.keypoints)
    
    def __getitem__(self, index):
        row = self.keypoints.iloc[index]
        img_path = self.img_dir / (row['name'] + '.jpg')
        img = Image.open(img_path).convert('RGB')
        img = np.array(img)

        keypoint = row.iloc[2:].values.reshape(-1, 2).astype(np.float32)

        if self.transform:
            transformed = self.transform(image=img, keypoints=keypoint)
            img = transformed['image']
            keypoint = np.array(transformed['keypoints'])

        mask = ((keypoint[:, 0] > 0) & (keypoint[:, 1] > 0))

        heatmaps = self._generate_heatmap(keypoint, mask)

        keypoint_rel = np.copy(keypoint)
        keypoint_rel[:, 0] = keypoint_rel[:, 0] / self.width
        keypoint_rel[:, 1] = keypoint_rel[:, 1] / self.height

        return {
            'image': img,                         
            'heatmap': torch.tensor(heatmaps),          
            'keypoint_rel': torch.tensor(keypoint_rel),
            'mask': torch.tensor(mask)
        }