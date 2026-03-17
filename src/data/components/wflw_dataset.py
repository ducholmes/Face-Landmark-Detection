import os
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image

class WFLWDataset(Dataset):
    def __init__(
        self,
        img_dir,
        landmark_path,
        transform = None,
        img_size=(256, 256),    
        heatmap_size=(64, 64),  
        sigma=2.0,  
        num_keypoints=98
    ):
        self.img_dir = Path(img_dir)

        self.landmarks = []

        with open(landmark_path, 'r') as f:
            lines = f.readline()

            for line in lines:
                parts = line.strip().split()

                landmark = np.array(parts[0:196], dtype=np.float32).reshape(-1, 2)
                print(landmark)
                landmark_name = parts[-1]

                self.landmarks.append((landmark, landmark_name))

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
        return len(self.landmarks)
    
    def __getitem__(self, index):
        landmark, landmark_name = self.landmarks[index]

        img_path = os.path.join(self.img_dir, landmark_name)

        img = Image.open(img_path)

        if self.transform:
            transformed = self.transform(img, landmark)

            img = transformed['image']
            landmark = np.array(transformed['keypoints'])

        mask = (landmark[:, 0] > 0) & (landmark[:, 1] > 0)

        heatmap = self._generate_heatmap(landmark, mask)

        keypoint_rel = np.copy(landmark)
        keypoint_rel[:, 0] = keypoint_rel[:, 0] / self.width
        keypoint_rel[:, 1] = keypoint_rel[:, 1] / self.height

        return {
            'image': img,                         
            'heatmap': torch.tensor(heatmap),          
            'keypoint_rel': torch.tensor(keypoint_rel),
            'mask': torch.tensor(mask)
        }

