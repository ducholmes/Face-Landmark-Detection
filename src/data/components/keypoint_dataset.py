from pathlib import Path

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image

class KeypointDataset(Dataset):
    def __init__(self,
        img_dir:str,
        csv_path:str,
        transform = None,
        img_size = (480, 640),
        num_keypoints = 76
    ):
        self.img_dir = Path(img_dir)
        self.keypoints = pd.read_csv(csv_path)
        self.transform = transform
        self.width, self.height = img_size
        self.num_keypoints = num_keypoints

    def __len__(self):
        return len(self.keypoints)-2
    
    def __getitem__(self, index):
        keypoint = self.keypoints.iloc[index]
        img_path = self.img_dir / keypoint['name']
        img = Image.open(img_path).convert('RGB')

        keypoint = keypoint.iloc[:, 2:].values.reshape(-1, 2).astype(np.float32)

        if self.transform:
            transformed = self.transform(image=img, keypoint=keypoint)

            img = transformed['image']
            keypoint = np.array(transformed['keypoints'])

        keypoint[:, 0] = keypoint[:, 0] / 256.0
        keypoint[:, 1] = keypoint[:, 1] / 256.0

        mask = ((keypoint[:, 0] == 0) & (keypoint[:, 1] == 0))
        keypoint[mask] =  -1

        return {
            'image' : img,
            'keypoint': torch.tensor(keypoint),
            'mask': torch.tensor(mask)
        }
