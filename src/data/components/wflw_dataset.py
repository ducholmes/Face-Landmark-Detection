import os
from pathlib import Path

import cv2
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
            lines = f.readlines()

            for line in lines:
                parts = line.strip().split()

                landmark = np.array(parts[0:196], dtype=np.float32).reshape(-1, 2)
                landmark_name = parts[-1]
                bounding_box = np.array(parts[196: 200], dtype=np.float32)

                self.landmarks.append((landmark, bounding_box, landmark_name))

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
    
    def crop_face_with_padding(self, image, bounding_box, padding_ratio=1.25, target_size=(256, 256)):
        x1, y1, x2, y2 = bounding_box
        w = x2 - x1
        h = y2 - y1
        
        center_x = x1 + w / 2
        center_y = y1 + h / 2
        
        aspect_ratio = target_size[0] / target_size[1]
        if w > h * aspect_ratio:
            h = w / aspect_ratio
        elif w < h * aspect_ratio:
            w = h * aspect_ratio
        
        w = w * padding_ratio
        h = h * padding_ratio
        
        new_x1 = center_x - w / 2
        new_y1 = center_y - h / 2
        new_x2 = center_x + w / 2
        new_y2 = center_y + h / 2
        
        cropped_img = image.crop(box=(new_x1, new_y1, new_x2, new_y2))
        cropped_img = np.array(cropped_img)
        
        return cropped_img, (new_x1, new_y1, new_x2, new_y2)

    def __len__(self):
        return len(self.landmarks)
    
    def __getitem__(self, index):
        landmark_orig, bounding_box, landmark_name = self.landmarks[index]
        landmark = landmark_orig.copy()

        img_path = os.path.join(self.img_dir, landmark_name)

        img = Image.open(img_path)

        cropped_img, (cropped_x1, cropped_y1, cropped_x2, cropped_y2) = self.crop_face_with_padding(img, bounding_box)
        landmark[:, 0] -= cropped_x1
        landmark[:, 1] -= cropped_y1

        if self.transform:
            transformed = self.transform(image=cropped_img, keypoints=landmark)

            cropped_img = transformed['image']
            landmark = np.array(transformed['keypoints'])

        mask = (landmark[:, 0] >= 0) & (landmark[:, 0] < self.width) & (landmark[:, 1] >= 0) & (landmark[:, 1] < self.height)

        heatmap = self._generate_heatmap(landmark, mask)

        keypoint_rel = np.copy(landmark)
        keypoint_rel[:, 0] = keypoint_rel[:, 0] / self.width
        keypoint_rel[:, 1] = keypoint_rel[:, 1] / self.height

        return {
            'image': cropped_img,                         
            'heatmap': torch.tensor(heatmap),          
            'keypoint_rel': torch.tensor(keypoint_rel),
            'mask': torch.tensor(mask)
        }

