import os
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image

class WFLWRegressionDataset(Dataset):
    def __init__(
        self,
        img_dir,
        landmark_path,
        transform=None,
        img_size=(256, 256),    
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
                bounding_box = np.array(parts[196:200], dtype=np.float32)

                self.landmarks.append((landmark, bounding_box, landmark_name))

        self.transform = transform
        self.width, self.height = img_size
        self.num_keypoints = num_keypoints

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
        
        cropped_img = cropped_img.resize(target_size, Image.BILINEAR)
        cropped_img = np.array(cropped_img)
        
        return cropped_img, (new_x1, new_y1, new_x2, new_y2)

    def __len__(self):
        return len(self.landmarks)
    
    def __getitem__(self, index):
        landmark_orig, bounding_box, landmark_name = self.landmarks[index]
        landmark = landmark_orig.copy()

        img_path = os.path.join(self.img_dir, landmark_name)
        img = Image.open(img_path).convert('RGB')

        cropped_img, (cropped_x1, cropped_y1, cropped_x2, cropped_y2) = self.crop_face_with_padding(
            img, bounding_box, target_size=(self.width, self.height)
        )
        
        scale_x = self.width / (cropped_x2 - cropped_x1)
        scale_y = self.height / (cropped_y2 - cropped_y1)

        landmark[:, 0] = (landmark[:, 0] - cropped_x1) * scale_x
        landmark[:, 1] = (landmark[:, 1] - cropped_y1) * scale_y

        if self.transform:
            transformed = self.transform(image=cropped_img, keypoints=landmark)
            cropped_img = transformed['image']
            landmark = np.array(transformed['keypoints'])

        mask = (landmark[:, 0] >= 0) & (landmark[:, 0] < self.width) & \
               (landmark[:, 1] >= 0) & (landmark[:, 1] < self.height)

        keypoint_norm = landmark.copy()
        keypoint_norm[:, 0] /= self.width
        keypoint_norm[:, 1] /= self.height

        keypoint_flatten = keypoint_norm.flatten()

        return {
            'image': cropped_img,                                         
            'keypoint_rel': torch.tensor(keypoint_flatten, dtype=torch.float32),
            'mask': torch.tensor(mask, dtype=torch.bool)
        }