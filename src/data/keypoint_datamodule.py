import random
from typing import Any, Tuple, Dict
from lightning import LightningDataModule
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset, random_split
from src.data.components.keypoint_dataset import KeypointDataset
from torchvision import transforms
import albumentations as A

class KeypointDataModule(LightningDataModule):
    def __init__(
        self,
        data_dir: str = '/data',
        train_val_test_split: Tuple[int, int, int]=[0.8, 0.1, 0.1],
        batch_size: int = 64,
        num_workers: int = 3,
        pin_memory: bool = False
    ) -> None:
        super().__init__()

        self.save_hyperparameters(logger=False)

        self.img_dir = data_dir + "/jpg"
        self.keypoint_path = data_dir + "/muct-landmarks/muct76-opencv.csv"

        self.train_transform = A.Compose([
            A.LongestMaxSize(max_size=256),
            A.PadIfNeeded(
                min_height=256,
                min_width=256,
                border_mode=0,
                value=(0,0,0),
            ),
            A.Normalize(),
            A.ToTensorV2()
        ], keypoint_params=A.KeypointParams(format='xy', remove_invisible=False))

        self.test_transform = A.Compose([
            A.LongestMaxSize(max_size=256),
            A.PadIfNeeded(
                min_height=256,
                min_width=256,
                border_mode=0,
                value=(0,0,0),
            ),
            A.Normalize(),
            A.ToTensorV2()
        ], keypoint_params=A.KeypointParams(format='xy', remove_invisible=False))

        self.data_train = None
        self.data_test = None
        self.data_val = None

        self.batch_size_per_device = None

    def train_dataloader(self) -> DataLoader[Any]:
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
        )

    def val_dataloader(self) -> DataLoader[Any]:
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def test_dataloader(self) -> DataLoader[Any]:
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )
    
    def setup(self, stage=None):
        if self.trainer is not None:
            if self.hparams.batch_size % self.trainer.world_size != 0:
                raise RuntimeError(
                    f"Batch size ({self.hparams.batch_size}) is not divisible by the number of devices ({self.trainer.world_size})."
                )
            self.batch_size_per_device = self.hparams.batch_size // self.trainer.world_size

        if not self.data_train and not self.data_val and not self.data_test:
            full_dataset = KeypointDataset(self.img_dir, self.keypoint_path)
            df = full_dataset.keypoints
            df['unique_subject'] = df['name'].str[:4]
            unique_subjects = df['unique_subject'].unique().tolist()

            random.seed(42)
            random.shuffle(unique_subjects)

            total_subjects = len(unique_subjects)

            train_size = int(self.hparams.train_val_test_split[0]*total_subjects)
            val_size = int(self.hparams.train_val_test_split[1]*total_subjects)

            train_subjects = set(unique_subjects[:train_size])
            val_subjects = set(unique_subjects[train_size:val_size+train_size])
            test_subjects = set(unique_subjects[val_size+train_size:])

            train_indices = df.index[df['unique_subject'].isin(train_subjects)].tolist()
            val_indices = df.index[df['unique_subject'].isin(val_subjects)].tolist()
            test_indices = df.index[df['unique_subject'].isin(test_subjects)].tolist()

            if stage=='fit' or stage == None:
                train_base = KeypointDataset(
                    img_dir = self.img_dir,
                    csv_path = self.keypoint_path,
                    transform=self.train_transform
                )

                self.data_train = Subset(train_base, train_indices)

                val_base = KeypointDataset(
                    img_dir = self.img_dir,
                    csv_path = self.keypoint_path,
                    transform=self.test_transform
                )

                self.data_val = Subset(val_base, val_indices)

            if stage=='test':
                test_base = KeypointDataset(
                    img_dir = self.img_dir,
                    csv_path = self.keypoint_path,
                    transform=self.test_transform
                )

                self.data_test = Subset(test_base, test_indices)

if __name__ == "__main__":
    _ = KeypointDataModule()