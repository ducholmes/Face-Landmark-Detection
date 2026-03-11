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
            total = len(full_dataset)

            indices = list(range(total))

            random.seed(42)
            random.shuffle(indices)

            train_size = int(self.hparams.train_val_test_split[0]*total)
            val_size = int(self.hparams.train_val_test_split[1]*total)

            train_indices = indices[:train_size]
            val_indices = indices[train_size:val_size+train_size]
            test_indices = indices[val_size+train_size:]

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