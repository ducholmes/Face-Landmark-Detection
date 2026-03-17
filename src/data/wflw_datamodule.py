import random
from typing import Any, Tuple, Dict
from lightning import LightningDataModule
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset, random_split
from torchvision import transforms
import albumentations as A
from src.data.components.wflw_dataset import WFLWDataset

class WFLWDataModule(LightningDataModule):
    def __init__(
        self,
        data_dir: str = '/data',
        train_val_test_split: Tuple[int, int, int]=[0.8, 0.2, 0],
        batch_size: int = 64,
        num_workers: int = 3,
        pin_memory: bool = False
    ) -> None:
        super().__init__()

        self.save_hyperparameters(logger=False)

        self.img_dir = data_dir + "/WFLW_images/WFLW_images"
        self.train_keypoint_path = data_dir + "/WFLW_annotations/WFLW_annotations/list_98pt_rect_attr_train_test/list_98pt_rect_attr_train.txt"
        self.test_keypoint_path = data_dir + "/WFLW_annotations/WFLW_annotations/list_98pt_rect_attr_train_test/list_98pt_rect_attr_test.txt"

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
            

            if stage=='fit' or stage == None:
                train_base = WFLWDataset(
                    img_dir = self.img_dir,
                    landmark_path=self.train_keypoint_path,
                    transform=self.train_transform
                )

                self.data_train = train_base

                val_base = WFLWDataset(
                    img_dir=self.img_dir,
                    landmark_path=self.test_keypoint_path,
                    transform=self.test_transform
                )

                self.data_val = val_base

            if stage=='test':
                test_base = WFLWDataset(
                    img_dir=self.img_dir,
                    landmark_path=self.test_keypoint_path,
                    transform=self.test_transform
                )

                self.data_test = test_base

if __name__ == "__main__":
    _ = WFLWDataModule()