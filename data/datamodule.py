import os

import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
from .dataset import LRWDataset
from .trans import VideoTransform


class DataModule(LightningDataModule):
    def __init__(self, hparams):
        super().__init__()
        self.save_hyperparameters(hparams)
        # self.cfg = cfg
        # self.cfg.gpus = torch.cuda.device_count()
        # self.total_gpus = self.cfg.gpus * self.cfg.trainer.num_nodes
        self.k = None
        self.last_batch = None
        self.in_channels=1

    def _dataloader(self, ds, shuffle=False, collate_fn=None):
        return DataLoader(
            ds,
            shuffle=shuffle,
            num_workers=self.hparams.workers,
            pin_memory=True,
            batch_size=self.hparams.batch_size,
            collate_fn=collate_fn,
        )
    
    def train_dataloader(self):
        ds_args = self.hparams.data.dataset
        train_data = LRWDataset(
            path=ds_args.root_dir,
            video_transform=VideoTransform("train"),
            num_words=self.hparams.words,
            in_channels=self.in_channels,
            seed=self.hparams.seed,
            words=self.hparams.words_list
        )

        # train_loader = DataLoader(train_data, shuffle=True, batch_size=self.hparams.batch_size, num_workers=self.hparams.workers, pin_memory=True, collate_fn=collate)
        # train_loader = DataLoader(train_data, shuffle=True, batch_size=self.hparams.batch_size, num_workers=self.hparams.workers, pin_memory=True)
        
        # return train_loader
        return self._dataloader(train_data, shuffle=True)

    def val_dataloader(self):
        ds_args = self.hparams.data.dataset
        val_data = LRWDataset(
            path=ds_args.root_dir,
            video_transform=VideoTransform("val"),
            num_words=self.hparams.words,
            in_channels=self.in_channels,
            mode='val',
            seed=self.hparams.seed,
            words=self.hparams.words_list
        )
        # val_loader = DataLoader(val_data, shuffle=False, batch_size=self.hparams.batch_size * 2, num_workers=self.hparams.workers, collate_fn=collate)
        # val_loader = DataLoader(val_data, shuffle=False, batch_size=self.hparams.batch_size * 2, num_workers=self.hparams.workers)
        
        # return val_loader
        return self._dataloader(val_data)

    def test_dataloader(self):
        ds_args = self.hparams.data.dataset
        test_data = LRWDataset(
            path=ds_args.root_dir,
            video_transform=VideoTransform("test"),
            num_words=self.hparams.words,
            in_channels=self.in_channels,
            mode='test',
            estimate_pose=False,
            seed=self.hparams.seed,
            words=self.hparams.words_list
        )
        # test_loader = DataLoader(test_data, shuffle=False, batch_size=self.hparams.batch_size * 2, num_workers=self.hparams.workers, collate_fn=collate)
        # test_loader = DataLoader(test_data, shuffle=False, batch_size=self.hparams.batch_size * 2, num_workers=self.hparams.workers)
        
        return self._dataloader(test_data)