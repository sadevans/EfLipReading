import logging
import os

import hydra
import torch
import psutil
from pytorch_lightning import Trainer
from model.model_module import ModelModule
from data.datamodule import DataModule


@hydra.main(version_base="1.3", config_path="configs", config_name="config")
def main(cfg):
    cfg.workers = psutil.cpu_count(logical=False) if cfg.workers == None else cfg.workers
    modelmodule = ModelModule(cfg, mode="test")
    datamodule_val = DataModule(cfg)
   
    trainer = Trainer(num_nodes=1, gpus=1)

    modelmodule.load_state_dict(torch.load(cfg.pretrained_model_path, map_location=lambda storage, loc: storage)["state_dict"])
    trainer.test(model=modelmodule, datamodule=datamodule_val)


if __name__ == "__main__":
    main()