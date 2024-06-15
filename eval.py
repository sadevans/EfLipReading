import logging
import os

import hydra
import torch

from pytorch_lightning import Trainer
from model_module import ModelModule
from data.datamodule import DataModule


@hydra.main(version_base="1.3", config_path="configs", config_name="config")
def main(cfg):
    modelmodule = ModelModule(cfg)
    datamodule_val = DataModule(cfg)
   
    trainer = Trainer(num_nodes=1, gpus=1)

    # Testing
    modelmodule.load_state_dict(torch.load(cfg.pretrained_model_path, map_location=lambda storage, loc: storage)["state_dict"])
    trainer.test(model=modelmodule, datamodule=datamodule_val)


if __name__ == "__main__":
    main()

# python eval.py data.modality=video data.dataset.root_dir="/media/sadevans/T7 Shield/PERSONAL/Diplom/datasets/mTedx/ru-ru/autoavsr_data" data.dataset.test_file=mtedx_ru_valid_transcript_lengths_seg24s_0to100.csv --pretrained-model-path="/media/sadevans/T7 Shield/PERSONAL/Diplom/experiments/11-04-2024_23-19-43/mtedx_ru/epoch=14.ckpt"