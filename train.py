from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint, EarlyStopping
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers import WandbLogger

from callbacks.ema_callback import EMACallback
from data.datamodule import DataModule
from model.model_module import ModelModule
from datetime import datetime

import hydra
import psutil
import torch
import wandb
import gc
import os


def configure_callbacks(args, save_checkpoint_dir):
    checkpoint_callback = ModelCheckpoint(
        monitor="val_acc",
        mode="max",
        dirpath=save_checkpoint_dir,
        save_last=True,
        filename="{epoch}",
        save_top_k=10,
    )
    lr_monitor = LearningRateMonitor(logging_interval="step")

    early_stop_callback = EarlyStopping(
        monitor='val_loss',
        min_delta=0.2,
        patience=3,
        mode='min',
    )

    if args.use_ema: 
        ema_callback = EMACallback(decay=args.ema_decay)
        return [checkpoint_callback, early_stop_callback, lr_monitor, ema_callback]
    return [checkpoint_callback, early_stop_callback, lr_monitor]


@hydra.main(version_base="1.3", config_path="configs", config_name="config")
def main(cfg):
    seed_everything(cfg.seed)

    name = f"exp_lr{cfg.lr}_batch_size{cfg.batch_size}_dropout{cfg.dropout}"
    ttime = datetime.now().strftime('%d-%m-%Y_%H-%M-%S')
    save_checkpoint_dir = os.makedirs(cfg.checkpoint_dir + '/' + ttime + '/' + name, exist_ok=True) if cfg.checkpoint_dir else None
    

    cfg.workers = psutil.cpu_count(logical=False) if cfg.workers == None else cfg.workers

    modelmodule = ModelModule(cfg)
    datamodule = DataModule(cfg)

    logger = WandbLogger(name=name, \
                         project=f'{cfg.name_exp}',\
                            save_dir=f"{cfg.checkpoint_dir}/{name}")
    logger.watch(model = modelmodule, log='gradients',log_graph=True)

    
    callbacks = configure_callbacks(cfg, save_checkpoint_dir)
    trainer = Trainer(
        logger=logger,
        gpus=-1,
        max_epochs=cfg.epochs,
        callbacks=callbacks,
        log_every_n_steps=5,
    )
    trainable_params = sum(p.numel() for p in modelmodule.parameters() if p.requires_grad)

    print(f"Trainable parameters: {trainable_params}")
    logger.log_hyperparams({"trainable_params": trainable_params})
    logger.log_hyperparams(cfg)

    if cfg.checkpoint != None:
        modelmodule.load_state_dict(torch.load(cfg.checkpoint, map_location=lambda storage, loc: storage)["state_dict"])

    trainer.fit(model=modelmodule, datamodule=datamodule)

if __name__ == "__main__":
    torch.cuda.empty_cache()
    gc.collect()
    main()