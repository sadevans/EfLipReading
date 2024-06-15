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

    ema_callback = EMACallback(decay=args.ema_decay)

    return [checkpoint_callback, early_stop_callback, lr_monitor, ema_callback]


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
        logs = trainer.validate(modelmodule, checkpoint=cfg.checkpoint)
        logger.log_metrics({'val_acc': logs['val_acc'], 'val_loss': logs['val_loss']})
        print(f"Initial val_acc: {logs['val_acc']:.4f}")


    trainer.fit(model=modelmodule, datamodule=datamodule)

if __name__ == "__main__":
    torch.cuda.empty_cache()
    gc.collect()
    main()

# if __name__ == "__main__":
#     torch.cuda.empty_cache()
#     gc.collect()
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--data', default="data/datasets/lrw")
#     parser.add_argument("--checkpoint_dir", type=str, default='data/checkpoints/lrw')
#     parser.add_argument("--checkpoint", type=str)
#     parser.add_argument("--batch_size", type=int, default=16)
#     parser.add_argument("--epochs", type=int, default=11)
#     parser.add_argument("--lr", type=float, default=1e-4)
#     parser.add_argument("--weight_decay", type=float, default=1e-5)
#     parser.add_argument("--words", type=int, default=10)
#     parser.add_argument("--workers", type=int, default=None)
#     parser.add_argument("--seed", type=int, default=42)
#     parser.add_argument("--in_channels", type=int, default=1)
#     parser.add_argument("--dropout", type=float, default=0.3)
#     parser.add_argument("--name_exp", type=str, default="exp")
#     parser.add_argument("--warmup_epochs", type=int, default=3)

#     args = parser.parse_args()
#     seed_everything(args.seed)

#     name = f"exp_lr{args.lr}_batch_size{args.batch_size}_dropout{args.dropout}"
#     ttime = datetime.now().strftime('%d-%m-%Y_%H-%M-%S')
#     save_checkpoint_dir = os.makedirs(args.checkpoint_dir + '/' + ttime + '/' + name, exist_ok=True) if args.checkpoint_dir else None
    

#     args.workers = psutil.cpu_count(logical=False) if args.workers == None else args.workers

#     modelmodule = ModelModule(hparams=args)
#     datamodule = DataModule(hparams=args)

#     # print(modelmodule.named_parameters)
#     # for param in modelmodule.parameters():
#     #     print(type(param), param.numel())

#     logger = WandbLogger(name=name, \
#                         #  project=f'lipreading_lrw_classification_{args.words}words',\
#                          project=f'{args.name_exp}',\
#                             save_dir=f"{args.checkpoint_dir}/{name}")
#     logger.watch(model = modelmodule, log='gradients',log_graph=True)

    
#     callbacks = configure_callbacks(args, save_checkpoint_dir)
#     trainer = Trainer(
#         logger=logger,
#         gpus=-1,
#         max_epochs=args.epochs,
#         callbacks=callbacks,
#         log_every_n_steps=5,
#     )
#     trainable_params = sum(p.numel() for p in modelmodule.parameters() if p.requires_grad)

#     print(f"Trainable parameters: {trainable_params}")
#     logger.log_hyperparams({"trainable_params": trainable_params})
#     logger.log_hyperparams(args)

#     if args.checkpoint != None:
#         logs = trainer.validate(modelmodule, checkpoint=args.checkpoint)
#         logger.log_metrics({'val_acc': logs['val_acc'], 'val_loss': logs['val_loss']})
#         print(f"Initial val_acc: {logs['val_acc']:.4f}")

#     # # trainer.fit(modelmodule, datamodule)
#     # # modelmodule = modelmodule.load_from_checkpoint("/media/sadevans/T7 Shield/PERSONAL/Diplom/datasets/LRW/checkpoints/epoch=9-v4.ckpt")
#     # trainer.fit(modelmodule)
#     trainer.fit(model=modelmodule, datamodule=datamodule)



#     # ckpt = torch.load("/media/sadevans/T7 Shield/PERSONAL/Diplom/datasets/LRW/checkpoints/epoch=9-v4.ckpt", map_location=lambda storage, loc: storage)["state_dict"]
#     # ckpt = torch.load("/media/sadevans/T7 Shield/PERSONAL/Diplom/datasets/LRW/checkpoints/WORDS/exp_lr0.0001_batch_size16_dropout0.4/lrw_100words_expw_conv3d/7750nc4d/checkpoints/epoch=11.ckpt", \
#     #                   map_location=lambda storage, loc: storage)["state_dict"]

#     # ckpt = torch.load("/media/sadevans/T7 Shield/PERSONAL/Diplom/datasets/LRW/checkpoints/WORDS/exp_lr0.0001_batch_size16_dropout0.4/lrw_100words_expw_conv3d_more/zsb1tosi/checkpoints/epoch=15.ckpt", \
#     #                   map_location=lambda storage, loc: storage)["state_dict"]
#     # ckpt = torch.load("/media/sadevans/T7 Shield/PERSONAL/Diplom/datasets/LRW/checkpoints/WORDS/exp_lr0.0001_batch_size16_dropout0.4/lrw_100words_expw_conv3d/en7f1r5e/checkpoints/epoch=11.ckpt", \
#     #                   map_location=lambda storage, loc: storage)["state_dict"]
    
#     # print(ckpt.keys())
#     # modelmodule.load_state_dict(ckpt)
#     # modelmodule.load_state_dict(torch.load("/media/sadevans/T7 Shield/PERSONAL/Diplom/datasets/LRW/checkpoints/epoch=9-v4.ckpt", map_location=lambda storage, loc: storage))
#     # trainer.test(modelmodule)

#     # logger.save_file(checkpoint_callback.last_checkpoint_path)

#     # python train_words.py --data "/media/sadevans/T7 Shield/PERSONAL/Diplom/datasets/LRW/lipread_mp4" --checkpoint_dir "/media/sadevans/T7 Shield/PERSONAL/Diplom/datasets/LRW/checkpoints" --lr 1e-6