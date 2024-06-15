import torch
import torch.nn as nn
from pytorch_lightning import LightningModule

from sklearn.metrics import f1_score, precision_score, recall_score, balanced_accuracy_score
import numpy as np
from scheduler import WarmupCosineScheduler
from torch import optim
import torch.nn.functional as F
import gc
from copy import deepcopy
from model.e2e import E2E
import yaml


class ModelModule(LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.save_hyperparameters(hparams)
        torch.cuda.empty_cache()
        gc.collect()
        self.save_hyperparameters(hparams)
        self.dropout_rate = self.hparams.dropout
        
        with open(f'data/labels/labels_{self.hparams.words}_seed{self.hparams.seed}.yaml', 'r') as file:
            self.labels_into_words = yaml.safe_load(file)

        self.in_channels = 1
        self.num_classes = self.hparams.words

        self.model = E2E(self.hparams.model, dropout=self.dropout_rate, in_channels=self.in_channels, \
                         num_classes=self.num_classes, efficient_net_size=self.hparams.efficientnet_size)
        
        self.best_val_acc = 0
        self.epoch = 0
        self.sum_batches = 0.0

        self.criterion = nn.NLLLoss()

        self.test_f1 = []
        self.test_precision = []
        self.test_recall = []
        self.test_acc = []


    def forward(self, sample):
        output = self.model(sample)
        preds = torch.exp(output)
        preds_ = torch.argmax(preds, dim=1)

        words = [self.labels_into_words[pr] for pr in preds_]

        return words

    
    def training_step(self, batch, batch_idx):
        return self.shared_step(batch, "train")


    def validation_step(self, batch, batch_idx):
        return self.shared_step(batch, "val")


    def shared_step(self, batch, mode):
        frames = batch['frames']
        labels = batch['label']
        # output = self.model(frames) if self.training or self.model_ema is None else self.model_ema.module(frames)
        output = self.model(frames)

        loss = self.criterion(output, labels.squeeze(1))
        acc = accuracy(output, labels.squeeze(1))

        if mode == "val":
            self.log("val_loss", loss, on_step=False, on_epoch=True, batch_size=self.hparams.batch_size, logger=True)
            self.log("val_acc", acc, on_step=False, on_epoch=True, batch_size=self.hparams.batch_size, logger=True)
            return {
            'val_loss': loss,
            'val_acc': acc,
            'predictions': torch.argmax(torch.exp(output), dim=1),
            'labels': labels.squeeze(dim=1),
            'words': batch['word'],
            }
            
        elif mode == "train":
            self.log("train_loss", loss, on_step=True, on_epoch=True, batch_size=self.hparams.batch_size, logger=True)
            self.log("train_acc", acc, on_step=True, on_epoch=True, batch_size=self.hparams.batch_size, logger=True)
            return {"loss": loss, "train_loss_step": loss, "train_acc_step": acc}


    # def on_before_backward(self, loss):
    #     if self.model_ema:
    #         self.model_ema.update(self.model)

    
    def test_step(self, batch, batch_idx):
        frames = batch['frames']
        labels = batch['label']
        words = batch['word']
        output = self.model(frames)

        loss = self.criterion(output, labels.squeeze(1))
        acc = accuracy(output, labels.squeeze(1))
        return {
            'test_loss': loss,
            'test_acc': acc,
            'predictions': torch.argmax(torch.exp(output), dim=1),
            'labels': labels.squeeze(dim=1),
            'words': batch['word'],
            }
    
    def test_epoch_end(self, outputs):
        predictions = torch.cat([x['predictions'] for x in outputs]).cpu().numpy()
        labels = torch.cat([x['labels'] for x in outputs]).cpu().numpy()
        words = np.concatenate([x['words'] for x in outputs])
        acc = np.array([x['test_acc'] for x in outputs])

        f1 = f1_score(labels, predictions, average="weighted")
        precision = precision_score(labels, predictions, average="weighted")
        recall = recall_score(labels, predictions, average="weighted")
        wacc = balanced_accuracy_score(labels, predictions)
    
        print(f"AVERAGE ACCURACY: {acc.mean()}")

        print(f"F1 score: {f1:.3f}")
        print(f"Precision: {precision:.3f}")
        print(f"Recall: {recall:.3f}")
        print(f"WAcc: {wacc:.3f}")


    def validation_epoch_end(self, outputs):
        predictions = torch.cat([x['predictions'] for x in outputs]).cpu().numpy()
        labels = torch.cat([x['labels'] for x in outputs]).cpu().numpy()
        words = np.concatenate([x['words'] for x in outputs])

        avg_acc = torch.FloatTensor([x['val_acc'] for x in outputs]).mean()
        avg_loss = torch.FloatTensor([x['val_loss'] for x in outputs]).mean()

        if self.best_val_acc < avg_acc:
            self.best_val_acc = avg_acc

        self.log("val_loss", avg_loss, on_step=False, on_epoch=True, batch_size=self.hparams.batch_size, logger=True)
        self.log("val_acc", avg_acc, on_step=False, on_epoch=True, batch_size=self.hparams.batch_size, logger=True)
        self.log("best_val_acc", self.best_val_acc, on_step=False, on_epoch=True, batch_size=self.hparams.batch_size, logger=True)

        return {
                'val_loss': avg_loss,
                'val_acc': avg_acc
                }
    

    def configure_optimizers(self):
        optimizer = optim.AdamW([{"name": "model", "params": self.parameters(), "lr": self.hparams.lr}], \
                                weight_decay=self.hparams.weight_decay, betas=(0.9, 0.98))
        scheduler = WarmupCosineScheduler(optimizer, self.hparams.warmup_epochs, self.hparams.epochs, len(self.trainer.datamodule.train_dataloader()))
        scheduler = {"scheduler": scheduler, "interval": "step", "frequency": 1}
        return [optimizer], [scheduler]


def accuracy(predictions, labels):
    preds = torch.exp(predictions)
    preds_ = torch.argmax(preds, dim=1)
    correct = (preds_ == labels).sum().item()
    accuracy = correct / labels.shape[0]
    return accuracy  