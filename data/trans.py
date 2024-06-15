import torch
import torchvision
import random


class FunctionalModule(torch.nn.Module):
    def __init__(self, functional):
        super().__init__()
        self.functional = functional

    def forward(self, input):
        return self.functional(input)


class VideoTransform:
    def __init__(self, subset):
        if subset == "train":
            self.video_pipeline = torch.nn.Sequential(
                FunctionalModule(lambda x: x / 255.0),
                torchvision.transforms.RandomCrop(88),
                torchvision.transforms.Grayscale(),
                # torchvision.transforms.Normalize(0.421, 0.165),
                torchvision.transforms.Normalize(0.4161, 0.1688)

            )
        elif subset == "val" or subset == "test":
            self.video_pipeline = torch.nn.Sequential(
                FunctionalModule(lambda x: x / 255.0),
                torchvision.transforms.CenterCrop(88),
                torchvision.transforms.Grayscale(),
                # torchvision.transforms.Normalize(0.421, 0.165),
                torchvision.transforms.Normalize(0.4161, 0.1688)
            )

    def __call__(self, sample):
        # sample: T x C x H x W
        return self.video_pipeline(sample)