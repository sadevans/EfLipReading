import os

import torch
import torchvision
import matplotlib.pyplot as plt
import random
# matplotlib qt

def cut_or_pad(data, size, dim=0):
    """
    Pads or trims the data along a dimension.
    """
    if data.size(dim) < size:
        padding = size - data.size(dim)
        data = torch.nn.functional.pad(data, (0, 0, 0, padding), "constant")
        size = data.size(dim)
    elif data.size(dim) > size:
        data = data[:size]
    assert data.size(dim) == size
    return data


def load_video(path):
    """
    rtype: torch, T x C x H x W
    """
    vid = torchvision.io.read_video(path, pts_unit="sec", output_format="THWC")[0]
    vid = vid.permute((0, 3, 1, 2))
    return vid


def build_word_list(directory, num_words, seed, words=None):
    random.seed(seed)
    if words is None:
        words = os.listdir(directory)
        words.sort()
        random.shuffle(words)
        words = words[:num_words]
    else:
        words.sort()
        random.shuffle(words)
    return words


class LRWDataset(torch.utils.data.Dataset):
    def __init__(self, path, video_transform, num_words=500, in_channels=1, mode="train", seed=42, words=None):
        self.video_transform = video_transform
        self.words = words
        self.seed = seed
        self.num_words = num_words
        self.in_channels = in_channels
        self.video_paths, self.files, self.labels, self.words = self.build_file_list(path, mode)


    def build_file_list(self, directory, mode):
        words = build_word_list(directory, self.num_words, seed=self.seed, words=self.words)
        # print(words)
        paths = []
        file_list = []
        labels = []
        for i, word in enumerate(words):
            dirpath = directory + "/{}/{}".format(word, mode)
            files = os.listdir(dirpath)
            for file in files:
                if file.endswith("mp4"):
                    path = dirpath + "/{}".format(file)
                    file_list.append(file)
                    paths.append(path)
                    labels.append(i)

        return paths, file_list, labels, words


    def __len__(self):
        return len(self.video_paths)


    def __getitem__(self, idx):
        label = self.labels[idx]
        video = load_video(self.video_paths[idx])
        video = self.video_transform(video)
        sample = {
            'frames': video,
            'label': torch.LongTensor([label]),
            'word': self.words[label]
        }
        return sample




# class LRWDataset(Dataset):
#     def __init__(self, path, num_words=500, in_channels=1, mode="train", augmentations=False, estimate_pose=False, seed=42, query=None):
#         self.seed = seed
#         self.num_words = num_words
#         self.in_channels = in_channels
#         self.query = query
#         self.augmentation = augmentations if mode == 'train' else False
#         self.poses = None
#         # if estimate_pose == False:
#         #     self.poses = self.head_poses(mode, query)
#         self.video_paths, self.files, self.labels, self.words = self.build_file_list(path, mode)
#         self.estimate_pose = estimate_pose

#     def head_poses(self, mode, query):
#         poses = {}
#         yaw_file = open(f"data/preprocess/lrw/{mode}.txt", "r")
#         content = yaw_file.read()
#         for line in content.splitlines():
#             file, yaw = line.split(",")
#             yaw = float(yaw)
#             if query == None or (query[0] <= yaw and query[1] > yaw):
#                 poses[file] = yaw
#         return poses

#     def build_file_list(self, directory, mode):
#         words = build_word_list(directory, self.num_words, seed=self.seed)
#         # print(words)
#         paths = []
#         file_list = []
#         labels = []
#         for i, word in enumerate(words):
#             dirpath = directory + "/{}/{}".format(word, mode)
#             files = os.listdir(dirpath)
#             for file in files:
#                 if file.endswith("mp4"):
#                     if self.poses != None and file not in self.poses:
#                         continue
#                     path = dirpath + "/{}".format(file)
#                     file_list.append(file)
#                     paths.append(path)
#                     labels.append(i)

#         return paths, file_list, labels, words

#     def build_tensor(self, frames):
#         # print(frames.shape[0])
#         temporalVolume = torch.FloatTensor(frames.shape[0], self.in_channels, 88, 88)
#         # temporalVolume = torch.FloatTensor(29, self.in_channels, 88, 88)

#         if(self.augmentation):
#             augmentations = transforms.Compose([
#                 StatefulRandomHorizontalFlip(0.5),
#             ])
#         else:
#             augmentations = transforms.Compose([])

#         if self.in_channels == 1:
#             transform = transforms.Compose([
#                 transforms.ToPILImage(),
#                 transforms.CenterCrop((88, 88)),
#                 augmentations,
#                 transforms.Grayscale(num_output_channels=1),
#                 transforms.ToTensor(),
#                 transforms.Normalize([0.4161, ], [0.1688, ]),
#             ])
#         elif self.in_channels == 3:
#             transform = transforms.Compose([
#                 transforms.ToPILImage(),
#                 transforms.CenterCrop((88, 88)),
#                 augmentations,
#                 transforms.ToTensor(),
#                 transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
#             ])

#         length = frames.shape[0]
#         # print(frames.shape, length-1)
#         for i in range(0, length):
#         # for i in range(0, 29):

#             # print(frames[i].shape)
#             frame = frames[i].permute(2, 0, 1)  # (C, H, W)
#             temporalVolume[i] = transform(frame)

#         temporalVolume = temporalVolume.transpose(1, 0)  # (C, D, H, W)
#         return temporalVolume

#     def __len__(self):
#         return len(self.video_paths)


#     def __getitem__(self, idx):
#         label = self.labels[idx]
#         file = self.files[idx]
#         video = torchvision.io.read_video(self.video_paths[idx], pts_unit='sec')[0]  # (Tensor[T, H, W, C])
#         frames = self.build_tensor(video)
#         sample = {
#             'frames': frames,
#             'label': torch.LongTensor([label]),
#             'word': self.words[label],
#             'length': frames.shape[1]
#         }
#         return sample