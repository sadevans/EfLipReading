import os
import yaml
import torch
import torchvision
import matplotlib.pyplot as plt
import random
# matplotlib qt
current_file_directory = os.path.abspath(__file__)


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

    labels_words = {}
    for i, word in enumerate(words):
        labels_words[i] = word
    dir = "/".join(current_file_directory.split('/')[:-2])
    with open(f'{dir}/model/labels/labels_{num_words}_seed{seed}.yaml', 'w') as file:
        yaml.dump(labels_words, file)
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