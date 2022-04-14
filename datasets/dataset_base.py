import torch
from torch.utils.data import Dataset
import numpy as np
import os.path as osp

from PIL import Image


def _decode_target(mask):
    h, w = mask.shape

    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[mask > 0] = [0, 255, 255]

    return img


class DatasetBase(Dataset):
    def __init__(self, root, split, transform=None):
        super(DatasetBase, self).__init__()

        self.root = root
        self.split = split
        self.data_dir = osp.join(self.root, split)
        self.index = self.collect_data_index()

        self.transform = transform

    def collect_data_index(self):
        raise NotImplementedError

    def read_image(self, idx):
        raise NotImplementedError

    def read_label(self, idx):
        raise NotImplementedError

    def __getitem__(self, item):
        idx = self.index[item]

        img = self.read_image(idx)
        gt = self.read_label(idx)

        # convert ndarray to PIL Image object
        img = Image.fromarray(img)
        gt = Image.fromarray(gt)

        if self.transform is not None:
            img, gt = self.transform(img, gt)

        return img, gt

    def __len__(self):
        return len(self.index)

    def decode_target(self, mask):
        return _decode_target(mask)
