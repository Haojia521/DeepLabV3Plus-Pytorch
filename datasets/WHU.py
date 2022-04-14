import numpy as np
import os.path as osp
from osgeo import gdal
import os
import cv2

from .dataset_base import DatasetBase


class DatasetWhu(DatasetBase):
    def __init__(self, root, split, transform=None):
        super(DatasetWhu, self).__init__(root, split, transform)

    def collect_data_index(self):
        return [name[:-4] for name in os.listdir(osp.join(self.data_dir, 'image'))]

    def read_image(self, idx):
        dataset = gdal.Open(osp.join(self.data_dir, 'image', idx + '.tif'), gdal.GA_ReadOnly)
        band_arr_list = [dataset.GetRasterBand(i + 1).ReadAsArray() for i in range(dataset.RasterCount)]
        img = np.dstack(band_arr_list)

        return img

    def read_label(self, idx):
        lbl = cv2.imread(osp.join(self.data_dir, 'label', idx + '.tif'), cv2.IMREAD_UNCHANGED) / 255
        return lbl


def WHU(root, transform=None, transform_val=None):
    train_dataset = DatasetWhu(root, 'train', transform)
    val_dataset = DatasetWhu(root, 'val', transform_val)

    return train_dataset, val_dataset


def WHU_test(root, transform=None):
    test_dataset = DatasetWhu(root, 'test', transform)

    return test_dataset
