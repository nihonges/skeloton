"""PyTorch dataset for daa"""
import os
import numpy as np
import pandas as pd
import torch
from skimage import io
from torch.utils.data import Dataset


class SkeletonPositions(Dataset):
    """key points dataset."""

    def __init__(self, csv_file_2d, csv_file_3d, image_root_dir, transform=None):
        """
        Args:
            csv_file_2d (string): Path to the csv file with 2d annotations.
            csv_file_3d (string): Path to the csv file with 3d annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.key_points_2d = pd.read_csv(csv_file_2d)
        self.key_points_3d = pd.read_csv(csv_file_3d)
        self.root_dir = image_root_dir
        self.transform = transform

    def __len__(self):
        return len(self.key_points_2d)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # img_name = os.path.join(self.root_dir,
        #                         'frame_' + str(self.key_points_2d.iloc[idx].name))

        img_name = os.path.join(self.root_dir,
                                'frame_' + str(self.key_points_2d.iloc[idx, 0]) + '.jpg')

        image = io.imread(img_name)

        key_points_2d = self.key_points_2d.iloc[idx, 2:]
        key_points_3d = self.key_points_3d.iloc[idx, 2:]

        key_points_2d = np.array([key_points_2d])
        key_points_3d = np.array([key_points_3d])

        key_points_2d = key_points_2d.astype('float').reshape(-1, 4)[:, :2]
        key_points_3d = key_points_3d.astype('float').reshape(-1, 5)[:, :3]

        sample = {'image': image, 'key_points_2d': key_points_2d,
                  'key_points_3d': key_points_3d}

        if self.transform:
            sample = self.transform(sample)

        return sample
