import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from skimage import io, transform
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
from tensorboardX import SummaryWriter
from itertools import product

def get_num_correct(preds, labels):#写成函数，输出预测对的个数
    return preds.argmax(dim=1).eq(labels).sum().item()



key_points_2d_path = r'I:\cvdata\daa_pose3d_test\keypoints_2d\vp11\run1_2018-05-24-13-44-01.ids_1.manual.csv'
key_points_3d_path = r'I:\cvdata\daa_pose3d_test\keypoints_3d\vp11\run1_2018-05-24-13-44-01.ids_1.triangulated.3d.csv'
images_root_path = r'I:\cvdata\daa_pose3d_test\gt_images\vp11\run1_2018-05-24-13-44-01.ids_1'

key_points_2d = pd.read_csv(key_points_2d_path)
key_points_3d = pd.read_csv(key_points_3d_path)


n = 150

img_name = 'frame_' + str(key_points_2d.iloc[n, 0]) + '.jpg'

position_2d = key_points_2d.iloc[n, 2:]
position_2d = np.asarray(position_2d)
position_2d = position_2d.astype('float').reshape(-1, 4)
position_2d = position_2d[:,:2]

position_3d = key_points_3d.iloc[n, 2:]
position_3d = np.asarray(position_3d)
position_3d = position_3d.astype('float').reshape(-1, 5)
position_3d = position_3d[:,:3]

# print('Image name: {}'.format(img_name))
#
# print('position_2d shape: {}'.format(position_2d.shape))
# print('First 4 position_2d: {}'.format(position_2d[:4]))
#
# print('position_3d shape: {}'.format(position_3d.shape))
# print('First 4 position_3d: {}'.format(position_3d[:4]))

def show_landmarks(image, key_points_2d):
    """Show image with landmarks"""
    plt.imshow(image)
    plt.scatter(key_points_2d[:, 0], key_points_2d[:, 1], s=10, marker='.', c='r')
    plt.pause(0.001)  # pause a bit so that plots are updated

# plt.figure()
# show_landmarks(io.imread(os.path.join(images_root_path, img_name)),
#                position_2d)
# plt.show()

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

        key_points_2d = key_points_2d.astype('float').reshape(-1, 4)[:,:2]
        key_points_3d = key_points_3d.astype('float').reshape(-1, 5)[:,:3]

        sample = {'image': image, 'key_points_2d': key_points_2d, 'key_points_3d' : key_points_3d}

        if self.transform:
            sample = self.transform(sample)

        return sample

daa_dataset = SkeletonPositions(csv_file_2d=key_points_2d_path,
                                csv_file_3d=key_points_3d_path,
                                image_root_dir=images_root_path)



# fig = plt.figure()
#
# for i in range(len(daa_dataset)):
#     sample = daa_dataset[i+150]
#
#     print(i, sample['image'].shape, sample['key_points_2d'].shape)
#
#     ax = plt.subplot(2, 2, i + 1)
#     plt.tight_layout()
#     ax.set_title('Sample #{}'.format(i))
#     ax.axis('off')
#     show_landmarks(sample['image'], sample['key_points_2d'])
#
#     if i == 3:
#         plt.show()
#         break

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, key_points_2d, key_points_3d = sample['image'], sample['key_points_2d'], sample['key_points_3d']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C x H x W
        image = image.transpose((2, 0, 1))
        return {'image': torch.from_numpy(image),
                'key_points_2d': torch.from_numpy(key_points_2d),
                'key_points_3d': torch.from_numpy(key_points_3d)}
#相当于train_set,要变成train_loader
transformed_dataset = SkeletonPositions(key_points_2d_path,
                                        key_points_3d_path,
                                        images_root_path,
                                        transform=transforms.Compose([
                                               ToTensor(),
                                           ]))

# for i in range(len(transformed_dataset)):
#     sample = transformed_dataset[i]
#
#     print(i, sample['image'].size(), sample['key_points_2d'].size())
#
#     if i == 3:
#         break
#相当于train_loader
dataloader = DataLoader(transformed_dataset, batch_size=4,
                        shuffle=True, num_workers=0)


# Helper function to show a batch
# def show_landmarks_batch(sample_batched):
#     """Show image with landmarks for a batch of samples."""
#     images_batch, key_points_2d_batch, key_points_3d_batch = \
#             sample_batched['image'], sample_batched['key_points_2d'], sample_batched['key_points_2d']
#     batch_size = len(images_batch)
#     im_size = images_batch.size(3)
#     grid_border_size = 2
#
#     grid = utils.make_grid(images_batch)
#     plt.imshow(grid.numpy().transpose((1, 2, 0)))
#
#     for i in range(batch_size):
#         plt.scatter(key_points_2d_batch[i, :, 0].numpy() + i * im_size + (i + 1) * grid_border_size,
#                     key_points_2d_batch[i, :, 1].numpy() + grid_border_size,
#                     s=10, marker='.', c='r')
#
#         plt.title('Batch from dataloader')
#
# # if you are using Windows, uncomment the next line and indent the for loop.
# # you might need to go back and change "num_workers" to 0.
#
# # if __name__ == '__main__':
# for i_batch, sample_batched in enumerate(dataloader):
#     print(i_batch, sample_batched['image'].size(),
#           sample_batched['key_points_2d'].size())
#
#     # observe 4th batch and stop.
#     if i_batch == 3:
#         plt.figure()
#         show_landmarks_batch(sample_batched)
#         plt.axis('off')
#         plt.ioff()
#         plt.show()
#         break

# sample_batched['key_points_2d'].flatten(start_dim = 1).float()
train_size = int(0.8 * len(transformed_dataset))
test_size = len(transformed_dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(transformed_dataset, [train_size, test_size])
# print(train_dataset[0])

# torch.cuda.is_available()
from main_tensor import main

class Options:
    action= 'All'
    ckpt= ''
    data_dir= 'data/'
    dropout= 0.5
    epochs= 50
    exp= 'example'
    is_train= True
    job= 0
    linear_size= 1024
    load= ''
    lr= 0.001
    lr_decay= 100000
    lr_gamma= 0.96
    max_norm= True
    num_stage= 2
    procrustes= False
    resume= False
    test= False
    test_batch= 64
    train_batch= 64
    use_hg= False
opt = Options()

main(opt, train_dataset, test_dataset)
