import spectral
import numpy as np
import torch
import torch.utils
import torch.utils.data
import os
from tqdm import tqdm
from sklearn import preprocessing
import utils


class HyperX(torch.utils.data.Dataset):
    """ Generic class for a hyperspectral scene """

    def __init__(self, data, gt, hyperparams,test=False):
        """
        Args:
            data: 3D hyperspectral image
            gt: 2D array of labels
            patch_size: int, size of the spatial neighbourhood
            center_pixel: bool, set to True to consider only the label of the
                          center pixel
            data_augmentation: bool, set to True to perform random flips
            supervision: 'full' or 'semi' supervised algorithms
        """
        super(HyperX, self).__init__()
        self.data = data
        self.label = gt
        # self.name = hyperparams['dataset']
        self.patch_size = hyperparams['patch_size']
        # self.patch_size = 5
        self.ignored_labels = [2]
        self.flip_augmentation = False
        self.radiation_augmentation =False
        self.mixture_augmentation = False
        self.center_pixel = True
        supervision = 'full'
        # Fully supervised : use all pixels with label not ignored
        if supervision == 'full':
            mask = np.ones_like(gt[:,:,0])
            for l in self.ignored_labels:
                mask[gt[:,:,0] == l] = 0
        x_pos, y_pos = np.nonzero(mask)
        p = self.patch_size // 2
        self.test=test
        self.indices = np.array([(x,y) for x,y in zip(x_pos, y_pos) if x > p-1 and x < data.shape[0] - p and y > p-1 and y < data.shape[1] - p])
        self.labels = [self.label[x,y] for x,y in self.indices]
        np.random.shuffle(self.indices)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        x, y = self.indices[i]
        x1, y1 = x - self.patch_size // 2, y - self.patch_size // 2
        x2, y2 = x1 + self.patch_size, y1 + self.patch_size

        data = self.data[x1:x2, y1:y2]
        label = self.label[x1:x2, y1:y2,:]


        # Copy the data into numpy arrays (PyTorch doesn't like numpy views)
        data = np.asarray(np.copy(data).transpose((2, 0, 1)), dtype='float32')
        label = np.asarray(np.copy(label), dtype='float32')
        



        # Load the data into PyTorch tensors
        data = torch.from_numpy(data)
        label = torch.from_numpy(label)
        # Extract the center label if needed
        if self.center_pixel and self.patch_size > 1:
            label = label[self.patch_size // 2, self.patch_size // 2]
            # label = label.mean(axis=0).mean(axis=0)

        # Add a fourth dimension for 3D CNN
        if self.patch_size > 1:
            # Make 4D data ((Batch x) Planes x Channels x Width x Height)
            data = data.unsqueeze(0)

        return data, label
