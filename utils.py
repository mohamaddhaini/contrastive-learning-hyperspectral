import numpy as np
import random
from sklearn.metrics import confusion_matrix
import sklearn.model_selection
import seaborn as sns
import itertools
import spectral
import transformations as aug
import importlib as im
im.reload(aug)
import datasets
im.reload(datasets)
import torch
import os
import matplotlib.pyplot as plt
import torch.utils.data as data

def sample_gt(gt, train_size, mode='random'):
    """Extract a fixed percentage of samples from an array of labels.

    Args:
        gt: a 2D array of int labels
        percentage: [0, 1] float
    Returns:
        train_gt, test_gt: 2D arrays of int labels

    """
    indices = np.nonzero(gt)
    X = list(zip(*indices)) # x,y features
    y = gt[indices].ravel() # classes
    train_gt = np.zeros_like(gt)
    test_gt = np.zeros_like(gt)
    if train_size > 1:
       train_size = int(train_size)
    
    if mode == 'random':
       train_indices, test_indices = sklearn.model_selection.train_test_split(X, train_size=train_size, stratify=y)
       train_indices = [list(t) for t in zip(*train_indices)]
       test_indices = [list(t) for t in zip(*test_indices)]
       train_gt[tuple(train_indices)] = gt[tuple(train_indices)]
       test_gt[tuple(test_indices)] = gt[tuple(test_indices)]
    elif mode == 'fixed':
       print("Sampling {} with train size = {}".format(mode, train_size))
       train_indices, test_indices = [], []
       for c in np.unique(gt):
           if c == 0:
              continue
           indices = np.nonzero(gt == c)
           X = list(zip(*indices)) # x,y features

           train, test = sklearn.model_selection.train_test_split(X, train_size=train_size)
           train_indices += train
           test_indices += test
       train_indices = [list(t) for t in zip(*train_indices)]
       test_indices = [list(t) for t in zip(*test_indices)]
       train_gt[train_indices] = gt[train_indices]
       test_gt[test_indices] = gt[test_indices]

    elif mode == 'disjoint':
        train_gt = np.copy(gt)
        test_gt = np.copy(gt)
        for c in np.unique(gt):
            mask = gt == c
            for x in range(gt.shape[0]):
                first_half_count = np.count_nonzero(mask[:x, :])
                second_half_count = np.count_nonzero(mask[x:, :])
                try:
                    # ratio = first_half_count / second_half_count
                    # if ratio > 0.9 * train_size and ratio < 1.1 * train_size:
                    if first_half_count >= train_size and second_half_count >= train_size:
                        break
                except ZeroDivisionError:
                    continue
            mask[:x, :] = 0
            train_gt[mask] = 0

        test_gt[train_gt > 0] = 0
    elif mode == 'custom':
        unique_classes = np.unique(gt)
        train_indices = []
        test_indices = []
        shape_ = gt.shape
        for c in unique_classes:
            if c==0:
                continue
            class_indices = np.where(gt.reshape(-1,) == c)[0]
            np.random.shuffle(class_indices)
            num_samples = len(class_indices)
            num_train_samples = min(train_size, num_samples)
            train_indices.extend(np.random.choice(class_indices, size=num_train_samples, replace=False))
        
        train_gt = np.zeros_like(gt).reshape(-1,)
        test_gt = np.copy(gt)
        train_gt[np.array(train_indices)] = gt.reshape(-1,)[np.array(train_indices)]
        train_gt = train_gt.reshape(shape_)
        test_gt[train_gt > 0]=0
        return train_gt, test_gt
    else:
        raise ValueError("{} sampling is not implemented yet.".format(mode))
    return train_gt, test_gt


def get_augmentation(trans_nm,wavelength=None):
    transform_dict_synthet = {
    'elastic': lambda x: aug.elastic_distortion(x, 50, 10),
    'atmospheric': lambda x: aug.libGeneratorSimpleAtmospheric(x, desiredWavelengths=wavelength),
    'scattering': lambda x: aug.libGeneratorHapke(x),
    'shift': lambda x: aug.shift_spectrum(x, 20, direction='right'),
    'flip' : lambda x: aug.spectral_flip(x),
    'erasure' : lambda x: aug.random_erasure(x),
    'neighbors' : lambda x: aug.generate_neighbors(x),
    'permutation' : lambda x: aug.group_random_permutation(x,20)
                }
    transform = []
    for tr in trans_nm:
        transform.append(transform_dict_synthet[tr])
    return transform


def combin_transformations(spectrums_array,transform, batch_size=5):
    """Apply random transformations to every `step`-th spectrum in the input array."""
    spectrums = spectrums_array.copy()
    num_rows = spectrums.shape[0]
    for i in range(0, num_rows, batch_size):
        # Randomly select a transformation to apply
        transformation = np.random.choice(transform)
        batch = spectrums[i:i+batch_size]
        transformed_batch = transformation(batch)
        spectrums[i:i+batch_size,:] = transformed_batch
    return spectrums

def patch_transform(patch,transform):
    trans_l=[]
    dims= patch.shape
    for i in range(len(patch)):
        
        temp = patch[i].squeeze(0).cpu().numpy().reshape(dims[2],-1).T
        trans = transform(temp)
        trans=trans.T.reshape(dims[2],dims[3],dims[4])
        trans_l.append(trans)
    trans_l = torch.from_numpy(np.stack(trans_l,axis=0).astype('float')).unsqueeze(1)
    return trans_l



def spatial_patch_transform(patch,transform,params):
    trans_l=[]
    dims= patch.shape
    for i in range(len(patch)):
        temp = patch[i].squeeze(0).cpu().numpy()
        trans = aug.spatial_transformation(temp,transform,params)
        trans_l.append(trans)
    trans_l = torch.from_numpy(np.stack(trans_l,axis=0).astype('float')).unsqueeze(1)
    return trans_l



def spatial_spectral_patch_transform(patch,transform1,transform2,params):
    trans_l=[]
    dims= patch.shape
    for i in range(len(patch)):
        temp = patch[i].squeeze(0).cpu().numpy()
        trans = aug.spatial_transformation(temp,transform1,params)
        trans = transform2(trans.reshape(dims[2],-1).T).T.reshape(dims[2],dims[3],dims[4])
        trans_l.append(trans)
    trans_l = torch.from_numpy(np.stack(trans_l,axis=0).astype('float')).unsqueeze(1)
    return trans_l


def split_data(urban,gt,abundance,hyperparams,train_size=1000,portion=0.5,image_shape=(100,100)):
    train_gt, test_gt = sample_gt(gt,train_size, mode='custom')
    train_gt, val_gt = sample_gt(train_gt, int(portion*train_size), mode='custom')


    abd_test = abundance.copy()
    abd_test[test_gt==0,:] = 2
    abd_test = abd_test.reshape(image_shape[0],image_shape[1],-1)
    # train_loader = data.DataLoader(train_dataset,batch_size=16,shuffle=True)

    abd_train = abundance.copy()
    abd_train[train_gt==0,:] = 2
    abd_train = abd_train.reshape(image_shape[0],image_shape[1],-1)

    abd_val = abundance.copy()
    abd_val[val_gt==0,:] = 2
    abd_val = abd_val.reshape(image_shape[0],image_shape[1],-1)


    train_dataset = datasets.HyperX(urban, abd_train,hyperparams)
    test_dataset = datasets.HyperX(urban, abd_test,hyperparams)
    val_dataset = datasets.HyperX(urban, abd_val,hyperparams)

    train_loader = data.DataLoader(train_dataset,batch_size=hyperparams['batch_size'],shuffle=True)
    test_loader = data.DataLoader(test_dataset,batch_size=256,shuffle=False)
    val_loader = data.DataLoader(val_dataset,batch_size=128,shuffle=True)

    return train_loader,test_loader,val_loader,abd_train,abd_test,abd_val



def set_seed(a):
    torch.manual_seed(a)
    torch.cuda.manual_seed_all(a)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(a)
    random.seed(a)
    os.environ['PYTHONHASHSEED'] = str(a)