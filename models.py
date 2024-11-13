import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.optim as optim
from torch.nn import init
# utils
import math
import os
import datetime
import matplotlib.pyplot as plt
import numpy as np
#from sklearn.externals 
from sklearn.metrics import r2_score
import joblib
from tqdm import tqdm
import contrast_learn
import importlib as im
im.reload(contrast_learn)
import utils
from datetime import datetime


class ChenEtAl(nn.Module):
    """
    DEEP FEATURE EXTRACTION AND CLASSIFICATION OF HYPERSPECTRAL IMAGES BASED ON
                        CONVOLUTIONAL NEURAL NETWORKS
    Yushi Chen, Hanlu Jiang, Chunyang Li, Xiuping Jia and Pedram Ghamisi
    IEEE Transactions on Geoscience and Remote Sensing (TGRS), 2017
    """
    @staticmethod
    def weight_init(m):
        # In the beginning, the weights are randomly initialized
        # with standard deviation 0.001
        if isinstance(m, nn.Linear) or isinstance(m, nn.Conv3d):
            init.normal_(m.weight, std=0.001)
            init.zeros_(m.bias)

    def __init__(self, input_channels, n_classes, patch_size=27, n_planes=32):
        super(ChenEtAl, self).__init__()
        self.input_channels = input_channels
        self.n_planes = n_planes
        self.patch_size = patch_size

        self.conv1 = nn.Conv3d(1, n_planes, (32, 3, 3))
        self.pool1 = nn.MaxPool3d((1, 1, 1))
        self.conv2 = nn.Conv3d(n_planes, n_planes, (32, 3, 3))
        self.pool2 = nn.MaxPool3d((1, 1, 1))
        self.conv3 = nn.Conv3d(n_planes, n_planes, (32, 1, 1))

        self.features_size = self._get_final_flattened_size()

        self.fc = nn.Linear(self.features_size, n_classes)

        self.dropout = nn.Dropout(p=0.5)

        self.apply(self.weight_init)

    def _get_final_flattened_size(self):
        with torch.no_grad():
            x = torch.zeros((1, 1, self.input_channels,
                             self.patch_size, self.patch_size))
            x = self.pool1(self.conv1(x))
            x = self.pool2(self.conv2(x))
            x = self.conv3(x)
            _, t, c, w, h = x.size()
        return t * c * w * h

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = self.dropout(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = self.dropout(x)
        x = F.relu(self.conv3(x))
        x = self.dropout(x)
        x1 = x.view(-1, self.features_size)
        x = self.fc(x1)
        return x,x1



class HamidaEtAl(nn.Module):
    """
    3-D Deep Learning Approach for Remote Sensing Image Classification
    Amina Ben Hamida, Alexandre Benoit, Patrick Lambert, Chokri Ben Amar
    IEEE TGRS, 2018
    https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=8344565
    """
    @staticmethod
    def weight_init(m):
        if isinstance(m, nn.Linear) or isinstance(m, nn.Conv3d):
            init.kaiming_normal_(m.weight)
            init.zeros_(m.bias)

    def __init__(self, input_channels, n_classes, patch_size=5, dilation=1):
        super(HamidaEtAl, self).__init__()
        # The first layer is a (3,3,3) kernel sized Conv characterized
        # by a stride equal to 1 and number of neurons equal to 20
        self.patch_size = patch_size
        self.input_channels = input_channels
        dilation = (dilation, 1, 1)

        if patch_size == 3:
            self.conv1 = nn.Conv3d(
                1, 20, (3, 3, 3), stride=(1, 1, 1), dilation=dilation, padding=1)
        else:
            self.conv1 = nn.Conv3d(
                1, 20, (3, 3, 3), stride=(1, 1, 1), dilation=dilation, padding=0)
        # Next pooling is applied using a layer identical to the previous one
        # with the difference of a 1D kernel size (1,1,3) and a larger stride
        # equal to 2 in order to reduce the spectral dimension
        self.pool1 = nn.Conv3d(
            20, 20, (3, 1, 1), dilation=dilation, stride=(2, 1, 1), padding=(1, 0, 0))
        # Then, a duplicate of the first and second layers is created with
        # 35 hidden neurons per layer.
        self.conv2 = nn.Conv3d(
            20, 35, (3, 3, 3), dilation=dilation, stride=(1, 1, 1), padding=(1, 0, 0))
        self.pool2 = nn.Conv3d(
            35, 35, (3, 1, 1), dilation=dilation, stride=(2, 1, 1), padding=(1, 0, 0))
        # Finally, the 1D spatial dimension is progressively reduced
        # thanks to the use of two Conv layers, 35 neurons each,
        # with respective kernel sizes of (1,1,3) and (1,1,2) and strides
        # respectively equal to (1,1,1) and (1,1,2)
        self.conv3 = nn.Conv3d(
            35, 35, (3, 1, 1), dilation=dilation, stride=(1, 1, 1), padding=(1, 0, 0))
        self.conv4 = nn.Conv3d(
            35, 35, (2, 1, 1), dilation=dilation, stride=(2, 1, 1), padding=(1, 0, 0))

        #self.dropout = nn.Dropout(p=0.5)

        self.features_size = self._get_final_flattened_size()
        # The architecture ends with a fully connected layer where the number
        # of neurons is equal to the number of input classes.
        self.fc = nn.Linear(self.features_size, n_classes)

        self.apply(self.weight_init)

    def _get_final_flattened_size(self):
        with torch.no_grad():
            x = torch.zeros((1, 1, self.input_channels,
                             self.patch_size, self.patch_size))
            x = self.pool1(self.conv1(x))
            x = self.pool2(self.conv2(x))
            x = self.conv3(x)
            x = self.conv4(x)
            _, t, c, w, h = x.size()
        return t * c * w * h

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x1 = x.view(-1, self.features_size)
        #x = self.dropout(x)
        x = self.fc(x1)
        return x,x1
    


class LuoEtAl(nn.Module):
    """
    HSI-CNN: A Novel Convolution Neural Network for Hyperspectral Image
    Yanan Luo, Jie Zou, Chengfei Yao, Tao Li, Gang Bai
    International Conference on Pattern Recognition 2018
    """

    @staticmethod
    def weight_init(m):
        if isinstance(m, (nn.Linear, nn.Conv2d, nn.Conv3d)):
            init.kaiming_uniform_(m.weight)
            init.zeros_(m.bias)

    def __init__(self, input_channels, n_classes, patch_size=3, n_planes=16):
        super(LuoEtAl, self).__init__()
        self.input_channels = input_channels
        self.patch_size = patch_size
        self.n_planes = n_planes

        # the 8-neighbor pixels [...] are fed into the Conv1 convolved by n1 kernels
        # and s1 stride. Conv1 results are feature vectors each with height of and
        # the width is 1. After reshape layer, the feature vectors becomes an image-like 
        # 2-dimension data.
        # Conv2 has 64 kernels size of 3x3, with stride s2.
        # After that, the 64 results are drawn into a vector as the input of the fully 
        # connected layer FC1 which has n4 nodes.
        # In the four datasets, the kernel height nk1 is 24 and stride s1, s2 is 9 and 1
        self.conv1 = nn.Conv3d(1, 8, (24, 3, 3), padding=0, stride=(9,1,1))
        self.conv2 = nn.Conv2d(1, 8, (3, 3), stride=(1, 1))

        self.features_size = self._get_final_flattened_size()

        self.fc = nn.Linear(self.features_size, n_classes)
        # self.fc2 = nn.Linear(512, n_classes)

        self.apply(self.weight_init)

    def _get_final_flattened_size(self):
        with torch.no_grad():
            x = torch.zeros((1, 1, self.input_channels,
                             self.patch_size, self.patch_size))
            x = self.conv1(x)
            b = x.size(0)
            x = x.view(b, 1, -1, self.n_planes)
            x = self.conv2(x)
            _, c, w, h = x.size()
        return c * w * h

    def forward(self, x):
        x = F.relu(self.conv1(x))
        b = x.size(0)
        x = x.view(b, 1, -1, self.n_planes)
        x = F.relu(self.conv2(x))
        x1 = x.view(-1, self.features_size)
        x = (self.fc(x1))
        # x = self.fc2(x)
        return x,x1

class LeeEtAl(nn.Module):
    """
    CONTEXTUAL DEEP CNN BASED HYPERSPECTRAL CLASSIFICATION
    Hyungtae Lee and Heesung Kwon
    IGARSS 2016
    """
    @staticmethod
    def weight_init(m):
        if isinstance(m, nn.Linear) or isinstance(m, nn.Conv3d):
            init.kaiming_uniform_(m.weight)
            init.zeros_(m.bias)

    def __init__(self, in_channels, n_classes,patch_size = 5):
        super(LeeEtAl, self).__init__()
        # The first convolutional layer applied to the input hyperspectral
        # image uses an inception module that locally convolves the input
        # image with two convolutional filters with different sizes
        # (1x1xB and 3x3xB where B is the number of spectral bands)

        self.in_channels = in_channels
        self.patch_size = patch_size
        self.conv_3x3 = nn.Conv3d(
            1, 128, (in_channels, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1))
        self.conv_1x1 = nn.Conv3d(
            1, 128, (in_channels, 1, 1), stride=(1, 1, 1), padding=0)

        # We use two modules from the residual learning approach
        # Residual block 1
        self.conv1 = nn.Conv2d(256, 128, (1, 1))
        self.conv2 = nn.Conv2d(128, 128, (1, 1))
        self.conv3 = nn.Conv2d(128, 128, (1, 1))

        # Residual block 2
        self.conv4 = nn.Conv2d(128, 128, (1, 1))
        self.conv5 = nn.Conv2d(128, 128, (1, 1))

        # The layer combination in the last three convolutional layers
        # is the same as the fully connected layers of Alexnet
        self.conv6 = nn.Conv2d(128, 128, (1, 1))
        self.conv7 = nn.Conv2d(128, 128, (1, 1))
        self.conv8 = nn.Conv2d(128, n_classes, (1, 1))

        self.lrn1 = nn.LocalResponseNorm(256)
        self.lrn2 = nn.LocalResponseNorm(128)

        # The 7 th and 8 th convolutional layers have dropout in training
        self.dropout = nn.Dropout(p=0.2)

        self.apply(self.weight_init)

        self.features_size = self._get_final_flattened_size()

        self.fc = nn.Linear(self.features_size, 64)
        self.fc1 = nn.Linear(64, n_classes)

    def _get_final_flattened_size(self):
        with torch.no_grad():
            x = torch.zeros((36, 1, self.in_channels,
                             self.patch_size, self.patch_size))
            x_3x3 = self.conv_3x3(x)
            x_1x1 = self.conv_1x1(x)
            x = torch.cat([x_3x3, x_1x1], dim=1)
            # Remove the third dimension of the tensor
            x = torch.squeeze(x)

            # Local Response Normalization
            x = F.relu(self.lrn1(x))

            # First convolution
            x = self.conv1(x)

            # Local Response Normalization
            x = F.relu(self.lrn2(x))

            # First residual block
            x_res = F.relu(self.conv2(x))
            x_res = self.conv3(x_res)
            x = F.relu(x + x_res)

            # Second residual block
            x_res = F.relu(self.conv4(x))
            x_res = self.conv5(x_res)
            x = F.relu(x + x_res)
            _, c, w, h = x.size()
        return c * w * h

    def forward(self, x):
        # Inception module
        x_3x3 = self.conv_3x3(x)
        x_1x1 = self.conv_1x1(x)
        x = torch.cat([x_3x3, x_1x1], dim=1)
        # Remove the third dimension of the tensor
        x = torch.squeeze(x)

        # Local Response Normalization
        x = F.relu(self.lrn1(x))

        # First convolution
        x = self.conv1(x)

        # Local Response Normalization
        x = F.relu(self.lrn2(x))

        # First residual block
        x_res = F.relu(self.conv2(x))
        x_res = self.conv3(x_res)
        x = F.relu(x + x_res)

        # Second residual block
        x_res = F.relu(self.conv4(x))
        x_res = self.conv5(x_res)
        x = F.relu(x + x_res)
        x = self.dropout(x)
        x1 = x.view(-1, self.features_size)
        
        x = (self.fc (x1))
        x = self.dropout(x)
        x = (self.fc1 (x))

        # x = F.relu(self.conv6(x))
        # x = self.dropout(x)
        # x = F.relu(self.conv7(x))
        # x = self.dropout(x)
        # x = self.conv8(x)


        return x,x1


class LiEtAl(nn.Module):
    """
    SPECTRAL–SPATIAL CLASSIFICATION OF HYPERSPECTRAL IMAGERY
            WITH 3D CONVOLUTIONAL NEURAL NETWORK
    Ying Li, Haokui Zhang and Qiang Shen
    MDPI Remote Sensing, 2017
    http://www.mdpi.com/2072-4292/9/1/67
    """
    @staticmethod
    def weight_init(m):
        if isinstance(m, nn.Linear) or isinstance(m, nn.Conv3d):
            init.xavier_uniform_(m.weight.data)
            init.constant_(m.bias.data, 0)

    def __init__(self, input_channels, n_classes, n_planes=2, patch_size=5):
        super(LiEtAl, self).__init__()
        self.input_channels = input_channels
        self.n_planes = n_planes
        self.patch_size = patch_size

        # The proposed 3D-CNN model has two 3D convolution layers (C1 and C2)
        # and a fully-connected layer (F1)
        # we fix the spatial size of the 3D convolution kernels to 3 × 3
        # while only slightly varying the spectral depth of the kernels
        # for the Pavia University and Indian Pines scenes, those in C1 and C2
        # were set to seven and three, respectively
        self.conv1 = nn.Conv3d(1, n_planes, (7, 3, 3), padding=(1, 0, 0))
        # the number of kernels in the second convolution layer is set to be
        # twice as many as that in the first convolution layer
        self.conv2 = nn.Conv3d(n_planes, 2 * n_planes,
                               (3, 3, 3), padding=(1, 0, 0))
        self.dropout = nn.Dropout(p=0.3)
        self.features_size = self._get_final_flattened_size()

        self.fc = nn.Linear(self.features_size,n_classes)
        self.fc2 = nn.Linear(64, n_classes)

        self.apply(self.weight_init)

    def _get_final_flattened_size(self):
        with torch.no_grad():
            x = torch.zeros((1, 1, self.input_channels,
                             self.patch_size, self.patch_size))
            x = self.conv1(x)
            x = self.conv2(x)
            _, t, c, w, h = x.size()
        return t * c * w * h

    def forward(self, x):
        x = (self.conv1(x))
        # x = self.dropout(x)
        x = (self.conv2(x))
        x = x.view(-1, self.features_size)
        # x = self.dropout(x)
        x1 = F.softmax(self.fc(x))
        # x1 = self.fc2(x1)
        # x1 = F.relu(x1)
        # x1 = (x1.T/torch.sum(x1,dim=1)).T
        return x1,x


def train(net, optimizer, criterion, data_loader, epoch,
          display_iter=100, device=torch.device('cpu'), display=None,wavelength=None,
          val_loader=None,test_loader=None, supervision='full',trans= ["permutation"],type='spectral',spatial_trans=None,contrast=1,tradeoff_cont=0.001,thresh=0.01,params=None):
    """
    Training loop to optimize a network for several epochs and a specified loss

    Args:
        net: a PyTorch model
        optimizer: a PyTorch optimizer
        data_loader: a PyTorch dataset loader
        epoch: int specifying the number of training epochs
        criterion: a PyTorch-compatible loss function, e.g. nn.CrossEntropyLoss
        device (optional): torch device to use (defaults to CPU)
        display_iter (optional): number of iterations before refreshing the
        display (False/None to switch off).
        scheduler (optional): PyTorch scheduler
        val_loader (optional): validation dataset
        supervision (optional): 'full' or 'semi'
    """

    if criterion is None:
        raise Exception("Missing criterion. You must specify a loss function.")

    net.to(device)

    save_epoch = epoch // 20 if epoch > 20 else 1


    losses = np.zeros(1000000)
    mean_losses = np.zeros(100000000)
    iter_ = 1
    loss_win, val_win = None, None
    val_accuracies = []
    train_accuracies = []
    
    transform = utils.get_augmentation(trans,wavelength=wavelength.reshape(-1,))
    cont_loss= contrast_learn.ContrastiveLoss()
    avg_loss = 0.
    number =0
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1,gamma=0.95)
    for e in tqdm(range(1, epoch + 1),total=epoch,miniters=display_iter, desc="Training the network",disable=True):
        # Set the network to training mode
        net.train(True)
        avg_loss = 0
        number =0

        # Run the training loop for one epoch
        for batch_idx, (data, target) in tqdm(enumerate(data_loader), total=len(data_loader),disable=True):
            # Load the data into the GPU if required
            data, target = data.float().to(device), target.float().to(device)

            
            tr1 = np.random.choice(spatial_trans)
            tr2 = np.random.choice(transform)

            if contrast==1 :
                if type=='spatial':
                    #spatial transformation
                    data_tr = utils.spatial_patch_transform(data,tr1,params).to(device).float()

                elif type=='spectral':
                    #  spectral transformation
                    data_tr = utils.patch_transform(data,tr2).to(device).float()

                else:
                    # spatial - spectral transformation
                    data_tr = utils.spatial_spectral_patch_transform(data,tr1,tr2,params).to(device).float()




            # plt.figure(dpi=200)
            # plt.imshow(utils.spatial_patch_transform(data,tr,params)[0,0,0,:,:])
            # plt.title('Flip Transform')
            # plt.show()
            # plt.figure(dpi=200)
            # plt.imshow(data.cpu().numpy()[0,0,0,:,:])
            # plt.title('Original')
            # break



            optimizer.zero_grad()
            output, feature_s = net(data)
            
            #target = target - 1
            loss_classif = criterion(output, target)
            if contrast==1 :
                _, feature_tr = net(data_tr)
                loss_contrast = cont_loss(feature_s,feature_tr,target,target,thresh)
                loss = loss_classif +  tradeoff_cont * loss_contrast
                # print(loss_classif,tradeoff_cont * loss_contrast)
            else:
                loss =  loss_classif
            loss.backward()
            # print(net.conv1.weight.grad)
            grad = net.fc.weight.grad
            # grad = net.fc1.weight.grad
            optimizer.step()
            # value=optimizer.param_groups[0]['lr']
            

            # avg_loss += torch.nn.MSELoss(reduction='sum')(output, target).item()
            avg_loss += torch.nn.L1Loss(reduction='sum')(output, target).item()
            number += output.size(0)
            losses[iter_] = loss.item()
            mean_losses[iter_] = np.mean(losses[max(0, iter_ - 100):iter_ + 1])

            # if display_iter and iter_ % display_iter == 0:
            #     string = 'Train (epoch {}/{}) [{}/{} ({:.0f}%)]\tLoss: {:.6f}'
            #     string = string.format(
            #         e, epoch, batch_idx *
            #         len(data), len(data) * len(data_loader),
            #         100. * batch_idx / len(data_loader), mean_losses[iter_])
            #     update = None if loss_win is None else 'append'
            #     tqdm.write(string)

        
            del(data, target, loss, output)
        avg_loss /= number
        train_accuracies.append(avg_loss)

        iter_ += 1
        # Update the scheduler

        if val_loader is not None:
            val_acc,_,val_r2,_ = val(net, val_loader, device=device)
            val_accuracies.append(val_acc)

        if display_iter and iter_ % display_iter == 0:
            current_time = datetime.now().time()

            # Format the time as "hour:minutes:seconds"
            formatted_time = current_time.strftime("%H:%M:%S")
            print(f'Train Loss :{avg_loss:.4f}, Val loss : {val_acc:.4f}, Val r2 : {val_r2:.4f} , grad :{grad.mean()} , time : {formatted_time}')


        # if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
        #     scheduler.step(metric)
        
        # elif scheduler is not None:
        #     scheduler.step()

    if test_loader is not None:
        test_mae,test_mse,test_r2,pred= val(net, test_loader, device=device)
    print(f'Train Loss :{avg_loss:.4f}, Val loss : {val_acc:.4f} , test loss: {test_mae:.4f},{test_mse:.4f} r2 : {test_r2:.4f}')
    return {'train loss':avg_loss,'val loss':val_acc,'test loss':{'mae':test_mae,'mse':test_mse},'train curve':train_accuracies,'val curve':val_accuracies,'prediction':pred}
    # return net,avg_loss,val_acc,test_acc,train_accuracies,val_accuracies


def val(net, data_loader, device='cpu'):
# TODO : fix me using metrics()
    net.eval()
    MAE = 0
    MSE = 0
    number=0
    output_l =  []
    target_l  = []


    for batch_idx, (data, target) in enumerate(data_loader):
        with torch.no_grad():
            # Load the data into the GPU if required
            data, target = data.float().to(device), target.to(device)
            output,_ = net(data)
            #target = target - 1
            MSE += torch.nn.MSELoss(reduction='sum')(output, target).item()
            MAE += torch.nn.L1Loss(reduction='sum')(output, target).item()
            number += output.size(0)
            output_l.append(output.detach().cpu().numpy())
            target_l.append(target.detach().cpu().numpy())

    MAE /= number
    MSE /= number
    output_l = np.vstack(output_l)
    target_l = np.vstack(target_l)
    r2 = r2_score(target_l ,output_l )
    return MAE,MSE,r2,output_l