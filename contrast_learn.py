
import torch
from torch import nn
from torch.autograd import grad
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, ToTensor
from tqdm import tqdm, trange
from torchvision import transforms, datasets
import os
import datetime
import sys
import numpy as np
import matplotlib.pyplot as plt  
import random


class ContrastiveLoss(nn.Module):
    def __init__(self, temperature=1.0):
        super(ContrastiveLoss, self).__init__()
        self.temperature = temperature
        self.criterion= torch.nn.BCEWithLogitsLoss()
        
    def forward(self, x1, x2,y1,y2,thresh=0.03):
        # Concatenate input data along batch dimension
        features = torch.cat([x1, x2], dim=0)
        output=torch.cat([y1, y2], dim=0)
        # Normalize feature vectors
        features = nn.functional.normalize(features, dim=1)
        # Compute cosine similarity matrix
        similarities = torch.matmul(features, features.T) / self.temperature
        # Create upper triangular matrix
        similarities = torch.triu(similarities,diagonal=1)
        # Create target labels (1 if samples are similar, 0 if samples are dissimilar)
        val = torch.cdist(output,output,p=2)
        mask = val<=thresh
        mask = torch.triu(mask,diagonal=1)
        # Compute log-softmax probabilities and cross-entropy loss
        logits = similarities
        loss = self.criterion(logits, mask.float())

        return loss