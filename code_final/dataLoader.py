from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy

plt.ion()   # interactive mode

# Data augmentation and normalization for training
# Just normalization for validation
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

data_dir = '../dataset/processedData/'
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                            data_transforms[x])
                  for x in ['train','val']}
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x],batch_size=16,
                                              shuffle=True,num_workers=4)
               for x in ['train','val']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train','val']}
calss_names = image_datasets['train'].classes

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# visualize a few images
def imshow(input,title=None):
    """imshow for Tensor"""
    input = input.numpy().transpose((1,2,0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    input = std * input + mean
    input = np.clip(input,0,1)
    plt.imshow(input)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)


if __name__ == '__main__':
    # Get a batch of training data
    inputs, classes = next(iter(dataloaders['train']))

    # make a grid from batch
    out = torchvision.utils.make_grid(inputs)

    imshow(out,title=[calss_names[x] for x in classes])