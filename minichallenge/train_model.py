import random

import numpy as np
import torch
from torchvision import datasets, transforms
from torchvision.io import read_image

from model_classes import ImageDataset

# Transformation of images ------------------------------------------------------------------------------------------- #
transform = transforms.Compose((
    transforms.Grayscale(1),
    transforms.Resize((255, 255)),
))

# Load training and validation data ---------------------------------------------------------------------------------- #
root = '../tmp/train/train/'
categories = list(np.loadtxt('category.csv', delimiter=',', skiprows=1, usecols=1, dtype=str))
n_categories = len(categories)
targets_train = np.loadtxt('data_train.csv', delimiter=',', usecols=0, dtype=int)
images_train = np.loadtxt('data_train.csv', delimiter=',', usecols=1, dtype=str)
targets_validation = np.loadtxt('data_validation.csv', delimiter=',', usecols=0, dtype=int)
images_validation = np.loadtxt('data_validation.csv', delimiter=',', usecols=1, dtype=str)

dataset_train = ImageDataset(
    root=root, images=list(images_train), targets=list(targets_validation), transform=transform
)
dataset_validation = ImageDataset(
    root=root, images=list(images_validation), targets=list(targets_validation), transform=transform
)

# Test batch (TODO - REMOVE) ----------------------------------------------------------------------------------------- #
from matplotlib import pyplot as plt
dataloader = torch.utils.data.DataLoader(dataset_train, batch_size=100, shuffle=False)
tmp = next(iter(dataloader))
pixel_data = np.asarray(tmp[0][2, :, :])
fig_grey_cat_trial = plt.figure()
ax_grey_cat_trial = fig_grey_cat_trial.add_subplot(111)
ax_grey_cat_trial.imshow(1. - pixel_data, cmap='Greys', interpolation='nearest', vmin=0., vmax=1.)
ax_grey_cat_trial.set_xticks([])
ax_grey_cat_trial.set_yticks([])
fig_grey_cat_trial.tight_layout()

plt.show()
