import random

import numpy as np
import torch
from torchvision import datasets, transforms
from torchvision.io import read_image

from model_classes import ImageDataset

# Get Data TODO - FIX THIS TO LOAD DATA SAVED FROM "create_data_sets" after that is implemented right
root = '../tmp/train/train/'
categories = list(np.loadtxt('category.csv', delimiter=',', skiprows=1, usecols=1, dtype=str))
n_categories = len(categories)
images_train = np.loadtxt('data_validation.csv', delimiter=',', usecols=0, dtype=str)

labels_str = list(np.loadtxt('train.csv', delimiter=',', skiprows=1, usecols=2, dtype=str))
targets = [categories.index(lab) for lab in labels_str]

# dataset = datasets.ImageFolder('../tmp/train', transform=transform)
dataset_raw = ImageDataset(
    root=root, images=list(image_names_arr), targets=targets, transform=transform
)

# Test batch
from matplotlib import pyplot as plt
dataloader = torch.utils.data.DataLoader(dataset_raw, batch_size=100, shuffle=False)
tmp = next(iter(dataloader))
pixel_data = np.asarray(tmp[0][2, :, :])
fig_grey_cat_trial = plt.figure()
ax_grey_cat_trial = fig_grey_cat_trial.add_subplot(111)
ax_grey_cat_trial.imshow(1. - pixel_data, cmap='Greys', interpolation='nearest', vmin=0., vmax=1.)
ax_grey_cat_trial.set_xticks([])
ax_grey_cat_trial.set_yticks([])
fig_grey_cat_trial.tight_layout()

plt.show()
