import random

import numpy as np
import torch
from torchvision import datasets, transforms

from model_classes import ImageFolderWithTargets

# Hyper Parameters
n_data_validation = 10_000
batch_size = None  # If None -> will set to 1 of each category per batch

# Get Data
categories = list(np.loadtxt('category.csv', delimiter=',', skiprows=1, usecols=1, dtype=str))
n_categories = len(categories)
labels_str = list(np.loadtxt('train.csv', delimiter=',', skiprows=1, usecols=2, dtype=str))
targets = [categories.index(lab) for lab in labels_str]

transform = transforms.Compose((
    transforms.Grayscale(1),
    transforms.Resize((255, 255)),
    transforms.ToTensor()
))

# dataset = datasets.ImageFolder('../tmp/train', transform=transform)
dataset_raw = ImageFolderWithTargets('../tmp/train', targets=targets, transform=transform)
samples_raw = dataset_raw.samples
samples_raw_arr = np.array(samples_raw, dtype=str)[:, 0]

# Put data into batches with 1 of each image ------------------------------------------------------------------------- #
batch_size = len(targets) if batch_size is None else batch_size
target_arr = np.asarray(targets, dtype=int)
dict_idces = {}
n_images_max = 0
n_images_min = np.Inf
for idx in range(n_categories):
    # Map target index to indicies in input data
    dict_idces[idx], = np.nonzero(target_arr == idx)
    n_images_max = max(n_images_max, len(dict_idces[idx]))
    n_images_min = min(n_images_min, len(dict_idces[idx]))

arr_idces = np.empty(shape=(n_categories, n_images_min))
for idx in range(n_categories):
    arr_idces[idx, :] = dict_idces[idx][:n_images_min]

n_images_validation = n_data_validation // len(categories)

# Create the validation dataset
image_order = list(np.arange(0, len(categories), 1, dtype=int))
index_order_validation = []
samples_validation = []

# for idx in range(n_images_validation):
#     random.shuffle(image_order)
#     new_sample_idces =
#     for
#     samples_validation.append(
#
#     )


# Create the training data set
index_order_train = []

# Test batch
from matplotlib import pyplot as plt
dataloader = torch.utils.data.DataLoader(dataset_raw, batch_size=batch_size, shuffle=False)
tmp = next(iter(dataloader))
pixel_data = np.asarray(tmp[0][2, 0, :, :])
fig_grey_cat_trial = plt.figure()
ax_grey_cat_trial = fig_grey_cat_trial.add_subplot(111)
ax_grey_cat_trial.imshow(1. - pixel_data, cmap='Greys', interpolation='nearest', vmin=0., vmax=1.)
ax_grey_cat_trial.set_xticks([])
ax_grey_cat_trial.set_yticks([])
fig_grey_cat_trial.tight_layout()

plt.show()
