import random

import numpy as np
from torchvision.io import read_image

N_IMAGES_VALIDATION = 100  # Number of images from each category to act as validation

# Get Data
root = '../tmp/train/train/'
categories = list(np.loadtxt('category.csv', delimiter=',', skiprows=1, usecols=1, dtype=str))
n_categories = len(categories)
image_names_arr = np.loadtxt('train.csv', delimiter=',', skiprows=1, usecols=1, dtype=str)
labels_str = list(np.loadtxt('train.csv', delimiter=',', skiprows=1, usecols=2, dtype=str))
targets = [categories.index(lab) for lab in labels_str]

images_valid = []
targets_valid = []

for target, image in zip(targets, image_names_arr):
    try:
        read_image(root + image)
        images_valid.append(image)
        targets_valid.append(target)
    except RuntimeError as _:
        continue

targets_valid_arr = np.asarray(targets_valid, dtype=int)
n_images_min = np.Inf
for idx in range(n_categories):
    n_images_min = min(n_images_min, len(np.nonzero(targets_valid_arr == idx)[0]))

# Sort images into sets of 100 by category
arr_idces = np.empty((n_categories, n_images_min), dtype=int)
idces_sorted = []
for idx in range(n_categories):
    arr_idces[idx, :] = np.nonzero(targets_valid_arr == idx)[0][:n_images_min]

# Shuffle images
image_order = list(np.arange(0, len(categories), 1, dtype=int))
for idx in range(n_images_min):
    random.shuffle(image_order)
    arr_idces[:, idx] = arr_idces[image_order, idx]

# One-dimensionalize
validation_idces = arr_idces[:, :N_IMAGES_VALIDATION].ravel(order='F')
train_idces = arr_idces[:, N_IMAGES_VALIDATION:].ravel(order='F')

data_validation = np.hstack((
    np.asarray(targets_valid, dtype=str)[validation_idces].reshape((-1, 1)),
    np.asarray(images_valid, dtype=str)[validation_idces].reshape((-1, 1))
                        ))
np.savetxt('data_validation.csv', data_validation, delimiter=',', fmt='%s')
data_train = np.hstack((
    np.asarray(targets_valid, dtype=str)[train_idces].reshape((-1, 1)),
    np.asarray(images_valid, dtype=str)[train_idces].reshape((-1, 1))
                        ))
np.savetxt('data_train.csv', data_train, delimiter=',', fmt='%s')
