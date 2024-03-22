import random

import numpy as np
import cv2

FRAC_VALIDATION = 0.1  # Fraction of data from each category to act as validation

# Get Data
root = '../tmp/minichallenge_data/train/'
categories = list(np.loadtxt('category.csv', delimiter=',', skiprows=1, usecols=1, dtype=str))
n_categories = len(categories)
image_names_arr = np.loadtxt('train.csv', delimiter=',', skiprows=1, usecols=1, dtype=str)
labels_str = list(np.loadtxt('train.csv', delimiter=',', skiprows=1, usecols=2, dtype=str))
targets = [categories.index(lab) for lab in labels_str]

images_valid = []
targets_valid = []

face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

for target, image in zip(targets, image_names_arr):
    try:
        img = cv2.imread(root + image, cv2.IMREAD_GRAYSCALE)
        faces = face_classifier.detectMultiScale(img, scaleFactor=1.1, minNeighbors=2)
        if len(faces) != 1:
            raise RuntimeError('Classifier did not find exactly 1 face!')

        images_valid.append(image)
        targets_valid.append(target)
    except RuntimeError as _:
        continue


# Separate images into training and validation set
images_valid_arr = np.asarray(images_valid, dtype=str)
targets_valid_arr_str = np.asarray(targets_valid, dtype=str)
targets_valid_arr = np.asarray(targets_valid, dtype=int)

idces_train = []
idces_validation = []

for category in range(n_categories):
    labels_cat = np.nonzero(targets_valid_arr == category)[0]
    idx_validation = int(len(labels_cat) * FRAC_VALIDATION)
    idces_train.extend(labels_cat[idx_validation:])
    idces_validation.extend(labels_cat[:idx_validation])

data_validation = np.hstack((
    targets_valid_arr_str[idces_validation].reshape((-1, 1)),
    images_valid_arr[idces_validation].reshape((-1, 1))
                        ))
np.savetxt('data_validation.csv', data_validation, delimiter=',', fmt='%s')
data_train = np.hstack((
    targets_valid_arr_str[idces_train].reshape((-1, 1)),
    images_valid_arr[idces_train].reshape((-1, 1))
                        ))
np.savetxt('data_train.csv', data_train, delimiter=',', fmt='%s')
