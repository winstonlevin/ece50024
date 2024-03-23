import warnings

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
targets_arr = np.asarray(targets, dtype=int)

images_valid = []
targets_valid = []

face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
n_images = len(targets)
n_faces = -np.ones(shape=(n_images,), dtype=int)  # Error = -1

# Parameters for face recognition
scale_factor = 1.1
min_neighbors = 2

warnings.filterwarnings('error')  # Raise warnings as an error
for idx, (target, image) in enumerate(zip(targets[:10], image_names_arr[:10])):
    if idx % 100 == 0:
        print(f'Trying image {idx+1}/{n_images}...')
    try:
        img = cv2.imread(root + image, cv2.IMREAD_GRAYSCALE)
        faces = face_classifier.detectMultiScale(img, scaleFactor=scale_factor, minNeighbors=min_neighbors)
        n_faces[idx] = len(faces)
    except RuntimeError as _:
        continue
warnings.resetwarnings()  # Stop raising warnings as an error

# Save results since this takes a LONG time to reproduce
np.savetxt(f'n_faces_SF{scale_factor}_minNeighbors{min_neighbors}.csv', n_faces, delimiter=',')

# Separate images into training and validation set
idces_valid = np.nonzero(n_faces <= 1)
images_valid_arr = image_names_arr[idces_valid]
targets_valid_arr = np.asarray(targets_arr[idces_valid], dtype=int)
targets_valid_arr_str = np.asarray(targets_valid_arr, dtype=str)

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
