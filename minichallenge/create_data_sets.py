import warnings
import os
import shutil

import numpy as np
import cv2

FRAC_VALIDATION = 0.1  # Fraction of data from each category to act as validation
SMALL_DATA_SET = False  # True -> test script on subset of data

# Get Data
root = '../tmp/minichallenge_data/train/'
categories = list(np.loadtxt('category.csv', delimiter=',', skiprows=1, usecols=1, dtype=str))
n_categories = len(categories)
image_names_arr = np.loadtxt('train.csv', delimiter=',', skiprows=1, usecols=1, dtype=str)
labels_str = list(np.loadtxt('train.csv', delimiter=',', skiprows=1, usecols=2, dtype=str))
targets = [categories.index(lab) for lab in labels_str]
targets_arr = np.asarray(targets, dtype=int)

dir_train = '../tmp/minichallenge_data/train_cropped/'
dir_test = '../tmp/minichallenge_data/test_cropped/'

# Make directories if they do not yet exist
os.makedirs(os.path.dirname(dir_train), exist_ok=True)
os.makedirs(os.path.dirname(dir_test), exist_ok=True)

face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')


def crop_data(_dir_uncropped, _dir_cropped, _img_names, _sf=1.1, _mn=5, convert_to_png=False):
    print(f'--- Cropping Images in {_dir_uncropped} ---')
    _n_images = len(_img_names)
    _success = np.ones(shape=(_n_images,), dtype=bool)
    for _idx, _image in enumerate(_img_names):
        if _idx % 100 == 0:
            print(f'Trying image {_idx + 1}/{_n_images}...')
        try:
            if convert_to_png:
                _image_png = _image.split('.')[0] + '.png'
                shutil.copyfile(_dir_uncropped + _image, _dir_uncropped + _image_png)
                _image = _image_png
            # Load Image
            _img = cv2.imread(_dir_uncropped + _image, cv2.IMREAD_GRAYSCALE)
            if _img is None:
                raise RuntimeError('Image unable to load!')

            # Find faces
            _faces, _confidence = face_classifier.detectMultiScale2(_img, scaleFactor=_sf, minNeighbors=_mn)
            if len(_faces) == 0:
                # No faces found, default to using whole image
                _x, _y = 0, 0
                _h, _w = _img.shape
            else:
                # Face(s) found, run classifier on most confident face
                _x, _y, _w, _h = _faces[np.asarray(_confidence).argmax()]

            # Pad cropped image to be square
            _img_cropped = _img[_y:_y + _h, _x:_x + _w]
            if _h > _w:
                dim = _h
                pad_offset = (_h - _w) // 2
                ave_color = np.mean(_img_cropped)
                _img_padded = ave_color * np.ones((dim, dim))
                _img_padded[:, pad_offset:pad_offset + _w] = _img_cropped
            elif _w > _h:
                dim = _w
                pad_offset = (_w - _h) // 2
                ave_color = np.mean(_img_cropped)
                _img_padded = ave_color * np.ones((dim, dim))
                _img_padded[pad_offset:pad_offset + _h, :] = _img_cropped
            else:
                _img_padded = _img_cropped

            # Save cropped image
            cv2.imwrite(_dir_cropped + _image, _img_padded)
        except RuntimeError as _:
            _success[_idx] = False
            continue

    return _success


# CROP TRAINING DATA ------------------------------------------------------------------------------------------------- #
if SMALL_DATA_SET:
    n_images = 100
else:
    n_images = len(targets)

# Try to crop out faces
successful_crop_train = np.zeros(shape=(len(targets),), dtype=bool)
successful_crop_train[:n_images] = crop_data(root, dir_train, image_names_arr[:n_images], _mn=20)

# TODO, try to save images which did not work as .png and then do it

# Save results since this takes a LONG time to reproduce
np.savetxt(dir_train + 'successful_crop.csv', successful_crop_train, delimiter=',')

# Separate images into training and validation set
images_valid_arr = image_names_arr[successful_crop_train]
targets_valid_arr = np.asarray(targets_arr[successful_crop_train], dtype=int)
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

# CROP TEST DATA ----------------------------------------------------------------------------------------------------- #
if not SMALL_DATA_SET:
    image_names_test_arr = np.asarray(np.arange(0, 4977, 1, dtype=int), dtype=str)
    root_test = '../tmp/minichallenge_data/test/'
    for idx in range(len(image_names_test_arr)):
        image_names_test_arr[idx] += '.jpg'
    np.savetxt('data_test.csv', image_names_test_arr, delimiter=',', fmt='%s')

    successful_crop_test = crop_data(root_test, dir_test, image_names_test_arr, _mn=20)
    successful_crop_test_str = np.asarray(successful_crop_test, dtype=str)
    data_test = np.hstack((
        successful_crop_test_str.reshape((-1, 1)),
        image_names_test_arr.reshape((-1, 1))
                            ))
    np.savetxt('data_test.csv', data_test, delimiter=',', fmt='%s')
    # TODO, try to save images which did not work as .png and then do it
