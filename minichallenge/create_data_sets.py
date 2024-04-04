from enum import Enum
import warnings
import os
import shutil

import numpy as np
import cv2
from PIL import Image

FRAC_VALIDATION = 0.1  # Fraction of data from each category to act as validation
SMALL_DATA_SET = False  # True -> test script on subset of data

# Get Data
root = '../tmp/minichallenge_data/train/'
categories = list(np.loadtxt('category.csv', delimiter=',', skiprows=1, usecols=1, dtype=str))
n_categories = len(categories)
image_names_arr = np.loadtxt('train.csv', delimiter=',', skiprows=1, usecols=1, dtype=str)
labels_str = list(np.loadtxt('train.csv', delimiter=',', skiprows=1, usecols=2, dtype=str))
targets = [categories.index(lab) for lab in labels_str]
targets_arr = np.asarray(targets, dtype=str)

dir_train = '../tmp/minichallenge_data/train_cropped_multifeature/'
dir_test = '../tmp/minichallenge_data/test_cropped_multifeature/'

# Make directories if they do not yet exist
os.makedirs(os.path.dirname(dir_train), exist_ok=True)
os.makedirs(os.path.dirname(dir_test), exist_ok=True)

face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
mouth_classifier = cv2.CascadeClassifier('haarcascade_mcs_mouth.xml')
nose_classifier = cv2.CascadeClassifier('haarcascade_mcs_nose.xml')
profile_classifier = cv2.CascadeClassifier('haarcascade_profileface.xml')

if SMALL_DATA_SET:
    # Try small set of data to ensure implementation is ok
    image_names_arr = image_names_arr[:100]
    targets_arr = targets_arr[:100]


def crop_data(_dir_uncropped, _dir_cropped, _img_names, _classifier, _sf=1.1, _mn=5, save_raw=True):
    print(f'--- Cropping Images in {_dir_uncropped} ---')
    _n_images = len(_img_names)
    _success = np.zeros(shape=(_n_images,), dtype=bool)
    for _idx, _image in enumerate(_img_names):
        # Attempt to load image and move on if it already exists
        try:
            with Image.open(_dir_cropped + _image) as _img_raw:
                _img_cropped = np.array(_img_raw.convert('L'))
                _success[_idx] = True
                continue
        except FileNotFoundError as _:
            if _idx % 100 == 0:
                print(f'Trying image {_idx + 1}/{_n_images}...')

        try:
            with Image.open(_dir_uncropped + _image) as _img_raw:
                _img = np.array(_img_raw.convert('L'))  # Create HxW NumPy array of grayscale image
            if _img is None:
                raise RuntimeError('Image unable to load!')

            # Find faces
            if _classifier is not None:
                _detections, _confidence = _classifier.detectMultiScale2(_img, scaleFactor=_sf, minNeighbors=_mn)
            else:
                _detections, _confidence = [], []
            if len(_detections) == 0:
                # No faces found, default to using whole image
                _x, _y = 0, 0
                _h, _w = _img.shape

                if not save_raw:
                    continue
            else:
                # Face(s) found, run classifier on most confident face
                _x, _y, _w, _h = _detections[np.asarray(_confidence).argmax()]
                _success[_idx] = True

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


def separate_train_and_validation(_images, _targets):
    _idces_train = []
    _idces_validation = []
    _targets_int = np.asarray(_targets, dtype=int)
    _targets_str = np.asarray(_targets, dtype=str)

    for _category in range(n_categories):
        _labels_cat = np.nonzero(_targets_int == _category)[0]
        _idx_validation = int(len(_labels_cat) * FRAC_VALIDATION)
        _idces_train.extend(_labels_cat[_idx_validation:])
        _idces_validation.extend(_labels_cat[:_idx_validation])

    _data_validation = np.hstack((
        _targets_str[_idces_validation].reshape((-1, 1)),
        _images[_idces_validation].reshape((-1, 1))
    ))
    _data_train = np.hstack((
        _targets_str[_idces_train].reshape((-1, 1)),
        _images[_idces_train].reshape((-1, 1))
    ))

    return _data_train, _data_validation


# CROP TRAINING DATA ------------------------------------------------------------------------------------------------- #
# # Raw images
# os.makedirs(os.path.dirname(dir_train + 'raw/'), exist_ok=True)
# crop_data(root, dir_train + 'raw/', image_names_arr, _classifier=None, save_raw=True)
# data_raw_train, data_raw_validation = separate_train_and_validation(
#     image_names_arr, targets_arr
# )
# np.savetxt('data_raw_train.csv', data_raw_train, delimiter=',', fmt='%s')
# np.savetxt('data_raw_validation.csv', data_raw_validation, delimiter=',', fmt='%s')
#
# # Try to crop out faces
# os.makedirs(os.path.dirname(dir_train + 'faces/'), exist_ok=True)
# successful_crop_face_frontal_train = crop_data(
#     root, dir_train + 'faces/', image_names_arr, _classifier=face_classifier, _mn=20, save_raw=False
# )
#
# # For pictures where face was not found, try finding the profile of the face
# successful_crop_face_train = successful_crop_face_frontal_train
# successful_crop_face_profile = crop_data(
#     root, dir_train + 'faces/', image_names_arr[~successful_crop_face_frontal_train], _classifier=profile_classifier, _mn=20,
#     save_raw=False
# )
# successful_crop_face_train[~successful_crop_face_train] = successful_crop_face_profile
#
# np.savetxt(dir_train + 'faces/' + 'successful_crop.csv', successful_crop_face_train, delimiter=',', fmt='%s')
# data_face_train, data_face_validation = separate_train_and_validation(
#     image_names_arr[successful_crop_face_train], targets_arr[successful_crop_face_train]
# )
# np.savetxt('data_face_train.csv', data_face_train, delimiter=',', fmt='%s')
# np.savetxt('data_face_validation.csv', data_face_validation, delimiter=',', fmt='%s')
#
# # Try to crop out nose
# os.makedirs(os.path.dirname(dir_train + 'nose/'), exist_ok=True)
# successful_crop_nose_train = crop_data(
#     root, dir_train + 'nose/', image_names_arr, _classifier=nose_classifier, _mn=20, save_raw=False
# )
# np.savetxt(dir_train + 'nose/' + 'successful_crop.csv', successful_crop_nose_train, delimiter=',', fmt='%s')
# data_nose_train, data_nose_validation = separate_train_and_validation(
#     image_names_arr[successful_crop_nose_train], targets_arr[successful_crop_nose_train]
# )
# np.savetxt('data_nose_train.csv', data_nose_train, delimiter=',', fmt='%s')
# np.savetxt('data_nose_validation.csv', data_nose_validation, delimiter=',', fmt='%s')

# # Try to crop out mouth
# os.makedirs(os.path.dirname(dir_train + 'mouth/'), exist_ok=True)
# successful_crop_mouth_train = crop_data(
#     root, dir_train + 'mouth/', image_names_arr, _classifier=mouth_classifier, _mn=20, save_raw=False
# )
# np.savetxt(dir_train + 'mouth/' + 'successful_crop.csv', successful_crop_mouth_train, delimiter=',', fmt='%s')
# data_mouth_train, data_mouth_validation = separate_train_and_validation(
#     image_names_arr[successful_crop_mouth_train], targets_arr[successful_crop_mouth_train]
# )
# np.savetxt('data_mouth_train.csv', data_mouth_train, delimiter=',', fmt='%s')
# np.savetxt('data_mouth_validation.csv', data_mouth_validation, delimiter=',', fmt='%s')

# CROP TEST DATA ----------------------------------------------------------------------------------------------------- #
if not SMALL_DATA_SET:  # TODO - update
    image_names_test_arr = np.asarray(np.arange(0, 4977, 1, dtype=int), dtype=str)
    root_test = '../tmp/minichallenge_data/test/'
    for idx in range(len(image_names_test_arr)):
        image_names_test_arr[idx] += '.jpg'

    # Save raw images
    os.makedirs(os.path.dirname(dir_test + 'raw/'), exist_ok=True)
    crop_data(root_test, dir_test + 'raw/', image_names_test_arr, _classifier=None, save_raw=True)
    np.savetxt('data_raw_test.csv', image_names_test_arr, delimiter=',', fmt='%s')

    # Try to crop out faces
    os.makedirs(os.path.dirname(dir_test + 'faces/'), exist_ok=True)
    successful_crop_face_frontal_test = crop_data(
        root_test, dir_test + 'faces/', image_names_test_arr, _classifier=face_classifier, _mn=20, save_raw=False
    )

    # For pictures where face was not found, try finding the profile of the face
    successful_crop_face_test = successful_crop_face_frontal_test
    successful_crop_face_profile_test = crop_data(
        root_test, dir_test + 'faces/', image_names_test_arr[~successful_crop_face_test], _classifier=profile_classifier,
        _mn=20,
        save_raw=False
    )
    successful_crop_face_test[~successful_crop_face_test] = successful_crop_face_profile_test

    np.savetxt(dir_test + 'faces/' + 'successful_crop.csv', successful_crop_face_test, delimiter=',', fmt='%s')
    np.savetxt('data_face_test.csv', image_names_test_arr[successful_crop_face_test], delimiter=',', fmt='%s')

    # Try to crop out nose
    os.makedirs(os.path.dirname(dir_test + 'nose/'), exist_ok=True)
    successful_crop_nose_test = crop_data(
        root_test, dir_test + 'nose/', image_names_test_arr, _classifier=nose_classifier, _mn=20, save_raw=False
    )
    np.savetxt(dir_test + 'nose/' + 'successful_crop.csv', successful_crop_nose_test, delimiter=',', fmt='%s')
    np.savetxt('data_nose_test.csv', image_names_test_arr[successful_crop_nose_test], delimiter=',', fmt='%s')

    # Try to crop out mouth
    os.makedirs(os.path.dirname(dir_test + 'mouth/'), exist_ok=True)
    successful_crop_mouth_test = crop_data(
        root_test, dir_test + 'mouth/', image_names_test_arr, _classifier=mouth_classifier, _mn=20, save_raw=False
    )
    np.savetxt('data_mouth_test.csv', image_names_test_arr[successful_crop_mouth_test], delimiter=',', fmt='%s')
