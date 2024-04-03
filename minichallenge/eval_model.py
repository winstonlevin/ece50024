import pickle

import numpy as np
import torch
from torchvision.io import ImageReadMode
from torchvision.transforms import transforms

from model_classes import ImageDataset

with open('canonical/model_9CNN.pickle', 'rb') as file:
    model = pickle.load(file)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

# Choose best params
validation_acc_arr = np.asarray(model.test_accuracies[-1]).flatten()
state_dict_arr = np.asarray(model.state_dicts[-1], dtype=object).flatten()
idx_use = validation_acc_arr.argmax()
model.load_state_dict(state_dict_arr[idx_use])
validation_acc = validation_acc_arr[idx_use]
validation_acc_raw = np.asarray(model.test_accuracies_raw[-1]).flatten()[idx_use]
validation_acc_face = np.asarray(model.test_accuracies_face[-1]).flatten()[idx_use]
validation_acc_nose = np.asarray(model.test_accuracies_nose[-1]).flatten()[idx_use]
validation_acc_mouth = np.asarray(model.test_accuracies_mouth[-1]).flatten()[idx_use]

if hasattr(model, 'n_pixels'):
    transform = transforms.Resize((model.n_pixels, model.n_pixels))
else:
    with open('canonical/transform_acc50.pickle', 'rb') as file:
        transform = pickle.load(file)

image_read_mode = ImageReadMode.GRAY

# Mapping from numerical categories to text category
categories_arr = np.loadtxt('category.csv', delimiter=',', skiprows=1, usecols=1, dtype=str)
n_categories = len(categories_arr)

# Test datasets
root_test = '../tmp/minichallenge_data/test_cropped_multifeature/'
images_test_raw = np.loadtxt('data_raw_test.csv', delimiter=',', dtype=str)
targets_test_raw = np.zeros(images_test_raw.shape, dtype=int)
images_test_face = np.loadtxt('data_face_test.csv', delimiter=',', dtype=str)
targets_test_face = np.zeros(images_test_raw.shape, dtype=int)
images_test_nose = np.loadtxt('data_nose_test.csv', delimiter=',', dtype=str)
targets_test_nose = np.zeros(images_test_raw.shape, dtype=int)
images_test_mouth = np.loadtxt('data_mouth_test.csv', delimiter=',', dtype=str)
targets_test_mouth = np.zeros(images_test_raw.shape, dtype=int)

# Determine which images have features
has_face_feature = np.zeros(shape=targets_test_raw.shape, dtype=bool)
has_nose_feature = np.zeros(shape=targets_test_raw.shape, dtype=bool)
has_mouth_feature = np.zeros(shape=targets_test_raw.shape, dtype=bool)
for idx, img in enumerate(images_test_raw):
    has_face_feature[idx] = img in images_test_face
    has_nose_feature[idx] = img in images_test_nose
    has_mouth_feature[idx] = img in images_test_mouth

dataset_test_raw = ImageDataset(
    root=root_test, images=images_test_raw, targets=targets_test_raw, n_targets=n_categories,
    transform=transform, image_read_mode=image_read_mode
)
dataset_test_face = ImageDataset(
    root=root_test, images=images_test_face, targets=targets_test_face, n_targets=n_categories,
    transform=transform, image_read_mode=image_read_mode
)
dataset_test_nose = ImageDataset(
    root=root_test, images=images_test_nose, targets=targets_test_nose, n_targets=n_categories,
    transform=transform, image_read_mode=image_read_mode
)
dataset_test_mouth = ImageDataset(
    root=root_test, images=images_test_mouth, targets=targets_test_mouth, n_targets=n_categories,
    transform=transform, image_read_mode=image_read_mode
)
batch_size = 64
test_loader_raw = torch.utils.data.DataLoader(dataset_test_raw, batch_size=batch_size, shuffle=False)
test_loader_face = torch.utils.data.DataLoader(dataset_test_face, batch_size=batch_size, shuffle=False)
test_loader_nose = torch.utils.data.DataLoader(dataset_test_nose, batch_size=batch_size, shuffle=False)
test_loader_mouth = torch.utils.data.DataLoader(dataset_test_mouth, batch_size=batch_size, shuffle=False)

# Run model on validation set with chosen parameters ----------------------------------------------------------------- #
root_validation = '../tmp/minichallenge_data/train_cropped_multifeature/'

# Validation data
targets_raw_validation = np.loadtxt('data_raw_validation.csv', delimiter=',', usecols=0, dtype=int)
images_raw_validation = np.loadtxt('data_raw_validation.csv', delimiter=',', usecols=1, dtype=str)
for idx in range(len(images_raw_validation)):
    images_raw_validation[idx] += 'raw/'
targets_face_validation = np.loadtxt('data_face_validation.csv', delimiter=',', usecols=0, dtype=int)
images_face_validation = np.loadtxt('data_face_validation.csv', delimiter=',', usecols=1, dtype=str)
for idx in range(len(images_face_validation)):
    images_face_validation[idx] += 'faces/'
targets_nose_validation = np.loadtxt('data_nose_validation.csv', delimiter=',', usecols=0, dtype=int)
images_nose_validation = np.loadtxt('data_nose_validation.csv', delimiter=',', usecols=1, dtype=str)
for idx in range(len(images_nose_validation)):
    images_nose_validation[idx] += 'nose/'
targets_mouth_validation = np.loadtxt('data_mouth_validation.csv', delimiter=',', usecols=0, dtype=int)
images_mouth_validation = np.loadtxt('data_mouth_validation.csv', delimiter=',', usecols=1, dtype=str)
for idx in range(len(images_mouth_validation)):
    images_mouth_validation[idx] += 'mouth/'

has_face_feature_validation = np.zeros(shape=targets_test_raw.shape, dtype=bool)
has_nose_feature_validation = np.zeros(shape=targets_test_raw.shape, dtype=bool)
has_mouth_feature_validation = np.zeros(shape=targets_test_raw.shape, dtype=bool)
for idx, img in enumerate(images_raw_validation):
    has_face_feature_validation[idx] = img in images_face_validation
    has_nose_feature_validation[idx] = img in images_nose_validation
    has_mouth_feature_validation[idx] = img in images_mouth_validation

idces_raw_validation = np.arange(0, len(targets_raw_validation), 1)
idces_face_validation, = np.nonzero(has_face_feature_validation)
idces_nose_validation, = np.nonzero(has_nose_feature_validation)
idces_mouth_validation, = np.nonzero(has_mouth_feature_validation)

dataset_validation_raw = ImageDataset(
    root=root_validation, images=list(images_raw_validation), targets=list(targets_raw_validation), n_targets=n_categories,
    transform=transform, image_read_mode=image_read_mode
)
dataset_validation_face = ImageDataset(
    root=root_validation, images=list(images_face_validation), targets=list(targets_face_validation), n_targets=n_categories,
    transform=transform, image_read_mode=image_read_mode
)
dataset_validation_nose = ImageDataset(
    root=root_validation, images=list(images_nose_validation), targets=list(targets_nose_validation), n_targets=n_categories,
    transform=transform, image_read_mode=image_read_mode
)
dataset_validation_mouth = ImageDataset(
    root=root_validation, images=list(images_mouth_validation), targets=list(targets_mouth_validation), n_targets=n_categories,
    transform=transform, image_read_mode=image_read_mode
)

validation_raw_loader = torch.utils.data.DataLoader(dataset_validation_raw, batch_size=batch_size, shuffle=False)
validation_face_loader = torch.utils.data.DataLoader(dataset_validation_face, batch_size=batch_size, shuffle=False)
validation_nose_loader = torch.utils.data.DataLoader(dataset_validation_nose, batch_size=batch_size, shuffle=False)
validation_mouth_loader = torch.utils.data.DataLoader(dataset_validation_mouth, batch_size=batch_size, shuffle=False)


def add_validation_prediction(_loader, _acc, _targets, _idces_predict):
    _predictions = -np.ones(shape=_targets.shape, dtype=int)
    n_validation = _targets.shape[0]
    idx = 0
    n_batches = len(_loader)
    print(f'--- Validating Model with {_acc / 100:%} Accuracy on Validation Set ---')
    for current_batch, (inputs, _) in enumerate(_loader, start=1):
        if current_batch % 10 == 1:
            print(f'Validation batch {current_batch}/{n_batches}...')
        inputs = inputs.to(device)
        outputs = model(inputs)
        _, prediction = torch.max(outputs.data, 1)
        n_predicted = len(prediction)
        _predictions[_idces_predict][idx:idx + n_predicted] = prediction.cpu()
        idx += n_predicted

    # Save compiled validation results
    celebrity_names_arr = categories_arr[_predictions]
    file_name_compilation = 'compiled_validation_results.csv'
    celeb_names_with_header = np.vstack((
        np.asarray((_acc,), dtype=str).reshape((-1, 1)),
        celebrity_names_arr.reshape((-1, 1))
    ))
    try:
        previous_results_all = np.loadtxt(file_name_compilation, dtype=str, delimiter=',').reshape(
            (n_validation + 1, -1))
        results_all = np.hstack((previous_results_all, celeb_names_with_header))
        print('Appending new results to compiled results...')
    except FileNotFoundError as _:
        print('Generating a compiled results...')
        results_all = celeb_names_with_header
    np.savetxt(file_name_compilation, results_all, fmt='%s', delimiter=',')


add_validation_prediction(validation_raw_loader, validation_acc_raw, targets_raw_validation, idces_raw_validation)
add_validation_prediction(validation_face_loader, validation_acc_face, targets_raw_validation, idces_face_validation)
add_validation_prediction(validation_nose_loader, validation_acc_nose, targets_raw_validation, idces_nose_validation)
add_validation_prediction(validation_mouth_loader, validation_acc_mouth, targets_raw_validation, idces_mouth_validation)


# Evaluate model ----------------------------------------------------------------------------------------------------- #
def add_evaluation(_loader, _acc, _targets, _idces_predict):
    _predictions = -np.ones(shape=_targets.shape, dtype=int)
    n_validation = _targets.shape[0]
    idx = 0
    n_batches = len(_loader)
    print(f'--- Evaluating Model with {_acc/100:%} Accuracy on Validation Set ---')
    for current_batch, (inputs, _) in enumerate(_loader, start=1):
        if current_batch % 10 == 1:
            print(f'Evaluating batch {current_batch}/{n_batches}...')
        inputs = inputs.to(device)
        outputs = model(inputs)
        _, prediction = torch.max(outputs.data, 1)
        n_predicted = len(prediction)
        _predictions[_idces_predict][idx:idx + n_predicted] = prediction.cpu()
        idx += n_predicted

    # Save compiled validation results (assume test acc. is 98% of validation acc.)
    celebrity_names_arr = categories_arr[_predictions]
    file_name_compilation = 'compiled_test_results.csv'
    celeb_names_with_header = np.vstack((
        np.asarray((0.98*_acc,), dtype=str).reshape((-1, 1)),
        celebrity_names_arr.reshape((-1, 1))
    ))
    try:
        previous_results_all = np.loadtxt(file_name_compilation, dtype=str, delimiter=',').reshape(
            (n_validation + 1, -1))
        results_all = np.hstack((previous_results_all, celeb_names_with_header))
        print('Appending new results to compiled results...')
    except FileNotFoundError as _:
        print('Generating a compiled results...')
        results_all = celeb_names_with_header
    np.savetxt(file_name_compilation, results_all, fmt='%s', delimiter=',')
