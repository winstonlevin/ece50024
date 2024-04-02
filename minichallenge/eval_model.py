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
validation_acc_arr = np.asarray(model.test_accuracies[-1])
if hasattr(model, 'state_dicts'):
    state_dict_arr = np.asarray(model.state_dicts[-1], dtype=object)

    model.load_state_dict(state_dict_arr.flatten()[validation_acc_arr.argmax()])
    validation_acc = validation_acc_arr.flatten()[validation_acc_arr.argmax()]
else:
    validation_acc = validation_acc_arr.flatten()[-1]

if hasattr(model, 'n_pixels'):
    transform = transforms.Resize((model.n_pixels, model.n_pixels))
else:
    with open('canonical/transform_acc50.pickle', 'rb') as file:
        transform = pickle.load(file)

image_read_mode = ImageReadMode.GRAY

# Mapping from numerical categories to text category
categories_arr = np.loadtxt('category.csv', delimiter=',', skiprows=1, usecols=1, dtype=str)
n_categories = len(categories_arr)

# Test dataset
root_test = '../tmp/minichallenge_data/test_cropped/'
images_test = np.loadtxt('data_test.csv', delimiter=',', usecols=1, dtype=str)
targets_test = np.empty(images_test.shape, dtype=int)

dataset_test = ImageDataset(
    root=root_test, images=images_test, targets=targets_test, n_targets=n_categories,
    transform=transform, image_read_mode=image_read_mode
)
batch_size = 64
test_loader = torch.utils.data.DataLoader(dataset_test, batch_size=batch_size, shuffle=False)

# Run model on validation set with chosen parameters ----------------------------------------------------------------- #
root_validation = '../tmp/minichallenge_data/train_cropped/'
images_validation = np.loadtxt('data_validation.csv', delimiter=',', usecols=1, dtype=str)
targets_validation = np.loadtxt('data_validation.csv', delimiter=',', usecols=0, dtype=int)
dataset_validation = ImageDataset(
    root=root_validation, images=images_validation, targets=targets_validation, n_targets=n_categories,
    transform=transform, image_read_mode=image_read_mode
)
validation_loader = torch.utils.data.DataLoader(dataset_validation, batch_size=batch_size, shuffle=False)
n_validation, = targets_validation.shape
prediction_validation = np.empty(shape=targets_validation.shape, dtype=int)

idx = 0
n_batches = len(validation_loader)
print(f'--- Validating Model with {validation_acc/100:%} Accuracy on Validation Set ---')
for current_batch, (inputs, _) in enumerate(validation_loader, start=1):
    if current_batch % 10 == 1:
        print(f'Validation batch {current_batch}/{n_batches}...')
    inputs = inputs.to(device)
    outputs = model(inputs)
    _, prediction = torch.max(outputs.data, 1)
    n_predicted = len(prediction)
    prediction_validation[idx:idx + n_predicted] = prediction.cpu()
    idx += n_predicted

# Save compiled validation results
celebrity_names_arr = categories_arr[prediction_validation]
file_name_compilation = 'compiled_validation_results.csv'
celeb_names_with_header = np.vstack((
    np.asarray((validation_acc,), dtype=str).reshape((-1, 1)),
    celebrity_names_arr.reshape((-1, 1))
))
try:
    previous_results_all = np.loadtxt(file_name_compilation, dtype=str, delimiter=',').reshape((n_validation+1, -1))
    results_all = np.hstack((previous_results_all, celeb_names_with_header))
    print('Appending new results to compiled results...')
except FileNotFoundError as _:
    print('Generating a compiled results...')
    results_all = celeb_names_with_header
np.savetxt(file_name_compilation, results_all, fmt='%s', delimiter=',')

# Evaluate model ----------------------------------------------------------------------------------------------------- #
idx = 0
n_batches = len(test_loader)
print(f'--- Evaluating Model with {validation_acc/100:%} Accuracy on Validation Set ---')
for current_batch, (inputs, _) in enumerate(test_loader, start=1):
    if current_batch % 10 == 1:
        print(f'Testing batch {current_batch}/{n_batches}...')
    inputs = inputs.to(device)
    outputs = model(inputs)
    _, prediction = torch.max(outputs.data, 1)
    n_predicted = len(prediction)
    targets_test[idx:idx + n_predicted] = prediction.cpu()
    idx += n_predicted

# Save data in correct format ---------------------------------------------------------------------------------------- #
image_id_arr = np.arange(0, len(images_test), 1)
celebrity_names_arr = categories_arr[targets_test]
file_name = 'test_results.csv'

# Append to running list of all results
# (Assume the test accuracy will by 98% of the validation accuracy)
file_name_compilation = 'compiled_test_results.csv'
celeb_names_with_header = np.vstack((
    np.asarray((0.98*validation_acc,), dtype=str).reshape((-1, 1)),
    celebrity_names_arr.reshape((-1, 1))
))
n_test = len(images_test)
try:
    previous_results_all = np.loadtxt(file_name_compilation, dtype=str, delimiter=',').reshape((n_test+1, -1))
    results_all = np.hstack((previous_results_all, celeb_names_with_header))
    print('Appending new results to compiled results...')
except FileNotFoundError as _:
    print('Generating a compiled results...')
    results_all = celeb_names_with_header
np.savetxt(file_name_compilation, results_all, fmt='%s', delimiter=',')

# # Compare to previous results and overwrite results
#
# data_to_save = np.hstack((
#     np.asarray(image_id_arr, dtype=str).reshape((-1, 1)),
#     celebrity_names_arr.reshape((-1, 1))
# ))
# try:
#     previous_celebrity_names = np.loadtxt(file_name, dtype=str, skiprows=1, usecols=1, delimiter=',')
#     num_changed = n_test - np.sum(celebrity_names_arr == previous_celebrity_names)
#     print(f'Saving new results to "{file_name}", '
#           f'{num_changed}/{n_test} ({num_changed/n_test:.2%}) changed from previous results.')
# except FileNotFoundError as _:
#     print(f'Saving results to "{file_name}".')
# np.savetxt(file_name, data_to_save, fmt='%s', delimiter=',', header='Id,Category', comments='')
