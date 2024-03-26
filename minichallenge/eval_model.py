import pickle

import numpy as np
import torch
from torchvision.io import ImageReadMode

from model_classes import ImageDataset

with open('canonical/model_acc50.pickle', 'rb') as file:
    model = pickle.load(file)
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

# Evaluate model ----------------------------------------------------------------------------------------------------- #
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
idx = 0
n_batches = len(test_loader)
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
data_to_save = np.hstack((
    np.asarray(image_id_arr, dtype=str).reshape((-1, 1)),
    celebrity_names_arr.reshape((-1, 1))
))
np.savetxt('test_results.csv', data_to_save, fmt='%s', delimiter=',', header='Id,Category', comments='')
