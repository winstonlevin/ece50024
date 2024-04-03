import os
import pickle
import time

import numpy as np
from matplotlib import pyplot as plt
import torch
from torch import nn
from torchvision import transforms
from torchvision.io import ImageReadMode

from model_classes import ImageDataset, ImageClassifier, train, validate

# Hyperparameters/Transformation of images --------------------------------------------------------------------------- #
# n_filters = 64
batch_size = 32
kernel_size = 3
pool_size = 2
n_dense_layers = 1
activation_type = 'LeakyReLU'
pool_type = 'AdaptiveAvgPool2d'
n_pixels = 128
n_filters = 64
n_convs_per_layer = (2, 2, 2)  # 3 layers -> 16x16 is final image width/height
n_conv_layers = len(n_convs_per_layer)
n_pixels_after_pooling = 2  # Pool 16x16 to 2x2 and leave 64x2x2 features
use_pool = True
learning_rate = 1e-3

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
criterion = nn.CrossEntropyLoss()

# Initialize the model, loss function, and optimizer
model = ImageClassifier(
    n_pixels=n_pixels, grayscale=True, n_pixel_after_pooling=n_pixels_after_pooling,
    n_filters=n_filters, kernel_size=3, pool_size=pool_size, pool_type=pool_type,
    n_conv_layers=n_conv_layers, n_dense_layers=n_dense_layers, activation_type=activation_type,
    n_convs_per_layer=n_convs_per_layer, use_pool=use_pool, n_outputs=n_categories
).to(device)
print(f'Model has:\n{model.n_features} Feat. and {model.n_parameters} Params.')
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

curriculum_epochs_max = [50]
include_no_image = True

image_read_mode = ImageReadMode.GRAY
transform_test = transforms.Resize((n_pixels, n_pixels))
transform_train = transform_test

# Load training and validation data ---------------------------------------------------------------------------------- #
root = '../tmp/minichallenge_data/train_cropped/'
categories = list(np.loadtxt('category.csv', delimiter=',', skiprows=1, usecols=1, dtype=str))
n_categories = len(categories)

# Training data
targets_raw_train = np.loadtxt('data_raw_train.csv', delimiter=',', usecols=0, dtype=int)
images_raw_train = np.loadtxt('data_raw_train.csv', delimiter=',', usecols=1, dtype=str)
for idx in range(len(images_raw_train)):
    images_raw_train[idx] += 'raw/'
targets_face_train = np.loadtxt('data_face_train.csv', delimiter=',', usecols=0, dtype=int)
images_face_train = np.loadtxt('data_face_train.csv', delimiter=',', usecols=1, dtype=str)
for idx in range(len(images_face_train)):
    images_face_train[idx] += 'faces/'
targets_nose_train = np.loadtxt('data_nose_train.csv', delimiter=',', usecols=0, dtype=int)
images_nose_train = np.loadtxt('data_nose_train.csv', delimiter=',', usecols=1, dtype=str)
for idx in range(len(images_nose_train)):
    images_nose_train[idx] += 'nose/'
targets_mouth_train = np.loadtxt('data_mouth_train.csv', delimiter=',', usecols=0, dtype=int)
images_mouth_train = np.loadtxt('data_mouth_train.csv', delimiter=',', usecols=1, dtype=str)
for idx in range(len(images_mouth_train)):
    images_mouth_train[idx] += 'mouth/'

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

# Make Data loaders -------------------------------------------------------------------------------------------------- #
# Add 1% no-category inputs
n_no_category = include_no_image * len(images_raw_train) // 100

images_train = list(images_raw_train)
images_train.extend(images_face_train)
images_train.extend(images_nose_train)
images_train.extend(images_mouth_train)
images_train.extend(n_no_category * [ImageDataset.NO_IMAGE])

targets_train = list(targets_raw_train)
targets_train.extend(targets_face_train)
targets_train.extend(targets_nose_train)
targets_train.extend(targets_mouth_train)
targets_train.extend(n_no_category * [n_categories + 1])

dataset_train = ImageDataset(
    root=root, images=images_train, targets=targets_train, n_targets=n_categories,
    transform=transform_train, image_read_mode=image_read_mode
)
dataset_validation_raw = ImageDataset(
    root=root, images=list(images_raw_validation), targets=list(targets_raw_validation), n_targets=n_categories,
    transform=transform_test, image_read_mode=image_read_mode
)
dataset_validation_face = ImageDataset(
    root=root, images=list(images_face_validation), targets=list(targets_face_validation), n_targets=n_categories,
    transform=transform_test, image_read_mode=image_read_mode
)
dataset_validation_nose = ImageDataset(
    root=root, images=list(images_nose_validation), targets=list(targets_nose_validation), n_targets=n_categories,
    transform=transform_test, image_read_mode=image_read_mode
)
dataset_validation_mouth = ImageDataset(
    root=root, images=list(images_mouth_validation), targets=list(targets_mouth_validation), n_targets=n_categories,
    transform=transform_test, image_read_mode=image_read_mode
)

train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
validation_raw_loader = torch.utils.data.DataLoader(dataset_validation_raw, batch_size=batch_size, shuffle=False)
validation_face_loader = torch.utils.data.DataLoader(dataset_validation_face, batch_size=batch_size, shuffle=False)
validation_nose_loader = torch.utils.data.DataLoader(dataset_validation_nose, batch_size=batch_size, shuffle=False)
validation_mouth_loader = torch.utils.data.DataLoader(dataset_validation_mouth, batch_size=batch_size, shuffle=False)

n_valid_raw = len(dataset_validation_raw)
n_valid_face = len(dataset_validation_face)
n_valid_nose = len(dataset_validation_nose)
n_valid_mouth = len(dataset_validation_mouth)
n_valid = n_valid_raw + n_valid_face + n_valid_nose + n_valid_mouth

# Train NN ----------------------------------------------------------------------------------------------------------- #
# Save Model
dir_save = '../tmp/models/'
current_time = time.gmtime()
date = f'{current_time.tm_year:04d}-{current_time.tm_mon:02d}-{current_time.tm_mday:02d}'
hour = f'{current_time.tm_hour:02d}-{current_time.tm_min:02d}-{current_time.tm_sec:02d}'
file_name_transform = f'Celebrity_transform_{date}_{hour}.pickle'
os.makedirs(os.path.dirname(dir_save), exist_ok=True)  # Make directory if it does not yet exist
with open(dir_save + file_name_transform, 'wb') as f:
    pickle.dump(transform_test, f, protocol=pickle.HIGHEST_PROTOCOL)

# Training loop
for curr_idx, epochs_max in enumerate(curriculum_epochs_max):
    print(f"\nCurriculum Stage #{curr_idx+1}")
    model.train_losses.append([])
    model.test_accuracies.append([])
    model.test_accuracies_raw.append([])
    model.test_accuracies_face.append([])
    model.test_accuracies_nose.append([])
    model.test_accuracies_mouth.append([])
    model.state_dicts.append([])
    for epoch in range(epochs_max):
        elapsed_time = -time.perf_counter()  # -t0
        train_loss = train(
            model, train_loader, optimizer, criterion, device, verbose=True, include_no_image=include_no_image
        )
        model.train_losses[-1].append(train_loss)
        acc_raw = validate(model, validation_raw_loader, device)
        acc_face = validate(model, validation_face_loader, device)
        acc_nose = validate(model, validation_nose_loader, device)
        acc_mouth = validate(model, validation_mouth_loader, device)

        acc_overall = (n_valid_raw * acc_raw
                       + n_valid_face * acc_face
                       + n_valid_nose * acc_nose
                       + n_valid_mouth * acc_mouth) / n_valid

        model.test_accuracies[-1].append(acc_overall)
        model.test_accuracies_raw[-1].append(acc_raw)
        model.test_accuracies_face[-1].append(acc_face)
        model.test_accuracies_nose[-1].append(acc_nose)
        model.test_accuracies_mouth[-1].append(acc_mouth)
        model.state_dicts[-1].append(model.state_dict().copy())

        elapsed_time += time.perf_counter()  # Result is tf - t0
        elapsed_hours = int(elapsed_time // 3600)
        elapsed_minutes = int((elapsed_time - 3600*elapsed_hours) // 60)
        elapsed_seconds = int(elapsed_time - 3600*elapsed_hours - 60 * elapsed_minutes)
        print(
            f"Epoch {epoch + 1}/{epochs_max}, Train Loss: {train_loss:.4f}, "
            f"Test Accuracy: {acc_overall/100:.2%}, "
            f"Epoch Time: {elapsed_hours:02d}:{elapsed_minutes:02d}:{elapsed_seconds:02d}"
        )
        if epoch > 2 and model.test_accuracies[-1][-1] < model.test_accuracies[-1][-2] < model.test_accuracies[-1][-3]:
            # Unsuccessful break (accuracy is diminishing, so overtraining is occurring)
            print('Overtraining detected on validation set, ending curriculum stage early!')
            break

        # Save Model
        current_time = time.gmtime()
        date = f'{current_time.tm_year:04d}-{current_time.tm_mon:02d}-{current_time.tm_mday:02d}'
        hour = f'{current_time.tm_hour:02d}-{current_time.tm_min:02d}-{current_time.tm_sec:02d}'
        file_name = f'Celebrity_model_epoch{epoch}_{date}_{hour}.pickle'
        with open(dir_save + file_name, 'wb') as f:
            pickle.dump(model, f, protocol=pickle.HIGHEST_PROTOCOL)

# Plotting the training loss and test accuracy ----------------------------------------------------------------------- #
fig_accuracy = plt.figure(figsize=(10, 5))

ax_loss = fig_accuracy.add_subplot(121)
ax_loss.grid()
ax_loss.plot(model.train_losses[-1])
ax_loss.set_xlabel('Epoch')
ax_loss.set_ylabel('Training Loss')
ax_loss.set_title('Training Loss vs. Epoch')

ax_accuracy = fig_accuracy.add_subplot(122)
ax_accuracy.grid()
ax_accuracy.plot(model.test_accuracies[-1])
ax_accuracy.set_xlabel('Epoch')
ax_accuracy.set_ylabel('Test Accuracy (%)')
ax_accuracy.set_title('Test Accuracy vs. Epoch')

fig_accuracy.tight_layout()

plt.show()
