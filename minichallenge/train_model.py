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
n_filters = 64
batch_size = 32
# n_pixels = 64
kernel_size = 3
pool_size = 2
n_dense_layers = 1
activation_type = 'LeakyReLU'
n_pixels = 128
n_convs_per_layer = (2, 2, 2,)  # 3 layers -> 16x16 is final image width/height
n_conv_layers = len(n_convs_per_layer)
n_pixels_after_pooling = 2  # Pool 16x16 to 2x2 and leave 64x2x2 features
use_pool = True
learning_rate = 1e-3

curriculum_n_categories = [100]
curriculum_accuracies = len(curriculum_n_categories) * [50]
curriculum_accuracies[-1] = 99
curriculum_epochs_max = len(curriculum_n_categories) * [10]
curriculum_epochs_max[-1] = 25
include_no_image = True
pad_image_data = False  # Ensure equal amount of data for each target

image_read_mode = ImageReadMode.GRAY
# # Get pseudo data from transformations
# transform_train = transforms.Compose((
#     transforms.RandomAutocontrast(0.05),  # Change contrast randomly
#     transforms.RandomPerspective(0.05),  # Distort image to practice multiple perpectives
#     transforms.RandomRotation(5),  # Rotate image to practice at multiple angles
#     transforms.Resize((n_pixels, n_pixels)),  # Ensure all images are same size
# ))
transform_test = transforms.Resize((n_pixels, n_pixels))
transform_train = transform_test

# Load training and validation data ---------------------------------------------------------------------------------- #
root = '../tmp/minichallenge_data/train_cropped/'
categories = list(np.loadtxt('category.csv', delimiter=',', skiprows=1, usecols=1, dtype=str))
n_categories = len(categories)
targets_train = np.loadtxt('data_train.csv', delimiter=',', usecols=0, dtype=int)
images_train = np.loadtxt('data_train.csv', delimiter=',', usecols=1, dtype=str)
targets_validation = np.loadtxt('data_validation.csv', delimiter=',', usecols=0, dtype=int)
images_validation = np.loadtxt('data_validation.csv', delimiter=',', usecols=1, dtype=str)

if pad_image_data:
    images_train_list = list(images_train)
    targets_train_list = list(targets_train)

    image_is_target = targets_train == np.arange(0, n_categories, 1).reshape((-1, 1))
    n_images = np.sum(image_is_target, axis=1)
    n_pad = np.max(n_images) - n_images

    for idx_pad in range(n_categories):
        # Append enough images so that all images are equally represented
        images_train_list.extend(list(images_train[image_is_target[idx_pad, :]][:n_pad[idx_pad]]))
        targets_train_list.extend(list(targets_train[image_is_target[idx_pad, :]][:n_pad[idx_pad]]))

    images_train_aug = np.asarray(images_train_list, dtype=str)
    targets_train_aug = np.asarray(targets_train_list, dtype=int)
else:
    images_train_aug = images_train
    targets_train_aug = targets_train

# Train NN ----------------------------------------------------------------------------------------------------------- #
# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
criterion = nn.CrossEntropyLoss()

# Initialize the model, loss function, and optimizer
model = ImageClassifier(
    n_pixels=n_pixels, grayscale=True, n_pixel_after_pooling=n_pixels_after_pooling,
    n_filters=n_filters, kernel_size=3, pool_size=pool_size,
    n_conv_layers=n_conv_layers, n_dense_layers=n_dense_layers, activation_type=activation_type,
    n_convs_per_layer=n_convs_per_layer, use_pool=use_pool, n_outputs=n_categories
).to(device)
print(f'Model has:\n{model.n_features} Feat. and {model.n_parameters} Params.')
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

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
for curr_idx, (n_cat, acc_min, epochs_max) in enumerate(zip(
        curriculum_n_categories, curriculum_accuracies, curriculum_epochs_max
)):
    print(f"\nCurriculum Stage #{curr_idx+1} [{n_cat} Categories, {acc_min/100:.0%} Accuracy]")
    # Progressively add more categories to identification to help model train
    idces_curriculum_train = targets_train < n_cat
    idces_curriculum_validation = targets_validation < n_cat

    # Add 1% no-category inputs
    n_no_category = include_no_image * (1 + idces_curriculum_train.sum() // 100)

    images_train_curr = list(images_train_aug[idces_curriculum_train])
    images_train_curr.extend(n_no_category * [ImageDataset.NO_IMAGE])
    targets_train_curr = list(targets_train_aug[idces_curriculum_train])
    targets_train_curr.extend(n_no_category * [n_categories + 1])

    dataset_train = ImageDataset(
        root=root, images=images_train_curr, targets=targets_train_curr, n_targets=n_categories,
        transform=transform_train, image_read_mode=image_read_mode
    )
    dataset_validation = ImageDataset(
        root=root, images=list(images_validation[idces_curriculum_validation]),
        targets=list(targets_validation[idces_curriculum_validation]), n_targets=n_categories,
        transform=transform_test, image_read_mode=image_read_mode
    )

    train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
    validation_loader = torch.utils.data.DataLoader(dataset_validation, batch_size=batch_size, shuffle=False)

    model.train_losses.append([])
    model.test_accuracies.append([])
    model.state_dicts.append([])
    for epoch in range(epochs_max):
        elapsed_time = -time.perf_counter()  # -t0
        train_loss = train(
            model, train_loader, optimizer, criterion, device, verbose=True, include_no_image=include_no_image
        )
        model.train_losses[-1].append(train_loss)
        test_accuracy = validate(model, validation_loader, device)
        model.test_accuracies[-1].append(test_accuracy)
        model.state_dicts[-1].append(model.state_dict().copy())

        elapsed_time += time.perf_counter()  # Result is tf - t0
        elapsed_hours = int(elapsed_time // 3600)
        elapsed_minutes = int((elapsed_time - 3600*elapsed_hours) // 60)
        elapsed_seconds = int(elapsed_time - 3600*elapsed_hours - 60 * elapsed_minutes)
        print(
            f"Epoch {epoch + 1}/{epochs_max}, Train Loss: {train_loss:.4f}, "
            f"Test Accuracy: {test_accuracy/100:.0%}/{acc_min/100:.0%}, "
            f"Epoch Time: {elapsed_hours:02d}:{elapsed_minutes:02d}:{elapsed_seconds:02d}"
        )
        if test_accuracy > acc_min:
            # Successful break (achieved desired accuracy)
            print('Desired accuracy achieved, ending curriculum stage.')
            break
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
