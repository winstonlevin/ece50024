import os
import pickle
import time
import random

import numpy as np
from matplotlib import pyplot as plt
import torch
from torch import nn
from torchvision import transforms
from torchvision.io import ImageReadMode

from model_classes import ImageDataset, ImageClassifier, train, validate

# Hyperparameters/Transformation of images --------------------------------------------------------------------------- #
n_features = 64
batch_size = 25  # 100 labels -> 4 batches/label cycle
n_pixels = 100
kernel_size = 3

epochs_max = 10
curriculum_n_categories = [2, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
curriculum_accuracies = len(curriculum_n_categories) * [80]
curriculum_accuracies[-1] = 99


image_read_mode = ImageReadMode.GRAY
transform = transforms.Compose((
    transforms.Resize((n_pixels, n_pixels)),
))

# Load training and validation data ---------------------------------------------------------------------------------- #
root = '../tmp/minichallenge_data/train/'
categories = list(np.loadtxt('category.csv', delimiter=',', skiprows=1, usecols=1, dtype=str))
n_categories = len(categories)
targets_train = np.loadtxt('data_train.csv', delimiter=',', usecols=0, dtype=int)
images_train = np.loadtxt('data_train.csv', delimiter=',', usecols=1, dtype=str)
targets_validation = np.loadtxt('data_validation.csv', delimiter=',', usecols=0, dtype=int)
images_validation = np.loadtxt('data_validation.csv', delimiter=',', usecols=1, dtype=str)

# Train NN ----------------------------------------------------------------------------------------------------------- #
# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
criterion = nn.CrossEntropyLoss()

# Initialize the model, loss function, and optimizer
model = ImageClassifier(n_features=n_features, n_outputs=n_categories, kernel_size=3).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training loop
for curr_idx, (n_cat, acc_min) in enumerate(zip(curriculum_n_categories, curriculum_accuracies)):
    print(f"Curriculum Stage #{curr_idx} [{n_cat} Categories, {acc_min:.0%} Accuracy]")
    # Progressively add more categories to identification to help model train
    idces_curriculum_train = targets_train < n_cat
    idces_curriculum_validation = targets_validation < n_cat

    dataset_train = ImageDataset(
        root=root, images=list(images_train[idces_curriculum_train]),
        targets=list(targets_train[idces_curriculum_train]),
        transform=transform, image_read_mode=image_read_mode
    )
    dataset_validation = ImageDataset(
        root=root, images=list(images_validation[idces_curriculum_validation]),
        targets=list(targets_validation[idces_curriculum_validation]),
        transform=transform, image_read_mode=image_read_mode
    )

    train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=batch_size, shuffle=False)
    validation_loader = torch.utils.data.DataLoader(dataset_validation, batch_size=batch_size, shuffle=False)

    train_losses, test_accuracies = [], []
    for epoch in range(epochs_max):
        train_loss = train(model, train_loader, optimizer, criterion, device, verbose=True)
        train_losses.append(train_loss)
        test_accuracy = validate(model, validation_loader, device)
        test_accuracies.append(test_accuracy)
        print(f"Epoch [{epoch + 1}/{epochs_max}], Train Loss: {train_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%")
        if test_accuracy > acc_min:
            break
    model.train_losses.append(train_losses)
    model.test_accuracies.append(test_accuracies)

# Save Model
current_time = time.gmtime()
date = f'{current_time.tm_year:04d}-{current_time.tm_mon:02d}-{current_time.tm_mday:02d}'
hour = f'{current_time.tm_hour:02d}-{current_time.tm_min:02d}-{current_time.tm_sec:02d}'
file_name = f'tmp/models/MNIST_{date}_{hour}.pickle'
os.makedirs(os.path.dirname(file_name), exist_ok=True)  # Make directory if it does not yet exist
with open(file_name, 'wb') as f:
    pickle.dump(model, f, protocol=pickle.HIGHEST_PROTOCOL)

# Plotting the training loss and test accuracy ----------------------------------------------------------------------- #
fig_accuracy = plt.figure(figsize=(10, 5))

ax_loss = fig_accuracy.add_subplot(121)
ax_loss.plot(model.train_losses)
ax_loss.set_xlabel('Epoch')
ax_loss.set_ylabel('Training Loss')
ax_loss.set_title('Training Loss vs. Epoch')

ax_accuracy = fig_accuracy.add_subplot(122)
ax_accuracy.plot(model.test_accuracies)
ax_accuracy.set_xlabel('Epoch')
ax_accuracy.set_ylabel('Test Accuracy (%)')
ax_accuracy.set_title('Test Accuracy vs. Epoch')

fig_accuracy.tight_layout()

plt.show()
