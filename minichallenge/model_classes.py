from typing import Optional, Callable

import torch
from torch import Tensor
import torch.nn as nn

from torchvision.io import read_image, ImageReadMode
from torch.utils.data import Dataset


class ImageDataset(Dataset):
    def __init__(self, root, images, targets, transform=None, image_read_mode=ImageReadMode.UNCHANGED):
        self.root = root
        self.targets = targets
        self.images = images
        self.transform = transform
        self.image_read_mode = image_read_mode

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        path = self.root + self.images[index]
        sample = read_image(path, self.image_read_mode) / 255  # Normalize image 0 = black, 1 = white
        # Pad image from [1xhxw] to square [1xsxs]
        if sample.shape[1] > sample.shape[2]:
            dim = sample.shape[1]
            pad_offset = (sample.shape[1] - sample.shape[2]) // 2
            sample_padded = torch.zeros((1, dim, dim))
            sample_padded[:, :, pad_offset:pad_offset+sample.shape[2]] = sample
        else:
            dim = sample.shape[2]
            pad_offset = (sample.shape[2] - sample.shape[1]) // 2
            sample_padded = torch.zeros((1, dim, dim))
            sample_padded[:, pad_offset:pad_offset+sample.shape[1], :] = sample
        if self.transform is not None:
            sample_padded = self.transform(sample_padded)
        target = self.targets[index]

        return sample_padded, target


class ImageClassifier(nn.Module):
    """
    Classifier for Image Data. The input data is a HxW greyscale image of a face. The output data is a classification
    from 0 to 99 representing a celebrity. The NN architecture is:

    (1) INPUT LAYER
        (1a.) 2D convolution of each pixel to a feature vector
        (1b.) ReLU activation of features

    (2) HIDDEN LAYER
        (2a.) 2D channel-wise convolution of feature vector
        (2b.) ReLU activation of features

    (3) POOL LAYER
        Pool features of each pixel into single feature vector

    (3) OUTPUT LAYER
        Linear Transformation from feature vector to 10 possible classifications.
    """
    def __init__(self, n_features: int = 64, kernel_size: int = 3, n_outputs: int = 100):

        super(ImageClassifier, self).__init__()
        self.n_features = n_features
        self.kernal_size = kernel_size
        self.n_outputs = n_outputs
        self.stride = self.kernal_size // 2
        self.padding = self.stride
        self.input_layer = nn.Sequential(
            nn.Conv2d(
                1, n_features, kernel_size=(kernel_size, kernel_size),
                padding=self.padding, stride=(self.stride, self.stride)
            ),  # in_channel (1 for grayscale), out_channels
            nn.ReLU(inplace=True)  # inplace=True argument modifies the input tensor directly, saving memory.
        )
        self.hidden_layer = nn.Sequential(
            nn.Conv2d(
                n_features, n_features, kernel_size=(kernel_size, kernel_size),
                padding=self.padding, stride=(self.stride, self.stride)
            ),
            nn.ReLU(inplace=True)
        )
        self.pool_layer = nn.AdaptiveAvgPool2d((1, 1))
        self.output_layer = nn.Linear(n_features, n_outputs)

        # Storage of training information
        self.train_losses = []
        self.test_accuracies = []

    def forward(self, _state):
        _state = self.input_layer(_state)  # Extract features from each pixel
        _state = self.hidden_layer(_state)  # Propagate NODE
        _state = self.pool_layer(_state)  # Pool features from each pixel
        _state = torch.flatten(_state, 1)  # Remove extra dimensions for linear transform
        _state = self.output_layer(_state)  # Convert pool of features to classifications
        return _state


def train(model, train_loader, optimizer, criterion, device, verbose=True):
    """
    Function to train the model, using the optimizer to tune the parameters.
    """
    model.train()
    running_loss = 0.0
    n_batches = len(train_loader)
    idx_report = max(1, int(n_batches / 10))
    if verbose:
        print(f'\n')

    for idx, (inputs, labels) in enumerate(train_loader):
        if verbose and idx % idx_report == 0:
            print(f'Batch #{idx+1}/{n_batches} (Ave. Loss = {running_loss / (idx+1):.4f})...')
        inputs, labels = inputs.to(device), labels.to(device).long()
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)

    return running_loss / n_batches


def validate(model, test_loader, device):
    """
    Function to evaluate the model using the validation dataset
    """
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    return accuracy
