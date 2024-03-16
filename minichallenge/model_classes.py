from typing import Optional, Callable

import torch
from torch import Tensor
import torch.nn as nn
from torchvision.datasets import ImageFolder


class ImageFolderWithTargets(ImageFolder):
    def __init__(self, root, targets, transform=None, is_valid_file=None):
        super(ImageFolderWithTargets, self).__init__(root, transform=transform, is_valid_file=is_valid_file)
        self.targets = targets

    def __getitem__(self, index):
        path, _ = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        target = self.targets[index]

        return sample, target
        # path, _ = self.samples[index]
        # sample = self.loader(path)
        # target = self.targets[index]
        # return sample, target


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
    def __init__(self, n_features: int = 64):

        super(ImageClassifier, self).__init__()
        self.n_features = n_features

        self.input_layer = nn.Sequential(
            nn.Conv2d(1, n_features, kernel_size=3, padding=1),  # in_channel (1 for grayscale), out_channels
            nn.ReLU(inplace=True)  # inplace=True argument modifies the input tensor directly, saving memory.
        )
        self.hidden_layer = nn.Sequential(
            nn.Conv2d(n_features, n_features, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.pool_layer = nn.AdaptiveAvgPool2d((1, 1))
        self.output_layer = nn.Linear(n_features, 10)

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
    idx_report = max(1, int(n_batches / 5))
    if verbose:
        print(f'\n')

    for idx, (inputs, labels) in enumerate(train_loader):
        if verbose and idx % idx_report == 0:
            print(f'Batch #{idx+1}/{n_batches} (Ave. Loss = {running_loss / (idx+1):.4f})...')
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)

    return running_loss / n_batches


def test(model, test_loader, device):
    """
    Function to evaluate the model using the testing dataset
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
