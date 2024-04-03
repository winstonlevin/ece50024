from typing import Union, Sequence
import numpy as np
import torch
import torch.nn as nn
import cv2
from torchvision.io import ImageReadMode
from torch.utils.data import Dataset


class ImageDataset(Dataset):
    NO_IMAGE = 'no_image'  # image string corresponding to NO image

    def __init__(self, root, images, targets, n_targets=None, transform=None, image_read_mode=ImageReadMode.UNCHANGED):
        self.root = root
        self.targets = targets
        self.images = images
        self.transform = transform
        self.image_read_mode = image_read_mode
        self.face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

        if n_targets is not None:
            self.n_targets = n_targets
        else:
            self.n_targets = 1 + int(torch.as_tensor(targets).max())

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        if self.images[index] == self.NO_IMAGE:
            return self.gen_noise_image()

        # Load image
        path = self.root + self.images[index]
        try:
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        except RuntimeError:
            return self.gen_noise_image()

        h, w = img.shape
        sample = torch.as_tensor(img).reshape((1, h, w)) / 255  # Normalize image, 0 = black, 1 = white

        # Apply transformation
        if self.transform is not None:
            sample = self.transform(sample)
        target = self.targets[index]

        return sample, target

    def gen_noise_image(self):
        # Generate noise and return no image index
        target = self.n_targets
        sample_padded = torch.rand((1, 100, 100))
        if self.transform is not None:
            sample_padded = self.transform(sample_padded)

        return sample_padded, target


class ImageClassifier(nn.Module):
    """
    Classifier for Image Data. The input data is a HxW greyscale image of a face. The output data is a classification
    from 0 to 99 representing a celebrity. The NN architecture is:

    (1) INPUT LAYER
        (1a.) 2D convolution of each pixel to a feature vector
        (1b.) ReLU activation of features

    (2) HIDDEN LAYER(S)
        (2a.) 2D channel-wise convolution of feature vector
        (2b.) ReLU activation of features
        (3b.) Pool

    (3) POOL LAYER
        Pool features of remaining pixels into single feature vector

    (3) OUTPUT LAYER
        Linear Transformation from feature vector to possible classifications.
    """
    def __init__(
            self,
            n_pixels: int, grayscale: bool, n_pixel_after_pooling: int = 1,
            n_filters: Union[int, Sequence] = 64, kernel_size: int = 3, pool_size: int = 2, pool_type: str = 'MaxPool2D',
            n_conv_layers: int = 1, n_dense_layers: int = 1, activation_type: str = 'ReLU',
            n_convs_per_layer: Union[int, Sequence] = 1, use_pool: Union[bool, Sequence] = True, n_outputs: int = 100
    ):

        super(ImageClassifier, self).__init__()
        self.n_pixels: int = n_pixels
        self.grayscale: bool = grayscale
        self.n_filters: int = n_filters
        self.kernel_size: int = kernel_size
        self.pool_size: int = pool_size
        self.pool_type: str = pool_type
        self.n_conv_layers: int = n_conv_layers
        self.n_dense_layers: int = n_dense_layers
        self.activation_type: str = activation_type
        self.n_outputs: int = n_outputs
        self.n_pixel_after_pooling: int = n_pixel_after_pooling

        if isinstance(n_filters, Sequence):
            self.n_filters = n_filters
        else:
            self.n_filters = self.n_conv_layers * [n_filters]

        if isinstance(n_convs_per_layer, Sequence):
            self.n_convs_per_layer = n_convs_per_layer
        else:
            self.n_convs_per_layer = self.n_conv_layers * [n_convs_per_layer]

        if isinstance(use_pool, Sequence):
            self.use_pool = use_pool
        else:
            self.use_pool = self.n_conv_layers * [use_pool]

        self.stride: int = self.kernel_size // 2
        self.padding: int = self.stride

        # Choose activation function based on type
        if self.activation_type.lower() == 'relu':
            def gen_activation_fn():
                return nn.ReLU(inplace=True)
        elif self.activation_type.lower() == 'elu':
            def gen_activation_fn():
                return nn.ELU(inplace=True)
        elif "leaky" in self.activation_type.lower():
            def gen_activation_fn():
                return nn.LeakyReLU(inplace=True)
        else:
            raise NotImplementedError(f'Activation type "{self.activation_type}" is not implemented!')

        # Choose pool function based on type
        if "max" in self.pool_type.lower():
            def gen_pool_fn(_):
                return nn.MaxPool2d(kernel_size=(self.pool_size, self.pool_size))
        elif "adapt" in self.pool_type.lower():
            gen_pool_fn = self.gen_adaptive_pool
        else:
            raise NotImplementedError(f'Pool type "{self.pool_type}" is not implemented!')

        # Generate convolutional layers ------------------------------------------------------------------------------ #
        self.conv_layers = nn.Sequential()
        size_conv_out = self.n_pixels
        for idx_conv in range(self.n_conv_layers):
            # Form specified number of CNN layers, where each CNN layer has specified number of convolutions/activations
            # followed by a pool if use_pool is True
            n_feat_in = 1 if idx_conv == 0 else self.n_filters[idx_conv - 1]
            convs = nn.Sequential()
            for idx_n_convs in range(self.n_convs_per_layer[idx_conv]):
                convs.extend((
                    nn.Conv2d(
                        n_feat_in, self.n_filters[idx_conv], kernel_size=(self.kernel_size, self.kernel_size),
                        padding=self.padding, stride=(self.stride, self.stride)),
                    gen_activation_fn()
                ))
                n_feat_in = self.n_filters[idx_conv]  # Ensure n_feat_in = 1 is overwritten
                size_conv_out = int(1 + (size_conv_out + 2*self.padding - (self.kernel_size - 1) - 1) // self.stride)

            if self.use_pool[idx_conv]:
                convs.append(gen_pool_fn(size_conv_out))
                size_conv_out //= self.pool_size
            self.conv_layers.append(convs)

        # Final pool layer into flat feature vector
        self.pool_layer = nn.Sequential(
            nn.AdaptiveAvgPool2d(output_size=self.n_pixel_after_pooling),
            nn.Flatten()
        )
        self.n_features = self.n_pixel_after_pooling ** 2 * self.n_filters[-1]

        # Generate dense layers to operate on feature vector --------------------------------------------------------- #
        self.dense_layers = nn.Sequential()
        for idx_out in range(self.n_dense_layers):
            if idx_out + 1 == self.n_dense_layers:
                # Final layer must map to outputs
                self.dense_layers.append(
                    nn.Linear(self.n_features, self.n_outputs)
                )
            else:
                # Intermediate layer with activation for non-linearity
                self.dense_layers.append(nn.Sequential(
                    nn.Linear(self.n_features, self.n_features),
                    gen_activation_fn()
                ))

        # Calculate the number of parameters
        self.n_parameters = 0
        for _param in list(self.parameters()):
            self.n_parameters += _param.numel()

        # Storage of training information
        self.train_losses = []
        self.test_accuracies = []
        self.state_dicts = []

    def forward(self, _state):
        _state = self.conv_layers(_state)  # Convolution of image into feature vector
        _state = self.pool_layer(_state)  # Pool features to flat feature vector
        _state = self.dense_layers(_state)  # Convert feature vector to classifications
        return _state

    def gen_adaptive_pool(self, input_size):
        output_size = int(input_size / self.pool_size)
        return nn.AdaptiveAvgPool2d(output_size=output_size)


def train(model, train_loader, optimizer, criterion, device, verbose=True, include_no_image=False):
    """
    Function to train the model, using the optimizer to tune the parameters.
    """
    model.train()
    running_loss = 0.0
    n_batches = len(train_loader)
    idx_report = max(1, int(n_batches / 10))
    if verbose:
        print('')  # 1 new line

    for idx, (inputs, labels) in enumerate(train_loader):
        if verbose and idx % idx_report == 0:
            print(f'Batch #{idx+1}/{n_batches} (Ave. Loss = {running_loss / (idx+1):.4f})...')
        inputs, labels = inputs.to(device), labels.to(device).long()
        optimizer.zero_grad()
        outputs = model(inputs)

        if include_no_image:
            # Loss must append the "no image" category at the end
            loss = criterion(torch.cat((outputs, 1 - outputs.max(dim=1, keepdim=True)[0]), dim=1), labels)
        else:
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
