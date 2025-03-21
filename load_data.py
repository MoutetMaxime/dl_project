import os

import numpy as np
import scipy.io as sio
import torch
import torchvision
import torchvision.transforms as transforms


def lire_alpha_digit(characters):
    """
    Load the binary alpha digit dataset and return the images corresponding to the specified characters.

    Parameters:
    - characters (list of str): List of characters to load. Lowercase characters only.

    Returns:
    - images (torch.Tensor): Tensor of images corresponding to the specified characters
    """
    # Load the dataset
    try:
        data_mat = sio.loadmat(os.path.join('data', 'binaryalphadigs.mat'))
    except FileNotFoundError:
        print("Dataset not found in 'data' folder. Trying to load from parent directory...")

        try:
            data_mat = sio.loadmat('binaryalphadigs.mat')
        except FileNotFoundError:
            print("Dataset not found in parent directory. Please download the dataset from the following link and place it in the 'data' folder: https://www.kaggle.com/datasets/angevalli/binary-alpha-digits?select=binaryalphadigs.mat")
        
    print("Dataset loaded successfully.")
    images = data_mat['dat']

    # Get labels and convert them to lowercase
    labels = data_mat['classlabels'].squeeze()
    labels = np.array([str(label[0]).lower() for label in labels])

    # Select the characters we want
    selected_characters = np.isin(labels, characters)

    images = images[selected_characters].reshape(-1, 1) # (nb_images, 1)
    images = np.array([img[0] for img in images])   # (nb_images, 20, 16)
    images = images.reshape(images.shape[0], -1)  # (nb_images, 320)

    return images


def load_mnist(data_dir="data/mnist"):
    """
    Downloads the MNIST dataset and returns the train and test sets as PyTorch tensors.

    Parameters:
    - data_dir (str): Directory where the dataset should be stored.

    Returns:
    - train_loader (torch.utils.data.DataLoader): DataLoader for training set.
    - test_loader (torch.utils.data.DataLoader): DataLoader for test set.
    """
    os.makedirs(data_dir, exist_ok=True)  # Ensure directory exists

    # # Define transformations (convert to tensor and normalize)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: (x >= 0.5).float())  # Apply binarization
    ])

    # Load training & test datasets
    train_dataset = torchvision.datasets.MNIST(root=data_dir, train=True, download=True, transform=transform)
    test_dataset = torchvision.datasets.MNIST(root=data_dir, train=False, download=True, transform=transform)

    return train_dataset, test_dataset
