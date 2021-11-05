"""
This module contains all the necessary functions to create data loaders
"""
import os
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader


def load_data(data_dir, mode):
    """
    Function to load dataset.
    :param data_dir: path to the data directory
    :param mode: type of data set to load - train | test
    :return: MFCC data and the corresponding annotations as tensors
    """

    mfccs = np.load(os.path.join(data_dir, '{:s}_mfccs.npy'.format(mode)))
    annotations = np.load(os.path.join(data_dir, '{:s}_annotations.npy'.format(mode)))

    # Convert numpy arrays to torch tensors
    data, target = map(torch.tensor, (mfccs.astype(np.float32),
                                      annotations.astype(np.float32)))

    return data, target


def normalize_mfccs(sample_mfcc, mfcc_mean, mfcc_std):
    """
    Function to normalize MFCCs data, according to mean and variance in train set
    :param sample_mfcc: MFCC data to be normalized
    :param mfcc_mean: train set MFCC mean
    :param mfcc_std: train set MFCC stadard deviation
    :return: normalized MFCC data
    """
    return (sample_mfcc - mfcc_mean) / mfcc_std


def make_training_loaders(data_dir):
    """
    Function to create data loaders for training and validation in batches of 64 samples.
    :param data_dir: path to the data directory
    :return: train and test data loaders
    """

    # Load train and test sets
    train_data, train_annotations = load_data(data_dir, 'train')
    test_data, test_annotations = load_data(data_dir, 'test')

    # Normalize the MFCC data using train mean and standard deviation
    mfcc_mean, mfcc_std = torch.mean(train_data), torch.std(train_data)
    train_data = normalize_mfccs(train_data, mfcc_mean, mfcc_std)
    test_data = normalize_mfccs(test_data, mfcc_mean, mfcc_std)

    # Create Datasets
    train_dataset = TensorDataset(train_data, train_annotations)
    test_dataset = TensorDataset(test_data, test_annotations)

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=64, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=64, drop_last=True)

    return train_loader, test_loader


def make_testing_loader(data_dir):
    """
    Function to create data loaders for training and validation in batches of 64 samples.
    :param data_dir: path to the data directory
    :return: test data loader
    """

    # Load train MFCCs and test MFCCs & annotations
    train_data, _ = load_data(data_dir, 'train')
    test_data, test_annotations = load_data(data_dir, 'test')

    # Normalize test MFCCs using train mean and standard deviation
    mfcc_mean, mfcc_std = torch.mean(train_data), torch.std(train_data)
    test_data = normalize_mfccs(test_data, mfcc_mean, mfcc_std)

    # Create data loaders
    test_dataset = TensorDataset(test_data, test_annotations)
    test_loader = DataLoader(test_dataset, batch_size=64, drop_last=True)

    return test_loader
