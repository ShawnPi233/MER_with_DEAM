"""
This module contains the Convolutional NN model architecture
"""
import torch
import torch.nn as nn


class AudioNet(nn.Module):
    """
    Model architecture and the forward-pass are defined in this class.
    """
    def __init__(self, params_dict):
        """
        :param params_dict: Dictionary with information about the architecture
        """
        super().__init__()

        in_ch = params_dict['in_ch']
        num_filters1 = params_dict['num_filters1']
        num_filters2 = params_dict['num_filters2']
        num_hidden = params_dict['num_hidden']
        out_size = params_dict['out_size']

        self._conv1 = nn.Sequential(nn.Conv1d(in_ch, num_filters1, 10, 1),
                                    nn.BatchNorm1d(num_filters1),
                                    nn.ReLU(),
                                    nn.AvgPool1d(2, 2))
        self._conv2 = nn.Sequential(nn.Conv1d(num_filters1, num_filters2, 10, 1),
                                    nn.BatchNorm1d(num_filters2),
                                    nn.ReLU(),
                                    nn.AvgPool1d(2, 2))
        self._pool = nn.AvgPool1d(10, 10)
        self._drop = nn.Dropout(0.5)
        self._act = nn.ReLU()
        self._fc1 = nn.Linear(29*num_filters2, num_hidden)
        self._fc2 = nn.Linear(num_hidden, out_size)

    def forward(self, x):

        x = self._conv1(x)
        x = self._conv2(x)
        x = self._pool(x)

        _, ch, w = x.shape
        x = x.view(-1, ch * w)

        x = self._fc1(x)
        x = self._drop(x)
        x = self._act(x)
        x = self._fc2(x)

        return x
