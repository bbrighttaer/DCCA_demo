# Author: bbsipingsoft
# Project: DCCA_demo
# Date: 4/24/19
# Time: 12:01 PM
# File: models.py


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch.nn as nn


class ViewModel(nn.Module):
    def __init__(self, input_dim, h_layers=(100,), out_dim=10):
        super(ViewModel, self).__init__()
        layers = []
        prev_in = input_dim
        for num_nodes in h_layers:
            layers.append(nn.Linear(prev_in, num_nodes))
            # layers.append(nn.BatchNorm1d(num_nodes))
            layers.append(nn.Sigmoid())
            prev_in = num_nodes

        # add output layer
        layers.append(nn.Linear(prev_in, out_dim))

        # construct model
        self.model = nn.Sequential(*layers)

        self.__canonical_weight = None

    def forward(self, x):
        outputs = self.model(x)
        return outputs

    @property
    def canonical_weight(self):
        return self.__canonical_weight

    @canonical_weight.setter
    def canonical_weight(self, weight):
        self.__canonical_weight = weight


class CnnViewModel(nn.Module):
    def __init__(self, out_dim=10):
        super(CnnViewModel, self).__init__()

        # Conv 1
        self.cnn1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5, stride=1, padding=2)
        self.relu1 = nn.ReLU()

        # Max pool 1
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)

        # Conv 2
        self.cnn2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, stride=1, padding=2)
        self.relu2 = nn.ReLU()

        # Max pool 2
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)

        # Fully connected 1
        self.fc1 = nn.Linear(32 * 7 * 7, out_dim)

    def forward(self, X):
        X = X.view(-1, 1, 28, 28)

        # conv 1
        out = self.cnn1(X)
        out = self.relu1(out)

        # maxpool 1
        out = self.maxpool1(out)

        # conv 2
        out = self.cnn2(out)
        out = self.relu2(out)

        # maxpool 2
        out = self.maxpool2(out)

        # flatten output
        out = out.view(out.size(0), -1)

        # Fully connected network
        out = self.fc1(out)

        return out


class NonSaturatingSigmoid(nn.Module):
    def __init__(self):
        super(NonSaturatingSigmoid, self).__init__()

    def forward(self, input):
        pass
