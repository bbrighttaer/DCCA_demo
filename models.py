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
            layers.append(nn.Sigmoid())
            prev_in = num_nodes

        # add output layer
        layers.append(nn.Linear(prev_in, out_dim))

        # construct model
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        outputs = self.model(x)
        return outputs


class NonSaturatingSigmoid(nn.Module):
    def __init__(self):
        super(NonSaturatingSigmoid, self).__init__()

    def forward(self, input):
        pass
