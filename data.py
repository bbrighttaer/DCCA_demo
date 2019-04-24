# Author: bbsipingsoft
# Project: DCCA_demo
# Date: 4/24/19
# Time: 11:59 AM
# File: data.py


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch
import torch.utils.data as ds


class Dataset(ds.Dataset):
    def __init__(self, data_x, data_y):
        self.X = torch.FloatTensor(data_x)
        self.y = torch.FloatTensor(data_y)
        self.size = len(data_y)

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        return self.X[index], self.y[index]
