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
    def __init__(self, views_data):
        self.X_list = [torch.FloatTensor(data_x) for data_x, _ in views_data]
        self.y_list = [torch.FloatTensor(data_y) for _, data_y in views_data]
        self.size = len(self.y_list[0])

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        queried_x, queried_y = [], []
        for view_x, view_y in zip(self.X_list, self.y_list):
            queried_x.append(view_x[index])
            queried_y.append(view_y[index])
        return queried_x, queried_y
