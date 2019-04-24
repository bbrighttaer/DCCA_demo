# Author: bbsipingsoft
# Project: DCCA_demo
# Date: 4/24/19
# Time: 12:18 PM
# File: dcca_iris.py


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch
from torch.optim import Adam
from torch.autograd import Variable
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

from utils.utils import construct_iris_views, create_view_loaders, visualize, svm_classify, process_evaluation_data
from models import ViewModel
from objective import dcca_objective

torch.manual_seed(0)
np.random.seed(0)
batch = 90


def train(view1, view2, v1_data, v2_data, lr, reg, n_iters, stats):
    """
    Trains the models for correlation maximization

    :param view1: Model 1
    :param view2: Model 2
    :param v1_data: data loaders for view 1
    :param v2_data: data loaders for view 2
    :param lr: learning rate
    :param reg: regularization parameter
    :param n_iters: number of total iterations
    :param stats: information about data set sizes
    :return: the correlation history
    """

    v1_train, v1_val = v1_data
    v2_train, v2_val = v2_data

    sz = stats[0]
    n_epochs = int(n_iters / float(sz / batch))
    corr_hist = []
    counter = 0

    optimizer1 = Adam(view1.parameters(), lr)
    optimizer2 = Adam(view2.parameters(), lr)

    for epoch in range(n_epochs):
        for i, ((x_v1, _), (x_v2, _)) in enumerate(zip(v1_train, v2_train)):
            x_v1 = Variable(x_v1, requires_grad=False)
            x_v2 = Variable(x_v2, requires_grad=False)

            pred1 = view1(x_v1)
            pred2 = view2(x_v2)

            corr, dH1, dH2 = dcca_objective(pred1, pred2, reg)
            corr_hist.append(corr)

            optimizer1.zero_grad()
            optimizer2.zero_grad()

            pred1.backward(-dH1)
            pred2.backward(-dH2)

            optimizer1.step()
            optimizer2.step()

            if counter % 100 == 0:
                print("Epoch={}/{}, iteration={}, corr={}".format(epoch, n_epochs-1, counter, corr))

            counter += 1
    return corr_hist


if __name__ == '__main__':
    df = pd.read_csv('iris.csv', header=0)
    df = df[:100][:]
    labels = list(set(df['Species']))
    species = [labels.index(sp) for sp in df['Species']]
    df['Species'] = species

    shuffle(df)

    v1_data, v2_data = construct_iris_views(df)

    # view 1
    v1_data_xtrain, v1_data_xvalid, v1_data_ytrain, v1_data_yvalid = train_test_split(v1_data[:, 0:2], v1_data[:, 2],
                                                                                      test_size=0.1)
    v1_data = ([v1_data_xtrain, v1_data_ytrain], [v1_data_xvalid, v1_data_yvalid])

    v1_dataloader_train, v1_dataloader_val, v1_stats = create_view_loaders(v1_data, (batch, 10, 1), has_test=False)

    # view 2
    v2_data_xtrain, v2_data_xvalid, v2_data_ytrain, v2_data_yvalid = train_test_split(v2_data[:, 0:2], v2_data[:, 2],
                                                                                      test_size=0.1)
    v2_data = ([v2_data_xtrain, v2_data_ytrain], [v2_data_xvalid, v2_data_yvalid])

    v2_dataloader_train, v2_dataloader_val, v2_stats = create_view_loaders(v2_data, (batch, 10, 1), has_test=False)

    latent_dim = 20
    view1 = ViewModel(input_dim=2, out_dim=latent_dim)
    view2 = ViewModel(input_dim=2, out_dim=latent_dim)
    corr_hist = train(view1, view2, (v1_dataloader_train, v1_dataloader_val), (v2_dataloader_train, v2_dataloader_val),
                      lr=1e-2, reg=1e-4, n_iters=5000, stats=v1_stats)
    kwargs = {'corr_hist': corr_hist}

    # evaluation
    sanity_check, valid_accuracy = svm_classify(v1_data, has_test=False)
    print("Linear SVC (baseline): Training set accuracy={}, validation set accuracy={}".format(sanity_check,
                                                                                               valid_accuracy))
    datasets = process_evaluation_data(view1, (v1_dataloader_train, v1_dataloader_val), latent_dim)
    sanity_check, valid_accuracy = svm_classify(datasets, has_test=False)
    print("DCCA: Training set accuracy={}, validation set accuracy={}".format(sanity_check, valid_accuracy))

    visualize('corr_iris.png', **kwargs)
