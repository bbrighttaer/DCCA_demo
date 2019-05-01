# Author: bbsipingsoft
# Project: DCCA_demo
# Date: 4/24/19
# Time: 12:18 PM
# File: dcca_iris.py


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from torch.autograd import Variable
from torch.optim import Adam
from torch.optim.lr_scheduler import ExponentialLR

from models import ViewModel
from objective import dcca_objective
from utils import construct_iris_views, create_view_loader, visualize, svm_classify, process_evaluation_data, evaluate

torch.manual_seed(0)
np.random.seed(0)
batch_sz = 90


def train(view1, view2, dloader, lr, reg, n_iters):
    """
    Trains the models for correlation maximization

    :param view1: Model 1
    :param view2: Model 2
    :param v1_data: data loaders for view 1
    :param v2_data: data loaders for view 2
    :param lr: learning rate
    :param reg: regularization parameter
    :param n_iters: number of total iterations
    :return: the correlation history
    """
    n_epochs = n_iters / float(len(dloader))
    n_epochs = int(n_epochs)
    corr_hist = []
    counter = 0

    optimizer1 = Adam(view1.parameters(), lr)
    optimizer2 = Adam(view2.parameters(), lr)

    sch1 = ExponentialLR(optimizer1, gamma=0.001)
    sch2 = ExponentialLR(optimizer2, gamma=0.001)

    for epoch in range(n_epochs):
        for Xs, _ in dloader:
            x_v1, x_v2 = Xs
            x_v1 = Variable(x_v1, requires_grad=False)
            x_v2 = Variable(x_v2, requires_grad=False)

            pred1 = view1(x_v1)
            pred2 = view2(x_v2)

            corr, dH1, dH2, A1, A2 = dcca_objective(pred1, pred2, reg)
            corr_hist.append(corr)
            view1.canonical_weight = A1
            view2.canonical_weight = A2

            optimizer1.zero_grad()
            optimizer2.zero_grad()

            pred1.backward(-dH1)
            pred2.backward(-dH2)

            optimizer1.step()
            optimizer2.step()

            if counter % 100 == 0:
                print("Epoch={}/{}, iteration={}, corr={}".format(epoch + 1, n_epochs, counter, corr))

            counter += 1

        sch1.step()
        sch2.step()
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

    # view 2
    v2_data_xtrain, v2_data_xvalid, v2_data_ytrain, v2_data_yvalid = train_test_split(v2_data[:, 0:2], v2_data[:, 2],
                                                                                      test_size=0.1)
    v2_data = ([v2_data_xtrain, v2_data_ytrain], [v2_data_xvalid, v2_data_yvalid])

    tr_ldr = create_view_loader(((v1_data_xtrain, v1_data_ytrain), (v2_data_xtrain, v2_data_ytrain)), batch_sz,
                                shuffle=True)
    val_ldr = create_view_loader(((v1_data_xvalid, v1_data_yvalid), (v2_data_xvalid, v2_data_yvalid)), 10,
                                 shuffle=False)

    latent_dim = 2
    view1 = ViewModel(input_dim=2, out_dim=latent_dim, h_layers=(200, 50))
    view2 = ViewModel(input_dim=2, out_dim=latent_dim, h_layers=(200, 50))
    corr_hist = train(view1, view2, tr_ldr, lr=1e-2, reg=1e-4, n_iters=5000)
    kwargs = {'corr_hist': corr_hist}

    # DCCA evaluation
    sanity_check, test_accuracy = evaluate(tr_ldr=tr_ldr, val_ldr=val_ldr, tt_ldr=None, model=view1,
                                           view_idx=0, has_test=False, dim=latent_dim)
    print("DCCA: training set accuracy={}, test set accuracy={}".format(sanity_check, test_accuracy))

    # Linear SVM baseline
    data_x = df[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']].values
    data_y = df['Species'].values
    train_x, valid_x, train_y, valid_y = train_test_split(data_x, data_y, test_size=0.1)
    tr_ldr = create_view_loader(((train_x, train_y),), batch_sz, shuffle=True)
    val_ldr = create_view_loader(((valid_x, valid_y),), 10, shuffle=False)
    sanity_check, test_accuracy = evaluate(tr_ldr=tr_ldr, val_ldr=val_ldr, tt_ldr=None, model=None, view_idx=0, dim=4,
                                           has_test=False)
    print("Linear SVC (baseline): training set accuracy={}, test set accuracy={}".format(sanity_check, test_accuracy))

    visualize('corr_iris.png', **kwargs)
