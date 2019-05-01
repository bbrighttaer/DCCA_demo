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
import torch
from torch.autograd import Variable
from torch.optim import Adam, RMSprop
from torch.optim.lr_scheduler import  ExponentialLR

from models import ViewModel
from objective import dcca_objective
from utils import load_data, create_view_loader, visualize, evaluate

torch.manual_seed(0)
np.random.seed(0)
batch = 1000


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
    data1 = load_data('noisymnist_view1.gz', 'https://www2.cs.uic.edu/~vnoroozi/noisy-mnist/noisymnist_view1.gz')
    data2 = load_data('noisymnist_view2.gz', 'https://www2.cs.uic.edu/~vnoroozi/noisy-mnist/noisymnist_view2.gz')

    train_data = (data1[0], data2[0])
    val_data = (data1[1], data2[1])
    test_data = (data1[2], data2[2])

    tr_ldr = create_view_loader(train_data, batch, shuffle=True)
    val_ldr = create_view_loader(val_data, batch, shuffle=False)
    tt_ldr = create_view_loader(test_data, batch, shuffle=False)

    latent_dim = 10
    in_dim = 784
    lyrs = (1024, 1024, 1024)
    view1 = ViewModel(input_dim=in_dim, out_dim=latent_dim, h_layers=lyrs)
    view2 = ViewModel(input_dim=in_dim, out_dim=latent_dim, h_layers=lyrs)
    # view1 = CnnViewModel(latent_dim)
    # view2 = CnnViewModel(latent_dim)
    corr_hist = train(view1, view2, tr_ldr, lr=1e-1, reg=1e-4, n_iters=5000)
    kwargs = {'corr_hist': corr_hist}

    # DCCA evaluation
    sanity_check, val_accuracy, test_accuracy = evaluate(tr_ldr=tr_ldr, val_ldr=val_ldr, tt_ldr=tt_ldr, model=view1,
                                                         view_idx=0, has_test=True, dim=latent_dim)
    print("DCCA: training set accuracy={}, validation set accuracy={}, test set accuracy={}".format(sanity_check,
                                                                                                    val_accuracy,
                                                                                                    test_accuracy))

    # Linear SVM baseline
    # sanity_check, val_accuracy, test_accuracy = evaluate(tr_ldr=tr_ldr, val_ldr=val_ldr, tt_ldr=tt_ldr, model=None,
    #                                                      view_idx=0, has_test=True, dim=784)
    # print("Linear SVC (baseline): training set accuracy={}, validation set accuracy={}, test set accuracy={}".format(
    #     sanity_check,
    #     val_accuracy,
    #     test_accuracy))

    visualize('corr_mnist.png', **kwargs)
