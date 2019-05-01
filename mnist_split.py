# Author: bbsipingsoft
# Project: DCCA_demo
# Date: 4/26/19
# Time: 12:42 PM
# File: mnist_split.py


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.optim.lr_scheduler import ExponentialLR
from torch.optim import Adam
from torch.autograd import Variable
import numpy as np

from utils import construct_mnist_views, create_view_loader, visualize, svm_classify, process_evaluation_data, evaluate
from models import ViewModel
from objective import dcca_objective
import os
from pathlib import Path

home_dir = str(Path.home())
ds_path = '.pytorch/datasets/'

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
    sch1 = ExponentialLR(optimizer1, gamma=0.01)
    sch2 = ExponentialLR(optimizer2, gamma=0.01)

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


def construct_views_data(num_views=2):
    train_dataset = dsets.MNIST(os.path.join(home_dir, ds_path), train=True, transform=transforms.ToTensor(),
                                download=True)
    test_dataset = dsets.MNIST(os.path.join(home_dir, ds_path), train=False, transform=transforms.ToTensor(),
                               download=True)
    train_dataset = construct_mnist_views(train_dataset, train_dataset.targets, num_views)
    test_dataset = construct_mnist_views(test_dataset, test_dataset.targets, num_views)
    tr_ldr = create_view_loader(train_dataset, batch_size=batch, shuffle=True)
    tt_ldr = create_view_loader(test_dataset, batch_size=batch, shuffle=False)
    return tr_ldr, tt_ldr


if __name__ == '__main__':
    train_dloader, test_dloader = construct_views_data()

    latent_dim = 50
    in_dim = 392
    view1 = ViewModel(input_dim=in_dim, out_dim=latent_dim, h_layers=(2038,))
    view2 = ViewModel(input_dim=in_dim, out_dim=latent_dim, h_layers=(1608,))
    corr_hist = train(view1, view2, train_dloader, lr=1e-2, reg=1e-4, n_iters=5000)
    kwargs = {'corr_hist': corr_hist}

    # DCCA evaluation
    sanity_check, test_accuracy = evaluate(tr_ldr=train_dloader, val_ldr=test_dloader, tt_ldr=None, model=view1,
                                           view_idx=0, has_test=False, dim=latent_dim)
    print("DCCA: training set accuracy={}, test set accuracy={}".format(sanity_check, test_accuracy))

    # Linear SVM baseline
    # train_dloader, test_dloader = construct_views_data(num_views=1)
    # sanity_check, test_accuracy = evaluate(tr_ldr=train_dloader, val_ldr=test_dloader, tt_ldr=None, model=None,
    #                                        view_idx=0, dim=784,
    #                                        has_test=False)
    # print("Linear SVC (baseline): training set accuracy={}, test set accuracy={}".format(sanity_check, test_accuracy))

    visualize('corr_mnist_split.png', **kwargs)
