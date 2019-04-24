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
import numpy as np

from utils.utils import load_data, create_view_loaders, visualize, svm_classify, process_evaluation_data
from models import ViewModel
from objective import dcca_objective

torch.manual_seed(0)
np.random.seed(0)
batch = 800


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

    v1_train = v1_data[0]
    v2_train = v2_data[0]

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
    data1 = load_data('noisymnist_view1.gz', 'https://www2.cs.uic.edu/~vnoroozi/noisy-mnist/noisymnist_view1.gz')
    data2 = load_data('noisymnist_view2.gz', 'https://www2.cs.uic.edu/~vnoroozi/noisy-mnist/noisymnist_view2.gz')

    view1_data = create_view_loaders(data1, batch_sizes=(batch, batch, batch))
    view2_data = create_view_loaders(data2, batch_sizes=(batch, batch, batch))

    latent_dim = 50
    in_dim = 784
    lyrs = (1024, 1024, 512, 256)
    view1 = ViewModel(input_dim=in_dim, out_dim=latent_dim, h_layers=lyrs)
    view2 = ViewModel(input_dim=in_dim, out_dim=latent_dim, h_layers=lyrs)
    corr_hist = train(view1, view2, view1_data, view2_data, lr=1e-2, reg=1e-4, n_iters=20000, stats=view1_data[3])
    kwargs = {'corr_hist': corr_hist}

    # evaluation
    datasets = process_evaluation_data(view1, view1_data[:3], latent_dim=latent_dim)
    sanity_check, valid_accuracy, test_accuracy = svm_classify(datasets, has_test=True)
    print("DCCA: training set accuracy={}, validation set accuracy={}, test set accuracy={}".format(sanity_check,
                                                                                                    valid_accuracy,
                                                                                                    test_accuracy))
    sanity_check, valid_accuracy, test_accuracy = svm_classify(data1, has_test=True)
    print("Linear SVC (baseline): training set accuracy={}, validation set accuracy={}, test set accuracy={}".format(
        sanity_check,
        valid_accuracy,
        test_accuracy))

    visualize('corr_mnist.png', **kwargs)
