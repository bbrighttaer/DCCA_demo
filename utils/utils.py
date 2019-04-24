# Author: bbsipingsoft
# Project: DCCA_demo
# Date: 4/24/19
# Time: 11:57 AM
# File: utils.py


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import gzip
import numpy as np
import sklearn.svm as svm
from sklearn.metrics import accuracy_score
from keras.utils.data_utils import get_file

from matplotlib import pyplot as plt
from torch.autograd import Variable

from data import Dataset, ds


# def load_data(path):
#     f = gzip.open(path, 'rb')
#     train_set, valid_set, test_set = load_pickle(f)
#     f.close()
#
#     train_set_x, train_set_y = make_numpy_array(train_set)
#     valid_set_x, valid_set_y = make_numpy_array(valid_set)
#     test_set_x, test_set_y = make_numpy_array(test_set)
#
#     return [(train_set_x, train_set_y), (valid_set_x, valid_set_y), (test_set_x, test_set_y)]


def load_data(data_file, url):
    """loads the data from the gzip pickled files, and converts to numpy arrays"""
    print('loading data ...')
    path = get_file(data_file, origin=url)
    f = gzip.open(path, 'rb')
    train_set, valid_set, test_set = load_pickle(f)
    f.close()

    train_set_x, train_set_y = make_numpy_array(train_set)
    valid_set_x, valid_set_y = make_numpy_array(valid_set)
    test_set_x, test_set_y = make_numpy_array(test_set)

    return [(train_set_x, train_set_y), (valid_set_x, valid_set_y), (test_set_x, test_set_y)]


def load_pickle(f):
    """
    loads and returns the content of a pickled file
    it handles the inconsistencies between the pickle packages available in Python 2 and 3
    """
    try:
        import cPickle as thepickle
    except ImportError:
        import _pickle as thepickle

    try:
        ret = thepickle.load(f, encoding='latin1')
    except TypeError:
        ret = thepickle.load(f)

    return ret


def make_numpy_array(data_xy):
    """converts the input to numpy arrays"""
    data_x, data_y = data_xy
    # data_x = np.asarray(data_x, dtype=theano.config.floatX)
    data_x = np.asarray(data_x, dtype='float32')
    data_y = np.asarray(data_y, dtype='int32')
    return data_x, data_y


def make_iterable(data, batch_size, shuffle=False):
    data_x, data_y = data
    dataset = Dataset(data_x, data_y)
    data_loader = ds.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return data_loader


def create_view_loaders(view_data, batch_sizes, has_test=True):
    tr_batch_size, v_batch_size, tt_batch_size = batch_sizes
    train_x, train_y = view_data[0]
    valid_x, valid_y = view_data[1]
    if has_test:
        test_x, test_y = view_data[2]

    batch_num_train = len(train_x) // tr_batch_size
    batch_num_valid = len(valid_x) // v_batch_size
    if has_test:
        batch_num_test = len(test_x) // tt_batch_size

    num_train = batch_num_train * tr_batch_size
    num_valid = batch_num_valid * v_batch_size
    if has_test:
        num_test = batch_num_test * tt_batch_size

    train_x, train_y = train_x[:num_train, :], train_y[:num_train]
    valid_x, valid_y = valid_x[:num_valid, :], valid_y[:num_valid]
    if has_test:
        test_x, test_y = test_x[:num_test, :], test_y[:num_test]

    train_loader = make_iterable((train_x, train_y), tr_batch_size, shuffle=True)
    valid_loader = make_iterable((valid_x, valid_y), v_batch_size)
    if has_test:
        test_loader = make_iterable((test_x, test_y), tt_batch_size)
    if has_test:
        return train_loader, valid_loader, test_loader, (num_train, num_valid, num_test)
    else:
        return train_loader, valid_loader, (num_train, num_valid)


def construct_iris_views(dataframe):
    view1 = dataframe[['SepalLengthCm', 'SepalWidthCm', 'Species']].values
    view2 = dataframe[['PetalLengthCm', 'PetalWidthCm', 'Species']].values
    return view1, view2


def visualize(path, **kwargs):
    if kwargs['corr_hist']:
        corr_hist = kwargs['corr_hist']
        fig = plt.figure()
        plt.plot(corr_hist)
        plt.ylabel('correlation')
        plt.xlabel('iteration')
        plt.savefig(path)
        plt.close(fig)


def svm_classify(data, C=1.0, has_test=True):
    """
    trains a linear SVM on the data
    input C specifies the penalty factor of SVM
    """
    train_data, train_label = data[0]
    valid_data, valid_label = data[1]
    if has_test:
        test_data, test_label = data[2]

    print('training SVM...')
    clf = svm.LinearSVC(C=C, dual=False)
    clf.fit(train_data, train_label.ravel())

    # sanity check
    p = clf.predict(train_data)
    san_acc = accuracy_score(train_label, p)

    p = clf.predict(valid_data)
    valid_acc = accuracy_score(valid_label, p)

    if has_test:
        p = clf.predict(test_data)
        test_acc = accuracy_score(test_label, p)
    if has_test:
        return [san_acc, valid_acc, test_acc]
    else:
        return [san_acc, valid_acc]


def process_evaluation_data(model, data_loaders, latent_dim):
    datasets = []
    for dloader in data_loaders:
        data_x = []
        data_y = []
        for X, y in dloader:
            X = Variable(X, requires_grad=False)
            out = model(X)
            data_x.append(out.data.numpy())
            data_y.append(y.data.numpy())
        datasets.append((np.array(data_x).reshape(-1, latent_dim), np.array(data_y).ravel()))
    return datasets
