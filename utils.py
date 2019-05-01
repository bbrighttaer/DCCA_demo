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
from keras.utils.data_utils import get_file
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score
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
    # train_set_x = train_set_x / 255
    valid_set_x, valid_set_y = make_numpy_array(valid_set)
    # valid_set_x = valid_set_x / 255
    test_set_x, test_set_y = make_numpy_array(test_set)
    # test_set_x = test_set_x / 255

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
    dataset = Dataset(data)
    return ds.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def create_view_loader(views_data, batch_size, shuffle=True):
    data = []

    for i, view_data in enumerate(views_data):
        x_data, y_data = view_data

        batches = len(x_data) // batch_size

        num_samples = batches * batch_size

        x_data, y_data = x_data[:num_samples, :], y_data[:num_samples]
        data.append((x_data, y_data))

    return make_iterable(data, batch_size, shuffle)


def construct_iris_views(dataframe):
    view1 = dataframe[['SepalLengthCm', 'SepalWidthCm', 'Species']].values
    view2 = dataframe[['PetalLengthCm', 'PetalWidthCm', 'Species']].values
    return view1, view2


def visualize(path, series):
    fig = plt.figure()
    legend = []
    for k in series.keys():
        plt.plot(series[k])
        legend.append(k)
    plt.ylabel('correlation')
    plt.xlabel('iteration')
    plt.legend(legend)
    plt.savefig(path)
    plt.close(fig)


def svm_classify(data, C=0.01, has_test=True):
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


def process_evaluation_data(dloader, dim, model=None, view_idx=0):
    data_x = []
    data_y = []
    for Xs, ys in dloader:
        v1_x = Xs[view_idx]
        v1_y = ys[view_idx]
        if model:
            v1_x = Variable(v1_x, requires_grad=False)
            v1_x = model(v1_x)
            canonical_weight = model.canonical_weight
            if canonical_weight is not None:
                v1_x = canonical_weight.t().mm(v1_x.t()).t()
        data_x.append(v1_x.data.numpy())
        data_y.append(v1_y.data.numpy())
    return np.array(data_x).reshape(-1, dim), np.array(data_y).ravel()


def construct_mnist_views(dataset, targets, num_views=2):
    shape = dataset.data.shape
    unrolled_dim = shape[1] * shape[2]
    data = dataset.data.view(shape[0], unrolled_dim)
    view_dim = unrolled_dim // num_views
    views_data = []

    last_index = 0
    for v in range(num_views):
        v_data = data[:, last_index:last_index + view_dim]
        views_data.append(v_data)
        last_index = last_index + view_dim

    views_dataset = []
    for v_data in views_data:
        v_dataset = (v_data.numpy(), targets.numpy())  # (X, y)
        views_dataset.append(v_dataset)

    return views_dataset


def evaluate(**kwargs):
    tr_ldr, val_ldr, tt_ldr = kwargs['tr_ldr'], kwargs['val_ldr'], kwargs['tt_ldr']
    dim = kwargs['dim']
    model = kwargs['model']
    view_idx = kwargs['view_idx']
    has_test = kwargs['has_test']  # boolean
    svm_train_dataset = process_evaluation_data(tr_ldr, dim, model, view_idx)
    svm_val_dataset = process_evaluation_data(val_ldr, dim, model, view_idx)
    if has_test:
        svm_test_dataset = process_evaluation_data(tt_ldr, dim, model, view_idx)
        sanity_check, val_accuracy, test_accuracy = svm_classify((svm_train_dataset, svm_val_dataset, svm_test_dataset),
                                                                 has_test=True)
        return sanity_check, val_accuracy, test_accuracy
    else:
        sanity_check, val_accuracy = svm_classify((svm_train_dataset, svm_val_dataset),
                                                  has_test=False)
        return sanity_check, val_accuracy
