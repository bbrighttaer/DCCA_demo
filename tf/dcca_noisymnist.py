# Author: bbsipingsoft
# Project: DCCA_demo
# Date: 5/1/19
# Time: 1:49 AM
# File: dcca_noisymnist.py


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import keras.backend as K
import numpy as np
from keras.layers import concatenate
from keras.models import Model

from tf.cca_layer import CCA
from tf.model import ViewModelFFN
from utils import load_data, svm_classify, visualize
from sklearn.preprocessing import StandardScaler


# seed = 2019
# np.random.seed(seed)
# tf.set_random_seed(seed)


def merge_models(sub_models, name=None):
    out_lyrs_lst = [mdl.output_layer for mdl in sub_models]
    merged_out_layer = concatenate(out_lyrs_lst, name='merged_output_layer')
    cca_lyr = CCA(name=name + '_cca_layer')(merged_out_layer)
    model = Model(inputs=[sub_mdl.input for sub_mdl in sub_models], outputs=cca_lyr)
    # optimizer = tf.keras.optimizers.Adam(lr=0.001)
    model.compile(optimizer='Adam', loss=loss_function, metrics=[mean_pred])
    return model


def loss_function(y_true, y_pred):
    print(y_pred)
    return y_pred


def mean_pred(y_true, y_pred):
    return K.mean(y_pred)


def prep_data(data, batch_size):
    data_new = []
    for i, ds in enumerate(data):
        x_data, y_data = ds

        batches = len(x_data) // batch_size

        num_samples = batches * batch_size

        x_data, y_data = x_data[:num_samples, :], y_data[:num_samples]
        data_new.append((x_data, y_data))
    return data_new


if __name__ == '__main__':
    v1_input_dim = 784
    v2_input_dim = 784
    latent_dim = 10

    n_iters = 5000
    batch_sz = 1000

    data1 = prep_data(
        load_data('noisymnist_view1.gz', 'https://www2.cs.uic.edu/~vnoroozi/noisy-mnist/noisymnist_view1.gz'), batch_sz)
    data2 = prep_data(
        load_data('noisymnist_view2.gz', 'https://www2.cs.uic.edu/~vnoroozi/noisy-mnist/noisymnist_view2.gz'), batch_sz)

    # view 1 data
    scl1 = StandardScaler()
    v1_train_x, v1_train_y = data1[0]
    scl1.fit(v1_train_x)
    scl1.transform(v1_train_x)
    v1_valid_x, v1_valid_y = data1[1]
    scl1.transform(v1_valid_x)
    v1_test_x, v1_test_y = data1[2]
    scl1.transform(v1_test_x)

    # view 2 data
    scl2 = StandardScaler()
    v2_train_x, v2_train_y = data2[0]
    scl2.fit(v2_train_x)
    scl2.transform(v2_train_x)
    v2_valid_x, v2_valid_y = data2[1]
    scl2.transform(v2_valid_x)
    v2_test_x, v2_test_y = data2[2]
    scl2.transform(v2_test_x)

    # models
    view1 = ViewModelFFN(v1_input_dim, latent_dim=latent_dim, hlyrs=(1024, 1024, 1024), name='view1')
    view2 = ViewModelFFN(v2_input_dim, latent_dim=latent_dim, hlyrs=(1024, 1024, 1024), name='view2')

    # models combination
    model = merge_models([view1, view2], name='merged_model')

    n_epochs = int(n_iters / float(len(v1_train_x) / batch_sz))
    hist = model.fit(x=[v1_train_x, v2_train_x],
                     y=np.zeros(len(v1_train_x)),
                     batch_size=batch_sz,
                     epochs=n_epochs,
                     verbose=1,
                     validation_data=([v1_valid_x, v2_valid_x], np.zeros(len(v1_valid_x))),
                     shuffle=True)

    v1_train_x_new = view1.predict(v1_train_x)
    v1_valid_x_new = view1.predict(v1_valid_x)
    v1_test_x_new = view1.predict(v1_test_x)
    sanity_check, valid_accuracy, test_accuracy = svm_classify(([v1_train_x_new, v1_train_y],
                                                                [v1_valid_x_new, v1_valid_y],
                                                                [v1_test_x_new, v1_test_y]),
                                                               has_test=True)
    print("DCCA: training set accuracy={:.2f}, validation set accuracy={:.2f}, test set accuracy={:.2f}"
          .format(sanity_check * 100, valid_accuracy * 100, test_accuracy * 100))

    # Linear SVM baseline
    sanity_check, valid_accuracy, test_accuracy = svm_classify(([v1_train_x, v1_train_y],
                                                                [v1_valid_x, v1_valid_y],
                                                                [v1_test_x, v1_test_y]),
                                                               has_test=True)
    print(
        "Linear SVC (baseline): training set accuracy={:.2f}, validation set accuracy={:.2f}, test set accuracy={:.2f}"
        .format(sanity_check * 100, valid_accuracy * 100, test_accuracy * 100))

    cor_hist = {k: np.abs(hist.history[k]) for k in hist.history.keys()}
    visualize('corr_noisy_mnist.png', series=cor_hist)
