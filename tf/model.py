# Author: bbsipingsoft
# Project: DCCA_demo
# Date: 4/30/19
# Time: 11:11 PM
# File: model.py


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from keras.layers import Input, Dense
from keras.models import Model


class ViewModelFFN(object):
    def __init__(self, in_dim, latent_dim=10, hlyrs=(500,), activation='sigmoid', name=None):
        self.latent_dim = latent_dim
        input_layer = Input(shape=(in_dim,), name=name + '_input_layer')
        curr_tensor = input_layer
        for i in range(len(hlyrs)):
            curr_tensor = Dense(hlyrs[i], activation=activation,
                                name=name + '_hidden_layer_' + str(i))(curr_tensor)
        output = Dense(units=latent_dim, activation='linear', name=name + '_output_layer')(curr_tensor)
        self.model = Model(input_layer, output, name=name + '_model')

        # canonical weight
        self.W = None

    @property
    def canonical_weight(self):
        return self.W

    @canonical_weight.setter
    def canonical_weight(self, w):
        self.W = w

    @property
    def layers(self):
        return self.model.layers

    @property
    def input(self):
        return self.model.input

    @property
    def output_layer(self):
        return self.model.output

    def predict(self, x):
        return self.model.predict(x)
