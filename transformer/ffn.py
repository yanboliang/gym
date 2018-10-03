from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from keras.layers import Dense, Dropout
from model_utils import WeightsRef


def feed_forward_network(x, params):
    filter_dense_layer = Dense(params.filter_size, use_bias=True, activation='relu')
    dropout_layer = Dropout(params.relu_dropout)
    output_dense_layer = Dense(params.hidden_size, use_bias=True)
    output = filter_dense_layer(x)
    output = dropout_layer(output)
    output = output_dense_layer(output)

    wr = WeightsRef()
    wr.filter_dense_layer = filter_dense_layer
    wr.output_dense_layer = output_dense_layer
    return output, wr
