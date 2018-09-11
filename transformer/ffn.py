from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from keras import backend as K
from keras.engine.base_layer import Layer
from keras.layers import Dense


class FeedFowardNetwork(Layer):

    def __init__(self,
                 hidden_size,
                 filter_size,
                 relu_dropout,
                 **kwargs):

        super(FeedFowardNetwork, self).__init__(**kwargs)

        self.hidden_size = hidden_size
        self.filter_size = filter_size
        self.relu_dropout = relu_dropout
        self.filter_dense_layer = Dense(filter_size, use_bias=True, activation='relu')
        self.output_dense_layer = Dense(hidden_size, use_bias=True)

    # def compute_output_shape(self, input_shape):
    #    return input_shape[0]

    # TODO: add padding support
    def call(self, inputs, train=True, **kwargs):
        # assert isinstance(inputs, (list, tuple))
        # inputs = inputs[0]

        # batch_size = K.shape(inputs)[0]
        # length = K.shape(inputs)[1]

        output = self.filter_dense_layer(inputs)
        if train:
            output = K.dropout(output, self.relu_dropout)
        output = self.output_dense_layer(output)
        return output

    # TODO: add get_config/from_config
