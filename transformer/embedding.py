from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from keras.initializers import RandomNormal

from model_utils import *


class EmbeddingSharedWeights(Layer):
    def __init__(self,
                 vocab_size,
                 hidden_size,
                 **kwargs):

        super(EmbeddingSharedWeights, self).__init__(**kwargs)
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.shared_weights = None

    def build(self, input_shape):
        self.shared_weights = self.add_weight(
            shape=(self.vocab_size, self.hidden_size),
            initializer=RandomNormal(0.0, self.hidden_size ** -0.5),
            name='embeddings',
            dtype='float32')
        self.built = True

    def compute_output_shape(self, input_shape):
        if len(input_shape) == 2: # (batch_size, length)
            return input_shape + (self.hidden_size,)
        elif len(input_shape) == 3: # (batch_size, length, hidden_size)
            return input_shape[0:2] + (self.vocab_size,)
        else:
            raise Exception("embedding compute output shape error.")

    def call(self, inputs, do_embedding=True):
        if do_embedding:
            if K.dtype(inputs) != 'int32':
                inputs = K.cast(inputs, 'int32')
            embeddings = K.gather(self.shared_weights, inputs)

            # Scale embedding by the sqrt of the hidden size
            embeddings *= self.hidden_size ** 0.5

            # Create binary array of size [batch_size, length]
            # where 1 = padding, 0 = not padding
            padding = get_padding(inputs)

            # Set all padding embedding values to 0
            embeddings *= K.expand_dims(1 - padding, -1)
            return embeddings
        else:
            return self.linear(inputs)

    def linear(self, x):
        shape = K.shape(x)
        batch_size = shape[0]
        length = shape[1]

        x = K.reshape(x, [batch_size * length, self.hidden_size])
        logits = K2.matmul(x, K.transpose(self.shared_weights))

        return K.reshape(logits, [batch_size, length, self.vocab_size])

