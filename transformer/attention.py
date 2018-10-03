from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from keras import backend as K
from keras.engine.base_layer import Layer
from keras.layers import Dense, Dropout

import backend2 as K2


class Attention(Layer):

    def __init__(self,
                 hidden_size,
                 num_heads,
                 attention_dropout,
                 is_self_attention,
                 **kwargs):
        if hidden_size % num_heads != 0:
            raise ValueError("Hidden size must be evenly divisible by the number of "
                             "heads.")

        super(Attention, self).__init__(**kwargs)
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.attention_dropout_layer = Dropout(attention_dropout)
        self.is_self_attention = is_self_attention

        # Layers for linearly projecting the queries, keys, and values.
        self.q_dense_layer = Dense(hidden_size, use_bias=False, name=self.name + "_q")
        self.k_dense_layer = Dense(hidden_size, use_bias=False, name=self.name + "_k")
        self.v_dense_layer = Dense(hidden_size, use_bias=False, name=self.name + "_v")

        # self.input_spec = [InputSpec(shape=(None, hidden_size)),
        #                   InputSpec(shape=(None, hidden_size))]
        self.output_dense_layer = Dense(hidden_size, use_bias=False,
                                                  name=self.name + "_output_transform")

    def split_heads(self, x):
        batch_size = K.shape(x)[0]
        length = K.shape(x)[1]

        # Calculate depth of last dimension after it has been split.
        depth = (self.hidden_size // self.num_heads)

        # Split the last dimension
        x = K.reshape(x, (batch_size, length, self.num_heads, depth))

        # Transpose the result
        return K.permute_dimensions(x, (0, 2, 1, 3))

    def combine_heads(self, x):
        batch_size = K.shape(x)[0]
        length = K.shape(x)[2]
        x = K.permute_dimensions(x, (0, 2, 1, 3))  # --> [batch, length, num_heads, depth]
        return K.reshape(x, (batch_size, length, self.hidden_size))

    def compute_output_shape(self, input_shape):
        return input_shape[0]

    # input [batch_size, seq_len, input_vec_size]
    def call(self, inputs, cache=None, **kwargs):
        if self.is_self_attention:
            x = inputs[0]
            y = inputs[0]
            bias = inputs[1]
        else:
            x = inputs[0]
            y = inputs[1]
            bias = inputs[2]

        q = self.q_dense_layer(x)
        k = self.k_dense_layer(y)
        v = self.v_dense_layer(y)

        if cache is not None:
            # Combine cached keys and values with new keys and values.
            k = K.concatenate([cache["k"], k], axis=1)
            v = K.concatenate([cache["v"], v], axis=1)

            # Update cache
            cache["k"] = k
            cache["v"] = v

        # [batch_size, seq_len, hidden_size]

        # Split q, k, v into heads.
        q = self.split_heads(q)
        k = self.split_heads(k)
        v = self.split_heads(v)

        # Scale q to prevent the dot product between q and k from growing too large.
        depth = (self.hidden_size // self.num_heads)
        q *= depth ** -0.5

        # Calculate dot product attention
        logits = K2.matmul(q, K.permute_dimensions(k, (0, 1, 3, 2)))
        logits += bias
        weights = K.softmax(logits, axis=-1)
        weights = self.attention_dropout_layer(weights)
        attention_output = K2.matmul(weights, v)

        # Recombine heads --> [batch_size, length, hidden_size]
        attention_output = self.combine_heads(attention_output)

        # Run the combined outputs through another linear projection layer.
        attention_output = self.output_dense_layer(attention_output)
        return attention_output

    # TODO: add get_config/from_config
