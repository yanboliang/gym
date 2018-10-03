from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math

from keras import backend as K
from keras.engine.base_layer import Layer

from keras.layers import Lambda

import backend2 as K2

_NEG_INF = -1e9


def get_padding(x):
    return K.cast(K.equal(x, 0), 'float32')


def padding_layer():
    return Lambda(lambda x: get_padding(x))


def _get_padding_bias(x):
    padding = get_padding(x)
    attention_bias = padding * _NEG_INF
    return K.expand_dims(K.expand_dims(attention_bias, axis=1), axis=1)


def get_padding_bias(x):
    return Lambda(lambda x: _get_padding_bias(x))(x)


def get_position_encoding(
        length,  # this is tensor
        hidden_size,
        min_timescale=1.0,
        max_timescale=1.0e4):
    """Return positional encoding.

    Calculates the position encoding as a mix of sine and cosine functions with
    geometrically increasing wavelengths.
    Defined and formulized in Attention is All You Need, section 3.5.

    Args:
      length: Sequence length.
      hidden_size: Size of the
      min_timescale: Minimum scale that will be applied at each position
      max_timescale: Maximum scale that will be applied at each position

    Returns:
      Tensor with shape [length, hidden_size]
    """
    position = K.cast(K2.range(0, length), 'float32')
    num_timescales = hidden_size // 2
    log_timescale_increment = (
        math.log(float(max_timescale) / float(min_timescale)) /
        (float(num_timescales) - 1.0))
    inv_timescales = min_timescale * K.exp(
        K.cast(K2.range(0, num_timescales), 'float32') * (-log_timescale_increment))
    scaled_time = K.expand_dims(position, 1) * K.expand_dims(inv_timescales, 0)
    signal = K.concatenate([K.sin(scaled_time), K.cos(scaled_time)], axis=1)
    return signal


def with_position_encoding(
        x,
        hidden_size,
        min_timescale=1.0,
        max_timescale=1.0e4):
    def with_position_encoding_func(embedded_inputs):
        length = K.shape(embedded_inputs)[1]
        return embedded_inputs + get_position_encoding(
            length, hidden_size, min_timescale, max_timescale)

    return Lambda(with_position_encoding_func)(x)


def get_decoder_self_attention_bias_from_len(length):
    valid_locs = K2.matrix_band_part(K.ones([length, length]), -1, 0)
    valid_locs = K.reshape(valid_locs, [1, 1, length, length])
    decoder_bias = _NEG_INF * (1.0 - valid_locs)
    return decoder_bias


def get_decoder_self_attention_bias(x):
    """Calculate bias for decoder that maintains model's autoregressive property.

    Creates a tensor that masks out locations that correspond to illegal
    connections, so prediction at position i cannot draw information from future
    positions.

    Args:
      length: int length of sequences in batch.

    Returns:
      float tensor of shape [1, 1, length, length]
    """
    def get_bias(embedded_inputs):
        length = K.shape(embedded_inputs)[1]
        return get_decoder_self_attention_bias_from_len(length)
    return Lambda(get_bias)(x)


def shift_decoder_input(x):
    def shift(embedded_inputs):
        decoder_inputs_shape = K.shape(embedded_inputs)
        length = decoder_inputs_shape[1]
        return K.slice(
            K2.pad(embedded_inputs, [[0, 0], [1, 0], [0, 0]]),
            (0, 0, 0),
            (decoder_inputs_shape[0], length, decoder_inputs_shape[2]))
    return Lambda(shift)(x)


class LayerNormalization(Layer):

    def __init__(self, hidden_size):
        super(LayerNormalization, self).__init__()
        self.hidden_size = hidden_size
        shape = [hidden_size]
        self.scale = K.variable(K.ones(shape=shape, dtype='float32'), dtype='float32')
        self.bias = K.variable(K.zeros(shape=shape, dtype='float32'), dtype='float32')

    def call(self, x, epsilon=1e-6):
        mean = K.mean(x, axis=[-1], keepdims=True)
        variance = K.mean(K.square(x - mean), axis=[-1], keepdims=True)
        norm_x = (x - mean) * K.pow(variance + epsilon, -0.5)
        return norm_x * self.scale + self.bias


class WeightsRef(object):
    pass
