import tensorflow as tf
import numpy as np
from keras import Input
from keras import backend as K

from official.transformer.model.attention_layer import Attention as TFAttention, SelfAttention as TFSelfAttention
from official.transformer.model.transformer import PrePostProcessingWrapper as TFPrePostProcessingWrapper
from official.transformer.model.model_utils import get_padding_bias as tf_get_padding_bias
from official.transformer.model.model_utils import get_decoder_self_attention_bias as tf_get_decoder_self_attention_bias

from attention import Attention as KAttention
from transformer import pre_post_processor_wrapper as k_pre_post_processor_wrapper
from model_utils import get_padding_bias as k_get_padding_bias
from model_utils import get_decoder_self_attention_bias_from_len as k_get_decoder_self_attention_bias_from_len
from test_utils import rel_cmp

if __name__ == '__main__':

    hidden_size = 6
    _batch_size = 8
    _seq_len_x = 4
    _seq_len_y = 5
    vocab_size = 6
    num_heads = 3

    class LayerParams(object):
        layer_postprocess_dropout = 0.5
        hidden_size = hidden_size

    params = LayerParams()

    # do this to test for symbolic shape size.
    batch_size = K.constant(_batch_size, dtype='int32')
    seq_len_x = K.constant(_seq_len_x, dtype='int32')
    seq_len_y = K.constant(_seq_len_y, dtype='int32')

    np.random.seed(1)
    weight_q = np.random.rand(hidden_size, hidden_size)
    weight_k = np.random.rand(hidden_size, hidden_size)
    weight_v = np.random.rand(hidden_size, hidden_size)
    weight_output = np.random.rand(hidden_size, hidden_size)

    scale_init = np.random.rand(hidden_size)
    bias_init = np.random.rand(hidden_size)

    # attention
    my_input_x = np.random.rand(_batch_size, _seq_len_x, hidden_size)
    my_input_y = np.random.rand(_batch_size, _seq_len_y, hidden_size)
    my_input_y_raw = np.random.randint(0, vocab_size, size=(_batch_size, _seq_len_y))

    tf_sess = tf.InteractiveSession()
    tf_attention = TFAttention(hidden_size=hidden_size, num_heads=num_heads,
                                attention_dropout=0.5, train=False)
    tf_input_x = tf.placeholder('float32', shape=(_batch_size, _seq_len_x, hidden_size))
    tf_input_y = tf.placeholder('float32', shape=(_batch_size, _seq_len_y, hidden_size))
    tf_input_y_raw = tf.placeholder('int32', shape=(_batch_size, _seq_len_y))

    # with tf.variable_scope(tf.get_variable_scope()):
    tf_output = tf_attention(tf_input_x, tf_input_y,
                             tf_get_padding_bias(tf_input_y_raw))

    # assign weight
    tf_sess.run([
        tf.assign(tf_attention.q_dense_layer.kernel, weight_q),
        tf.assign(tf_attention.k_dense_layer.kernel, weight_k),
        tf.assign(tf_attention.v_dense_layer.kernel, weight_v),
        tf.assign(tf_attention.output_dense_layer.kernel, weight_output)
    ])

    tf_res = tf_sess.run(tf_output, feed_dict={
        tf_input_x: my_input_x,
        tf_input_y: my_input_y,
        tf_input_y_raw: my_input_y_raw
    })

    print("tf Attention output:")
    print(tf_res)

    tf_self_attention = TFSelfAttention(hidden_size=hidden_size, num_heads=num_heads,
                                         attention_dropout=0.5, train=False)
    tf_wrapper = TFPrePostProcessingWrapper(tf_self_attention, params, train=False)

    # with tf.variable_scope(tf.get_variable_scope()):
    bias = tf_get_decoder_self_attention_bias(seq_len_y)
    tf_self_attention_output = tf_wrapper(tf_input_y, bias=bias)

    # assign weight
    tf_sess.run([
        tf.assign(tf_self_attention.q_dense_layer.kernel, weight_q),
        tf.assign(tf_self_attention.k_dense_layer.kernel, weight_k),
        tf.assign(tf_self_attention.v_dense_layer.kernel, weight_v),
        tf.assign(tf_self_attention.output_dense_layer.kernel, weight_output),
        tf.assign(tf_wrapper.layer_norm.scale, scale_init),
        tf.assign(tf_wrapper.layer_norm.bias, bias_init),
    ])

    tf_self_attention_res = tf_sess.run(tf_self_attention_output, feed_dict={
        tf_input_y: my_input_y
    })
    print("tf self Attention output:")
    print(tf_self_attention_res)

    K.set_learning_phase(0)

    k_attention = KAttention(hidden_size=hidden_size,
                             num_heads=num_heads,
                             attention_dropout=0.5,
                             is_self_attention=False)
    k_input_x = Input(shape=(_seq_len_x, hidden_size))
    k_input_y = Input(shape=(_seq_len_y, hidden_size))
    k_input_y_raw = Input(shape=(_seq_len_y,))

    k_output = k_attention([k_input_x, k_input_y,
                            k_get_padding_bias(k_input_y_raw)])
    # assign keras weight
    tf_sess.run([
        tf.assign(k_attention.q_dense_layer.kernel, weight_q),
        tf.assign(k_attention.k_dense_layer.kernel, weight_k),
        tf.assign(k_attention.v_dense_layer.kernel, weight_v),
        tf.assign(k_attention.output_dense_layer.kernel, weight_output),
    ])
    k_run = K.function([k_input_x, k_input_y, k_input_y_raw], [k_output])
    k_res = k_run([my_input_x, my_input_y, my_input_y_raw])[0]

    print("keras Attention output:")
    print(k_res)

    k_self_attention = KAttention(hidden_size=hidden_size,
                                  num_heads=num_heads,
                                  attention_dropout=0.5,
                                  is_self_attention=True)
    # k_wrapper = KPrePostProcessingWrapper(k_self_attention, params)
    bias = k_get_decoder_self_attention_bias_from_len(seq_len_y)
    # k_self_attention_output = k_wrapper([k_input_y, bias], train=False)
    def self_attention_processor(inputs):
        return k_self_attention(inputs), k_self_attention
    k_self_attention_output, k_wrapper =\
        k_pre_post_processor_wrapper(self_attention_processor, [k_input_y, bias], params)
    # assign keras weight
    tf_sess.run([
        tf.assign(k_self_attention.q_dense_layer.kernel, weight_q),
        tf.assign(k_self_attention.k_dense_layer.kernel, weight_k),
        tf.assign(k_self_attention.v_dense_layer.kernel, weight_v),
        tf.assign(k_self_attention.output_dense_layer.kernel, weight_output),
        tf.assign(k_wrapper.layer_norm.scale, scale_init),
        tf.assign(k_wrapper.layer_norm.bias, bias_init),
    ])
    k_self_attention_run = K.function([k_input_y], [k_self_attention_output])
    k_self_attention_res = k_self_attention_run([my_input_y])[0]

    print("keras self Attention output:")
    print(k_self_attention_res)

    assert rel_cmp(tf_res, k_res), "attention test FAILed."

    assert rel_cmp(tf_self_attention_res, k_self_attention_res), "self attention test FAILed."

    print("Test PASSed.")
    tf_sess.close()

