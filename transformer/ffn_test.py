from official.transformer.model.ffn_layer import FeedFowardNetwork as TFFeedFowardNetwork
from official.transformer.model.transformer import PrePostProcessingWrapper as TFPrePostProcessingWrapper

from ffn import feed_forward_network as Kfeed_forward_network
from transformer import pre_post_processor_wrapper as Kpre_post_processor_wrapper

import tensorflow as tf
import numpy as np

from keras import Input
from keras import backend as K

from test_utils import rel_cmp

if __name__ == '__main__':

    hidden_size = 4
    filter_size = 6
    _batch_size = 8
    _seq_len = 5

    class LayerParams(object):
        layer_postprocess_dropout = 0.3
        hidden_size = hidden_size
        filter_size = filter_size
        relu_dropout = 0.3


    params = LayerParams()

    np.random.seed(1)

    my_input = np.random.rand(_batch_size, _seq_len, hidden_size)
    weight_filter = np.random.rand(hidden_size, filter_size)
    weight_bias_filter = np.random.rand(filter_size)
    weight_output = np.random.rand(filter_size, hidden_size)
    weight_bias_output = np.random.rand(hidden_size)

    scale_init = np.random.rand(hidden_size)
    bias_init = np.random.rand(hidden_size)

    tf_sess = tf.InteractiveSession()

    tf_ffn = TFFeedFowardNetwork(hidden_size, filter_size, relu_dropout=0.5, train=False)
    tf_input = tf.placeholder('float32', shape=(_batch_size, _seq_len, hidden_size))

    tf_wrapper = TFPrePostProcessingWrapper(tf_ffn, params, train=False)
    tf_output = tf_wrapper(tf_input)

    tf_sess.run([
        tf.assign(tf_ffn.filter_dense_layer.kernel, weight_filter),
        tf.assign(tf_ffn.filter_dense_layer.bias, weight_bias_filter),
        tf.assign(tf_ffn.output_dense_layer.kernel, weight_output),
        tf.assign(tf_ffn.output_dense_layer.bias, weight_bias_output),
        tf.assign(tf_wrapper.layer_norm.scale, scale_init),
        tf.assign(tf_wrapper.layer_norm.bias, bias_init),
    ])

    tf_res = tf_sess.run(tf_output, feed_dict={tf_input: my_input})

    print("tf output:")
    print(tf_res)

    K.set_learning_phase(0)
    # k_ffn = KFeedFowardNetwork(hidden_size, filter_size, relu_dropout=0.5)
    k_input = Input(shape=(_seq_len, hidden_size))
    # k_wrapper = KPrePostProcessingWrapper(k_ffn, params)
    # k_output = k_wrapper(k_input, train=False)

    def ffn_processor(inputs):
        return Kfeed_forward_network(inputs, params)
    k_output, k_wrapper = Kpre_post_processor_wrapper(ffn_processor, k_input, params)
    k_ffn = k_wrapper.layer

    tf_sess.run([
        tf.assign(k_ffn.filter_dense_layer.kernel, weight_filter),
        tf.assign(k_ffn.filter_dense_layer.bias, weight_bias_filter),
        tf.assign(k_ffn.output_dense_layer.kernel, weight_output),
        tf.assign(k_ffn.output_dense_layer.bias, weight_bias_output),
        tf.assign(k_wrapper.layer_norm.scale, scale_init),
        tf.assign(k_wrapper.layer_norm.bias, bias_init),
    ])

    k_run = K.function([k_input], [k_output])
    k_res = k_run([my_input])[0]

    print("keras output:")
    print(k_res)

    assert rel_cmp(tf_res, k_res), "ffn test FAILed."

    print("test PASSed.")

    tf_sess.close()
