import tensorflow as tf
import numpy as np
from keras import Input
from keras import backend as K

from official.transformer.model.transformer import Transformer as TFTransformer
from official.transformer.model import model_utils as tf_model_utils

from transformer import transformer as ktransformer, transformer_predict as ktransformer_predict
import model_utils as k_model_utils
from test_utils import printoptions, rel_cmp

test_obj = {
    "i_list": [],
    "ids_list": [],
}

hidden_size = 18
_batch_size = 100
_seq_len_x = 17
_seq_len_y = 18
vocab_size = 9
num_heads = 3
num_hidden_layers = 3
filter_size = 7

# do this to test for symbolic shape size.
batch_size = K.constant(_batch_size, dtype='int32')
seq_len_x = K.constant(_seq_len_x, dtype='int32')
seq_len_y = K.constant(_seq_len_y, dtype='int32')


class Params(object):
    initializer_gain = 1.0
    attention_dropout = 0.5
    layer_postprocess_dropout = 0.5
    relu_dropout = 0.5
    hidden_size = hidden_size
    vocab_size = vocab_size
    num_heads = num_heads
    num_hidden_layers = num_hidden_layers
    filter_size = filter_size
    label_smoothing = 0.1
    extra_decode_length = 1
    beam_size = 3
    alpha = 0.6


params = Params()

np.random.seed(1)

embedding_weight = np.random.rand(vocab_size, hidden_size)

encoder_layer_norm_scale_init = [[np.ones((hidden_size,),dtype='float32') + 0.05 * np.random.rand(hidden_size)
                                  for _ in range(3)] for i in range(num_hidden_layers)]
encoder_layer_norm_bias_init = [[np.zeros((hidden_size,),dtype='float32') + 0.05 * np.random.rand(hidden_size)
                                 for _ in range(3)] for i in range(num_hidden_layers)]
decoder_layer_norm_scale_init = [[np.ones((hidden_size,),dtype='float32') + 0.05 * np.random.rand(hidden_size)
                                  for _ in range(4)] for i in range(num_hidden_layers)]
decoder_layer_norm_bias_init = [[np.zeros((hidden_size,),dtype='float32') + 0.05 * np.random.rand(hidden_size)
                                 for _ in range(4)] for i in range(num_hidden_layers)]


def gen_attention_weight():
    return [np.random.rand(hidden_size, hidden_size) for _ in range(num_hidden_layers)]


encoder_self_attention_weight_q = gen_attention_weight()
encoder_self_attention_weight_k = gen_attention_weight()
encoder_self_attention_weight_v = gen_attention_weight()
encoder_self_attention_weight_output = gen_attention_weight()

decoder_self_attention_weight_q = gen_attention_weight()
decoder_self_attention_weight_k = gen_attention_weight()
decoder_self_attention_weight_v = gen_attention_weight()
decoder_self_attention_weight_output = gen_attention_weight()

decoder_encoder_attention_weight_q = gen_attention_weight()
decoder_encoder_attention_weight_k = gen_attention_weight()
decoder_encoder_attention_weight_v = gen_attention_weight()
decoder_encoder_attention_weight_output = gen_attention_weight()

ffn_encoder_weight_filter = [np.random.rand(hidden_size, filter_size) for _ in range(num_hidden_layers)]
ffn_encoder_weight_bias_filter = [np.random.rand(filter_size) for _ in range(num_hidden_layers)]
ffn_encoder_weight_output = [np.random.rand(filter_size, hidden_size) for _ in range(num_hidden_layers)]
ffn_encoder_weight_bias_output = [np.random.rand(hidden_size) for _ in range(num_hidden_layers)]

ffn_decoder_weight_filter = [np.random.rand(hidden_size, filter_size) for _ in range(num_hidden_layers)]
ffn_decoder_weight_bias_filter = [np.random.rand(filter_size) for _ in range(num_hidden_layers)]
ffn_decoder_weight_output = [np.random.rand(filter_size, hidden_size) for _ in range(num_hidden_layers)]
ffn_decoder_weight_bias_output = [np.random.rand(hidden_size) for _ in range(num_hidden_layers)]


def set_input_padding(raw):
    nsize = len(raw)
    rlen = len(raw[0])
    assert rlen > 3
    for i in range(nsize):
        tlen = np.random.randint(3, rlen + 1)
        raw[i][tlen - 1] = 1.0
        for j in range(tlen, rlen):
            raw[i][j] = 0.0


my_input_x_raw = np.random.randint(2, vocab_size, size=(_batch_size, _seq_len_x))
my_input_y_raw = np.random.randint(2, vocab_size, size=(_batch_size, _seq_len_y))
set_input_padding(my_input_x_raw)
set_input_padding(my_input_y_raw)


def get_tf_assign_for_encoder_layer(transformer, layer_idx):
    layer = transformer.encoder_stack.layers[layer_idx]
    attention_layer = layer[0].layer
    ffn_layer = layer[1].layer
    norm_layer = [
        layer[0].layer_norm,
        layer[1].layer_norm,
        transformer.encoder_stack.output_normalization
    ]
    norm_layer_assign = []
    for i in range(3):
        norm_layer_assign.append(tf.assign(norm_layer[i].scale,
                                           encoder_layer_norm_scale_init[layer_idx][i]))
        norm_layer_assign.append(tf.assign(norm_layer[i].bias,
                                           encoder_layer_norm_bias_init[layer_idx][i]))
    return [
        tf.assign(attention_layer.q_dense_layer.kernel, encoder_self_attention_weight_q[layer_idx]),
        tf.assign(attention_layer.k_dense_layer.kernel, encoder_self_attention_weight_k[layer_idx]),
        tf.assign(attention_layer.v_dense_layer.kernel, encoder_self_attention_weight_v[layer_idx]),
        tf.assign(attention_layer.output_dense_layer.kernel, encoder_self_attention_weight_output[layer_idx]),

        tf.assign(ffn_layer.filter_dense_layer.kernel, ffn_encoder_weight_filter[layer_idx]),
        tf.assign(ffn_layer.filter_dense_layer.bias, ffn_encoder_weight_bias_filter[layer_idx]),
        tf.assign(ffn_layer.output_dense_layer.kernel, ffn_encoder_weight_output[layer_idx]),
        tf.assign(ffn_layer.output_dense_layer.bias, ffn_encoder_weight_bias_output[layer_idx]),
    ] + norm_layer_assign


def get_tf_assign_for_decoder_layer(transformer, layer_idx):
    layer = transformer.decoder_stack.layers[layer_idx]
    self_attention_layer = layer[0].layer
    ed_attention_layer = layer[1].layer
    ffn_layer = layer[2].layer
    # print("decoder self_attention_layer: " + str(self_attention_layer))
    norm_layer = [
        layer[0].layer_norm,
        layer[1].layer_norm,
        layer[2].layer_norm,
        transformer.decoder_stack.output_normalization
    ]
    norm_layer_assign = []
    for i in range(4):
        norm_layer_assign.append(tf.assign(norm_layer[i].scale,
                                           decoder_layer_norm_scale_init[layer_idx][i]))
        norm_layer_assign.append(tf.assign(norm_layer[i].bias,
                                           decoder_layer_norm_bias_init[layer_idx][i]))

    return [
        tf.assign(self_attention_layer.q_dense_layer.kernel, decoder_self_attention_weight_q[layer_idx]),
        tf.assign(self_attention_layer.k_dense_layer.kernel, decoder_self_attention_weight_k[layer_idx]),
        tf.assign(self_attention_layer.v_dense_layer.kernel, decoder_self_attention_weight_v[layer_idx]),
        tf.assign(self_attention_layer.output_dense_layer.kernel, decoder_self_attention_weight_output[layer_idx]),

        tf.assign(ed_attention_layer.q_dense_layer.kernel, decoder_encoder_attention_weight_q[layer_idx]),
        tf.assign(ed_attention_layer.k_dense_layer.kernel, decoder_encoder_attention_weight_k[layer_idx]),
        tf.assign(ed_attention_layer.v_dense_layer.kernel, decoder_encoder_attention_weight_v[layer_idx]),
        tf.assign(ed_attention_layer.output_dense_layer.kernel, decoder_encoder_attention_weight_output[layer_idx]),

        tf.assign(ffn_layer.filter_dense_layer.kernel, ffn_encoder_weight_filter[layer_idx]),
        tf.assign(ffn_layer.filter_dense_layer.bias, ffn_encoder_weight_bias_filter[layer_idx]),
        tf.assign(ffn_layer.output_dense_layer.kernel, ffn_encoder_weight_output[layer_idx]),
        tf.assign(ffn_layer.output_dense_layer.bias, ffn_encoder_weight_bias_output[layer_idx]),
    ] + norm_layer_assign


def get_assign_list(transformer):
    assign_list = [
        tf.assign(transformer.embedding_softmax_layer.shared_weights, embedding_weight)
    ]
    for i in range(num_hidden_layers):
        assign_list += get_tf_assign_for_encoder_layer(transformer, i)
        assign_list += get_tf_assign_for_decoder_layer(transformer, i)
    return assign_list


if __name__ == '__main__':

    tf_sess = tf.InteractiveSession()

    tf_transformer = TFTransformer(params, train=False)
    tf_input_x_raw = tf.placeholder('int32', shape=(_batch_size, _seq_len_x))
    tf_input_y_raw = tf.placeholder('int32', shape=(_batch_size, _seq_len_y))
    tf_output = tf_transformer(tf_input_x_raw, tf_input_y_raw)

    tf_sess.run(tf.global_variables_initializer())
    tf_assign_list = get_assign_list(tf_transformer)
    assert len(tf_assign_list) == len(list(set(tf_assign_list)))
    tf_sess.run(tf_assign_list)

    tf_res = tf_sess.run(tf_output, feed_dict={
        tf_input_x_raw: my_input_x_raw,
        tf_input_y_raw: my_input_y_raw
    })
    print("tf output:")
    with printoptions(precision=3, suppress=True):
        print(tf_res)

    tf_embedded_inputs = tf_transformer.embedding_softmax_layer(tf_input_x_raw)
    tf_pos_encoding = tf_model_utils.get_position_encoding(seq_len_x, tf_transformer.params.hidden_size)
    tf_embedding_inputs = tf_embedded_inputs + tf_pos_encoding

    tf_attention_bias = tf_model_utils.get_padding_bias(tf_input_x_raw)
    tf_encoder_outputs = tf_transformer.encode(tf_input_x_raw, tf_attention_bias)

    tf_pred = tf_transformer(tf_input_x_raw)["outputs"]
    tf_pred_res = tf_sess.run(tf_pred, feed_dict={tf_input_x_raw: my_input_x_raw})
    print("tf prediction:")
    with printoptions(threshold=2000):
        print(tf_pred_res)

    # K.set_learning_phase(0) default learning phase value is False.
    k_transformer = ktransformer(params)
    k_input_x_raw = Input(shape=(_seq_len_x,))
    k_input_y_raw = Input(shape=(_seq_len_y,))

    #k_embedded_inputs = k_transformer.embedding_softmax_layer(k_input_x_raw)
    #k_pos_encoding = k_model_utils._get_position_encoding(seq_len_x, k_transformer.params.hidden_size)
    #k_embedding_inputs = k_embedded_inputs + k_pos_encoding

    #k_attention_bias = k_model_utils._get_padding_bias(k_input_x_raw)
    # k_encoder_outputs = k_transformer.encode(k_input_x_raw, k_attention_bias, train=False)

    k_output = k_transformer([k_input_x_raw, k_input_y_raw])

    tf_sess.run(tf.global_variables_initializer())
    tf_sess.run(get_assign_list(k_transformer))

    k_run = K.function([k_input_x_raw, k_input_y_raw], [k_output])
    k_res = k_run([my_input_x_raw, my_input_y_raw])[0]
    print("k output:")
    with printoptions(precision=3, suppress=True):
        print(k_res)

    max_err = np.max(np.abs(tf_res - k_res) / (np.abs(tf_res) + 1e-99))
    print("max err: " + str(max_err))
    assert rel_cmp(tf_res, k_res, rel_tol=1e-1)

    """
    k_pred_result = k_transformer._get_predict_function()([my_input_x_raw, None])[0]
    print("k prediction:")
    with printoptions(threshold=2000):
        print(k_pred_result)

    assert rel_cmp(tf_pred_res.astype(float), k_pred_result.astype(float))
    """

    k_pred_result = ktransformer_predict(k_transformer, params, my_input_x_raw)
    print("k prediction:")
    with printoptions(threshold=2000):
        print(k_pred_result)

    print("test PASSed.")
    tf_sess.close()
