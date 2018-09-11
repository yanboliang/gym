import tensorflow as tf
import numpy as np
from keras import Input
from keras import backend as K

from official.transformer.model.transformer import LayerNormalization as TFLayerNormalization

from transformer import LayerNormalization as KLayerNormalization
from test_utils import rel_cmp

if __name__ == '__main__':

    hidden_size = 4
    filter_size = 6
    batch_size = 8
    seq_len = 5

    np.random.seed(1)

    scale_init = np.random.rand(hidden_size)
    bias_init = np.random.rand(hidden_size)

    my_input = np.random.rand(batch_size, seq_len, hidden_size)

    tf_sess = tf.InteractiveSession()

    tf_norm = TFLayerNormalization(hidden_size)
    tf_input = tf.placeholder('float32', shape=(batch_size, seq_len, hidden_size))
    tf_output = tf_norm(tf_input)
    tf_sess.run([
        tf.assign(tf_norm.scale, scale_init),
        tf.assign(tf_norm.bias, bias_init),
    ])

    tf_res = tf_sess.run(tf_output, feed_dict={tf_input: my_input})
    print("tf output:")
    print(tf_res)

    k_norm = KLayerNormalization(hidden_size)
    k_input = Input(shape=(seq_len, hidden_size))
    k_output = k_norm(k_input)
    tf_sess.run([
        tf.assign(k_norm.scale, scale_init),
        tf.assign(k_norm.bias, bias_init),
    ])

    k_run = K.function([k_input], [k_output])
    k_res = k_run([my_input])[0]

    print("keras output:")
    print(k_res)

    assert rel_cmp(tf_res, k_res), "ffn test FAILed."

    print("test PASSed.")

    tf_sess.close()



