from official.transformer.utils import metrics as tf_metrics

import tensorflow as tf

from transformer import get_transformer_loss
from transformer_test import Params as KParams

from keras import Input
from keras import backend as K

from test_utils import rel_cmp

import numpy as np


def tf_loss_fn(logits, targets, label_smoothing, vocab_size):

    xentropy, weights = tf_metrics.padded_cross_entropy_loss(
        logits, targets, label_smoothing, vocab_size)
    # Compute the weighted mean of the cross entropy losses
    loss = tf.reduce_sum(xentropy) / tf.reduce_sum(weights)

    return loss, xentropy, weights


def k_loss_fn(logits, targets, label_smoothing, vocab_size):
    params = KParams()
    params.vocab_size = vocab_size
    params.label_smoothing = label_smoothing

    loss_fn = get_transformer_loss(params, with_xent=True)
    return loss_fn(targets, logits)


if __name__ == '__main__':

    _batch_size = 8
    _seq_len_y = 5
    vocab_size = 7
    label_smoothing = 0.1

    np.random.seed(1)

    my_logits = np.random.rand(_batch_size, _seq_len_y, vocab_size)
    my_targets = np.random.randint(0, vocab_size, size=(_batch_size, _seq_len_y))

    sess = tf.InteractiveSession()
    tf_logits = tf.placeholder('float32', shape=(_batch_size, _seq_len_y, vocab_size))
    tf_targets = tf.placeholder('int32', shape=(_batch_size, _seq_len_y))
    tf_loss, tf_xent, tf_weight = tf_loss_fn(tf_logits, tf_targets, label_smoothing, vocab_size)
    tf_loss_res = sess.run([tf_loss, tf_xent, tf_weight], feed_dict={
        tf_logits: my_logits,
        tf_targets: my_targets
    })
    print("tf_loss_res: " + str(tf_loss_res))

    k_logits = Input(shape=(_seq_len_y, vocab_size))
    k_targets = Input(shape=(_seq_len_y,), dtype='int32')
    k_loss = k_loss_fn(k_logits, k_targets, label_smoothing, vocab_size)
    k_loss_res = K.function([k_logits, k_targets], list(k_loss))([my_logits, my_targets])
    print("k_loss_res: " + str(k_loss_res))

    rel_cmp(tf_loss_res[0], k_loss_res[0])

    print("test PASSed.")

    sess.close()
