import tensorflow as tf
import numpy as np
from keras import Input
from keras import backend as K

from official.transformer.model.embedding_layer import EmbeddingSharedWeights as TFEmbeddingSharedWeights

from embedding import EmbeddingSharedWeights as KEmbeddingSharedWeights
from test_utils import rel_cmp

if __name__ == '__main__':

    vocab_size = 6
    hidden_size = 4
    _batch_size = 8
    _seq_len = 5

    # do this to test for symbolic shape size.
    batch_size = K.constant(_batch_size, dtype='int32')
    seq_len = K.constant(_seq_len, dtype='int32')

    np.random.seed(1)
    my_weight = np.random.rand(vocab_size, hidden_size)
    my_input = np.random.randint(0, vocab_size, size=(_batch_size, _seq_len))
    my_embedding_input = np.random.rand(_batch_size, _seq_len, hidden_size)

    tf_sess = tf.InteractiveSession()

    tf_embeddingLayer = TFEmbeddingSharedWeights(vocab_size, hidden_size)
    tf_input = tf.placeholder('int32', shape=(_batch_size, _seq_len))
    tf_embedding_input = tf.placeholder('float32', shape=(_batch_size, _seq_len, hidden_size))
    tf_output = tf_embeddingLayer(tf_input)
    tf_logits_output = tf_embeddingLayer.linear(tf_embedding_input)

    tf_sess.run(tf.global_variables_initializer())
    tf_assign_op = tf.assign(tf_embeddingLayer.shared_weights, my_weight)
    tf_sess.run(tf_assign_op)

    tf_res = tf_sess.run(tf_output, feed_dict={tf_input: my_input})
    tf_logits_res = tf_sess.run(tf_logits_output,
                                feed_dict={tf_embedding_input: my_embedding_input})
    print("tf EmbeddingSharedWeights output:")
    print(tf_res)
    print("tf EmbeddingSharedWeights logits output:")
    print(tf_logits_res)

    k_embeddingLayer = KEmbeddingSharedWeights(vocab_size, hidden_size)
    k_input = Input(shape=(_seq_len,))
    k_embedding_input = Input(shape=(_seq_len, hidden_size))
    k_output = k_embeddingLayer(k_input)
    k_logits_output = k_embeddingLayer.linear(k_embedding_input)

    tf_sess.run(tf.global_variables_initializer())
    k_assign_op = tf.assign(k_embeddingLayer.shared_weights, my_weight)
    tf_sess.run(k_assign_op)

    k_run = K.function([k_input], [k_output])
    k_logits_run = K.function([k_embedding_input], [k_logits_output])

    k_res = k_run([my_input])[0]
    k_logits_res = k_logits_run([my_embedding_input])[0]
    print("k EmbeddingSharedWeights output:")
    print(k_res)
    print("k EmbeddingSharedWeights logits output:")
    print(k_logits_res)

    tf_sess.close()

    assert rel_cmp(tf_res, k_res), "embedding test FAILed."
    assert rel_cmp(tf_logits_res, k_logits_res), "embedding logits test FAILed."

    print("test PASSed.")



