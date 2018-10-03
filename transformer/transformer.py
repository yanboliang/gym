from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from keras import Input, Model
from keras.layers import Dropout

from attention import Attention
from ffn import feed_forward_network
from embedding import EmbeddingSharedWeights
from model_utils import *
from backend2 import sequence_beam_search


EOS_ID = 1


def transformer(params):
    src_input = Input(shape=(None,), dtype='int32')
    target_input = Input(shape=(None,), dtype='int32')

    embedding_softmax_layer = EmbeddingSharedWeights(params.vocab_size, params.hidden_size)

    attention_bias = get_padding_bias(src_input)
    encoder_inputs = embedding_softmax_layer(src_input, do_embedding=True)

    encoder_inputs = with_position_encoding(encoder_inputs, params.hidden_size)
    encoder_inputs = Dropout(params.layer_postprocess_dropout)(encoder_inputs)

    encoder_outputs, encoder_stack_wr = encoder_stack(encoder_inputs, attention_bias, params)

    decoder_inputs = embedding_softmax_layer(target_input, do_embedding=True)
    decoder_self_attention_bias = get_decoder_self_attention_bias(decoder_inputs)
    decoder_inputs = shift_decoder_input(decoder_inputs)
    decoder_inputs = with_position_encoding(decoder_inputs, hidden_size=params.hidden_size)
    decoder_inputs = Dropout(params.layer_postprocess_dropout)(decoder_inputs)

    decoder_outputs, decoder_stack_wr = decoder_stack(decoder_inputs,
                                                      encoder_outputs,
                                                      decoder_self_attention_bias,
                                                      attention_bias,
                                                      params)

    logits = embedding_softmax_layer(decoder_outputs, do_embedding=False)
    model = Model([src_input, target_input], logits)
    # Used for testing
    model.embedding_softmax_layer = embedding_softmax_layer
    model.encoder_stack = encoder_stack_wr
    model.decoder_stack = decoder_stack_wr
    return model


def get_transformer_loss(params, with_xent=False):
    smoothing = params.label_smoothing
    vocab_size = params.vocab_size

    def transformer_loss(y_true, y_pred):
        """
        :param y_true: labels: Tensor of size [batch_size, length_labels]
        :param y_pred: logits: Tensor of size [batch_size, length_logits, vocab_size]
        :return: loss
        """
        confidence = 1.0 - smoothing
        low_confidence = (1.0 - confidence) / float(vocab_size - 1)

        logits = y_pred

        y_pred_shape = K.shape(y_pred)
        y_true = K.reshape(y_true, [y_pred_shape[0], y_pred_shape[1]])

        y_true = K.cast(y_true, dtype='int32')
        y_true_ohe = K.one_hot(y_true, num_classes=vocab_size)

        soft_targets = y_true_ohe * (confidence - low_confidence) + low_confidence
        xentropy = K.categorical_crossentropy(soft_targets, logits, from_logits=True)

        # Calculate the best (lowest) possible value of cross entropy, and
        # subtract from the cross entropy loss.
        normalizing_constant = -(
                confidence * math.log(confidence) + float(vocab_size - 1) *
                low_confidence * math.log(low_confidence + 1e-20))
        xentropy = xentropy - normalizing_constant

        weights = 1.0 - K.squeeze(K.slice(y_true_ohe, (0, 0, 0), (-1, -1, 1)), axis=2)

        xentropy = xentropy * weights

        loss = K.sum(xentropy) / K.sum(weights)
        if with_xent:
            return loss, xentropy, weights
        else:
            return loss

    return transformer_loss


def transformer_predict(model, params, src_data):
    src_input = Input(shape=(None,), dtype="int32")
    src_input_shape = K.shape(src_input)
    batch_size = src_input_shape[0]
    input_length = src_input_shape[1]
    max_decode_length = input_length + params.extra_decode_length

    repeated_src_input = Lambda(lambda x: K.repeat_elements(x, params.beam_size, axis=0))(src_input)

    # Create initial set of IDs that will be passed into symbols_to_logits_fn.
    initial_ids = K.zeros([batch_size], dtype='int32')

    def get_symbols_to_logits_fn(model, src_input):
        def symbols_to_logits_fn(ids, i, cache):
            """Generate logits for next potential IDs.

            Args:
              ids: Current decoded sequences.
                int tensor with shape [batch_size * beam_size, i + 1]
              i: Loop index
              cache: dictionary of values storing the encoder output, encoder-decoder
                attention bias, and previous decoder attention values.

            Returns:
              Tuple of
                (logits with shape [batch_size * beam_size, vocab_size],
                 updated cache values)
            """
            target_input = ids
            logits = model([repeated_src_input, target_input])
            logits = K.slice(logits, (0, i, 0), (-1, 1, -1))
            logits = K.squeeze(logits, axis=1)
            return logits, cache
        return symbols_to_logits_fn

    # Use beam search to find the top beam_size sequences and scores.
    decoded_ids, scores = sequence_beam_search(
        symbols_to_logits_fn=get_symbols_to_logits_fn(model, src_input),
        initial_ids=initial_ids,
        initial_cache={},
        vocab_size=params.vocab_size,
        beam_size=params.beam_size,
        alpha=params.alpha,
        max_decode_length=max_decode_length,
        eos_id=EOS_ID)

    # Get the top sequence for each batch element
    top_decoded_ids = K.slice(decoded_ids, (0, 0, 1), (-1, 1, -1))
    top_decoded_ids = K.squeeze(top_decoded_ids, 1)

    predict_fn = K.function([src_input, K.learning_phase()], [top_decoded_ids])

    result = predict_fn([src_data, 0.])[0]
    return result


def pre_post_processor_wrapper(processor, inputs, params):
    layer_norm = LayerNormalization(params.hidden_size)
    dropout_layer = Dropout(params.layer_postprocess_dropout)
    if isinstance(inputs, list):
        x = inputs[0]
    else:
        x = inputs
    y = layer_norm(x)

    if isinstance(inputs, list):
        processor_inputs = [y] + inputs[1:]
    else:
        processor_inputs = y
    y, iwr = processor(processor_inputs)
    y = dropout_layer(y)
    y = Lambda(lambda x: x[0] + x[1])([x, y])

    wr = WeightsRef()
    wr.layer_norm = layer_norm
    wr.layer = iwr
    return y, wr


def encoder_stack(embedded_input, attention_bias, params):

    def self_attention_processor(inputs):
        self_attention_layer = Attention(params.hidden_size,
                                         params.num_heads,
                                         params.attention_dropout,
                                         is_self_attention=True)
        return self_attention_layer(inputs), self_attention_layer

    def ffn_processor(inputs):
        return feed_forward_network(inputs, params)

    output_normalization = LayerNormalization(params.hidden_size)

    wr = WeightsRef()
    wr.layers = []
    wr.output_normalization = output_normalization

    y = embedded_input
    for _ in range(params.num_hidden_layers):
        y, aiwr = pre_post_processor_wrapper(self_attention_processor,
                                             [y, attention_bias],
                                             params)
        y, fiwr = pre_post_processor_wrapper(ffn_processor, y, params)
        wr.layers.append([aiwr, fiwr])

    return output_normalization(y), wr


def decoder_stack(decoder_inputs,
                  encoder_outputs,
                  decoder_self_attention_bias,
                  attention_bias,
                  params):

    def self_attention_processor(inputs):
        self_attention_layer = Attention(params.hidden_size,
                                         params.num_heads,
                                         params.attention_dropout,
                                         is_self_attention=True)
        return self_attention_layer(inputs), self_attention_layer

    def enc_dec_attention_processor(inputs):
        enc_dec_attention_layer = Attention(params.hidden_size,
                                            params.num_heads,
                                            params.attention_dropout,
                                            is_self_attention=False)
        return enc_dec_attention_layer(inputs), enc_dec_attention_layer

    def ffn_processor(inputs):
        return feed_forward_network(inputs, params)

    output_normalization = LayerNormalization(params.hidden_size)

    wr = WeightsRef()
    wr.layers = []
    wr.output_normalization = output_normalization

    y = decoder_inputs
    for _ in range(params.num_hidden_layers):
        y, saiwr = pre_post_processor_wrapper(self_attention_processor,
                                             [y, decoder_self_attention_bias],
                                             params)
        y, aiwr = pre_post_processor_wrapper(enc_dec_attention_processor,
                                             [y, encoder_outputs, attention_bias],
                                             params)

        y, fiwr = pre_post_processor_wrapper(ffn_processor, y, params)
        wr.layers.append([saiwr, aiwr, fiwr])

    return output_normalization(y), wr
