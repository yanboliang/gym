from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from keras import Input, Model
from keras.layers import Wrapper
from keras import backend as K

from attention import Attention, SelfAttention
from ffn import FeedFowardNetwork
from embedding import EmbeddingSharedWeights
from model_utils import *
from backend2 import sequence_beam_search

EOS_ID = 1


class Transformer(Model):
    def __init__(self, params):
        super(Transformer, self).__init__(name='transformer')

        self.params = params
        self.embedding_softmax_layer = EmbeddingSharedWeights(
            params.vocab_size, params.hidden_size)
        self.encoder_stack = EncoderStack(params)
        self.decoder_stack = DecoderStack(params)

    def call(self, inputs, train=True):
        assert isinstance(inputs, (list, tuple)) and len(inputs) == 2
        x = inputs[0]
        targets = inputs[1]
        attention_bias = get_padding_bias(x)

        encoder_outputs = self.encode(x, attention_bias, train=train)
        logits = self.decode(targets, encoder_outputs, attention_bias, train=train)
        return logits

    def compute_output_shape(self, input_shape):
        assert len(input_shape) == 2
        return input_shape[0] + (self.params.hidden_size,)

    def encode(self, inputs, attention_bias, train):
        embedded_inputs = self.embedding_softmax_layer(inputs)
        inputs_padding = get_padding(inputs)

        length = K.shape(embedded_inputs)[1]
        pos_encoding = get_position_encoding(length, self.params.hidden_size)
        encoder_inputs = embedded_inputs + pos_encoding

        if train:
            encoder_inputs = K.dropout(
                encoder_inputs, self.params.layer_postprocess_dropout)

        # TODO: add input padding
        return self.encoder_stack([encoder_inputs, attention_bias], train=train)

    def decode(self, targets, encoder_outputs, attention_bias, train):
        decoder_inputs = self.embedding_softmax_layer(targets)
        decoder_inputs_shape = K.shape(decoder_inputs)
        length = decoder_inputs_shape[1]
        decoder_inputs = K.slice(K2.pad(decoder_inputs, [[0, 0], [1, 0], [0, 0]]),
                                 (0, 0, 0),
                                 (decoder_inputs_shape[0],
                                  length,
                                  decoder_inputs_shape[2]))
        # decoder_inputs = K2.pad(decoder_inputs, [[0, 0], [1, 0], [0, 0]])[:, :-1, :]
        decoder_inputs += get_position_encoding(length, self.params.hidden_size)
        if train:
            decoder_inputs = K.dropout(
                decoder_inputs, self.params.layer_postprocess_dropout)
        decoder_self_attention_bias = get_decoder_self_attention_bias(length)
        outputs = self.decoder_stack(
            [decoder_inputs,
             encoder_outputs,
             decoder_self_attention_bias,
             attention_bias], train=train)
        logits = self.embedding_softmax_layer.linear(outputs)
        return logits

    @property
    def uses_learning_phase(self):
        return True

    def get_loss(self):

        smoothing = self.params.label_smoothing
        vocab_size = self.params.vocab_size

        def transformer_loss(y_true, y_pred, with_xent=False):
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

    def _get_symbols_to_logits_fn(self, max_decode_length):
        timing_signal = get_position_encoding(
            max_decode_length + 1, self.params.hidden_size)
        decoder_self_attention_bias = get_decoder_self_attention_bias(
            max_decode_length)

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
            # Set decoder input to the last generated IDs
            # ids_shape = K.shape(ids)
            decoder_input = K.slice(ids, (0, i), (-1, 1))

            # Preprocess decoder input by getting embeddings and adding timing signal.
            decoder_input = self.embedding_softmax_layer(decoder_input)
            decoder_input += K.slice(timing_signal, (i, 0), (1, -1))

            self_attention_bias = K.slice(decoder_self_attention_bias,
                                          (0, 0, i, 0),
                                          (-1, -1, 1, i + 1))

            decoder_outputs = self.decoder_stack(
                [decoder_input,
                 cache.get("encoder_outputs"),
                 self_attention_bias,
                 cache.get("encoder_decoder_attention_bias")], cache=cache, train=False)
            logits = self.embedding_softmax_layer.linear(decoder_outputs)
            logits = K.squeeze(logits, axis=1)
            return logits, cache
        return symbols_to_logits_fn

    def _get_predict_function(self):
        # inputs: int tensor with shape [batch_size, input_length].
        inputs = Input(shape=(None,), dtype="int32")
        attention_bias = get_padding_bias(inputs)

        # Run the inputs through the encoder layer to map the symbol
        # representations to continuous representations.
        encoder_outputs = self.encode(inputs, attention_bias, train=False)

        eo_shape = K.shape(encoder_outputs)
        batch_size = eo_shape[0]
        input_length = eo_shape[1]
        max_decode_length = input_length + self.params.extra_decode_length

        symbols_to_logits_fn = self._get_symbols_to_logits_fn(max_decode_length)

        # Create initial set of IDs that will be passed into symbols_to_logits_fn.
        initial_ids = K.zeros([batch_size], dtype='int32')

        # Create cache storing decoder attention values for each layer.
        cache = {
            "layer_%d" % layer: {
                "k": K.zeros([batch_size, 0, self.params.hidden_size]),
                "v": K.zeros([batch_size, 0, self.params.hidden_size]),
            } for layer in range(self.params.num_hidden_layers)}

        # Add encoder output and attention bias to the cache.
        cache["encoder_outputs"] = encoder_outputs
        cache["encoder_decoder_attention_bias"] = attention_bias

        # Use beam search to find the top beam_size sequences and scores.
        decoded_ids, scores = sequence_beam_search(
            symbols_to_logits_fn=symbols_to_logits_fn,
            initial_ids=initial_ids,
            initial_cache=cache,
            vocab_size=self.params.vocab_size,
            beam_size=self.params.beam_size,
            alpha=self.params.alpha,
            max_decode_length=max_decode_length,
            eos_id=EOS_ID)

        # Get the top sequence for each batch element
        top_decoded_ids = K.slice(decoded_ids, (0, 0, 1), (-1, 1, -1))
        top_decoded_ids = K.squeeze(top_decoded_ids, 1)

        # force 2 inputs to match fit inputs, the second input is always None when predict.
        predict_function = K.function([inputs, None],
                                      [top_decoded_ids], name='predict_function')
        return predict_function

    def _make_predict_function(self):
        if self.predict_function is None:
            self.predict_function = self._get_predict_function()


class PrePostProcessingWrapper(Wrapper):

    def __init__(self,
                 layer,
                 params,
                 **kwargs):
        super(PrePostProcessingWrapper, self).__init__(layer, **kwargs)

        self.layer = layer
        self.postprocess_dropout = params.layer_postprocess_dropout

        self.layer_norm = LayerNormalization(params.hidden_size)

    def compute_output_shape(self, input_shape):
        if isinstance(input_shape, list):
            return input_shape[0]
        else:
            return input_shape

    def call(self, inputs, train=True, **kwargs):
        if isinstance(inputs, (list, tuple)):
            x = inputs[0]
        else:
            x = inputs

        # Preprocessing: apply layer normalization
        y = self.layer_norm(x)

        # Get layer output
        if isinstance(inputs, (list, tuple)):
            layer_inputs = [y] + inputs[1:]
        else:
            layer_inputs = y
        y = self.layer(layer_inputs, train=train, **kwargs)

        # Postprocessing: apply dropout and residual connection
        if train:
            y = K.dropout(y, self.postprocess_dropout)
        return x + y


class EncoderStack(Layer):

    def __init__(self, params):
        super(EncoderStack, self).__init__()
        self.layers = []
        for _ in range(params.num_hidden_layers):
            # Create sublayers for each layer.
            self_attention_layer = SelfAttention(params.hidden_size,
                                             params.num_heads,
                                             params.attention_dropout)
            feed_forward_network = FeedFowardNetwork(params.hidden_size,
                                                     params.filter_size,
                                                     params.relu_dropout)
            self.layers.append([
                PrePostProcessingWrapper(self_attention_layer, params),
                PrePostProcessingWrapper(feed_forward_network, params)])

        # Create final layer normalization layer.
        self.output_normalization = LayerNormalization(params.hidden_size)

    def compute_output_shape(self, input_shape):
        return input_shape[0]

    # input: x [batch_size, seq_len, hidden_size], attention_bias
    def call(self, inputs, train=False):
        assert isinstance(inputs, (list, tuple)) and len(inputs) == 2
        y = inputs[0]
        attention_bias = inputs[1]
        for n, layer in enumerate(self.layers):
            # Run inputs through the sublayers.
            self_attention_layer = layer[0]
            feed_forward_network = layer[1]

            y = self_attention_layer([y, attention_bias], train=train)
            y = feed_forward_network(y, train=train)

        return self.output_normalization(y)

    # TODO: add get_config/from_config


class DecoderStack(Layer):

    def __init__(self, params):
        super(DecoderStack, self).__init__()
        self.layers = []
        for _ in range(params.num_hidden_layers):
            self_attention_layer = SelfAttention(params.hidden_size,
                                                 params.num_heads,
                                                 params.attention_dropout)
            enc_dec_attention_layer = Attention(params.hidden_size,
                                                params.num_heads,
                                                params.attention_dropout)
            feed_forward_network = FeedFowardNetwork(params.hidden_size,
                                                     params.filter_size,
                                                     params.relu_dropout)
            self.layers.append([
                PrePostProcessingWrapper(self_attention_layer, params),
                PrePostProcessingWrapper(enc_dec_attention_layer, params),
                PrePostProcessingWrapper(feed_forward_network, params)])

        # Create final layer normalization layer.
        self.output_normalization = LayerNormalization(params.hidden_size)

    def compute_output_shape(self, input_shape):
        return input_shape[0]

    def call(self, inputs, cache=None, train=False):
        assert isinstance(inputs, (list, tuple)) and len(inputs) == 4
        decoder_inputs = inputs[0]
        encoder_outputs = inputs[1]
        decoder_self_attention_bias = inputs[2]
        attention_bias = inputs[3]

        y = decoder_inputs
        for n, layer in enumerate(self.layers):
            # Run inputs through the sublayers.
            self_attention_layer = layer[0]
            enc_dec_attention_layer = layer[1]
            feed_forward_network = layer[2]

            layer_name = "layer_%d" % n
            layer_cache = cache[layer_name] if cache is not None else None

            y = self_attention_layer(
                [y, decoder_self_attention_bias],
                cache=layer_cache,
                train=train)
            y = enc_dec_attention_layer(
                [y, encoder_outputs, attention_bias], train=train)
            y = feed_forward_network(y, train=train)

        return self.output_normalization(y)

    # TODO: add get_config/from_config
