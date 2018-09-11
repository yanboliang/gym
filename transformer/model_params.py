
class TransformerBaseParams(object):
    initializer_gain = 1.0
    attention_dropout = 0.1
    layer_postprocess_dropout = 0.1
    relu_dropout = 0.1
    hidden_size = 512
    vocab_size = 100
    num_heads = 8
    num_hidden_layers = 6
    filter_size = 2048
    label_smoothing = 0.1
    extra_decode_length = 5
    beam_size = 4
    alpha = 0.6
