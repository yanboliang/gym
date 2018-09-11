import tensorflow as tf

from beam_search import sequence_beam_search as tf_sequence_beam_search


def matmul(a, b):
    return tf.matmul(a, b)


def matrix_band_part(input, num_lower, num_upper):
    return tf.matrix_band_part(input, num_lower, num_upper)


def pad(tensor, paddings, constant_values=0):
    return tf.pad(tensor, paddings, mode='CONSTANT', constant_values=constant_values)


def range(start, limit, delta=1, dtype=None):
    return tf.range(start, limit, delta, dtype)


def sequence_beam_search(
        symbols_to_logits_fn, initial_ids, initial_cache, vocab_size, beam_size,
        alpha, max_decode_length, eos_id):
    """Search for sequence of subtoken ids with the largest probability.

    Args:
      symbols_to_logits_fn: A function that takes in ids, index, and cache as
        arguments. The passed in arguments will have shape:
          ids -> [batch_size * beam_size, index]
          index -> [] (scalar)
          cache -> nested dictionary of tensors [batch_size * beam_size, ...]
        The function must return logits and new cache.
          logits -> [batch * beam_size, vocab_size]
          new cache -> same shape/structure as inputted cache
      initial_ids: Starting ids for each batch item.
        int32 tensor with shape [batch_size]
      initial_cache: dict containing starting decoder variables information
      vocab_size: int size of tokens
      beam_size: int number of beams
      alpha: float defining the strength of length normalization
      max_decode_length: maximum length to decoded sequence
      eos_id: int id of eos token, used to determine when a sequence has finished

    Returns:
      Top decoded sequences [batch_size, beam_size, max_decode_length]
      sequence scores [batch_size, beam_size]
    """
    return tf_sequence_beam_search(symbols_to_logits_fn,
                                   initial_ids,
                                   initial_cache,
                                   vocab_size,
                                   beam_size,
                                   alpha,
                                   max_decode_length,
                                   eos_id)
