import tensorflow as tf


def segment_sum(data, segment_ids, num_segments=None, sorted=False):
    if sorted:
        return tf.math.segment_sum(data, segment_ids)
    else:
        return tf.math.unsorted_segment_sum(data, segment_ids, num_segments)


def top_k(x, k, sorted=False):
    return tf.math.top_k(x, k, sorted=sorted)
