from keras.src import backend
from keras.src import tree


def make_batch(values):
    from keras.src import ops

    if not values:
        raise ValueError("Cannot batch 0 values. Please file a bug.")

    with backend.device_scope("cpu"):
        return tree.map_structure(lambda *xs: ops.stack(xs), *values)


def make_string_batch(values):
    from keras.src import ops

    if not values:
        raise ValueError("Cannot batch 0 values. Please file a bug.")

    def batch_fn(*xs):
        if isinstance(xs[0], str):
            if backend.backend() == "tensorflow":
                import tensorflow as tf

                xs = [tf.convert_to_tensor(x, dtype=tf.string) for x in xs]
                xs = tf.stack(xs)
            return xs
        else:
            return ops.stack(xs)

    with backend.device_scope("cpu"):
        return tree.map_structure(batch_fn, *values)
