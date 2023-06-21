import tensorflow as tf

from keras_core.backend.common import standardize_dtype
from keras_core.backend.config import floatx
from keras_core.random.seed_generator import SeedGenerator
from keras_core.random.seed_generator import draw_seed
from keras_core.random.seed_generator import make_default_seed


def tf_draw_seed(seed):
    # TF ops only accept int32/64 seeds but our base seed is uint32.
    return tf.cast(draw_seed(seed), dtype="int32")


def normal(shape, mean=0.0, stddev=1.0, dtype=None, seed=None):
    dtype = dtype or floatx()
    seed = tf_draw_seed(seed)
    return tf.random.stateless_normal(
        shape=shape, mean=mean, stddev=stddev, dtype=dtype, seed=seed
    )


def uniform(shape, minval=0.0, maxval=1.0, dtype=None, seed=None):
    dtype = dtype or floatx()
    seed = tf_draw_seed(seed)
    return tf.random.stateless_uniform(
        shape=shape,
        minval=minval,
        maxval=maxval,
        dtype=dtype,
        seed=seed,
    )


def categorical(logits, num_samples, dtype="int64", seed=None):
    seed = tf_draw_seed(seed)
    output = tf.random.stateless_categorical(logits, num_samples, seed=seed)
    return tf.cast(output, dtype)


def randint(shape, minval, maxval, dtype="int32", seed=None):
    intemediate_dtype = dtype
    if standardize_dtype(dtype) not in ["int32", "int64"]:
        intemediate_dtype = "int64"
    seed = tf_draw_seed(seed)
    output = tf.random.stateless_uniform(
        shape=shape,
        minval=minval,
        maxval=maxval,
        dtype=intemediate_dtype,
        seed=seed,
    )
    return tf.cast(output, dtype)


def truncated_normal(shape, mean=0.0, stddev=1.0, dtype=None, seed=None):
    dtype = dtype or floatx()
    seed = tf_draw_seed(seed)
    return tf.random.stateless_truncated_normal(
        shape=shape, mean=mean, stddev=stddev, dtype=dtype, seed=seed
    )


def dropout(inputs, rate, noise_shape=None, seed=None):
    seed = tf_draw_seed(seed)
    return tf.nn.experimental.stateless_dropout(
        inputs,
        rate=rate,
        noise_shape=noise_shape,
        seed=seed,
    )
