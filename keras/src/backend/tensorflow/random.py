import tensorflow as tf

from keras.src.backend.common import standardize_dtype
from keras.src.backend.config import floatx
from keras.src.random.seed_generator import SeedGenerator
from keras.src.random.seed_generator import draw_seed
from keras.src.random.seed_generator import make_default_seed


def normal(shape, mean=0.0, stddev=1.0, dtype=None, seed=None):
    dtype = dtype or floatx()
    seed = draw_seed(seed)
    return tf.random.stateless_normal(
        shape=shape, mean=mean, stddev=stddev, dtype=dtype, seed=seed
    )


def uniform(shape, minval=0.0, maxval=1.0, dtype=None, seed=None):
    dtype = dtype or floatx()
    seed = draw_seed(seed)
    return tf.random.stateless_uniform(
        shape=shape,
        minval=tf.cast(minval, dtype),
        maxval=tf.cast(maxval, dtype),
        dtype=dtype,
        seed=seed,
    )


def categorical(logits, num_samples, dtype="int64", seed=None):
    seed = draw_seed(seed)
    output = tf.random.stateless_categorical(logits, num_samples, seed=seed)
    return tf.cast(output, dtype)


def randint(shape, minval, maxval, dtype="int32", seed=None):
    intemediate_dtype = dtype
    if standardize_dtype(dtype) not in ["int32", "int64"]:
        intemediate_dtype = "int64"
    seed = draw_seed(seed)
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
    seed = draw_seed(seed)
    return tf.random.stateless_truncated_normal(
        shape=shape, mean=mean, stddev=stddev, dtype=dtype, seed=seed
    )


def _get_concrete_noise_shape(inputs, noise_shape):
    if noise_shape is None:
        return tf.shape(inputs)

    concrete_inputs_shape = tf.shape(inputs)
    concrete_noise_shape = []
    for i, value in enumerate(noise_shape):
        concrete_noise_shape.append(
            concrete_inputs_shape[i] if value is None else value
        )
    return concrete_noise_shape


def dropout(inputs, rate, noise_shape=None, seed=None):
    seed = draw_seed(seed)
    noise_shape = _get_concrete_noise_shape(inputs, noise_shape)
    return tf.nn.experimental.stateless_dropout(
        inputs,
        rate=rate,
        noise_shape=noise_shape,
        seed=seed,
    )


def shuffle(x, axis=0, seed=None):
    from keras.src.backend.tensorflow.numpy import swapaxes

    seed = draw_seed(seed)
    if axis == 0:
        return tf.random.experimental.stateless_shuffle(x, seed=seed)
    x = swapaxes(x, axis1=0, axis2=axis)
    x = tf.random.experimental.stateless_shuffle(x, seed=seed)
    x = swapaxes(x, axis1=0, axis2=axis)
    return x


def gamma(shape, alpha, dtype=None, seed=None):
    dtype = dtype or floatx()
    seed = draw_seed(seed)
    # TODO: `tf.random.stateless_gamma` doesn't support bfloat16
    intemediate_dtype = dtype
    if standardize_dtype(dtype) == "bfloat16":
        intemediate_dtype = "float32"
    return tf.cast(
        tf.random.stateless_gamma(
            shape,
            alpha=alpha,
            dtype=intemediate_dtype,
            seed=seed,
        ),
        dtype,
    )


def binomial(shape, counts, probabilities, dtype=None, seed=None):
    dtype = dtype or floatx()
    seed = draw_seed(seed)
    # TODO: `tf.random.stateless_binomial` doesn't support bfloat16
    intemediate_dtype = dtype
    if standardize_dtype(dtype) == "bfloat16":
        intemediate_dtype = "float32"
    return tf.cast(
        tf.random.stateless_binomial(
            shape=shape,
            seed=seed,
            counts=counts,
            probs=probabilities,
            output_dtype=intemediate_dtype,
        ),
        dtype,
    )


def beta(shape, alpha, beta, dtype=None, seed=None):
    dtype = dtype or floatx()
    # since tensorflow doesn't offer a beta distribution function
    # so we'll use the formula U(a,b) = (X(a) / (X(a) + Y(b)),
    # where U(a,b) is a beta-distributed random variable with
    # parameters a and b, and X(a) and Y(b) are gamma-distributed
    # random variables with parameters a and b respectively.

    # Additionally, we'll use two different seeds for our two
    # gamma random variables to prevent any unintended
    # dependencies and correlations between the generated values
    # due to the usage of same seed.
    seed_1 = draw_seed(seed)
    # The choice of 12 is totally arbitrary, as we're
    # incrementing the first drawn seed by a CONSTANT to
    # ensure deterministic results.
    seed_2 = seed_1 + 12

    # TODO: `tf.random.stateless_gamma` doesn't support bfloat16
    intemediate_dtype = dtype
    if standardize_dtype(dtype) == "bfloat16":
        intemediate_dtype = "float32"
    alpha = tf.convert_to_tensor(alpha, dtype=intemediate_dtype)
    beta = tf.convert_to_tensor(beta, dtype=intemediate_dtype)

    # tensorflow's tf.random.stateless_gamma has a bit of unconventional
    # implementation of the stateless_gamma function where it checks the
    # broadcastability of alpha's shape with ONLY the RIGHTMOST dimension of
    # the specified output shape instead of considering the whole.
    # Consequently, it then results in errors for perfectly broadcastable shapes
    # such as for output shape of (2, 3) and alpha shape of (1, 3)
    # So to resolve this, we explicitly broadcast alpha and beta to shape before
    # passing them to the stateless_gamma function.
    alpha = tf.broadcast_to(alpha, shape)
    beta = tf.broadcast_to(beta, shape)

    gamma_a = tf.cast(
        tf.random.stateless_gamma(
            shape=shape, seed=seed_1, alpha=alpha, dtype=intemediate_dtype
        ),
        dtype,
    )
    gamma_b = tf.cast(
        tf.random.stateless_gamma(
            shape=shape, seed=seed_2, alpha=beta, dtype=intemediate_dtype
        ),
        dtype,
    )
    sample = gamma_a / (gamma_a + gamma_b)
    return sample
