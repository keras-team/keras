import jax

from keras.src.backend.config import floatx
from keras.src.random.seed_generator import SeedGenerator
from keras.src.random.seed_generator import draw_seed
from keras.src.random.seed_generator import make_default_seed


def jax_draw_seed(seed):
    if isinstance(seed, jax.Array):
        return seed
    else:
        return draw_seed(seed)


def normal(shape, mean=0.0, stddev=1.0, dtype=None, seed=None):
    dtype = dtype or floatx()
    seed = jax_draw_seed(seed)
    sample = jax.random.normal(seed, shape=shape, dtype=dtype)
    return sample * stddev + mean


def uniform(shape, minval=0.0, maxval=1.0, dtype=None, seed=None):
    dtype = dtype or floatx()
    seed = jax_draw_seed(seed)
    return jax.random.uniform(
        seed, shape=shape, dtype=dtype, minval=minval, maxval=maxval
    )


def categorical(logits, num_samples, dtype="int32", seed=None):
    seed = jax_draw_seed(seed)
    output_shape = list(logits.shape)
    output_shape[1] = num_samples
    output_shape = tuple(output_shape)
    output = jax.random.categorical(
        seed, logits[..., None], shape=output_shape, axis=1
    )
    return output.astype(dtype)


def randint(shape, minval, maxval, dtype="int32", seed=None):
    seed = jax_draw_seed(seed)
    return jax.random.randint(
        seed, shape=shape, dtype=dtype, minval=minval, maxval=maxval
    )


def truncated_normal(shape, mean=0.0, stddev=1.0, dtype=None, seed=None):
    dtype = dtype or floatx()
    seed = jax_draw_seed(seed)
    sample = jax.random.truncated_normal(
        seed, shape=shape, lower=-2.0, upper=2.0, dtype=dtype
    )
    return sample * stddev + mean


def _get_concrete_noise_shape(inputs, noise_shape):
    if noise_shape is None:
        return inputs.shape

    concrete_inputs_shape = inputs.shape
    concrete_noise_shape = []
    for i, value in enumerate(noise_shape):
        concrete_noise_shape.append(
            concrete_inputs_shape[i] if value is None else value
        )
    return concrete_noise_shape


def dropout(inputs, rate, noise_shape=None, seed=None):
    seed = jax_draw_seed(seed)
    keep_prob = 1.0 - rate
    # The `noise_shape` may contain `None` so we need to convert it
    # into a concrete shape before passing it on to jax.
    noise_shape = _get_concrete_noise_shape(inputs, noise_shape)
    mask = jax.random.bernoulli(seed, p=keep_prob, shape=noise_shape)
    mask = jax.numpy.broadcast_to(mask, inputs.shape)
    return jax.lax.select(
        mask, inputs / keep_prob, jax.numpy.zeros_like(inputs)
    )


def shuffle(x, axis=0, seed=None):
    seed = jax_draw_seed(seed)
    return jax.random.permutation(seed, x, axis, independent=True)


def gamma(shape, alpha, dtype=None, seed=None):
    seed = jax_draw_seed(seed)
    dtype = dtype or floatx()
    return jax.random.gamma(seed, alpha, shape=shape, dtype=dtype)


def binomial(shape, counts, probabilities, dtype=None, seed=None):
    dtype = dtype or floatx()
    seed = jax_draw_seed(seed)
    # jax doesn't accept python lists as arguments
    counts = jax.numpy.array(counts)
    probabilities = jax.numpy.array(probabilities)
    sample = jax.random.binomial(
        key=seed, n=counts, p=probabilities, shape=shape, dtype=dtype
    )
    return sample


def beta(shape, alpha, beta, dtype=None, seed=None):
    dtype = dtype or floatx()
    seed = jax_draw_seed(seed)
    # jax doesn't accept python lists as arguments
    alpha = jax.numpy.array(alpha)
    beta = jax.numpy.array(beta)
    sample = jax.random.beta(
        key=seed, a=alpha, b=beta, shape=shape, dtype=dtype
    )
    return sample
