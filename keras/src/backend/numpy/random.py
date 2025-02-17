import numpy as np

from keras.src.backend.config import floatx
from keras.src.backend.numpy.nn import softmax
from keras.src.random.seed_generator import SeedGenerator
from keras.src.random.seed_generator import draw_seed
from keras.src.random.seed_generator import make_default_seed


def normal(shape, mean=0.0, stddev=1.0, dtype=None, seed=None):
    dtype = dtype or floatx()
    seed = draw_seed(seed)
    rng = np.random.default_rng(seed)
    return rng.normal(size=shape, loc=mean, scale=stddev).astype(dtype)


def uniform(shape, minval=0.0, maxval=1.0, dtype=None, seed=None):
    dtype = dtype or floatx()
    seed = draw_seed(seed)
    rng = np.random.default_rng(seed)
    return rng.uniform(size=shape, low=minval, high=maxval).astype(dtype)


def categorical(logits, num_samples, dtype="int64", seed=None):
    seed = draw_seed(seed)
    rng = np.random.default_rng(seed)
    output = []
    for logits_instance in logits:
        probabilities = softmax(logits_instance)
        classes = np.arange(logits_instance.shape[-1])
        samples = rng.choice(classes, size=num_samples, p=probabilities)
        output.append(samples)
    return np.array(output).astype(dtype)


def randint(shape, minval, maxval, dtype="int32", seed=None):
    seed = draw_seed(seed)
    rng = np.random.default_rng(seed)
    output = rng.integers(low=minval, high=maxval, size=shape, dtype=dtype)
    return output


def truncated_normal(shape, mean=0.0, stddev=1.0, dtype=None, seed=None):
    dtype = dtype or floatx()
    seed = draw_seed(seed)
    rng = np.random.default_rng(seed)

    lower_bound = mean - 2 * stddev
    upper_bound = mean + 2 * stddev

    flat_shape = np.prod(shape)
    random_numbers = np.empty(0)

    # loop until we have enough valid numbers to fill our desired shape
    while random_numbers.shape[0] < flat_shape:
        # Generate a batch of random numbers from a normal distribution
        batch = rng.normal(loc=mean, scale=stddev, size=flat_shape)

        # Filter the numbers to keep only those within the specified bounds
        valid = batch[(batch >= lower_bound) & (batch <= upper_bound)]

        # Append the valid numbers to the result array
        random_numbers = np.append(random_numbers, valid)

    # Truncate the result array to the desired size and reshape it
    return random_numbers[:flat_shape].astype(dtype).reshape(shape)


def dropout(inputs, rate, noise_shape=None, seed=None):
    dtype = inputs.dtype
    seed = draw_seed(seed)

    keep_prob = 1.0 - rate

    # If noise_shape is not provided, use the shape of inputs
    if noise_shape is None:
        noise_shape = inputs.shape
    else:
        # If noise_shape is provided, replace None with corresponding
        # input shape
        noise_shape = [
            n if n is not None else inputs.shape[i]
            for i, n in enumerate(noise_shape)
        ]

    rng = np.random.default_rng(seed)
    mask = rng.uniform(size=noise_shape) < keep_prob
    mask = np.broadcast_to(mask, inputs.shape)
    return np.where(
        mask, (inputs / keep_prob).astype(dtype), np.zeros_like(inputs)
    )


def shuffle(x, axis=0, seed=None):
    seed = draw_seed(seed)
    rng = np.random.default_rng(seed)
    return rng.permuted(x, axis=axis)


def gamma(shape, alpha, dtype=None, seed=None):
    dtype = dtype or floatx()
    seed = draw_seed(seed)
    rng = np.random.default_rng(seed)
    return rng.gamma(alpha, scale=1.0, size=shape).astype(dtype)


def binomial(shape, counts, probabilities, dtype=None, seed=None):
    dtype = dtype or floatx()
    seed = draw_seed(seed)
    rng = np.random.default_rng(seed)
    sample = rng.binomial(n=counts, p=probabilities, size=shape).astype(dtype)
    return sample


def beta(shape, alpha, beta, dtype=None, seed=None):
    dtype = dtype or floatx()
    seed = draw_seed(seed)
    rng = np.random.default_rng(seed)
    sample = rng.beta(a=alpha, b=beta, size=shape).astype(dtype)
    return sample
