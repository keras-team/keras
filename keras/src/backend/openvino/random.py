import numpy as np
import openvino.runtime.opset14 as ov_opset
from openvino import Type

from keras.src.backend.config import floatx
from keras.src.backend.openvino.core import OPENVINO_DTYPES
from keras.src.backend.openvino.core import OpenVINOKerasTensor
from keras.src.backend.openvino.core import convert_to_numpy
from keras.src.random.seed_generator import SeedGenerator
from keras.src.random.seed_generator import draw_seed
from keras.src.random.seed_generator import make_default_seed


def normal(shape, mean=0.0, stddev=1.0, dtype=None, seed=None):
    dtype = dtype or floatx()
    seed = draw_seed(seed)
    rng = np.random.default_rng(seed.data)
    normal_const = rng.normal(size=shape, loc=mean, scale=stddev).astype(dtype)
    return OpenVINOKerasTensor(ov_opset.constant(normal_const).output(0))


def uniform(shape, minval=0.0, maxval=1.0, dtype=None, seed=None):
    dtype = dtype or floatx()
    ov_type = OPENVINO_DTYPES[dtype]
    seed = draw_seed(seed)
    if isinstance(seed, OpenVINOKerasTensor):
        seed1, seed2 = convert_to_numpy(seed)
    else:
        seed1, seed2 = draw_seed(seed).data
    minval_const = ov_opset.constant(minval, dtype=dtype)
    maxval_const = ov_opset.constant(maxval, dtype=dtype)
    if isinstance(shape, tuple):
        shape = list(shape)
    output_shape_const = ov_opset.constant(shape, dtype=Type.i32)
    random_uniform = ov_opset.random_uniform(
        output_shape_const, minval_const, maxval_const, ov_type, seed1, seed2
    )
    return OpenVINOKerasTensor(random_uniform.output(0))


def categorical(logits, num_samples, dtype="int64", seed=None):
    raise NotImplementedError(
        "`categorical` is not supported with openvino backend"
    )


def randint(shape, minval, maxval, dtype="int32", seed=None):
    raise NotImplementedError(
        "`randint` is not supported with openvino backend"
    )


def truncated_normal(shape, mean=0.0, stddev=1.0, dtype=None, seed=None):
    dtype = dtype or floatx()
    seed = draw_seed(seed)
    rng = np.random.default_rng(seed.data)

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
    np_array_res = random_numbers[:flat_shape].astype(dtype).reshape(shape)
    return OpenVINOKerasTensor(ov_opset.constant(np_array_res).output(0))


def dropout(inputs, rate, noise_shape=None, seed=None):
    raise NotImplementedError(
        "`dropout` is not supported with openvino backend"
    )


def shuffle(x, axis=0, seed=None):
    raise NotImplementedError(
        "`shuffle` is not supported with openvino backend"
    )


def gamma(shape, alpha, dtype=None, seed=None):
    raise NotImplementedError("`gamma` is not supported with openvino backend")


def binomial(shape, counts, probabilities, dtype=None, seed=None):
    raise NotImplementedError(
        "`binomial` is not supported with openvino backend"
    )


def beta(shape, alpha, beta, dtype=None, seed=None):
    raise NotImplementedError("`beta` is not supported with openvino backend")
