import numpy as np
import openvino.opset14 as ov_opset
from openvino import Type

from keras.src.backend.config import floatx
from keras.src.backend.openvino.core import OPENVINO_DTYPES
from keras.src.backend.openvino.core import OpenVINOKerasTensor
from keras.src.backend.openvino.core import convert_to_numpy
from keras.src.backend.openvino.core import get_ov_output
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
    seed_val = draw_seed(seed)
    if isinstance(seed_val, OpenVINOKerasTensor):
        seed_data = convert_to_numpy(seed_val)
    else:
        seed_data = seed_val.data
    rng = np.random.default_rng(seed_data)
    random_values = rng.uniform(minval, maxval, size=shape).astype(dtype)
    return OpenVINOKerasTensor(ov_opset.constant(random_values).output(0))


def categorical(logits, num_samples, dtype="int64", seed=None):
    dtype = dtype or "int64"
    ov_dtype = OPENVINO_DTYPES[dtype]
    logits = get_ov_output(logits)

    zero_const = ov_opset.constant(0, Type.i32).output(0)
    one_const = ov_opset.constant(1, Type.i32).output(0)
    neg_one_const = ov_opset.constant(-1, Type.i32).output(0)

    # Compute probabilities and cumulative sum
    probs = ov_opset.softmax(logits, axis=-1).output(0)
    cumsum_probs = ov_opset.cumsum(probs, neg_one_const).output(0)

    # Get shape and compute batch dimensions
    logits_shape = ov_opset.shape_of(logits, Type.i32).output(0)
    rank = ov_opset.shape_of(logits_shape, Type.i32).output(0)
    rank_scalar = ov_opset.squeeze(rank, zero_const).output(0)
    rank_minus_1 = ov_opset.subtract(rank_scalar, one_const).output(0)

    # Extract batch shape (all dimensions except last)
    batch_indices = ov_opset.range(
        zero_const, rank_minus_1, one_const, output_type=Type.i32
    ).output(0)
    batch_shape = ov_opset.gather(logits_shape, batch_indices, axis=0).output(0)

    # Create final shape [batch_dims..., num_samples]
    num_samples_const = ov_opset.constant([num_samples], Type.i32).output(0)
    final_shape = ov_opset.concat(
        [batch_shape, num_samples_const], axis=0
    ).output(0)

    seed_tensor = draw_seed(seed)
    if isinstance(seed_tensor, OpenVINOKerasTensor):
        seed1, seed2 = convert_to_numpy(seed_tensor)
    else:
        seed1, seed2 = seed_tensor.data

    probs_dtype = probs.get_element_type()
    zero_float = ov_opset.constant(0.0, probs_dtype).output(0)
    one_float = ov_opset.constant(1.0, probs_dtype).output(0)

    rand = ov_opset.random_uniform(
        final_shape, zero_float, one_float, probs_dtype, seed1, seed2
    ).output(0)

    rand_unsqueezed = ov_opset.unsqueeze(rand, neg_one_const).output(0)
    cumsum_unsqueezed = ov_opset.unsqueeze(cumsum_probs, one_const).output(0)

    # Count how many cumulative probabilities each random number exceeds
    greater = ov_opset.greater(rand_unsqueezed, cumsum_unsqueezed).output(0)
    samples = ov_opset.reduce_sum(
        ov_opset.convert(greater, Type.i32).output(0), neg_one_const
    ).output(0)

    result = ov_opset.convert(samples, ov_dtype).output(0)
    return OpenVINOKerasTensor(result)


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
