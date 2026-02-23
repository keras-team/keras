import numpy as np
import openvino.opset15 as ov_opset
from openvino import Type

from keras.src.backend.config import floatx
from keras.src.backend.openvino import numpy as ov_numpy
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
    dtype = dtype or "int32"
    ov_dtype = OPENVINO_DTYPES[dtype]
    seed_val = draw_seed(seed)
    if isinstance(seed_val, OpenVINOKerasTensor):
        seed1, seed2 = convert_to_numpy(seed_val)
    else:
        seed1, seed2 = seed_val.data
    if ov_dtype in (Type.i64, Type.u64, Type.u32):
        gen_dtype = Type.i64
    else:
        gen_dtype = Type.i32
    if isinstance(shape, (list, tuple)):
        shape = ov_opset.constant(list(shape), Type.i32).output(0)
    elif isinstance(shape, OpenVINOKerasTensor):
        shape = shape.output
    elif isinstance(shape, int):
        shape = ov_opset.constant([shape], Type.i32).output(0)
    else:
        shape = get_ov_output(shape, Type.i32)
    minval = get_ov_output(minval, gen_dtype)
    maxval = get_ov_output(maxval, gen_dtype)
    if minval.get_element_type() != gen_dtype:
        minval = ov_opset.convert(minval, gen_dtype).output(0)
    if maxval.get_element_type() != gen_dtype:
        maxval = ov_opset.convert(maxval, gen_dtype).output(0)
    rand = ov_opset.random_uniform(
        shape, minval, maxval, gen_dtype, seed1, seed2
    ).output(0)
    if ov_dtype != gen_dtype:
        result = ov_opset.convert(rand, ov_dtype).output(0)
    else:
        result = rand
    return OpenVINOKerasTensor(result)


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
    seed_tensor = draw_seed(seed)
    if isinstance(seed_tensor, OpenVINOKerasTensor):
        seed1, seed2 = convert_to_numpy(seed_tensor)
    else:
        seed1, seed2 = seed_tensor.data
    x_ov = get_ov_output(x)
    x_shape = x_ov.get_partial_shape()
    rank = x_shape.rank.get_length()
    if axis < 0:
        axis += rank
    shape_tensor = ov_opset.shape_of(x_ov, Type.i32).output(0)
    dim_size = ov_opset.gather(
        shape_tensor,
        ov_opset.constant([axis], Type.i32).output(0),
        ov_opset.constant(0, Type.i32).output(0),
    ).output(0)
    min_val = ov_opset.constant(0.0, Type.f32).output(0)
    max_val = ov_opset.constant(1.0, Type.f32).output(0)
    rand_shape = ov_opset.reshape(
        dim_size, ov_opset.constant([1], Type.i32).output(0), False
    ).output(0)
    rand_values = ov_opset.random_uniform(
        rand_shape, min_val, max_val, Type.f32, seed1, seed2
    ).output(0)
    indices = ov_numpy.argsort(OpenVINOKerasTensor(rand_values), axis=0)
    return ov_numpy.take(x, indices, axis=axis)


def _const(val, dtype):
    if dtype == Type.bf16:
        return ov_opset.convert(
            ov_opset.constant(val, Type.f32), Type.bf16
        ).output(0)
    return ov_opset.constant(val, dtype).output(0)


def _random_normal(shape, dtype, seed1, seed2):
    zero = _const(0.0, dtype)
    one = _const(1.0, dtype)
    two_pi = _const(2 * np.pi, dtype)
    minus_two = _const(-2.0, dtype)
    epsilon = _const(1e-7, dtype)
    u1 = ov_opset.random_uniform(shape, zero, one, dtype, seed1, seed2).output(
        0
    )
    u2 = ov_opset.random_uniform(
        shape, zero, one, dtype, seed1 + 123, seed2
    ).output(0)
    u1 = ov_opset.add(u1, epsilon).output(0)
    mag = ov_opset.sqrt(ov_opset.multiply(minus_two, ov_opset.log(u1))).output(
        0
    )
    angle = ov_opset.multiply(two_pi, u2).output(0)
    z0 = ov_opset.multiply(mag, ov_opset.cos(angle)).output(0)
    return z0


def gamma(shape, alpha, dtype=None, seed=None):
    dtype = dtype or floatx()
    ov_dtype = OPENVINO_DTYPES[dtype]
    seed_val = draw_seed(seed)
    if isinstance(seed_val, OpenVINOKerasTensor):
        seed1, seed2 = convert_to_numpy(seed_val)
    else:
        seed1, seed2 = seed_val.data
    seed1 = int(seed1)
    seed2 = int(seed2)
    if isinstance(shape, (list, tuple)):
        shape = ov_opset.constant(list(shape), Type.i32).output(0)
    elif isinstance(shape, OpenVINOKerasTensor):
        shape = shape.output
    else:
        shape = get_ov_output(shape, Type.i32)
    alpha = get_ov_output(alpha, ov_dtype)
    one = _const(1.0, ov_dtype)
    one_third = _const(1.0 / 3.0, ov_dtype)
    zero = _const(0.0, ov_dtype)
    is_small_alpha = ov_opset.less(alpha, one).output(0)
    alpha_boosted = ov_opset.select(
        is_small_alpha, ov_opset.add(alpha, one), alpha
    ).output(0)
    d = ov_opset.subtract(alpha_boosted, one_third).output(0)
    c = ov_opset.divide(
        one,
        ov_opset.sqrt(ov_opset.multiply(_const(9.0, ov_dtype), d)),
    ).output(0)
    samples = ov_opset.broadcast(zero, shape).output(0)
    mask = ov_opset.broadcast(
        ov_opset.constant(False, Type.boolean), shape
    ).output(0)
    num_iters = 10
    for i in range(num_iters):
        iter_seed = seed1 + i * 1000
        x = _random_normal(shape, ov_dtype, iter_seed, seed2)
        cx = ov_opset.multiply(c, x).output(0)
        v_base = ov_opset.add(one, cx).output(0)
        v = ov_opset.power(v_base, _const(3.0, ov_dtype)).output(0)
        v_pos = ov_opset.greater(v, zero).output(0)
        u = ov_opset.random_uniform(
            shape, zero, one, ov_dtype, iter_seed + 500, seed2
        ).output(0)
        x2 = ov_opset.multiply(x, x).output(0)
        x4 = ov_opset.multiply(x2, x2).output(0)
        c1_val = ov_opset.subtract(
            one, ov_opset.multiply(_const(0.0331, ov_dtype), x4)
        ).output(0)
        accept1 = ov_opset.less(u, c1_val).output(0)
        v_safe = ov_opset.select(v_pos, v, one).output(0)
        log_u = ov_opset.log(u).output(0)
        log_v = ov_opset.log(v_safe).output(0)
        term2 = ov_opset.multiply(
            d, ov_opset.add(ov_opset.subtract(one, v), log_v)
        ).output(0)
        rhs = ov_opset.add(
            ov_opset.multiply(_const(0.5, ov_dtype), x2), term2
        ).output(0)
        accept2 = ov_opset.less(log_u, rhs).output(0)
        accepted = ov_opset.logical_or(accept1, accept2).output(0)
        accepted = ov_opset.logical_and(accepted, v_pos).output(0)
        dv = ov_opset.multiply(d, v).output(0)
        update_mask = ov_opset.logical_and(
            ov_opset.logical_not(mask), accepted
        ).output(0)
        samples = ov_opset.select(update_mask, dv, samples).output(0)
        mask = ov_opset.logical_or(mask, accepted).output(0)
    u_final = ov_opset.random_uniform(
        shape, zero, one, ov_dtype, seed1 + 9999, seed2
    ).output(0)
    pow_exp = ov_opset.divide(one, alpha).output(0)
    u_pow = ov_opset.power(u_final, pow_exp).output(0)
    adjusted_samples = ov_opset.multiply(samples, u_pow).output(0)
    final_samples = ov_opset.select(
        is_small_alpha, adjusted_samples, samples
    ).output(0)
    return OpenVINOKerasTensor(final_samples)


def binomial(shape, counts, probabilities, dtype=None, seed=None):
    raise NotImplementedError(
        "`binomial` is not supported with openvino backend"
    )


def beta(shape, alpha, beta, dtype=None, seed=None):
    raise NotImplementedError("`beta` is not supported with openvino backend")
