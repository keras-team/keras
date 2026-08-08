import contextlib

import paddle

from keras.src.backend.config import floatx
from keras.src.backend.paddle.core import convert_to_numpy
from keras.src.backend.paddle.core import convert_to_tensor
from keras.src.backend.paddle.core import needs_reduced_precision_upcast
from keras.src.backend.paddle.core import to_paddle_dtype
from keras.src.random.seed_generator import SeedGenerator
from keras.src.random.seed_generator import draw_seed
from keras.src.random.seed_generator import make_default_seed


def _draw_seed_value(seed):
    """Draw a distinct integer seed for a single random op call.

    `draw_seed` returns a state tensor `[seed, counter]` where only the
    counter increments between calls; combine both elements so that each
    call produces a distinct seed.
    """
    seed_state = draw_seed(seed)
    if isinstance(seed_state, paddle.Tensor):
        seed_state = convert_to_numpy(seed_state)
    first_seed, second_seed = int(seed_state[0]), int(seed_state[1])
    return int(first_seed + second_seed) & 0xFFFFFFFF


@contextlib.contextmanager
def _rng_scope(seed):
    """Seed paddle's RNG for a single draw, then restore its prior state.

    Paddle has no per-op generator argument, so we seed the global
    generators and restore their previous states afterwards to avoid
    permanently clobbering paddle's global RNG.
    """
    prev_rng_state = paddle.get_rng_state()
    paddle.seed(_draw_seed_value(seed))
    try:
        yield
    finally:
        paddle.set_rng_state(prev_rng_state)


def normal(shape, mean=0.0, stddev=1.0, dtype=None, seed=None):
    dtype = dtype or floatx()
    paddle_dtype = to_paddle_dtype(dtype)
    with _rng_scope(seed):
        sample = paddle.normal(mean=mean, std=stddev, shape=shape)
    return sample.cast(paddle_dtype)


def uniform(shape, minval=0.0, maxval=1.0, dtype=None, seed=None):
    dtype = dtype or floatx()
    paddle_dtype = to_paddle_dtype(dtype)
    with _rng_scope(seed):
        sample = paddle.uniform(shape=shape, min=minval, max=maxval)
    return sample.cast(paddle_dtype)


def categorical(logits, num_samples, dtype="int64", seed=None):
    logits = convert_to_tensor(logits)
    probs = paddle.nn.functional.softmax(logits, axis=-1)
    with _rng_scope(seed):
        sample = paddle.multinomial(
            probs, num_samples=num_samples, replacement=True
        )
    return sample.cast(to_paddle_dtype(dtype))


def randint(shape, minval, maxval, dtype="int32", seed=None):
    paddle_dtype = to_paddle_dtype(dtype)
    # `paddle.randint` only implements int32/int64, so draw as int64 and
    # narrow afterwards.
    with _rng_scope(seed):
        sample = paddle.randint(minval, maxval, shape=shape, dtype="int64")
    return sample.cast(paddle_dtype)


def truncated_normal(shape, mean=0.0, stddev=1.0, dtype=None, seed=None):
    dtype = dtype or floatx()
    paddle_dtype = to_paddle_dtype(dtype)
    # Take a larger standard normal dist, discard values outside 2 * stddev
    # Offset by mean and stddev.
    # Always draw in float32: `take_along_axis` has no CPU kernel for
    # float16 / bfloat16.
    x = normal(
        tuple(shape) + (4,), mean=0, stddev=1, dtype="float32", seed=seed
    )
    valid = (x > -2) & (x < 2)
    indexes = paddle.argmax(valid.cast("int32"), axis=-1, keepdim=True)
    trunc_x = paddle.take_along_axis(x, indexes, axis=-1).squeeze(-1)
    trunc_x = trunc_x * stddev + mean
    return trunc_x.cast(paddle_dtype)


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
    inputs = convert_to_tensor(inputs)
    if rate == 1.0:
        return paddle.zeros_like(inputs)
    if rate == 0.0:
        return inputs
    keep_prob = 1.0 - rate
    noise_shape = _get_concrete_noise_shape(inputs, noise_shape)
    with _rng_scope(seed):
        keep_mask = paddle.bernoulli(
            paddle.full(noise_shape, keep_prob, dtype="float32")
        ).cast(paddle.bool)
    keep_mask = paddle.broadcast_to(keep_mask, inputs.shape)
    # Neither `where` nor `divide` has a CPU kernel for float16 / bfloat16,
    # so run the scaling in float32 and cast the result back.
    orig_dtype = inputs.dtype
    upcast = needs_reduced_precision_upcast(inputs)
    if upcast:
        inputs = inputs.cast("float32")
    outputs = paddle.where(
        keep_mask,
        inputs / keep_prob,
        paddle.zeros_like(inputs),
    )
    if upcast:
        outputs = outputs.cast(orig_dtype)
    return outputs


def shuffle(x, axis=0, seed=None):
    x = convert_to_tensor(x)
    with _rng_scope(seed):
        indices = paddle.randperm(x.shape[axis])
    return paddle.index_select(x, indices, axis=axis)


def gamma(shape, alpha, dtype=None, seed=None):
    dtype = dtype or floatx()
    paddle_dtype = to_paddle_dtype(dtype)
    # `paddle.standard_gamma` draws with rate 1, which is what Keras wants.
    alpha = paddle.broadcast_to(convert_to_tensor(alpha).cast("float32"), shape)
    with _rng_scope(seed):
        sample = paddle.standard_gamma(alpha)
    return sample.cast(paddle_dtype)


def binomial(shape, counts, probabilities, dtype=None, seed=None):
    dtype = dtype or floatx()
    paddle_dtype = to_paddle_dtype(dtype)
    counts = paddle.broadcast_to(
        convert_to_tensor(counts).cast("float32"), shape
    )
    probabilities = paddle.broadcast_to(
        convert_to_tensor(probabilities).cast("float32"), shape
    )
    with _rng_scope(seed):
        sample = paddle.binomial(counts, probabilities)
    return sample.cast(paddle_dtype)


def beta(shape, alpha, beta, dtype=None, seed=None):
    dtype = dtype or floatx()
    paddle_dtype = to_paddle_dtype(dtype)
    alpha = paddle.broadcast_to(convert_to_tensor(alpha).cast("float32"), shape)
    beta = paddle.broadcast_to(convert_to_tensor(beta).cast("float32"), shape)
    # Build Beta(a, b) from two Gamma draws:
    #   X ~ Gamma(a, 1), Y ~ Gamma(b, 1)  =>  X / (X + Y) ~ Beta(a, b)
    # `paddle.distribution.Beta.sample` does not honour the seeded global
    # RNG, so sample the gammas directly instead.
    with _rng_scope(seed):
        x = paddle.standard_gamma(alpha)
        y = paddle.standard_gamma(beta)
    sample = x / (x + y)
    return sample.cast(paddle_dtype)


def seed_generator():
    return SeedGenerator(seed=make_default_seed())
