import mlx.core as mx

from keras.src.backend.config import floatx
from keras.src.backend.mlx.core import convert_to_tensor
from keras.src.backend.mlx.core import to_mlx_dtype
from keras.src.random.seed_generator import draw_seed


def mlx_draw_seed(seed):
    if isinstance(seed, mx.array):
        return seed
    else:
        return draw_seed(seed)


def normal(shape, mean=0.0, stddev=1.0, dtype=None, seed=None):
    dtype = dtype or floatx()
    dtype = to_mlx_dtype(dtype)
    seed = mlx_draw_seed(seed)
    return mx.random.normal(
        shape=shape, loc=mean, scale=stddev, dtype=dtype, key=seed
    )


def uniform(shape, minval=0.0, maxval=1.0, dtype=None, seed=None):
    dtype = dtype or floatx()
    dtype = to_mlx_dtype(dtype)
    seed = mlx_draw_seed(seed)
    return mx.random.uniform(
        low=minval, high=maxval, shape=shape, dtype=dtype, key=seed
    )


def categorical(logits, num_samples, dtype="int32", seed=None):
    logits = convert_to_tensor(logits)
    seed = mlx_draw_seed(seed)
    output = mx.random.categorical(logits, num_samples=num_samples, key=seed)
    return output.astype(to_mlx_dtype(dtype))


def randint(shape, minval, maxval, dtype="int32", seed=None):
    seed = mlx_draw_seed(seed)
    dtype = to_mlx_dtype(dtype)

    return mx.random.randint(
        low=minval, high=maxval, shape=shape, dtype=dtype, key=seed
    )


def truncated_normal(shape, mean=0.0, stddev=1.0, dtype=None, seed=None):
    dtype = dtype or floatx()
    dtype = to_mlx_dtype(dtype)
    seed = mlx_draw_seed(seed)
    sample = mx.random.truncated_normal(
        lower=-2.0, upper=2.0, shape=shape, dtype=dtype, key=seed
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
    seed = mlx_draw_seed(seed)
    keep_prob = 1.0 - rate
    # The `noise_shape` may contain `None` so we need to convert it
    # into a concrete shape before passing it on to jax.
    noise_shape = _get_concrete_noise_shape(inputs, noise_shape)
    mask = mx.random.bernoulli(p=keep_prob, shape=noise_shape, key=seed)
    mask = mx.broadcast_to(mask, inputs.shape)

    return mask * (inputs / keep_prob)


def shuffle(x, axis=0, seed=None):
    seed = mlx_draw_seed(seed)
    x = convert_to_tensor(x)
    return mx.random.permutation(x, axis=axis, key=seed)


def gamma(shape, alpha, dtype=None, seed=None):
    # Ref: jax.random.gamma
    # Ref: Marsaglia and Tsang method for generating gamma variables
    # Algorithm description can be found here:
    # https://en.wikipedia.org/wiki/Gamma_distribution#Random_variate_generation
    if isinstance(shape, int):
        shape = (shape,)

    dtype = to_mlx_dtype(dtype)

    if alpha <= 0:
        raise ValueError(
            "Invalid for argument `alpha`. Alpha must "
            f"be > 0, received alpha={alpha}"
        )

    key = mlx_draw_seed(seed)
    # if alpha < 1, apply Gamma(alpha) = Gamma(alpha+1) * U^(1/alpha)
    # where U is a random uniform variable
    if alpha < 1.0:
        gamma_alpha_plus_1 = gamma(shape, alpha + 1, dtype, seed=seed)

        key, ukey = mx.random.split(key)
        u = mx.random.uniform(key=ukey, shape=shape)

        return (gamma_alpha_plus_1 * (u ** (1.0 / alpha))).astype(dtype)

    # alpha >= 1 vectorized
    # constants
    d = alpha - 1.0 / 3.0
    c = 1.0 / mx.sqrt(9.0 * d)

    done = mx.zeros(shape=shape, dtype=mx.bool_)  # track acceptance
    results = mx.zeros(shape=shape, dtype=dtype)

    all_done = False
    while not all_done:
        key, key_x, key_u = mx.random.split(key, 3)

        x = mx.random.normal(key=key_x, shape=shape)
        u = mx.random.uniform(key=key_u, shape=shape)

        v_ = 1.0 + c * x

        not_done_mask = mx.logical_not(done)

        # get mask of v_ <= 0 to reject bad values in log(v_ ** 3)
        positive_mask = (v_ > 0.0) & not_done_mask
        v = v_ * v_ * v_

        # log(u) < 0.5*x^2 + d*(1 - v + log(v))
        log_u = mx.log(u)
        rhs = 0.5 * x * x + d * (1.0 - v + mx.log(v))

        accept_mask = positive_mask & (log_u < rhs)

        # store accepted d*v
        new_samples = d * v
        results = mx.where(accept_mask, new_samples, results)

        done = mx.logical_or(done, accept_mask)
        all_done = mx.all(done)

    return results.astype(dtype)


def beta(shape, alpha, beta, dtype=None, seed=None):
    # beta distribution using Gamma(alpha) / (Gamma(alpha) + Gamma(beta))
    dtype = to_mlx_dtype(dtype)

    if isinstance(shape, int):
        shape = (shape,)

    alpha_arr = mx.array(alpha, dtype=mx.float32)
    beta_arr = mx.array(beta, dtype=mx.float32)

    if mx.any(alpha_arr <= 0.0):
        raise ValueError(
            "Invalid value for argument `alpha`. All alpha "
            f"values must be > 0, received alpha={alpha}"
        )
    if mx.any(beta_arr <= 0.0):
        raise ValueError(
            "Invalid value for argument `beta`. All beta "
            f"values must be > 0, received beta={beta}"
        )
    if alpha_arr.shape != beta_arr.shape:
        raise ValueError(
            "Invalid shapes received for `beta` and `alpha`. "
            "Alpha and beta must both be scalar values or "
            f"the same shape, received alpha shape: {alpha_arr.shape}, "
            f"beta shape: {beta_arr.shape}"
        )

    key = mlx_draw_seed(seed)
    if alpha_arr.size == 1 and beta_arr.size == 1:
        alpha_scalar = alpha_arr.item()
        beta_scalar = beta_arr.item()
        key_x, key_y = mx.random.split(key, 2)
        x = gamma(shape, alpha_scalar, dtype=dtype, seed=key_x)
        y = gamma(shape, beta_scalar, dtype=dtype, seed=key_y)
        return (x / (x + y)).astype(dtype)
    else:
        # how can we check shape and alpha/beta shapes are broadcastable?
        # are scalar values okay when output shape dimension > 1?
        if len(shape) != alpha_arr.ndim:
            raise ValueError(
                "Output shape and `alpha` and `beta` shapes cannot be "
                f"broadcast. Received shape={shape} and alpha and "
                f"beta shapes={alpha.shape}"
            )

        def _sample_gamma(shape, a, b, key):
            carry_key, key_x = mx.random.split(key)
            x = gamma(shape, a, dtype=dtype, seed=key_x)

            carry_key, key_y = mx.random.split(carry_key)
            y = gamma(shape, b, dtype=dtype, seed=key_y)

            return x / (x + y), carry_key

        sample_shape = tuple(
            s for s, a in zip(shape, alpha_arr.shape) if s != a
        )
        carry_key = key
        results = []
        for a, b in zip(alpha_arr.flatten(), beta_arr.flatten()):
            result, carry_key = _sample_gamma(sample_shape, a, b, carry_key)
            results.append(result)

        result = mx.stack(results, axis=-1)
        return result.reshape(shape).astype(dtype)


def binomial(shape, counts, probabilities, dtype=None, seed=None):
    # Binomial(n, p) distribution by summing n Bernoulli(p) samples
    dtype = to_mlx_dtype(dtype)
    key = mlx_draw_seed(seed)

    if isinstance(shape, int):
        shape = (shape,)

    # counts will be handled as ints below
    counts_arr = mx.array(counts, dtype=mx.float32)
    probs_arr = mx.array(probabilities, dtype=mx.float32)

    # should we validate against counts and probabilities?
    if mx.any(counts_arr < 0.0):
        raise ValueError(
            "Invalid value for argument `counts`. All counts "
            f"must be >= 0, received counts={counts}"
        )
    if mx.any(probs_arr < 0.0) or mx.any(probs_arr > 1.0):
        raise ValueError(
            "Invalid value for argument `probabilities`. "
            "All probabilities must be in [0, 1], received "
            f"probabilities={probabilities}"
        )

    # broadcast counts and probs to `shape``
    zeros_for_bcast = mx.zeros(shape=shape, dtype=mx.float32)
    counts_bcast = counts_arr + zeros_for_bcast
    probs_bcast = probs_arr + zeros_for_bcast

    flat_size = 1
    for dim in shape:
        flat_size *= dim

    counts_flat = counts_bcast.reshape((flat_size,))
    probs_flat = probs_bcast.reshape((flat_size,))
    out_flat = mx.zeros((flat_size,), dtype=dtype)

    # for each element in flattened arrays
    # draw a single Binomial(n_i, p_i) sample by summing n_i Bernoulli draws
    carry_key = key
    for i in range(flat_size):
        n_i = counts_flat[i].astype(mx.int64).item()
        p_i = probs_flat[i].item()

        if n_i == 0:
            out_flat[i] = 0
            continue

        carry_key, subkey = mx.random.split(carry_key)
        bernoulli_samples = mx.random.bernoulli(key=subkey, shape=(n_i,), p=p_i)
        binomial_val = mx.sum(bernoulli_samples, axis=0)
        out_flat[i] = binomial_val

    out = out_flat.reshape(shape)
    return out.astype(dtype)
