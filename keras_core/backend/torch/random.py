import torch
import torch.nn.functional as tnn

from keras_core.backend.config import floatx
from keras_core.backend.torch.core import convert_to_tensor
from keras_core.backend.torch.core import get_device
from keras_core.backend.torch.core import to_torch_dtype
from keras_core.random.seed_generator import SeedGenerator
from keras_core.random.seed_generator import draw_seed
from keras_core.random.seed_generator import make_default_seed


def torch_seed_generator(seed, device="cpu"):
    seed_val, _ = draw_seed(seed)
    generator = torch.Generator(device=device)
    generator.manual_seed(int(seed_val))
    return generator


def normal(shape, mean=0.0, stddev=1.0, dtype=None, seed=None):
    dtype = dtype or floatx()
    dtype = to_torch_dtype(dtype)
    generator = torch_seed_generator(seed)
    return torch.normal(
        mean, stddev, size=shape, generator=generator, dtype=dtype
    ).to(get_device())


def categorical(logits, num_samples, dtype="int32", seed=None):
    logits = convert_to_tensor(logits)
    dtype = to_torch_dtype(dtype)
    generator = torch_seed_generator(seed, device=get_device())
    return torch.multinomial(
        logits,
        num_samples,
        replacement=True,
        generator=generator,
    ).type(dtype)


def uniform(shape, minval=0.0, maxval=1.0, dtype=None, seed=None):
    dtype = dtype or floatx()
    dtype = to_torch_dtype(dtype)
    generator = torch_seed_generator(seed)
    if len(shape) == 0:
        shape = (1,)
    output = (maxval - minval) * torch.rand(
        *shape, generator=generator, dtype=dtype
    ) + minval
    return output.to(get_device())


def randint(shape, minval, maxval, dtype="int32", seed=None):
    dtype = to_torch_dtype(dtype)
    generator = torch_seed_generator(seed)
    return torch.randint(
        low=minval, high=maxval, size=shape, generator=generator, dtype=dtype
    )


def truncated_normal(shape, mean=0.0, stddev=1.0, dtype=None, seed=None):
    # Take a larger standard normal dist, discard values outside 2 * stddev
    # Offset by mean and stddev
    x = normal(shape + (4,), mean=0, stddev=1, dtype=dtype, seed=seed)
    valid = (x > -2) & (x < 2)
    indexes = valid.max(-1, keepdim=True)[1]
    trunc_x = torch.empty(shape)
    trunc_x.data.copy_(x.gather(-1, indexes).squeeze(-1))
    trunc_x.data.mul_(stddev).add_(mean)
    return trunc_x.to(get_device())


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
    seed, _ = draw_seed(seed)
    generator = torch.Generator()
    generator.manual_seed(int(seed))

    keep_prob = 1.0 - rate
    noise_shape = _get_concrete_noise_shape(inputs, noise_shape)
    keep_prob_matrix = torch.full(noise_shape, keep_prob)
    mask = torch.bernoulli(keep_prob_matrix, generator=generator).bool()
    mask = torch.broadcast_to(mask, inputs.shape)
    mask = mask.to(get_device())
    return torch.where(
        mask, inputs / keep_prob, torch.zeros_like(inputs, dtype=inputs.dtype)
    )
