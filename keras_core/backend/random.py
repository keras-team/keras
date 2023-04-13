import random as python_random

from keras_core.backend import backend


class SeedGenerator:
    def __init__(self, seed):
        from keras_core.backend import Variable

        if not isinstance(seed, int):
            raise ValueError(
                "Argument `seed` must be an integer. " f"Received: seed={seed}"
            )
        seed = seed or make_default_seed()
        self.state = Variable([seed, 0], dtype="uint32", trainable=False)


def make_default_seed():
    return python_random.randint(1, int(1e9))


def draw_seed(seed):
    from keras_core.backend import convert_to_tensor

    if isinstance(seed, SeedGenerator):
        new_seed_value = seed.state.value
        seed.state.assign(
            seed.state + convert_to_tensor([0, 1], dtype="uint32")
        )
        return new_seed_value
    elif isinstance(seed, int):
        return convert_to_tensor([seed, 0], dtype="uint32")
    elif seed is None:
        return convert_to_tensor([make_default_seed(), 0], dtype="uint32")
    raise ValueError(
        "Argument `seed` must be either an integer "
        "or an instance of `SeedGenerator`. "
        f"Received: seed={seed} (of type {type(seed)})"
    )


if backend() == "jax":
    from keras_core.backend.jax.random import *
else:
    from keras_core.backend.tensorflow.random import *