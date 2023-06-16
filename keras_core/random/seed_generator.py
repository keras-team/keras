import random as python_random

import numpy as np

from keras_core import backend
from keras_core.api_export import keras_core_export


@keras_core_export("keras_core.random.SeedGenerator")
class SeedGenerator:
    def __init__(self, seed):
        from keras_core.backend import Variable

        if seed is None:
            seed = make_default_seed()
        if not isinstance(seed, int):
            raise ValueError(
                "Argument `seed` must be an integer. " f"Received: seed={seed}"
            )

        def seed_initializer(*args, **kwargs):
            from keras_core.backend import convert_to_tensor

            dtype = kwargs.get("dtype", None)
            return convert_to_tensor([seed, 0], dtype=dtype)

        self.state = Variable(
            seed_initializer,
            shape=(2,),
            dtype="uint32",
            trainable=False,
            name="seed_generator_state",
        )


def make_default_seed():
    return python_random.randint(1, int(1e9))


def draw_seed(seed):
    from keras_core.backend import convert_to_tensor

    if isinstance(seed, SeedGenerator):
        seed_state = seed.state
        # Use * 1 to create a copy
        new_seed_value = seed_state.value * 1
        seed.state.assign(seed_state + np.array([0, 1], dtype="uint32"))
        if backend.backend() == "torch":
            return backend.convert_to_numpy(new_seed_value)
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
