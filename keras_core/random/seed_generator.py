import random as python_random

import numpy as np

import keras_core.backend
from keras_core.api_export import keras_core_export


@keras_core_export("keras_core.random.SeedGenerator")
class SeedGenerator:
    """Generates variable seeds upon each call to a RNG-using function.

    In Keras, all RNG-using methods (such as `keras_core.random.normal()`)
    are stateless, meaning that if you pass an integer seed to them
    (such as `seed=42`), they will return the same values at each call.
    In order to get different values at each call, you must use a
    `SeedGenerator` instead as the seed argument. The `SeedGenerator`
    instead is stateful.

    Example:

    ```python
    seed_gen = keras_core.random.SeedGenerator(seed=42)
    values = keras_core.random.normal(shape=(2, 3), seed=seed_gen)
    new_values = keras_core.random.normal(shape=(2, 3), seed=seed_gen)
    ```
    """

    def __init__(self, seed, **kwargs):
        custom_backend = kwargs.pop("backend", None)
        if kwargs:
            raise ValueError(f"Unrecognized keyword arguments: {kwargs}")
        if custom_backend is not None:
            self.backend = custom_backend
        else:
            self.backend = keras_core.backend

        if seed is None:
            seed = make_default_seed()
        if not isinstance(seed, int):
            raise ValueError(
                "Argument `seed` must be an integer. " f"Received: seed={seed}"
            )

        def seed_initializer(*args, **kwargs):
            dtype = kwargs.get("dtype", None)
            return self.backend.convert_to_tensor([seed, 0], dtype=dtype)

        self.state = self.backend.Variable(
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
        increment = seed.backend.convert_to_tensor(
            np.array([0, 1]), dtype="uint32"
        )
        seed.state.assign(seed_state + increment)
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
