import random as python_random

import numpy as np

from keras_core import backend
from keras_core.api_export import keras_core_export
from keras_core.backend.common import global_state


@keras_core_export("keras_core.random.SeedGenerator")
class SeedGenerator:
    """Generates variable seeds upon each call to a RNG-using function.

    In Keras, all RNG-using methods (such as `keras_core.random.normal()`)
    are stateless, meaning that if you pass an integer seed to them
    (such as `seed=42`), they will return the same values at each call.
    In order to get different values at each call, you must use a
    `SeedGenerator` instead as the seed argument. The `SeedGenerator`
    object is stateful.

    Example:

    ```python
    seed_gen = keras_core.random.SeedGenerator(seed=42)
    values = keras_core.random.normal(shape=(2, 3), seed=seed_gen)
    new_values = keras_core.random.normal(shape=(2, 3), seed=seed_gen)
    ```

    Usage in a layer:

    ```python
    class Dropout(keras_core.Layer):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.seed_generator = keras_core.random.SeedGenerator(1337)

        def call(self, x, training=False):
            if training:
                return keras_core.random.dropout(
                    x, rate=0.5, seed=self.seed_generator
                )
            return x
    ```
    """

    def __init__(self, seed=None, **kwargs):
        custom_backend = kwargs.pop("backend", None)
        if kwargs:
            raise ValueError(f"Unrecognized keyword arguments: {kwargs}")
        if custom_backend is not None:
            self.backend = custom_backend
        else:
            self.backend = backend

        self._initial_seed = seed
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

    def next(self, ordered=True):
        seed_state = self.state
        # Use * 1 to create a copy
        new_seed_value = seed_state.value * 1
        if ordered:
            increment = self.backend.convert_to_tensor(
                np.array([0, 1]), dtype="uint32"
            )
            self.state.assign(seed_state + increment)
        else:
            # This produces a sequence of near-unique numbers
            # between 0 and 1M
            self.state.assign((seed_state + 1) * 5387 % 933199)
        return new_seed_value


def global_seed_generator():
    gen = global_state.get_global_attribute("global_seed_generator")
    if gen is None:
        gen = SeedGenerator()
        global_state.set_global_attribute("global_seed_generator", gen)
    return gen


@keras_core_export("keras_core.random.global_rng_state")
def global_rng_state():
    """Returns the state variable for the default global RNG.

    Returns:
        A `KerasVariable` with shape `(2,)` and dtype `uint32`.

    This is the global state used by unseeded (`seed=None`) Keras
    random ops, e.g. `keras_core.random.normal(shape=(), seed=None)`.

    In JAX, if you're using unseeded random ops, be mindful that
    their outputs will be unchanged across different calls of
    traced function (e.g. a `jax.jit`-transformed function) since
    traced functions in JAX are fully stateless. To get
    different outputs across different calls, you will need to pass the
    global RNG state in and out of the function boundary, like this:

    ```python
    @jax.jit
    def random_numbers(seed):
        rng_state = keras_core.random.global_rng_state()
        rng_state.assign(seed)
        x = keras_core.random.normal((), seed=None)
        y = keras_core.random.normal((), seed=None)
        return x, y, rng_state.value

    seed = jax.numpy.array([0, 0])
    x, y, seed = random_numbers(seed)
    new_x, new_y, seed = random_numbers(seed)
    ```
    """
    return global_seed_generator().state


def make_default_seed():
    return python_random.randint(1, int(1e9))


def draw_seed(seed):
    from keras_core.backend import convert_to_tensor

    if isinstance(seed, SeedGenerator):
        return seed.next()
    elif isinstance(seed, int):
        return convert_to_tensor([seed, 0], dtype="uint32")
    elif seed is None:
        return global_seed_generator().next(ordered=False)
    raise ValueError(
        "Argument `seed` must be either an integer "
        "or an instance of `SeedGenerator`. "
        f"Received: seed={seed} (of type {type(seed)})"
    )
