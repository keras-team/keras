import random as python_random

import numpy as np

from keras.src import backend
from keras.src.api_export import keras_export
from keras.src.backend.common import global_state
from keras.src.utils import jax_utils
from keras.src.utils.naming import auto_name


@keras_export("keras.random.SeedGenerator")
class SeedGenerator:
    """Generates variable seeds upon each call to a function generating
    random numbers.

    In Keras, all random number generators (such as
    `keras.random.normal()`) are stateless, meaning that if you pass an
    integer seed to them (such as `seed=42`), they will return the same
    values for repeated calls. To get different values for each
    call, a `SeedGenerator` providing the state of the random generator
    has to be used.

    Note that all the random number generators have a default seed of None,
    which implies that an internal global SeedGenerator is used.
    If you need to decouple the RNG from the global state you can provide
    a local `StateGenerator` with either a deterministic or random initial
    state.

    Remark concerning the JAX backen: Note that the use of a local
    `StateGenerator` as seed argument is required for JIT compilation of
    RNG with the JAX backend, because the use of global state is not
    supported.

    Example:

    ```python
    seed_gen = keras.random.SeedGenerator(seed=42)
    values = keras.random.normal(shape=(2, 3), seed=seed_gen)
    new_values = keras.random.normal(shape=(2, 3), seed=seed_gen)
    ```

    Usage in a layer:

    ```python
    class Dropout(keras.Layer):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.seed_generator = keras.random.SeedGenerator(1337)

        def call(self, x, training=False):
            if training:
                return keras.random.dropout(
                    x, rate=0.5, seed=self.seed_generator
                )
            return x
    ```
    """

    def __init__(self, seed=None, name=None, **kwargs):
        if name is None:
            name = auto_name(self.__class__.__name__)
        self.name = name

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

        with self.backend.name_scope(self.name, caller=self):
            self.state = self.backend.Variable(
                seed_initializer,
                shape=(2,),
                dtype=self.backend.random_seed_dtype(),
                trainable=False,
                aggregation="none",
                name="seed_generator_state",
            )

    def next(self, ordered=True):
        seed_state = self.state
        # Use * 1 to create a copy
        new_seed_value = seed_state.value * 1
        if ordered:
            increment = self.backend.convert_to_tensor(
                np.array([0, 1]), dtype=seed_state.dtype
            )
            self.state.assign(self.backend.numpy.add(seed_state, increment))
        else:
            # This produces a sequence of near-unique numbers
            # between 0 and 1M
            self.state.assign((seed_state + 1) * 5387 % 933199)
        return new_seed_value

    def get_config(self):
        return {"seed": self._initial_seed}

    @classmethod
    def from_config(cls, config):
        return cls(**config)


def global_seed_generator():
    if jax_utils.is_in_jax_tracing_scope():
        raise ValueError(
            "[JAX RNG] When tracing a JAX function, "
            "you should only use seeded random ops, e.g. "
            "you should create a `SeedGenerator` instance, attach it "
            "to your layer/model, and pass the instance as the `seed` "
            "argument when calling random ops. Unseeded random ops "
            "would get incorrectly traced by JAX and would become constant "
            "after tracing. Example:\n\n"
            "```\n"
            "# Make sure to set the seed generator as a layer attribute\n"
            "self.seed_generator = keras.random.SeedGenerator(seed=1337)\n"
            "...\n"
            "out = keras.random.normal(shape=(1,), seed=self.seed_generator)\n"
            "```"
        )
    gen = global_state.get_global_attribute("global_seed_generator")
    if gen is None:
        gen = SeedGenerator()
        global_state.set_global_attribute("global_seed_generator", gen)
    return gen


def make_default_seed():
    return python_random.randint(1, int(1e9))


def draw_seed(seed):
    from keras.src.backend import convert_to_tensor
    from keras.src.backend import random_seed_dtype

    if isinstance(seed, SeedGenerator):
        return seed.next()
    elif isinstance(seed, int):
        return convert_to_tensor([seed, 0], dtype=random_seed_dtype())
    elif seed is None:
        return global_seed_generator().next(ordered=False)
    raise ValueError(
        "Argument `seed` must be either an integer "
        "or an instance of `SeedGenerator`. "
        f"Received: seed={seed} (of type {type(seed)})"
    )
