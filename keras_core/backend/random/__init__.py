from keras_core.backend import backend

from keras_core.backend.random.random_seed_generator import RandomSeedGenerator
from keras_core.backend.random.random_seed_generator import draw_seed
from keras_core.backend.random.random_seed_generator import make_default_seed

if backend() == "jax":
    from keras_core.backend.jax.random import *
else:
    from keras_core.backend.tensorflow.random import *
