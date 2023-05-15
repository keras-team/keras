from keras_core.backend.config import floatx
from keras_core.random.seed_generator import SeedGenerator
from keras_core.random.seed_generator import draw_seed
from keras_core.random.seed_generator import make_default_seed


def normal(shape, mean=0.0, stddev=1.0, dtype=None, seed=None):
    raise NotImplementedError


def uniform(shape, minval=0.0, maxval=1.0, dtype=None, seed=None):
    raise NotImplementedError


def truncated_normal(shape, mean=0.0, stddev=1.0, dtype=None, seed=None):
    raise NotImplementedError


def dropout(inputs, rate, noise_shape=None, seed=None):
    raise NotImplementedError
