from keras.src import backend
from keras.src.backend import config


def is_in_jax_tracing_scope(x=None):
    if config.backend() == "jax":
        import jax

        if x is None:
            x = backend.ops.numpy.ones(())
        return isinstance(x, jax.core.Tracer)
    return False
