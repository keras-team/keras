from keras.src import backend
from keras.src.backend.config import is_nnx_backend_enabled


def is_in_jax_tracing_scope(x=None):
    if backend.backend() == "jax":
        if x is None:
            x = backend.numpy.ones(())
        for c in x.__class__.__mro__:
            if c.__name__ == "Tracer" and c.__module__.startswith("jax"):
                return True
    return False


def jit(*args, **kwargs):
    import jax

    def decorator(func):
        if is_nnx_backend_enabled():
            from flax import nnx

            return nnx.jit(func, *args, **kwargs)
        else:
            return jax.jit(func, *args, **kwargs)

    return decorator
