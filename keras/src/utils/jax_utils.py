from keras.src import backend


def is_in_jax_tracing_scope(x=None):
    if backend.backend() == "jax":
        import jax

        if x is None:
            x = backend.numpy.ones(())
        return isinstance(x, jax.core.Tracer)
    return False
