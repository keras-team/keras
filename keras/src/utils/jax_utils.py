from keras.src import backend


def is_in_jax_tracing_scope(x=None):
    if backend.backend() == "jax":
        if x is None:
            x = backend.numpy.ones(())
        if x.__class__.__name__ == "DynamicJaxprTracer":
            return True
    return False
