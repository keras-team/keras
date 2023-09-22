from keras import backend


def is_in_jax_tracing_scope():
    if backend.backend() == "jax":
        x = backend.numpy.ones(())
        if x.__class__.__name__ == "DynamicJaxprTracer":
            return True
    return False
