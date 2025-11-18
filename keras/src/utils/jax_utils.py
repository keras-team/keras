from keras.src import backend


def is_in_jax_tracing_scope(x=None):
    if backend.backend() == "jax":
        if x is None:
            x = backend.numpy.ones(())
        for c in x.__class__.__mro__:
            if c.__name__ == "Tracer" and c.__module__.startswith("jax"):
                return True
    return False


def get_jax_random_seed(seed=None):
    if is_in_jax_tracing_scope():
        # Constant dummy seed for Tracing
        seed = 0
    else:
        # Gathering seed from a seed generator
        seed = backend.random.draw_seed(None)[0]
    return seed


# Create a lightweight class that only provides shape/dtype info
class JAXTracingSeedGenerator:
    def __init__(self):
        self._shape = (2,)
        self._dtype = "uint32"

    def next(self, ordered=False):
        # Return a dummy key for tracing
        return backend.random.jax.random.key(0)
