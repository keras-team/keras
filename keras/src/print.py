import keras.backend
from keras.src.api_export import keras_export

# Unique source of truth for the version number.
__version__ = "3.10.0"
_print = print


@keras_export("keras.print")
def print(*args, **kwargs):
    backend = keras.backend.backend()
    if backend == "jax":
        import jax  # noqa: E402

        print_fn = jax.debug.print
    elif backend == "tensorflow":
        import tensorflow as tf  # noqa: E402

        print_fn = tf.print
    else:
        print_fn = _print
    # TODO:
    # "torch"
    #   pytorch.org/docs/stable/generated/torch.set_printoptions.html ?
    # "openvino"
    # "numpy"
    return print_fn(*args, **kwargs)
