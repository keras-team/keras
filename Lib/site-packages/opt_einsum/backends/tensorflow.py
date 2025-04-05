"""Required functions for optimized contractions of numpy arrays using tensorflow."""

from opt_einsum.helpers import has_array_interface
from opt_einsum.sharing import to_backend_cache_wrap

__all__ = ["to_tensorflow", "build_expression", "evaluate_constants"]

_CACHED_TF_DEVICE = None


def _get_tensorflow_and_device():
    global _CACHED_TF_DEVICE

    if _CACHED_TF_DEVICE is None:
        import tensorflow as tf  # type: ignore

        try:
            eager = tf.executing_eagerly()
        except AttributeError:
            try:
                eager = tf.contrib.eager.in_eager_mode()
            except AttributeError:
                eager = False

        device = tf.test.gpu_device_name()
        if not device:
            device = "cpu"

        _CACHED_TF_DEVICE = tf, device, eager

    return _CACHED_TF_DEVICE


@to_backend_cache_wrap(constants=True)
def to_tensorflow(array, constant=False):
    """Convert a numpy array to a ``tensorflow.placeholder`` instance."""
    tf, device, eager = _get_tensorflow_and_device()

    if eager:
        if has_array_interface(array):
            with tf.device(device):
                return tf.convert_to_tensor(array)

        return array

    if has_array_interface(array):
        if constant:
            return tf.convert_to_tensor(array)

        return tf.placeholder(array.dtype, array.shape)

    return array


# Standard graph mode


def build_expression_graph(arrays, expr):
    """Build a tensorflow function based on ``arrays`` and ``expr``."""
    tf, _, _ = _get_tensorflow_and_device()

    placeholders = [to_tensorflow(array) for array in arrays]
    graph = expr._contract(placeholders, backend="tensorflow")

    def tensorflow_contract(*arrays):
        session = tf.get_default_session()
        # only want to feed placeholders - constant tensors already have values
        feed_dict = {p: a for p, a in zip(placeholders, arrays) if p.op.type == "Placeholder"}
        return session.run(graph, feed_dict=feed_dict)

    return tensorflow_contract


def evaluate_constants_graph(const_arrays, expr):
    """Convert constant arguments to tensorflow constants, and perform any
    possible constant contractions. Requires evaluating a tensorflow graph.
    """
    tf, _, _ = _get_tensorflow_and_device()

    # compute the partial graph of new inputs
    const_arrays = [to_tensorflow(x, constant=True) for x in const_arrays]
    new_ops, new_contraction_list = expr(*const_arrays, backend="tensorflow", evaluate_constants=True)

    # evaluate the new inputs and convert back to tensorflow, maintaining None as non-consts
    session = tf.get_default_session()
    new_consts = iter(session.run([x for x in new_ops if x is not None]))
    new_ops = [None if x is None else to_tensorflow(next(new_consts), constant=True) for x in new_ops]

    return new_ops, new_contraction_list


# Eager execution mode


def build_expression_eager(_, expr):
    """Build a eager tensorflow function based on ``arrays`` and ``expr``."""

    def tensorflow_eager_contract(*arrays):
        return expr._contract([to_tensorflow(x) for x in arrays], backend="tensorflow").numpy()

    return tensorflow_eager_contract


def evaluate_constants_eager(const_arrays, expr):
    """Convert constant arguments to tensorflow_eager arrays, and perform any
    possible constant contractions.
    """
    return expr(*[to_tensorflow(x) for x in const_arrays], backend="tensorflow", evaluate_constants=True)


# Dispatch to eager or graph mode


def build_expression(arrays, expr):
    _, _, eager = _get_tensorflow_and_device()
    fn = build_expression_eager if eager else build_expression_graph
    return fn(arrays, expr)


def evaluate_constants(const_arrays, expr):
    _, _, eager = _get_tensorflow_and_device()
    fn = evaluate_constants_eager if eager else evaluate_constants_graph
    return fn(const_arrays, expr)
