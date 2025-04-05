"""Required functions for optimized contractions of numpy arrays using theano."""

from opt_einsum.helpers import has_array_interface
from opt_einsum.sharing import to_backend_cache_wrap

__all__ = ["to_theano", "build_expression", "evaluate_constants"]


@to_backend_cache_wrap(constants=True)
def to_theano(array, constant=False):
    """Convert a numpy array to ``theano.tensor.TensorType`` instance."""
    import theano  # type: ignore

    if has_array_interface(array):
        if constant:
            return theano.tensor.constant(array)

        return theano.tensor.TensorType(dtype=array.dtype, broadcastable=[False] * len(array.shape))()

    return array


def build_expression(arrays, expr):
    """Build a theano function based on ``arrays`` and ``expr``."""
    import theano

    in_vars = [to_theano(array) for array in arrays]
    out_var = expr._contract(in_vars, backend="theano")

    # don't supply constants to graph
    graph_ins = [x for x in in_vars if not isinstance(x, theano.tensor.TensorConstant)]
    graph = theano.function(graph_ins, out_var)

    def theano_contract(*arrays):
        return graph(*[x for x in arrays if not isinstance(x, theano.tensor.TensorConstant)])

    return theano_contract


def evaluate_constants(const_arrays, expr):
    # compute the partial graph of new inputs
    const_arrays = [to_theano(x, constant=True) for x in const_arrays]
    new_ops, new_contraction_list = expr(*const_arrays, backend="theano", evaluate_constants=True)

    # evaluate the new inputs and convert to theano shared tensors
    new_ops = [None if x is None else to_theano(x.eval(), constant=True) for x in new_ops]

    return new_ops, new_contraction_list
