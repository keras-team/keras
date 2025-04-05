"""Required functions for optimized contractions of numpy arrays using pytorch."""

from opt_einsum.helpers import has_array_interface
from opt_einsum.parser import convert_to_valid_einsum_chars
from opt_einsum.sharing import to_backend_cache_wrap

__all__ = [
    "transpose",
    "einsum",
    "tensordot",
    "to_torch",
    "build_expression",
    "evaluate_constants",
]

_TORCH_DEVICE = None
_TORCH_HAS_TENSORDOT = None

_torch_symbols_base = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"


def _get_torch_and_device():
    global _TORCH_DEVICE
    global _TORCH_HAS_TENSORDOT

    if _TORCH_DEVICE is None:
        import torch  # type: ignore

        device = "cuda" if torch.cuda.is_available() else "cpu"
        _TORCH_DEVICE = torch, device
        _TORCH_HAS_TENSORDOT = hasattr(torch, "tensordot")

    return _TORCH_DEVICE


def transpose(a, axes):
    """Normal torch transpose is only valid for 2D matrices."""
    return a.permute(*axes)


def einsum(equation, *operands, **kwargs):
    """Variadic version of torch.einsum to match numpy api."""
    # rename symbols to support PyTorch 0.4.1 and earlier,
    # which allow only symbols a-z.
    equation = convert_to_valid_einsum_chars(equation)

    torch, _ = _get_torch_and_device()
    return torch.einsum(equation, operands)


def tensordot(x, y, axes=2):
    """Simple translation of tensordot syntax to einsum."""
    torch, _ = _get_torch_and_device()

    if _TORCH_HAS_TENSORDOT:
        return torch.tensordot(x, y, dims=axes)

    xnd = x.ndimension()
    ynd = y.ndimension()

    # convert int argument to (list[int], list[int])
    if isinstance(axes, int):
        axes = range(xnd - axes, xnd), range(axes)

    # convert (int, int) to (list[int], list[int])
    if isinstance(axes[0], int):
        axes = (axes[0],), axes[1]
    if isinstance(axes[1], int):
        axes = axes[0], (axes[1],)

    # initialize empty indices
    x_ix = [None] * xnd
    y_ix = [None] * ynd
    out_ix = []

    # fill in repeated indices
    available_ix = iter(_torch_symbols_base)
    for ax1, ax2 in zip(*axes):
        repeat = next(available_ix)
        x_ix[ax1] = repeat
        y_ix[ax2] = repeat

    # fill in the rest, and maintain output order
    for i in range(xnd):
        if x_ix[i] is None:
            leave = next(available_ix)
            x_ix[i] = leave
            out_ix.append(leave)
    for i in range(ynd):
        if y_ix[i] is None:
            leave = next(available_ix)
            y_ix[i] = leave
            out_ix.append(leave)

    # form full string and contract!
    einsum_str = "{},{}->{}".format(*map("".join, (x_ix, y_ix, out_ix)))
    return einsum(einsum_str, x, y)


@to_backend_cache_wrap
def to_torch(array):
    torch, device = _get_torch_and_device()

    if has_array_interface(array):
        return torch.from_numpy(array).to(device)

    return array


def build_expression(_, expr):  # pragma: no cover
    """Build a torch function based on ``arrays`` and ``expr``."""

    def torch_contract(*arrays):
        torch_arrays = [to_torch(x) for x in arrays]
        torch_out = expr._contract(torch_arrays, backend="torch")

        if torch_out.device.type == "cpu":
            return torch_out.numpy()

        return torch_out.cpu().numpy()

    return torch_contract


def evaluate_constants(const_arrays, expr):
    """Convert constant arguments to torch, and perform any possible constant
    contractions.
    """
    const_arrays = [to_torch(x) for x in const_arrays]
    return expr(*const_arrays, backend="torch", evaluate_constants=True)
