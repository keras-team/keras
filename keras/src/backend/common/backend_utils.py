import functools
import operator
import re
import warnings


def _convert_conv_tranpose_padding_args_from_keras_to_jax(
    kernel_size, stride, dilation_rate, padding, output_padding
):
    """Convert the padding arguments from Keras to the ones used by JAX.
    JAX starts with an shape of size `(input-1) * stride - kernel_size + 2`,
    then adds `left_pad` on the left, and `right_pad` on the right.
    In Keras, the `padding` argument determines a base shape, to which
    `output_padding` is added on the right. If `output_padding` is None, it will
    be given a default value.
    """

    assert padding.lower() in {"valid", "same"}
    kernel_size = (kernel_size - 1) * dilation_rate + 1

    if padding.lower() == "valid":
        # If output_padding is None, we fill it so that the shape of the output
        # is `(input-1)*s + max(kernel_size, stride)`
        output_padding = (
            max(kernel_size, stride) - kernel_size
            if output_padding is None
            else output_padding
        )
        left_pad = kernel_size - 1
        right_pad = kernel_size - 1 + output_padding

    else:
        if output_padding is None:
            # When output_padding is None, we want the shape of the output to
            # be `input * s`, therefore a total padding of
            # `stride + kernel_size - 2`
            pad_len = stride + kernel_size - 2
        else:
            # When output_padding is filled, we want the shape of the output to
            # be `(input-1)*stride + kernel_size%2 + output_padding`
            pad_len = kernel_size + kernel_size % 2 - 2 + output_padding
        left_pad = min(pad_len // 2 + pad_len % 2, kernel_size - 1)
        right_pad = pad_len - left_pad

    return left_pad, right_pad


def _convert_conv_tranpose_padding_args_from_keras_to_torch(
    kernel_size, stride, dilation_rate, padding, output_padding
):
    """Convert the padding arguments from Keras to the ones used by Torch.
    Torch starts with an output shape of `(input-1) * stride + kernel_size`,
    then removes `torch_padding` from both sides, and adds
    `torch_output_padding` on the right.
    Because in Torch the output_padding can only be added to the right,
    consistency with Tensorflow is not always possible. In particular this is
    the case when both the Torch padding and output_padding values are
    strictly positive.
    """
    assert padding.lower() in {"valid", "same"}
    original_kernel_size = kernel_size
    kernel_size = (kernel_size - 1) * dilation_rate + 1

    if padding.lower() == "valid":
        # If output_padding is None, we fill it so that the shape of the output
        # is `(i-1)*s + max(k, s)`
        output_padding = (
            max(kernel_size, stride) - kernel_size
            if output_padding is None
            else output_padding
        )
        torch_padding = 0
        torch_output_padding = output_padding

    else:
        # When output_padding is None, we want the shape of the output to be
        # `input * s`, otherwise we use the value provided.
        output_padding = (
            stride - kernel_size % 2
            if output_padding is None
            else output_padding
        )
        torch_padding = max(
            -((kernel_size % 2 - kernel_size + output_padding) // 2), 0
        )
        torch_output_padding = (
            2 * torch_padding + kernel_size % 2 - kernel_size + output_padding
        )

    if torch_padding > 0 and torch_output_padding > 0:
        warnings.warn(
            f"You might experience inconsistencies across backends when "
            f"calling conv transpose with kernel_size={original_kernel_size}, "
            f"stride={stride}, dilation_rate={dilation_rate}, "
            f"padding={padding}, output_padding={output_padding}."
        )

    if torch_output_padding >= stride:
        raise ValueError(
            f"The padding arguments (padding={padding}) and "
            f"output_padding={output_padding}) lead to a Torch "
            f"output_padding ({torch_output_padding}) that is greater than "
            f"strides ({stride}). This is not supported. You can change the "
            f"padding arguments, kernel or stride, or run on another backend. "
        )

    return torch_padding, torch_output_padding


def compute_conv_transpose_padding_args_for_jax(
    input_shape,
    kernel_shape,
    strides,
    padding,
    output_padding,
    dilation_rate,
):
    num_spatial_dims = len(input_shape) - 2
    kernel_spatial_shape = kernel_shape[:-2]

    jax_padding = []
    for i in range(num_spatial_dims):
        output_padding_i = (
            output_padding
            if output_padding is None or isinstance(output_padding, int)
            else output_padding[i]
        )
        strides_i = strides if isinstance(strides, int) else strides[i]
        dilation_rate_i = (
            dilation_rate
            if isinstance(dilation_rate, int)
            else dilation_rate[i]
        )
        (
            pad_left,
            pad_right,
        ) = _convert_conv_tranpose_padding_args_from_keras_to_jax(
            kernel_size=kernel_spatial_shape[i],
            stride=strides_i,
            dilation_rate=dilation_rate_i,
            padding=padding,
            output_padding=output_padding_i,
        )
        jax_padding.append((pad_left, pad_right))

    return jax_padding


def compute_conv_transpose_padding_args_for_torch(
    input_shape,
    kernel_shape,
    strides,
    padding,
    output_padding,
    dilation_rate,
):
    num_spatial_dims = len(input_shape) - 2
    kernel_spatial_shape = kernel_shape[:-2]

    torch_paddings = []
    torch_output_paddings = []
    for i in range(num_spatial_dims):
        output_padding_i = (
            output_padding
            if output_padding is None or isinstance(output_padding, int)
            else output_padding[i]
        )
        strides_i = strides if isinstance(strides, int) else strides[i]
        dilation_rate_i = (
            dilation_rate
            if isinstance(dilation_rate, int)
            else dilation_rate[i]
        )
        (
            torch_padding,
            torch_output_padding,
        ) = _convert_conv_tranpose_padding_args_from_keras_to_torch(
            kernel_size=kernel_spatial_shape[i],
            stride=strides_i,
            dilation_rate=dilation_rate_i,
            padding=padding,
            output_padding=output_padding_i,
        )
        torch_paddings.append(torch_padding)
        torch_output_paddings.append(torch_output_padding)

    return torch_paddings, torch_output_paddings


def _get_output_shape_given_tf_padding(
    input_size, kernel_size, strides, padding, output_padding, dilation_rate
):
    if input_size is None:
        return None

    assert padding.lower() in {"valid", "same"}

    kernel_size = (kernel_size - 1) * dilation_rate + 1

    if padding.lower() == "valid":
        output_padding = (
            max(kernel_size, strides) - kernel_size
            if output_padding is None
            else output_padding
        )
        return (input_size - 1) * strides + kernel_size + output_padding

    else:
        if output_padding is None:
            return input_size * strides
        else:
            return (input_size - 1) * strides + kernel_size % 2 + output_padding


def compute_conv_transpose_output_shape(
    input_shape,
    kernel_size,
    filters,
    strides,
    padding,
    output_padding=None,
    data_format="channels_last",
    dilation_rate=1,
):
    num_spatial_dims = len(input_shape) - 2
    kernel_spatial_shape = kernel_size

    if isinstance(output_padding, int):
        output_padding = (output_padding,) * len(kernel_spatial_shape)
    if isinstance(strides, int):
        strides = (strides,) * num_spatial_dims
    if isinstance(dilation_rate, int):
        dilation_rate = (dilation_rate,) * num_spatial_dims

    if data_format == "channels_last":
        input_spatial_shape = input_shape[1:-1]
    else:
        input_spatial_shape = input_shape[2:]

    output_shape = []
    for i in range(num_spatial_dims):
        current_output_padding = (
            None if output_padding is None else output_padding[i]
        )

        shape_i = _get_output_shape_given_tf_padding(
            input_size=input_spatial_shape[i],
            kernel_size=kernel_spatial_shape[i],
            strides=strides[i],
            padding=padding,
            output_padding=current_output_padding,
            dilation_rate=dilation_rate[i],
        )
        output_shape.append(shape_i)

    if data_format == "channels_last":
        output_shape = [input_shape[0]] + output_shape + [filters]
    else:
        output_shape = [input_shape[0], filters] + output_shape
    return output_shape


def canonicalize_axis(axis, num_dims):
    """Canonicalize an axis in [-num_dims, num_dims) to [0, num_dims)."""
    axis = operator.index(axis)
    if not -num_dims <= axis < num_dims:
        raise ValueError(
            f"axis {axis} is out of bounds for an array with dimension "
            f"{num_dims}."
        )
    if axis < 0:
        axis = axis + num_dims
    return axis


def standardize_axis_for_numpy(axis):
    """Standardize an axis to a tuple if it is a list in the numpy backend."""
    return tuple(axis) if isinstance(axis, list) else axis


def to_tuple_or_list(value):
    """Convert the non-`None` value to either a tuple or a list."""
    if value is None:
        return value
    if not isinstance(value, (int, tuple, list)):
        raise ValueError(
            "`value` must be an integer, tuple or list. "
            f"Received: value={value}"
        )
    if isinstance(value, int):
        return (value,)
    return value


### Code for ops.vectorize() used for TF and torch backends.

# See http://docs.scipy.org/doc/numpy/reference/c-api.generalized-ufuncs.html
_DIMENSION_NAME = r"\w+"
_CORE_DIMENSION_LIST = "(?:{0:}(?:,{0:})*)?".format(_DIMENSION_NAME)
_ARGUMENT = rf"\({_CORE_DIMENSION_LIST}\)"
_ARGUMENT_LIST = "{0:}(?:,{0:})*".format(_ARGUMENT)
_SIGNATURE = "^{0:}->{0:}$".format(_ARGUMENT_LIST)


def _vectorize_parse_gufunc_signature(
    signature,
):
    if not re.match(_SIGNATURE, signature):
        raise ValueError(f"not a valid gufunc signature: {signature}")
    args, retvals = (
        [
            tuple(re.findall(_DIMENSION_NAME, arg))
            for arg in re.findall(_ARGUMENT, arg_list)
        ]
        for arg_list in signature.split("->")
    )
    return args, retvals


def _vectorize_update_dim_sizes(dim_sizes, shape, core_dims, is_input=True):
    num_core_dims = len(core_dims)
    if is_input:
        if len(shape) < num_core_dims:
            raise ValueError(
                f"input with shape {shape} does not "
                "have enough dimensions for all core "
                f"dimensions {core_dims}"
            )
    else:
        if len(shape) != num_core_dims:
            raise ValueError(
                f"output shape {shape} does not "
                f"match core dimensions {core_dims}"
            )

    core_shape = shape[-num_core_dims:] if core_dims else ()
    for dim, size in zip(core_dims, core_shape):
        if dim not in dim_sizes:
            dim_sizes[dim] = size
        elif size != dim_sizes[dim]:
            raise ValueError(
                f"inconsistent size for core dimension {dim}: "
                f"{size} vs {dim_sizes[dim]}"
            )


def _vectorize_parse_input_dimensions(
    args,
    input_core_dims,
):
    from keras.src import ops

    if len(args) != len(input_core_dims):
        raise TypeError(
            "wrong number of positional arguments: "
            f"expected {len(input_core_dims)}, got {len(args)}"
        )
    shapes = []
    dim_sizes: dict[str, int] = {}
    for arg, core_dims in zip(args, input_core_dims):
        _vectorize_update_dim_sizes(
            dim_sizes, arg.shape, core_dims, is_input=True
        )
        ndim = arg.ndim - len(core_dims)
        shapes.append(arg.shape[:ndim])
    broadcast_shape = shapes[0]
    for s in shapes:
        broadcast_shape = ops.broadcast_shapes(broadcast_shape, s)
    return broadcast_shape, dim_sizes


def _vectorize_check_output_dims(
    func,
    dim_sizes,
    expected_output_core_dims,
):
    from keras.src import ops

    def wrapped(*args):
        out = func(*args)
        if isinstance(out, (list, tuple)):
            out_shapes = [ops.shape(x) for x in out]
        else:
            out_shapes = [out.shape]

        if expected_output_core_dims is None:
            output_core_dims = [()] * len(out_shapes)
        else:
            output_core_dims = expected_output_core_dims
            if len(output_core_dims) > 1 and not isinstance(out, tuple):
                raise TypeError(
                    "output must be a tuple when multiple outputs "
                    f"are expected, got: {out}"
                )
            if len(out_shapes) != len(output_core_dims):
                raise TypeError(
                    "wrong number of output arguments: "
                    f"expected {len(output_core_dims)}, got {len(out_shapes)}"
                )

        sizes = dict(dim_sizes)
        for shape, core_dims in zip(out_shapes, output_core_dims):
            _vectorize_update_dim_sizes(sizes, shape, core_dims, is_input=False)

        return out

    return wrapped


def _vectorize_apply_excluded(func, excluded, args, kwargs):
    if not excluded:
        return func, args, kwargs

    dynamic_args = [arg for i, arg in enumerate(args) if i not in excluded]
    dynamic_kwargs = {
        key: val for key, val in kwargs.items() if key not in excluded
    }
    static_args = [
        (i, args[i])
        for i in sorted(e for e in excluded if isinstance(e, int))
        if i < len(args)
    ]
    static_kwargs = {key: val for key, val in kwargs.items() if key in excluded}

    def new_func(*args, **kwargs):
        args = list(args)
        for i, arg in static_args:
            args.insert(i, arg)
        return func(*args, **kwargs, **static_kwargs)

    return new_func, dynamic_args, dynamic_kwargs


def vectorize_impl(pyfunc, vmap_fn, *, excluded=None, signature=None):
    """Implementation adapted from JAX and NumPy."""

    from keras.src import ops

    excluded = None or set()

    @functools.wraps(pyfunc)
    def wrapped(*args, **kwargs):
        excluded_func, args, kwargs = _vectorize_apply_excluded(
            pyfunc, excluded, args, kwargs
        )

        if signature is not None:
            input_core_dims, output_core_dims = (
                _vectorize_parse_gufunc_signature(signature)
            )
        else:
            input_core_dims = [()] * len(args)
            output_core_dims = None

        none_args = {i for i, arg in enumerate(args) if arg is None}
        if any(none_args):
            if any(input_core_dims[i] != () for i in none_args):
                raise ValueError(
                    f"Cannot pass None at locations {none_args} "
                    f"with signature={signature}"
                )
            excluded_func, args, _ = _vectorize_apply_excluded(
                excluded_func, none_args, args, {}
            )
            input_core_dims = [
                dim
                for i, dim in enumerate(input_core_dims)
                if i not in none_args
            ]

        args = tuple(map(ops.convert_to_tensor, args))

        broadcast_shape, dim_sizes = _vectorize_parse_input_dimensions(
            args, input_core_dims
        )
        checked_func = _vectorize_check_output_dims(
            excluded_func, dim_sizes, output_core_dims
        )
        squeezed_args = []
        rev_filled_shapes = []
        for arg, core_dims in zip(args, input_core_dims):
            noncore_shape = arg.shape[: arg.ndim - len(core_dims)]

            pad_ndim = len(broadcast_shape) - len(noncore_shape)
            filled_shape = pad_ndim * (1,) + noncore_shape
            rev_filled_shapes.append(filled_shape[::-1])

            squeeze_indices = tuple(
                i for i, size in enumerate(noncore_shape) if size == 1
            )
            squeezed_arg = ops.squeeze(arg, axis=squeeze_indices)
            squeezed_args.append(squeezed_arg)

        vectorized_func = checked_func
        dims_to_expand = []
        for negdim, axis_sizes in enumerate(zip(*rev_filled_shapes)):
            in_axes = tuple(None if size == 1 else 0 for size in axis_sizes)
            if all(axis is None for axis in in_axes):
                dims_to_expand.append(len(broadcast_shape) - 1 - negdim)
            else:
                vectorized_func = vmap_fn(vectorized_func, in_axes)
        result = vectorized_func(*squeezed_args)

        if not dims_to_expand:
            return result
        elif isinstance(result, tuple):
            return tuple(
                ops.expand_dims(r, axis=dims_to_expand) for r in result
            )
        else:
            return ops.expand_dims(result, axis=dims_to_expand)

    return wrapped
