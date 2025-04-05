# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import itertools
import math
from typing import Sequence

import numpy as np

from onnx.reference.op_run import OpRun


def get_pad_shape(
    auto_pad: str,
    input_spatial_shape: Sequence[int],
    kernel_spatial_shape: Sequence[int],
    strides_spatial: Sequence[int],
    output_spatial_shape: Sequence[int],
) -> Sequence[int]:
    spatial_dims = len(input_spatial_shape)
    pad_shape = [0] * spatial_dims
    strides_spatial = strides_spatial or [1] * spatial_dims
    if auto_pad in ("SAME_UPPER", "SAME_LOWER"):
        for i in range(spatial_dims):
            pad_shape[i] = (
                (output_spatial_shape[i] - 1) * strides_spatial[i]
                + kernel_spatial_shape[i]
                - input_spatial_shape[i]
            )
    elif auto_pad == "VALID":
        pass

    return pad_shape


def get_pad_with_auto_pad(auto_pad: str, pad_shape: Sequence[int]) -> Sequence[int]:
    spatial_dims = len(pad_shape)
    if auto_pad == "SAME_UPPER":
        pads = [pad_shape[i] // 2 for i in range(spatial_dims)] + [
            pad_shape[i] - pad_shape[i] // 2 for i in range(spatial_dims)
        ]
    elif auto_pad == "SAME_LOWER":
        pads = [pad_shape[i] - pad_shape[i] // 2 for i in range(spatial_dims)] + [
            pad_shape[i] // 2 for i in range(spatial_dims)
        ]
    else:
        pads = [0] * spatial_dims * 2  # no padding
    return pads


def get_output_shape_explicit_padding(
    pads: Sequence[int],
    input_spatial_shape: Sequence[int],
    kernel_spatial_shape: Sequence[int],
    strides_spatial: Sequence[int],
    dilations: Sequence[int] | None = None,
    ceil_mode: bool = False,
) -> tuple[Sequence[int], Sequence[int]]:
    """Compute output shape according to:
    https://pytorch.org/docs/stable/generated/torch.nn.MaxPool1d.html?highlight=max+pool#torch.nn.MaxPool1d
    Pads are used to calculate output shape. Use output shape in turn to calculate the actual pads
    that are used to pad the input tensor so that computation in pool() will not cause out of bound error.
    Here is the detail. Thinking kernel as a sliding window, its size:
    sw = dilation * (kernel - 1) + 1
    l_out = (l_in + pad[0] + pad[1] - sw) / stride + 1 # (ceiled if ceil_mode is True)
    l_in_required = (l_out - 1) * stride + sw

    l_in_required is used to for computation in pool() which may be larger than padded l_in, because of ceiling.
    as an example, l_in = 3, kernel = 2, stride = 2, dilation = 1, pad = [0, 0], then
    sw = dilation * (kernel - 1) + 1 = 1 * (2 - 1) + 1 = 2
    l_out = ceil((l_in + pad[0] + pad[1] - sw) / stride + 1) = ceil((3 + 0 + 0 - 1 * (2 - 1) - 1) / 2 + 1) = 2
    l_in_required = (l_out - 1) * stride + sw = (2 - 1) * 2 + 2 = 4
    l_in_required (= 4) is not equal to l_in (= 3), so we need to pad the input tensor to l_in_required to make sure that
    the sliding window does not go out-of-bound w.r.t. input tensor. Otherwise pool() will fail.
    """
    output_spatial_shape = [0] * len(input_spatial_shape)
    pads = pads or [0] * len(input_spatial_shape) * 2
    strides_spatial = strides_spatial or [1] * len(input_spatial_shape)
    dims = len(input_spatial_shape)
    if dilations is None:
        dilations = np.ones([dims], dtype=np.int64)

    for dim in range(dims):
        dim_size = (
            input_spatial_shape[dim]
            + pads[dim]
            + pads[dims + dim]
            - dilations[dim] * (kernel_spatial_shape[dim] - 1)
            - 1
        ) / strides_spatial[dim] + 1

        if ceil_mode:
            output_spatial_shape[dim] = int(np.ceil(dim_size))
        else:
            output_spatial_shape[dim] = int(np.floor(dim_size))

    pads_spatial_shape_new = pads[:]
    for dim in range(dims):
        sliding_window_size = (kernel_spatial_shape[dim] - 1) * dilations[dim] + 1
        actual_padded_input_size = (output_spatial_shape[dim] - 1) * strides_spatial[
            dim
        ] + sliding_window_size
        extra_pad = (
            actual_padded_input_size
            - input_spatial_shape[dim]
            - pads[dim]
            - pads[dims + dim]
        )
        if extra_pad > 0:
            pads_spatial_shape_new[dim] += extra_pad // 2
            pads_spatial_shape_new[dims + dim] += extra_pad - extra_pad // 2

    return output_spatial_shape, pads_spatial_shape_new


def get_output_shape_auto_pad(
    auto_pad: str,
    input_spatial_shape: Sequence[int],
    kernel_spatial_shape: Sequence[int],
    strides_spatial: Sequence[int],
) -> Sequence[int]:
    """https://www.tensorflow.org/api_docs/python/tf/keras/layers/AveragePooling2D
    output_shape = math.floor((input_shape - 1) / strides) + 1  (SAME)
    output_shape = math.floor((input_shape - pool_size) / strides) + 1 (VALID)
    IMPORTANT: this function assumes ceil_mode is False. In tenforflow, ceil_mode is always False.
    However, ONNX spec allow ceil_mode to be True because ORT does handle the case.
    """
    strides_spatial = strides_spatial or [1] * len(input_spatial_shape)
    out_shape = [0] * len(input_spatial_shape)
    for i in range(len(input_spatial_shape)):
        if auto_pad in ("SAME_UPPER", "SAME_LOWER"):
            out_shape[i] = (
                math.floor((input_spatial_shape[i] - 1) / strides_spatial[i]) + 1
            )
        elif auto_pad == "VALID":
            out_shape[i] = (
                math.floor(
                    (input_spatial_shape[i] - kernel_spatial_shape[i])
                    / strides_spatial[i]
                )
                + 1
            )
        # if auto_pad is NOTSET, explicite padding should be used
        else:
            raise ValueError(
                "auto_pad can only be NOTSET, SAME_UPPER, SAME_LOWER, or VALID"
            )
    # pads = get_pad_shape(auto_pad, input_spatial_shape, kernel_shape, strides_spatial, out_shape)

    return out_shape


def lp_pool(x: np.array, p: int) -> float:
    y = 0
    for v in np.nditer(x):
        y += abs(v) ** p
    return y ** (1.0 / p)


def pool(
    padded: np.ndarray,
    x_shape: Sequence[int],
    kernel: Sequence[int],
    strides: Sequence[int],
    out_shape: Sequence[int],
    pooling_type: str,
    pads: Sequence[int] | None = None,
    dilations: Sequence[int] | None = None,
    count_include_pad: int = 0,
    p: int = 1,
) -> np.ndarray:
    """This function is used to calculate the pooling result of a padded tensor
    padded: the padded tensor
    x_shape: the shape of the original tensor in [N, C, *spatial_shape]
    kernel: the pooling kernel
    strides: the strides
    out_shape: the shape of the output tensor
    pooling_type: the pooling type, can be "AVG", "LPPOOL", or "MAX"
    pads: the padding in an order of head_pad_1, head_pad_2, ..., tail_pad_1, tail_pad_2, ...
    dilations: the dilation
    count_include_pad: whether to include the padding in the calculation of average and lp pooling
    p: the p value for lp pooling
    """
    spatial_size = len(x_shape) - 2
    y = np.zeros([x_shape[0], x_shape[1], *list(out_shape)], dtype=padded.dtype)
    if dilations is None:
        dilations = np.ones([spatial_size], dtype=np.int64)
    if pads is None:
        pads = np.zeros([spatial_size * 2], dtype=np.int64)
    elif len(pads) == 1:
        pads = pads * spatial_size * 2
    strides = strides or [1] * spatial_size

    def lp_pool_p(x):
        return lp_pool(x, p)

    for shape in itertools.product(
        range(x_shape[0]),
        range(x_shape[1]),
        *[
            range(
                int(
                    (
                        x_shape[i + 2]
                        + pads[i]
                        + pads[i + spatial_size]
                        - (1 + (kernel[i] - 1) * dilations[i])
                    )
                    / strides[i]
                    + 1
                )
            )
            for i in range(spatial_size)
        ],
    ):
        window = padded[shape[0], shape[1]]
        window_vals = np.array(
            [
                window[i]
                for i in list(
                    itertools.product(
                        *[
                            range(
                                strides[i] * shape[i + 2],
                                strides[i] * shape[i + 2]
                                + (1 + (kernel[i] - 1) * dilations[i]),
                                dilations[i],
                            )
                            for i in range(spatial_size)
                        ]
                    )
                )
            ]
        )
        if pooling_type == "AVG":
            f = np.average
        elif pooling_type == "MAX":
            f = np.max
        elif pooling_type == "LPPOOL":
            f = lp_pool_p
        else:
            raise NotImplementedError(
                f"Pooling type {pooling_type} does not support. Should be AVG, MAX"
            )

        if count_include_pad == 1 and (pooling_type in {"AVG", "LPPOOL"}):
            y[shape] = f(window_vals)
        else:
            y[shape] = f(window_vals[np.where(~np.isnan(window_vals))])
    return y.astype(padded.dtype)


class CommonPool(OpRun):
    def _run(
        self,
        pooling_type,
        count_include_pad,
        x,
        auto_pad=None,
        ceil_mode=None,
        dilations=None,
        kernel_shape=None,
        pads=None,
        strides=None,
        p=None,
    ):
        x_shape = np.shape(x)
        pading_value = np.nan if pooling_type == "MAX" or count_include_pad == 0 else 0

        if auto_pad in ["SAME_UPPER", "SAME_LOWER", "VALID"]:
            assert (
                ceil_mode is None or ceil_mode == 0
            ), "ceil_mode is not supported with auto_pad"
            out_shape = get_output_shape_auto_pad(
                auto_pad, x.shape[2:], kernel_shape, strides
            )
            pads_shape = get_pad_shape(
                auto_pad, x_shape[2:], kernel_shape, strides, out_shape
            )
            pads = get_pad_with_auto_pad(auto_pad, pads_shape)
            n_dims = len(pads) // 2
            pads_np = [(pads[i], pads[i + n_dims]) for i in range(n_dims)]
            padded = np.pad(
                x,
                ((0, 0), (0, 0), *pads_np),
                mode="constant",
                constant_values=pading_value,
            )
            y = pool(
                padded,
                x_shape,
                kernel_shape,
                strides,
                out_shape,
                pooling_type,
                pads,
                dilations,
                count_include_pad,
                p,
            )
            return (y,)
        else:
            out_shape, pads = get_output_shape_explicit_padding(
                pads, x_shape[2:], kernel_shape, strides, dilations, ceil_mode
            )
            # convert pads from [x1_begin, x2_begin,...,x1_end, x2_end,...] to [(x1_begin, x1_end), (x2_begin, x2_end),...]
            n_dims = len(pads) // 2
            pads_np = [(pads[i], pads[i + n_dims]) for i in range(n_dims)]
            padded = np.pad(
                x,
                ((0, 0), (0, 0), *pads_np),
                mode="constant",
                constant_values=pading_value,
            )
            y = pool(
                padded,
                x_shape,
                kernel_shape,
                strides,
                out_shape,
                pooling_type,
                pads,
                dilations,
                count_include_pad,
                p,
            )
            return (y,)
