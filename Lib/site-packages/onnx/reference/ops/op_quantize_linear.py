# Copyright (c) ONNX Project Contributors

# SPDX-License-Identifier: Apache-2.0


from __future__ import annotations

from typing import ClassVar

import numpy as np

from onnx import TensorProto, subbyte
from onnx._custom_element_types import (
    float8e4m3fn,
    float8e4m3fnuz,
    float8e5m2,
    float8e5m2fnuz,
    int4,
    uint4,
)
from onnx.helper import (
    float32_to_float8e4m3,
    float32_to_float8e5m2,
    np_dtype_to_tensor_dtype,
    tensor_dtype_to_np_dtype,
)
from onnx.reference.op_run import OpRun


def reshape_input(
    value: np.ndarray,
    shape: tuple[int, ...],
    axis: int | None = None,
    block_size: int | None = None,
) -> np.ndarray:
    """Reshape/Replicate scale/zero-point to be broadcastable to shape.

    Args:
        value: the array to be reshaped/replicated
        shape: the rarget shape
        axis: quantization axis, applicable for per-axis and blocked quantization
        block_size: size of quantization block, applicable only for blocked quantization

    Returns:
        value array after reshape/replicate according to quantization mode.
    """
    if len(value.shape) == 0:
        return value
    if len(value.shape) > 0 and value.size == 1:
        return value[0]
    if not block_size:
        assert len(value.shape) == 1
        dims = [1] * len(shape)
        try:
            dims[axis] = value.size
            return value.reshape(tuple(dims))
        except IndexError as e:
            raise IndexError(
                f"axis is out of boundary, axis={axis}, "
                f"value.shape={value.shape}, shape={shape}."
            ) from e

    if block_size <= 0:
        raise ValueError("block_size must be a positive integer.")

    # repeat scale to get elementwise scale
    value = np.repeat(value, repeats=block_size, axis=axis)
    if (
        shape[axis] != value.shape[axis]
    ):  # block_size does not divide x, handle the remainder block
        value = value.take(indices=range(shape[axis]), axis=axis)
    if value.shape != shape:
        raise ValueError(
            "Invalid shapes for Blocked Quantization. Input 2 shape should identical to Input 1 shape, except for one dimension, in which blocking is performed"
        )
    assert np.broadcast_shapes(shape, value.shape) == shape
    return value


class _CommonQuantizeLinear(OpRun):
    float32_to_float8e4m3 = np.vectorize(float32_to_float8e4m3)
    float32_to_float8e5m2 = np.vectorize(float32_to_float8e5m2)
    quant_integer_ranges: ClassVar[dict[TensorProto.DataType, tuple[int]]] = {
        TensorProto.UINT8: (0, 255),
        TensorProto.INT8: (-128, 127),
        TensorProto.UINT16: (0, 65535),
        TensorProto.INT16: (-32768, 32767),
    }
    quant_types = (
        TensorProto.UINT8,
        TensorProto.INT8,
        TensorProto.UINT16,
        TensorProto.INT16,
        TensorProto.UINT4,
        TensorProto.INT4,
        TensorProto.FLOAT8E4M3FN,
        TensorProto.FLOAT8E4M3FNUZ,
        TensorProto.FLOAT8E5M2,
        TensorProto.FLOAT8E5M2FNUZ,
    )

    def get_zero_point_type(self, zero_point: np.ndarray) -> int:
        zero_point_type = None
        if (
            zero_point.dtype == float8e4m3fn
            and zero_point.dtype.descr[0][0] == "e4m3fn"
        ):
            zero_point_type = TensorProto.FLOAT8E4M3FN
        elif (
            zero_point.dtype == float8e4m3fnuz
            and zero_point.dtype.descr[0][0] == "e4m3fnuz"
        ):
            zero_point_type = TensorProto.FLOAT8E4M3FNUZ
        elif zero_point.dtype == float8e5m2 and zero_point.dtype.descr[0][0] == "e5m2":
            zero_point_type = TensorProto.FLOAT8E5M2
        elif (
            zero_point.dtype == float8e5m2fnuz
            and zero_point.dtype.descr[0][0] == "e5m2fnuz"
        ):
            zero_point_type = TensorProto.FLOAT8E5M2FNUZ
        elif zero_point.dtype == uint4 and zero_point.dtype.descr[0][0] == "uint4":
            zero_point_type = TensorProto.UINT4
        elif zero_point.dtype == int4 and zero_point.dtype.descr[0][0] == "int4":
            zero_point_type = TensorProto.INT4
        else:
            zero_point_type = np_dtype_to_tensor_dtype(zero_point.dtype)
        return zero_point_type

    def _run(
        self,
        x: np.ndarray,
        y_scale: np.ndarray,
        zero_point: np.ndarray | None = None,
        axis: int = 1,
        saturate: bool = True,
        block_size: int | None = None,
        output_dtype: np.dtype | None = None,
    ) -> tuple[np.ndarray]:
        y_scale = reshape_input(y_scale, x.shape, axis, block_size)

        # Determine output data type
        tensor_type = output_dtype
        if zero_point is not None:
            zero_point_type = self.get_zero_point_type(zero_point)
            if output_dtype and output_dtype != zero_point_type:
                raise ValueError(
                    f"Mismatched output data-types: output_dtype={output_dtype}, zero_point type={zero_point_type}"
                )
            tensor_type = zero_point_type
        tensor_type = tensor_type or TensorProto.UINT8

        if tensor_type not in _CommonQuantizeLinear.quant_types:
            raise ValueError(
                f"Unexpected type: output_dtype={tensor_type} is not a supported quantized type."
            )

        # Compute
        zero_point = (
            reshape_input(zero_point, x.shape, axis, block_size)
            if zero_point is not None
            else 0
        )
        x = x / y_scale

        if tensor_type in _CommonQuantizeLinear.quant_integer_ranges:
            xi = np.rint(x).astype(np.int32)
            xi += zero_point
            dtype = tensor_dtype_to_np_dtype(tensor_type)
            quant_range = _CommonQuantizeLinear.quant_integer_ranges[tensor_type]
            return (np.clip(xi, quant_range[0], quant_range[1]).astype(dtype),)

        if tensor_type == TensorProto.FLOAT8E4M3FN:
            f8 = _CommonQuantizeLinear.float32_to_float8e4m3(x, saturate=saturate)
            return (f8.astype(float8e4m3fn),)  # type: ignore[attr-defined]

        if tensor_type == TensorProto.FLOAT8E4M3FNUZ:
            f8 = _CommonQuantizeLinear.float32_to_float8e4m3(
                x, uz=True, saturate=saturate
            )
            return (f8.astype(float8e4m3fnuz),)  # type: ignore[attr-defined]

        if tensor_type == TensorProto.FLOAT8E5M2:
            f8 = _CommonQuantizeLinear.float32_to_float8e5m2(x, saturate=saturate)
            return (f8.astype(float8e5m2),)  # type: ignore[attr-defined]

        if tensor_type == TensorProto.FLOAT8E5M2FNUZ:
            f8 = _CommonQuantizeLinear.float32_to_float8e5m2(
                x, fn=True, uz=True, saturate=saturate
            )
            return (f8.astype(float8e5m2fnuz),)  # type: ignore[attr-defined]

        if tensor_type in (TensorProto.UINT4, TensorProto.INT4):
            xi = np.rint(x).astype(np.int32)
            xi += zero_point
            single_func = lambda x: subbyte.float32_to_4bit_unpacked(  # noqa: E731
                x, signed=(tensor_type == TensorProto.INT4)
            )
            func = np.vectorize(single_func)
            i4 = func(xi)
            return (i4,)  # type: ignore[attr-defined]

        raise ValueError(
            f"Unexpected type: output_dtype={tensor_type} is not a supported quantized type."
        )


class QuantizeLinear_10(_CommonQuantizeLinear):
    def _run(self, x, y_scale, zero_point=None, axis=None):  # type: ignore
        if len(y_scale.shape) > 1:
            raise ValueError("Input 2 must be a vector or a number.")
        return super()._run(x, y_scale, zero_point, axis=axis)  # type: ignore


class QuantizeLinear_19(_CommonQuantizeLinear):
    def _run(self, x, y_scale, zero_point=None, axis=None, saturate=None):  # type: ignore
        if len(y_scale.shape) > 1:
            raise ValueError("Input 2 must be a vector or a number.")
        return super()._run(x, y_scale, zero_point, axis=axis, saturate=saturate)  # type: ignore


class QuantizeLinear_21(_CommonQuantizeLinear):
    def _run(self, *args, axis=None, saturate=None, block_size=None, output_dtype=None):  # type: ignore
        # args: x, y_scale, zero_point
        return super()._run(
            *args,
            axis=axis,
            saturate=saturate,
            block_size=block_size,
            output_dtype=output_dtype,
        )  # type: ignore
