# Copyright (c) ONNX Project Contributors

# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import numpy as np

from onnx import TensorProto
from onnx._custom_element_types import (
    float8e4m3fn,
    float8e4m3fnuz,
    float8e5m2,
    float8e5m2fnuz,
    int4,
    uint4,
)
from onnx.helper import np_dtype_to_tensor_dtype
from onnx.numpy_helper import float8e4m3_to_float32, float8e5m2_to_float32
from onnx.reference.op_run import OpRun
from onnx.reference.ops.op_quantize_linear import reshape_input


class _CommonDequantizeLinear(OpRun):
    def get_x_type(self, x: np.ndarray) -> int:
        tensor_dtype = None
        if x.dtype == float8e4m3fn and x.dtype.descr[0][0] == "e4m3fn":
            tensor_dtype = TensorProto.FLOAT8E4M3FN
        elif x.dtype == float8e4m3fnuz and x.dtype.descr[0][0] == "e4m3fnuz":
            tensor_dtype = TensorProto.FLOAT8E4M3FNUZ
        elif x.dtype == float8e5m2 and x.dtype.descr[0][0] == "e5m2":
            tensor_dtype = TensorProto.FLOAT8E5M2
        elif x.dtype == float8e5m2fnuz and x.dtype.descr[0][0] == "e5m2fnuz":
            tensor_dtype = TensorProto.FLOAT8E5M2FNUZ
        elif x.dtype == uint4 and x.dtype.descr[0][0] == "uint4":
            tensor_dtype = TensorProto.UINT4
        elif x.dtype == int4 and x.dtype.descr[0][0] == "int4":
            tensor_dtype = TensorProto.INT4
        else:
            tensor_dtype = np_dtype_to_tensor_dtype(x.dtype)
        return tensor_dtype

    def _run(
        self,
        x: np.ndarray,
        x_scale: np.ndarray,
        x_zero_point: np.ndarray | None = None,
        axis: int | None = None,
        block_size: int | None = None,
    ):  # type: ignore
        x_type = self.get_x_type(x)
        fp8_type = x_type in {
            TensorProto.FLOAT8E4M3FN,
            TensorProto.FLOAT8E4M3FNUZ,
            TensorProto.FLOAT8E5M2,
            TensorProto.FLOAT8E5M2FNUZ,
        }
        if x_zero_point is not None and not fp8_type:
            zero_type = self.get_x_type(x_zero_point)
            if x_type != zero_type:
                raise ValueError(
                    f"Type mismatch {x_type} != {zero_type} in DequantizeLinear."
                )

            dx = x.astype(np.float32) - reshape_input(
                x_zero_point, x.shape, axis, block_size
            )
        else:
            if fp8_type and x_zero_point is not None:
                u_x_zero_point = x_zero_point.astype(np.uint8)
                umi = u_x_zero_point.min()
                uma = u_x_zero_point.max()
                if umi != uma or umi != np.uint8(0):
                    raise ValueError(
                        "x_zero_point is not null but should be zero for float8 types."
                    )
            if x_type == TensorProto.FLOAT8E4M3FN:
                dx = float8e4m3_to_float32(x)
            elif x_type == TensorProto.FLOAT8E4M3FNUZ:
                dx = float8e4m3_to_float32(x, uz=True)
            elif x_type == TensorProto.FLOAT8E5M2:
                dx = float8e5m2_to_float32(x)
            elif x_type == TensorProto.FLOAT8E5M2FNUZ:
                dx = float8e5m2_to_float32(x, fn=True, uz=True)
            else:
                dx = x.astype(np.float32)
        y = dx * reshape_input(x_scale, x.shape, axis, block_size)
        return (y.astype(x_scale.dtype),)


class DequantizeLinear_19(_CommonDequantizeLinear):
    def _run(self, x, x_scale, x_zero_point=None, axis=None):
        if len(x_scale.shape) > 1:
            raise ValueError("Input 2 must be a vector or a number.")
        return super()._run(x, x_scale, x_zero_point, axis)


class DequantizeLinear_21(_CommonDequantizeLinear):
    def _run(self, *args, axis=None, block_size=None):  # type: ignore
        # args: x, y_scale, zero_point
        return super()._run(*args, axis=axis, block_size=block_size)  # type: ignore
