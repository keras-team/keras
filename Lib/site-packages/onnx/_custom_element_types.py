# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0
"""This module defines custom dtypes not supported by numpy.
Function :func:`onnx.numpy_helper.from_array`
and :func:`onnx.numpy_helper.to_array` are using them
to convert arrays from/to these types.
Class :class:`onnx.reference.ReferenceEvalutor` also uses them.
To create such an array for unit test for example, it is convenient to write
something like the following:

.. exec_code::

    import numpy as np
    from onnx import TensorProto
    from onnx.reference.ops.op_cast import Cast_19 as Cast

    tensor_bfloat16 = Cast.eval(np.array([0, 1], dtype=np.float32), to=TensorProto.BFLOAT16)

The numpy representation dtypes used below are meant for internal use. They may change in the
future based on the industry standardization of these numpy types.
"""

from __future__ import annotations

import numpy as np

import onnx

#: Defines a bfloat16 as a uint16.
bfloat16 = np.dtype((np.uint16, {"bfloat16": (np.uint16, 0)}))

#: Defines float 8 e4m3fn type, see See :ref:`onnx-detail-float8` for technical details.
float8e4m3fn = np.dtype((np.uint8, {"e4m3fn": (np.uint8, 0)}))

#: Defines float 8 e4m3fnuz type, see See :ref:`onnx-detail-float8` for technical details.
float8e4m3fnuz = np.dtype((np.uint8, {"e4m3fnuz": (np.uint8, 0)}))

#: Defines float 8 e5m2 type, see See :ref:`onnx-detail-float8` for technical details.
float8e5m2 = np.dtype((np.uint8, {"e5m2": (np.uint8, 0)}))

#: Defines float 8 e5m2fnuz type, see See :ref:`onnx-detail-float8` for technical details.
float8e5m2fnuz = np.dtype((np.uint8, {"e5m2fnuz": (np.uint8, 0)}))

#: Defines int4, see See :ref:`onnx-detail-int4` for technical details.
#: Do note that one integer is stored using a byte and therefore is twice bigger
#: than its onnx size.
uint4 = np.dtype((np.uint8, {"uint4": (np.uint8, 0)}))

#: Defines int4, see See :ref:`onnx-detail-int4` for technical details.
#: Do note that one integer is stored using a byte and therefore is twice bigger
#: than its onnx size.
int4 = np.dtype((np.int8, {"int4": (np.int8, 0)}))

mapping_name_to_data_type = {
    "bfloat16": onnx.TensorProto.BFLOAT16,
    "e4m3fn": onnx.TensorProto.FLOAT8E4M3FN,
    "e4m3fnuz": onnx.TensorProto.FLOAT8E4M3FNUZ,
    "e5m2": onnx.TensorProto.FLOAT8E5M2,
    "e5m2fnuz": onnx.TensorProto.FLOAT8E5M2FNUZ,
    "int4": onnx.TensorProto.INT4,
    "uint4": onnx.TensorProto.UINT4,
}
