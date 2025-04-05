#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from ._llvm_ops_gen import *
from ._llvm_enum_gen import *
from .._mlir_libs._mlirDialectsLLVM import *
from ..ir import Value
from ._ods_common import get_op_result_or_op_results as _get_op_result_or_op_results


def mlir_constant(value, *, loc=None, ip=None) -> Value:
    return _get_op_result_or_op_results(
        ConstantOp(res=value.type, value=value, loc=loc, ip=ip)
    )
