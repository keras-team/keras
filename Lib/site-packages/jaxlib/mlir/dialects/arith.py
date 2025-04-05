#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from ._arith_ops_gen import *
from ._arith_ops_gen import _Dialect
from ._arith_enum_gen import *
from array import array as _array
from typing import overload

try:
    from ..ir import *
    from ._ods_common import (
        get_default_loc_context as _get_default_loc_context,
        _cext as _ods_cext,
        get_op_result_or_op_results as _get_op_result_or_op_results,
    )

    from typing import Any, List, Union
except ImportError as e:
    raise RuntimeError("Error loading imports from extension module") from e


def _isa(obj: Any, cls: type):
    try:
        cls(obj)
    except ValueError:
        return False
    return True


def _is_any_of(obj: Any, classes: List[type]):
    return any(_isa(obj, cls) for cls in classes)


def _is_integer_like_type(type: Type):
    return _is_any_of(type, [IntegerType, IndexType])


def _is_float_type(type: Type):
    return _is_any_of(type, [BF16Type, F16Type, F32Type, F64Type])


@_ods_cext.register_operation(_Dialect, replace=True)
class ConstantOp(ConstantOp):
    """Specialization for the constant op class."""

    @overload
    def __init__(self, value: Attribute, *, loc=None, ip=None):
        ...

    @overload
    def __init__(
        self, result: Type, value: Union[int, float, _array], *, loc=None, ip=None
    ):
        ...

    def __init__(self, result, value, *, loc=None, ip=None):
        if value is None:
            assert isinstance(result, Attribute)
            super().__init__(result, loc=loc, ip=ip)
            return

        if isinstance(value, int):
            super().__init__(IntegerAttr.get(result, value), loc=loc, ip=ip)
        elif isinstance(value, float):
            super().__init__(FloatAttr.get(result, value), loc=loc, ip=ip)
        elif isinstance(value, _array):
            if 8 * value.itemsize != result.element_type.width:
                raise ValueError(
                    f"Mismatching array element ({8 * value.itemsize}) and type ({result.element_type.width}) width."
                )
            if value.typecode in ["i", "l", "q"]:
                super().__init__(DenseIntElementsAttr.get(value, type=result))
            elif value.typecode in ["f", "d"]:
                super().__init__(DenseFPElementsAttr.get(value, type=result))
            else:
                raise ValueError(f'Unsupported typecode: "{value.typecode}".')
        else:
            super().__init__(value, loc=loc, ip=ip)

    @classmethod
    def create_index(cls, value: int, *, loc=None, ip=None):
        """Create an index-typed constant."""
        return cls(
            IndexType.get(context=_get_default_loc_context(loc)), value, loc=loc, ip=ip
        )

    @property
    def type(self):
        return self.results[0].type

    @property
    def value(self):
        return Attribute(self.operation.attributes["value"])

    @property
    def literal_value(self) -> Union[int, float]:
        if _is_integer_like_type(self.type):
            return IntegerAttr(self.value).value
        elif _is_float_type(self.type):
            return FloatAttr(self.value).value
        else:
            raise ValueError("only integer and float constants have literal values")


def constant(
    result: Type, value: Union[int, float, Attribute, _array], *, loc=None, ip=None
) -> Value:
    return _get_op_result_or_op_results(ConstantOp(result, value, loc=loc, ip=ip))
