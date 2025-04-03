#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from ._mlir_libs._mlir.ir import *
from ._mlir_libs._mlir.ir import _GlobalDebug
from ._mlir_libs._mlir import register_type_caster, register_value_caster
from ._mlir_libs import (
    get_dialect_registry,
    append_load_on_create_dialect,
    get_load_on_create_dialects,
)


# Convenience decorator for registering user-friendly Attribute builders.
def register_attribute_builder(kind, replace=False):
    def decorator_builder(func):
        AttrBuilder.insert(kind, func, replace=replace)
        return func

    return decorator_builder


@register_attribute_builder("AffineMapAttr")
def _affineMapAttr(x, context):
    return AffineMapAttr.get(x)


@register_attribute_builder("IntegerSetAttr")
def _integerSetAttr(x, context):
    return IntegerSetAttr.get(x)


@register_attribute_builder("BoolAttr")
def _boolAttr(x, context):
    return BoolAttr.get(x, context=context)


@register_attribute_builder("DictionaryAttr")
def _dictAttr(x, context):
    return DictAttr.get(x, context=context)


@register_attribute_builder("IndexAttr")
def _indexAttr(x, context):
    return IntegerAttr.get(IndexType.get(context=context), x)


@register_attribute_builder("I1Attr")
def _i1Attr(x, context):
    return IntegerAttr.get(IntegerType.get_signless(1, context=context), x)


@register_attribute_builder("I8Attr")
def _i8Attr(x, context):
    return IntegerAttr.get(IntegerType.get_signless(8, context=context), x)


@register_attribute_builder("I16Attr")
def _i16Attr(x, context):
    return IntegerAttr.get(IntegerType.get_signless(16, context=context), x)


@register_attribute_builder("I32Attr")
def _i32Attr(x, context):
    return IntegerAttr.get(IntegerType.get_signless(32, context=context), x)


@register_attribute_builder("I64Attr")
def _i64Attr(x, context):
    return IntegerAttr.get(IntegerType.get_signless(64, context=context), x)


@register_attribute_builder("SI1Attr")
def _si1Attr(x, context):
    return IntegerAttr.get(IntegerType.get_signed(1, context=context), x)


@register_attribute_builder("SI8Attr")
def _si8Attr(x, context):
    return IntegerAttr.get(IntegerType.get_signed(8, context=context), x)


@register_attribute_builder("SI16Attr")
def _si16Attr(x, context):
    return IntegerAttr.get(IntegerType.get_signed(16, context=context), x)


@register_attribute_builder("SI32Attr")
def _si32Attr(x, context):
    return IntegerAttr.get(IntegerType.get_signed(32, context=context), x)


@register_attribute_builder("SI64Attr")
def _si64Attr(x, context):
    return IntegerAttr.get(IntegerType.get_signed(64, context=context), x)


@register_attribute_builder("UI1Attr")
def _ui1Attr(x, context):
    return IntegerAttr.get(IntegerType.get_unsigned(1, context=context), x)


@register_attribute_builder("UI8Attr")
def _ui8Attr(x, context):
    return IntegerAttr.get(IntegerType.get_unsigned(8, context=context), x)


@register_attribute_builder("UI16Attr")
def _ui16Attr(x, context):
    return IntegerAttr.get(IntegerType.get_unsigned(16, context=context), x)


@register_attribute_builder("UI32Attr")
def _ui32Attr(x, context):
    return IntegerAttr.get(IntegerType.get_unsigned(32, context=context), x)


@register_attribute_builder("UI64Attr")
def _ui64Attr(x, context):
    return IntegerAttr.get(IntegerType.get_unsigned(64, context=context), x)


@register_attribute_builder("F32Attr")
def _f32Attr(x, context):
    return FloatAttr.get_f32(x, context=context)


@register_attribute_builder("F64Attr")
def _f64Attr(x, context):
    return FloatAttr.get_f64(x, context=context)


@register_attribute_builder("StrAttr")
def _stringAttr(x, context):
    return StringAttr.get(x, context=context)


@register_attribute_builder("SymbolNameAttr")
def _symbolNameAttr(x, context):
    return StringAttr.get(x, context=context)


@register_attribute_builder("SymbolRefAttr")
def _symbolRefAttr(x, context):
    if isinstance(x, list):
        return SymbolRefAttr.get(x, context=context)
    else:
        return FlatSymbolRefAttr.get(x, context=context)


@register_attribute_builder("FlatSymbolRefAttr")
def _flatSymbolRefAttr(x, context):
    return FlatSymbolRefAttr.get(x, context=context)


@register_attribute_builder("UnitAttr")
def _unitAttr(x, context):
    if x:
        return UnitAttr.get(context=context)
    else:
        return None


@register_attribute_builder("ArrayAttr")
def _arrayAttr(x, context):
    return ArrayAttr.get(x, context=context)


@register_attribute_builder("AffineMapArrayAttr")
def _affineMapArrayAttr(x, context):
    return ArrayAttr.get([_affineMapAttr(v, context) for v in x])


@register_attribute_builder("BoolArrayAttr")
def _boolArrayAttr(x, context):
    return ArrayAttr.get([_boolAttr(v, context) for v in x])


@register_attribute_builder("DictArrayAttr")
def _dictArrayAttr(x, context):
    return ArrayAttr.get([_dictAttr(v, context) for v in x])


@register_attribute_builder("FlatSymbolRefArrayAttr")
def _flatSymbolRefArrayAttr(x, context):
    return ArrayAttr.get([_flatSymbolRefAttr(v, context) for v in x])


@register_attribute_builder("I32ArrayAttr")
def _i32ArrayAttr(x, context):
    return ArrayAttr.get([_i32Attr(v, context) for v in x])


@register_attribute_builder("I64ArrayAttr")
def _i64ArrayAttr(x, context):
    return ArrayAttr.get([_i64Attr(v, context) for v in x])


@register_attribute_builder("I64SmallVectorArrayAttr")
def _i64SmallVectorArrayAttr(x, context):
    return _i64ArrayAttr(x, context=context)


@register_attribute_builder("IndexListArrayAttr")
def _indexListArrayAttr(x, context):
    return ArrayAttr.get([_i64ArrayAttr(v, context) for v in x])


@register_attribute_builder("F32ArrayAttr")
def _f32ArrayAttr(x, context):
    return ArrayAttr.get([_f32Attr(v, context) for v in x])


@register_attribute_builder("F64ArrayAttr")
def _f64ArrayAttr(x, context):
    return ArrayAttr.get([_f64Attr(v, context) for v in x])


@register_attribute_builder("StrArrayAttr")
def _strArrayAttr(x, context):
    return ArrayAttr.get([_stringAttr(v, context) for v in x])


@register_attribute_builder("SymbolRefArrayAttr")
def _symbolRefArrayAttr(x, context):
    return ArrayAttr.get([_symbolRefAttr(v, context) for v in x])


@register_attribute_builder("DenseF32ArrayAttr")
def _denseF32ArrayAttr(x, context):
    return DenseF32ArrayAttr.get(x, context=context)


@register_attribute_builder("DenseF64ArrayAttr")
def _denseF64ArrayAttr(x, context):
    return DenseF64ArrayAttr.get(x, context=context)


@register_attribute_builder("DenseI8ArrayAttr")
def _denseI8ArrayAttr(x, context):
    return DenseI8ArrayAttr.get(x, context=context)


@register_attribute_builder("DenseI16ArrayAttr")
def _denseI16ArrayAttr(x, context):
    return DenseI16ArrayAttr.get(x, context=context)


@register_attribute_builder("DenseI32ArrayAttr")
def _denseI32ArrayAttr(x, context):
    return DenseI32ArrayAttr.get(x, context=context)


@register_attribute_builder("DenseI64ArrayAttr")
def _denseI64ArrayAttr(x, context):
    return DenseI64ArrayAttr.get(x, context=context)


@register_attribute_builder("DenseBoolArrayAttr")
def _denseBoolArrayAttr(x, context):
    return DenseBoolArrayAttr.get(x, context=context)


@register_attribute_builder("TypeAttr")
def _typeAttr(x, context):
    return TypeAttr.get(x, context=context)


@register_attribute_builder("TypeArrayAttr")
def _typeArrayAttr(x, context):
    return _arrayAttr([TypeAttr.get(t, context=context) for t in x], context)


@register_attribute_builder("MemRefTypeAttr")
def _memref_type_attr(x, context):
    return _typeAttr(x, context)


try:
    import numpy as np

    @register_attribute_builder("F64ElementsAttr")
    def _f64ElementsAttr(x, context):
        return DenseElementsAttr.get(
            np.array(x, dtype=np.float64),
            type=F64Type.get(context=context),
            context=context,
        )

    @register_attribute_builder("I32ElementsAttr")
    def _i32ElementsAttr(x, context):
        return DenseElementsAttr.get(
            np.array(x, dtype=np.int32),
            type=IntegerType.get_signless(32, context=context),
            context=context,
        )

    @register_attribute_builder("I64ElementsAttr")
    def _i64ElementsAttr(x, context):
        return DenseElementsAttr.get(
            np.array(x, dtype=np.int64),
            type=IntegerType.get_signless(64, context=context),
            context=context,
        )

    @register_attribute_builder("IndexElementsAttr")
    def _indexElementsAttr(x, context):
        return DenseElementsAttr.get(
            np.array(x, dtype=np.int64),
            type=IndexType.get(context=context),
            context=context,
        )

except ImportError:
    pass
