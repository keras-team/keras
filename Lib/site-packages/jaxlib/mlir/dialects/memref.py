#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
import operator
from itertools import accumulate
from typing import Optional

from ._memref_ops_gen import *
from ._ods_common import _dispatch_mixed_values, MixedValues
from .arith import ConstantOp, _is_integer_like_type
from ..ir import Value, MemRefType, StridedLayoutAttr, ShapedType, Operation


def _is_constant_int_like(i):
    return (
        isinstance(i, Value)
        and isinstance(i.owner, Operation)
        and isinstance(i.owner.opview, ConstantOp)
        and _is_integer_like_type(i.type)
    )


def _is_static_int_like(i):
    return (
        isinstance(i, int) and not ShapedType.is_dynamic_size(i)
    ) or _is_constant_int_like(i)


def _infer_memref_subview_result_type(
    source_memref_type, offsets, static_sizes, static_strides
):
    source_strides, source_offset = source_memref_type.get_strides_and_offset()
    # "canonicalize" from tuple|list -> list
    offsets, static_sizes, static_strides, source_strides = map(
        list, (offsets, static_sizes, static_strides, source_strides)
    )

    if not all(
        all(_is_static_int_like(i) for i in s)
        for s in [
            static_sizes,
            static_strides,
            source_strides,
        ]
    ):
        raise ValueError(
            "Only inferring from python or mlir integer constant is supported."
        )

    for s in [offsets, static_sizes, static_strides]:
        for idx, i in enumerate(s):
            if _is_constant_int_like(i):
                s[idx] = i.owner.opview.literal_value

    if any(not _is_static_int_like(i) for i in offsets + [source_offset]):
        target_offset = ShapedType.get_dynamic_size()
    else:
        target_offset = source_offset
        for offset, target_stride in zip(offsets, source_strides):
            target_offset += offset * target_stride

    target_strides = []
    for source_stride, static_stride in zip(source_strides, static_strides):
        target_strides.append(source_stride * static_stride)

    # If default striding then no need to complicate things for downstream ops (e.g., expand_shape).
    default_strides = list(accumulate(static_sizes[1:][::-1], operator.mul))[::-1] + [1]
    if target_strides == default_strides and target_offset == 0:
        layout = None
    else:
        layout = StridedLayoutAttr.get(target_offset, target_strides)
    return (
        offsets,
        static_sizes,
        static_strides,
        MemRefType.get(
            static_sizes,
            source_memref_type.element_type,
            layout,
            source_memref_type.memory_space,
        ),
    )


_generated_subview = subview


def subview(
    source: Value,
    offsets: MixedValues,
    sizes: MixedValues,
    strides: MixedValues,
    *,
    result_type: Optional[MemRefType] = None,
    loc=None,
    ip=None,
):
    if offsets is None:
        offsets = []
    if sizes is None:
        sizes = []
    if strides is None:
        strides = []
    source_strides, source_offset = source.type.get_strides_and_offset()
    if result_type is None and all(
        all(_is_static_int_like(i) for i in s) for s in [sizes, strides, source_strides]
    ):
        # If any are arith.constant results then this will canonicalize to python int
        # (which can then be used to fully specify the subview).
        (
            offsets,
            sizes,
            strides,
            result_type,
        ) = _infer_memref_subview_result_type(source.type, offsets, sizes, strides)
    elif result_type is None:
        raise ValueError(
            "mixed static/dynamic offset/sizes/strides requires explicit result type."
        )

    offsets, _packed_offsets, static_offsets = _dispatch_mixed_values(offsets)
    sizes, _packed_sizes, static_sizes = _dispatch_mixed_values(sizes)
    strides, _packed_strides, static_strides = _dispatch_mixed_values(strides)

    return _generated_subview(
        result_type,
        source,
        offsets,
        sizes,
        strides,
        static_offsets,
        static_sizes,
        static_strides,
        loc=loc,
        ip=ip,
    )
