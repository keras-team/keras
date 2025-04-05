#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from typing import (
    List as _List,
    Optional as _Optional,
    Sequence as _Sequence,
    Tuple as _Tuple,
    Type as _Type,
    Union as _Union,
)

from .._mlir_libs import _mlir as _cext
from ..ir import (
    ArrayAttr,
    Attribute,
    BoolAttr,
    DenseI64ArrayAttr,
    IntegerAttr,
    IntegerType,
    OpView,
    Operation,
    ShapedType,
    Value,
)

__all__ = [
    "equally_sized_accessor",
    "get_default_loc_context",
    "get_op_result_or_value",
    "get_op_results_or_values",
    "get_op_result_or_op_results",
    "segmented_accessor",
]


def segmented_accessor(elements, raw_segments, idx):
    """
    Returns a slice of elements corresponding to the idx-th segment.

      elements: a sliceable container (operands or results).
      raw_segments: an mlir.ir.Attribute, of DenseI32Array subclass containing
          sizes of the segments.
      idx: index of the segment.
    """
    segments = _cext.ir.DenseI32ArrayAttr(raw_segments)
    start = sum(segments[i] for i in range(idx))
    end = start + segments[idx]
    return elements[start:end]


def equally_sized_accessor(
    elements, n_simple, n_variadic, n_preceding_simple, n_preceding_variadic
):
    """
    Returns a starting position and a number of elements per variadic group
    assuming equally-sized groups and the given numbers of preceding groups.

      elements: a sequential container.
      n_simple: the number of non-variadic groups in the container.
      n_variadic: the number of variadic groups in the container.
      n_preceding_simple: the number of non-variadic groups preceding the current
          group.
      n_preceding_variadic: the number of variadic groups preceding the current
          group.
    """

    total_variadic_length = len(elements) - n_simple
    # This should be enforced by the C++-side trait verifier.
    assert total_variadic_length % n_variadic == 0

    elements_per_group = total_variadic_length // n_variadic
    start = n_preceding_simple + n_preceding_variadic * elements_per_group
    return start, elements_per_group


def get_default_loc_context(location=None):
    """
    Returns a context in which the defaulted location is created. If the location
    is None, takes the current location from the stack, raises ValueError if there
    is no location on the stack.
    """
    if location is None:
        # Location.current raises ValueError if there is no current location.
        return _cext.ir.Location.current.context
    return location.context


def get_op_result_or_value(
    arg: _Union[
        _cext.ir.OpView, _cext.ir.Operation, _cext.ir.Value, _cext.ir.OpResultList
    ]
) -> _cext.ir.Value:
    """Returns the given value or the single result of the given op.

    This is useful to implement op constructors so that they can take other ops as
    arguments instead of requiring the caller to extract results for every op.
    Raises ValueError if provided with an op that doesn't have a single result.
    """
    if isinstance(arg, _cext.ir.OpView):
        return arg.operation.result
    elif isinstance(arg, _cext.ir.Operation):
        return arg.result
    elif isinstance(arg, _cext.ir.OpResultList):
        return arg[0]
    else:
        assert isinstance(arg, _cext.ir.Value)
        return arg


def get_op_results_or_values(
    arg: _Union[
        _cext.ir.OpView,
        _cext.ir.Operation,
        _Sequence[_Union[_cext.ir.OpView, _cext.ir.Operation, _cext.ir.Value]],
    ]
) -> _Union[_Sequence[_cext.ir.Value], _cext.ir.OpResultList]:
    """Returns the given sequence of values or the results of the given op.

    This is useful to implement op constructors so that they can take other ops as
    lists of arguments instead of requiring the caller to extract results for
    every op.
    """
    if isinstance(arg, _cext.ir.OpView):
        return arg.operation.results
    elif isinstance(arg, _cext.ir.Operation):
        return arg.results
    else:
        return [get_op_result_or_value(element) for element in arg]


def get_op_result_or_op_results(
    op: _Union[_cext.ir.OpView, _cext.ir.Operation],
) -> _Union[_cext.ir.Operation, _cext.ir.OpResult, _Sequence[_cext.ir.OpResult]]:
    if isinstance(op, _cext.ir.OpView):
        op = op.operation
    return (
        list(get_op_results_or_values(op))
        if len(op.results) > 1
        else get_op_result_or_value(op)
        if len(op.results) > 0
        else op
    )

ResultValueTypeTuple = _cext.ir.Operation, _cext.ir.OpView, _cext.ir.Value
ResultValueT = _Union[ResultValueTypeTuple]
VariadicResultValueT = _Union[ResultValueT, _Sequence[ResultValueT]]

StaticIntLike = _Union[int, IntegerAttr]
ValueLike = _Union[Operation, OpView, Value]
MixedInt = _Union[StaticIntLike, ValueLike]

IntOrAttrList = _Sequence[_Union[IntegerAttr, int]]
OptionalIntList = _Optional[_Union[ArrayAttr, IntOrAttrList]]

BoolOrAttrList = _Sequence[_Union[BoolAttr, bool]]
OptionalBoolList = _Optional[_Union[ArrayAttr, BoolOrAttrList]]

MixedValues = _Union[_Sequence[_Union[StaticIntLike, ValueLike]], ArrayAttr, ValueLike]

DynamicIndexList = _Sequence[_Union[MixedInt, _Sequence[MixedInt]]]


def _dispatch_dynamic_index_list(
    indices: _Union[DynamicIndexList, ArrayAttr],
) -> _Tuple[_List[ValueLike], _Union[_List[int], ArrayAttr], _List[bool]]:
    """Dispatches a list of indices to the appropriate form.

    This is similar to the custom `DynamicIndexList` directive upstream:
    provided indices may be in the form of dynamic SSA values or static values,
    and they may be scalable (i.e., as a singleton list) or not. This function
    dispatches each index into its respective form. It also extracts the SSA
    values and static indices from various similar structures, respectively.
    """
    dynamic_indices = []
    static_indices = [ShapedType.get_dynamic_size()] * len(indices)
    scalable_indices = [False] * len(indices)

    # ArrayAttr: Extract index values.
    if isinstance(indices, ArrayAttr):
        indices = [idx for idx in indices]

    def process_nonscalable_index(i, index):
        """Processes any form of non-scalable index.

        Returns False if the given index was scalable and thus remains
        unprocessed; True otherwise.
        """
        if isinstance(index, int):
            static_indices[i] = index
        elif isinstance(index, IntegerAttr):
            static_indices[i] = index.value  # pytype: disable=attribute-error
        elif isinstance(index, (Operation, Value, OpView)):
            dynamic_indices.append(index)
        else:
            return False
        return True

    # Process each index at a time.
    for i, index in enumerate(indices):
        if not process_nonscalable_index(i, index):
            # If it wasn't processed, it must be a scalable index, which is
            # provided as a _Sequence of one value, so extract and process that.
            scalable_indices[i] = True
            assert len(index) == 1
            ret = process_nonscalable_index(i, index[0])
            assert ret

    return dynamic_indices, static_indices, scalable_indices


# Dispatches `MixedValues` that all represents integers in various forms into
# the following three categories:
#   - `dynamic_values`: a list of `Value`s, potentially from op results;
#   - `packed_values`: a value handle, potentially from an op result, associated
#                      to one or more payload operations of integer type;
#   - `static_values`: an `ArrayAttr` of `i64`s with static values, from Python
#                      `int`s, from `IntegerAttr`s, or from an `ArrayAttr`.
# The input is in the form for `packed_values`, only that result is set and the
# other two are empty. Otherwise, the input can be a mix of the other two forms,
# and for each dynamic value, a special value is added to the `static_values`.
def _dispatch_mixed_values(
    values: MixedValues,
) -> _Tuple[_List[Value], _Union[Operation, Value, OpView], DenseI64ArrayAttr]:
    dynamic_values = []
    packed_values = None
    static_values = None
    if isinstance(values, ArrayAttr):
        static_values = values
    elif isinstance(values, (Operation, Value, OpView)):
        packed_values = values
    else:
        static_values = []
        for size in values or []:
            if isinstance(size, int):
                static_values.append(size)
            else:
                static_values.append(ShapedType.get_dynamic_size())
                dynamic_values.append(size)
        static_values = DenseI64ArrayAttr.get(static_values)

    return (dynamic_values, packed_values, static_values)


def _get_value_or_attribute_value(
    value_or_attr: _Union[any, Attribute, ArrayAttr]
) -> any:
    if isinstance(value_or_attr, Attribute) and hasattr(value_or_attr, "value"):
        return value_or_attr.value
    if isinstance(value_or_attr, ArrayAttr):
        return _get_value_list(value_or_attr)
    return value_or_attr


def _get_value_list(
    sequence_or_array_attr: _Union[_Sequence[any], ArrayAttr]
) -> _Sequence[any]:
    return [_get_value_or_attribute_value(v) for v in sequence_or_array_attr]


def _get_int_array_attr(
    values: _Optional[_Union[ArrayAttr, IntOrAttrList]]
) -> ArrayAttr:
    if values is None:
        return None

    # Turn into a Python list of Python ints.
    values = _get_value_list(values)

    # Make an ArrayAttr of IntegerAttrs out of it.
    return ArrayAttr.get(
        [IntegerAttr.get(IntegerType.get_signless(64), v) for v in values]
    )


def _get_int_array_array_attr(
    values: _Optional[_Union[ArrayAttr, _Sequence[_Union[ArrayAttr, IntOrAttrList]]]]
) -> ArrayAttr:
    """Creates an ArrayAttr of ArrayAttrs of IntegerAttrs.

    The input has to be a collection of a collection of integers, where any
    Python _Sequence and ArrayAttr are admissible collections and Python ints and
    any IntegerAttr are admissible integers. Both levels of collections are
    turned into ArrayAttr; the inner level is turned into IntegerAttrs of i64s.
    If the input is None, an empty ArrayAttr is returned.
    """
    if values is None:
        return None

    # Make sure the outer level is a list.
    values = _get_value_list(values)

    # The inner level is now either invalid or a mixed sequence of ArrayAttrs and
    # Sequences. Make sure the nested values are all lists.
    values = [_get_value_list(nested) for nested in values]

    # Turn each nested list into an ArrayAttr.
    values = [_get_int_array_attr(nested) for nested in values]

    # Turn the outer list into an ArrayAttr.
    return ArrayAttr.get(values)
