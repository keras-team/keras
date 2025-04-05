#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception


from ._scf_ops_gen import *
from ._scf_ops_gen import _Dialect
from .arith import constant

try:
    from ..ir import *
    from ._ods_common import (
        get_op_result_or_value as _get_op_result_or_value,
        get_op_results_or_values as _get_op_results_or_values,
        _cext as _ods_cext,
    )
except ImportError as e:
    raise RuntimeError("Error loading imports from extension module") from e

from typing import Optional, Sequence, Union


@_ods_cext.register_operation(_Dialect, replace=True)
class ForOp(ForOp):
    """Specialization for the SCF for op class."""

    def __init__(
        self,
        lower_bound,
        upper_bound,
        step,
        iter_args: Optional[Union[Operation, OpView, Sequence[Value]]] = None,
        *,
        loc=None,
        ip=None,
    ):
        """Creates an SCF `for` operation.

        - `lower_bound` is the value to use as lower bound of the loop.
        - `upper_bound` is the value to use as upper bound of the loop.
        - `step` is the value to use as loop step.
        - `iter_args` is a list of additional loop-carried arguments or an operation
          producing them as results.
        """
        if iter_args is None:
            iter_args = []
        iter_args = _get_op_results_or_values(iter_args)

        results = [arg.type for arg in iter_args]
        super().__init__(
            results, lower_bound, upper_bound, step, iter_args, loc=loc, ip=ip
        )
        self.regions[0].blocks.append(self.operands[0].type, *results)

    @property
    def body(self):
        """Returns the body (block) of the loop."""
        return self.regions[0].blocks[0]

    @property
    def induction_variable(self):
        """Returns the induction variable of the loop."""
        return self.body.arguments[0]

    @property
    def inner_iter_args(self):
        """Returns the loop-carried arguments usable within the loop.

        To obtain the loop-carried operands, use `iter_args`.
        """
        return self.body.arguments[1:]


@_ods_cext.register_operation(_Dialect, replace=True)
class IfOp(IfOp):
    """Specialization for the SCF if op class."""

    def __init__(self, cond, results_=None, *, hasElse=False, loc=None, ip=None):
        """Creates an SCF `if` operation.

        - `cond` is a MLIR value of 'i1' type to determine which regions of code will be executed.
        - `hasElse` determines whether the if operation has the else branch.
        """
        if results_ is None:
            results_ = []
        operands = []
        operands.append(cond)
        results = []
        results.extend(results_)
        super().__init__(results, cond, loc=loc, ip=ip)
        self.regions[0].blocks.append(*[])
        if hasElse:
            self.regions[1].blocks.append(*[])

    @property
    def then_block(self):
        """Returns the then block of the if operation."""
        return self.regions[0].blocks[0]

    @property
    def else_block(self):
        """Returns the else block of the if operation."""
        return self.regions[1].blocks[0]


def for_(
    start,
    stop=None,
    step=None,
    iter_args: Optional[Sequence[Value]] = None,
    *,
    loc=None,
    ip=None,
):
    if step is None:
        step = 1
    if stop is None:
        stop = start
        start = 0
    params = [start, stop, step]
    for i, p in enumerate(params):
        if isinstance(p, int):
            p = constant(IndexType.get(), p)
        elif isinstance(p, float):
            raise ValueError(f"{p=} must be int.")
        params[i] = p

    start, stop, step = params

    for_op = ForOp(start, stop, step, iter_args, loc=loc, ip=ip)
    iv = for_op.induction_variable
    iter_args = tuple(for_op.inner_iter_args)
    with InsertionPoint(for_op.body):
        if len(iter_args) > 1:
            yield iv, iter_args, for_op.results
        elif len(iter_args) == 1:
            yield iv, iter_args[0], for_op.results[0]
        else:
            yield iv
