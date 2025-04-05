#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from ._func_ops_gen import *
from ._func_ops_gen import _Dialect

try:
    from ..ir import *
    from ._ods_common import (
        get_default_loc_context as _get_default_loc_context,
        _cext as _ods_cext,
    )

    import inspect

    from typing import Any, List, Optional, Sequence, Union
except ImportError as e:
    raise RuntimeError("Error loading imports from extension module") from e

ARGUMENT_ATTRIBUTE_NAME = "arg_attrs"
RESULT_ATTRIBUTE_NAME = "res_attrs"


@_ods_cext.register_operation(_Dialect, replace=True)
class ConstantOp(ConstantOp):
    """Specialization for the constant op class."""

    @property
    def type(self):
        return self.results[0].type


@_ods_cext.register_operation(_Dialect, replace=True)
class FuncOp(FuncOp):
    """Specialization for the func op class."""

    def __init__(
        self, name, type, *, visibility=None, body_builder=None, loc=None, ip=None
    ):
        """
        Create a FuncOp with the provided `name`, `type`, and `visibility`.
        - `name` is a string representing the function name.
        - `type` is either a FunctionType or a pair of list describing inputs and
          results.
        - `visibility` is a string matching `public`, `private`, or `nested`. None
          implies private visibility.
        - `body_builder` is an optional callback, when provided a new entry block
          is created and the callback is invoked with the new op as argument within
          an InsertionPoint context already set for the block. The callback is
          expected to insert a terminator in the block.
        """
        sym_name = StringAttr.get(str(name))

        # If the type is passed as a tuple, build a FunctionType on the fly.
        if isinstance(type, tuple):
            type = FunctionType.get(inputs=type[0], results=type[1])

        type = TypeAttr.get(type)
        sym_visibility = (
            StringAttr.get(str(visibility)) if visibility is not None else None
        )
        super().__init__(sym_name, type, sym_visibility=sym_visibility, loc=loc, ip=ip)
        if body_builder:
            entry_block = self.add_entry_block()
            with InsertionPoint(entry_block):
                body_builder(self)

    @property
    def is_external(self):
        return len(self.regions[0].blocks) == 0

    @property
    def body(self):
        return self.regions[0]

    @property
    def type(self):
        return FunctionType(TypeAttr(self.attributes["function_type"]).value)

    @property
    def visibility(self):
        return self.attributes["sym_visibility"]

    @property
    def name(self) -> StringAttr:
        return StringAttr(self.attributes["sym_name"])

    @property
    def entry_block(self):
        if self.is_external:
            raise IndexError("External function does not have a body")
        return self.regions[0].blocks[0]

    def add_entry_block(self, arg_locs: Optional[Sequence[Location]] = None):
        """
        Add an entry block to the function body using the function signature to
        infer block arguments.
        Returns the newly created block
        """
        if not self.is_external:
            raise IndexError("The function already has an entry block!")
        self.body.blocks.append(*self.type.inputs, arg_locs=arg_locs)
        return self.body.blocks[0]

    @property
    def arg_attrs(self):
        if ARGUMENT_ATTRIBUTE_NAME not in self.attributes:
            return ArrayAttr.get([DictAttr.get({}) for _ in self.type.inputs])
        return ArrayAttr(self.attributes[ARGUMENT_ATTRIBUTE_NAME])

    @arg_attrs.setter
    def arg_attrs(self, attribute: Union[ArrayAttr, list]):
        if isinstance(attribute, ArrayAttr):
            self.attributes[ARGUMENT_ATTRIBUTE_NAME] = attribute
        else:
            self.attributes[ARGUMENT_ATTRIBUTE_NAME] = ArrayAttr.get(
                attribute, context=self.context
            )

    @property
    def arguments(self):
        return self.entry_block.arguments

    @property
    def result_attrs(self):
        return self.attributes[RESULT_ATTRIBUTE_NAME]

    @result_attrs.setter
    def result_attrs(self, attribute: ArrayAttr):
        self.attributes[RESULT_ATTRIBUTE_NAME] = attribute

    @classmethod
    def from_py_func(
        FuncOp,
        *inputs: Type,
        results: Optional[Sequence[Type]] = None,
        name: Optional[str] = None,
    ):
        """Decorator to define an MLIR FuncOp specified as a python function.

        Requires that an `mlir.ir.InsertionPoint` and `mlir.ir.Location` are
        active for the current thread (i.e. established in a `with` block).

        When applied as a decorator to a Python function, an entry block will
        be constructed for the FuncOp with types as specified in `*inputs`. The
        block arguments will be passed positionally to the Python function. In
        addition, if the Python function accepts keyword arguments generally or
        has a corresponding keyword argument, the following will be passed:
          * `func_op`: The `func` op being defined.

        By default, the function name will be the Python function `__name__`. This
        can be overriden by passing the `name` argument to the decorator.

        If `results` is not specified, then the decorator will implicitly
        insert a `ReturnOp` with the `Value`'s returned from the decorated
        function. It will also set the `FuncOp` type with the actual return
        value types. If `results` is specified, then the decorated function
        must return `None` and no implicit `ReturnOp` is added (nor are the result
        types updated). The implicit behavior is intended for simple, single-block
        cases, and users should specify result types explicitly for any complicated
        cases.

        The decorated function can further be called from Python and will insert
        a `CallOp` at the then-current insertion point, returning either None (
        if no return values), a unary Value (for one result), or a list of Values).
        This mechanism cannot be used to emit recursive calls (by construction).
        """

        def decorator(f):
            from . import func

            # Introspect the callable for optional features.
            sig = inspect.signature(f)
            has_arg_func_op = False
            for param in sig.parameters.values():
                if param.kind == param.VAR_KEYWORD:
                    has_arg_func_op = True
                if param.name == "func_op" and (
                    param.kind == param.POSITIONAL_OR_KEYWORD
                    or param.kind == param.KEYWORD_ONLY
                ):
                    has_arg_func_op = True

            # Emit the FuncOp.
            implicit_return = results is None
            symbol_name = name or f.__name__
            function_type = FunctionType.get(
                inputs=inputs, results=[] if implicit_return else results
            )
            func_op = FuncOp(name=symbol_name, type=function_type)
            with InsertionPoint(func_op.add_entry_block()):
                func_args = func_op.entry_block.arguments
                func_kwargs = {}
                if has_arg_func_op:
                    func_kwargs["func_op"] = func_op
                return_values = f(*func_args, **func_kwargs)
                if not implicit_return:
                    return_types = list(results)
                    assert return_values is None, (
                        "Capturing a python function with explicit `results=` "
                        "requires that the wrapped function returns None."
                    )
                else:
                    # Coerce return values, add ReturnOp and rewrite func type.
                    if return_values is None:
                        return_values = []
                    elif isinstance(return_values, tuple):
                        return_values = list(return_values)
                    elif isinstance(return_values, Value):
                        # Returning a single value is fine, coerce it into a list.
                        return_values = [return_values]
                    elif isinstance(return_values, OpView):
                        # Returning a single operation is fine, coerce its results a list.
                        return_values = return_values.operation.results
                    elif isinstance(return_values, Operation):
                        # Returning a single operation is fine, coerce its results a list.
                        return_values = return_values.results
                    else:
                        return_values = list(return_values)
                    func.ReturnOp(return_values)
                    # Recompute the function type.
                    return_types = [v.type for v in return_values]
                    function_type = FunctionType.get(
                        inputs=inputs, results=return_types
                    )
                    func_op.attributes["function_type"] = TypeAttr.get(function_type)

            def emit_call_op(*call_args):
                call_op = func.CallOp(
                    return_types, FlatSymbolRefAttr.get(symbol_name), call_args
                )
                if return_types is None:
                    return None
                elif len(return_types) == 1:
                    return call_op.result
                else:
                    return call_op.results

            wrapped = emit_call_op
            wrapped.__name__ = f.__name__
            wrapped.func_op = func_op
            return wrapped

        return decorator


func = FuncOp.from_py_func


@_ods_cext.register_operation(_Dialect, replace=True)
class CallOp(CallOp):
    """Specialization for the call op class."""

    def __init__(
        self,
        calleeOrResults: Union[FuncOp, List[Type]],
        argumentsOrCallee: Union[List, FlatSymbolRefAttr, str],
        arguments: Optional[List] = None,
        *,
        loc=None,
        ip=None,
    ):
        """Creates an call operation.

        The constructor accepts three different forms:

          1. A function op to be called followed by a list of arguments.
          2. A list of result types, followed by the name of the function to be
             called as string, following by a list of arguments.
          3. A list of result types, followed by the name of the function to be
             called as symbol reference attribute, followed by a list of arguments.

        For example

            f = func.FuncOp("foo", ...)
            func.CallOp(f, [args])
            func.CallOp([result_types], "foo", [args])

        In all cases, the location and insertion point may be specified as keyword
        arguments if not provided by the surrounding context managers.
        """

        # TODO: consider supporting constructor "overloads", e.g., through a custom
        # or pybind-provided metaclass.
        if isinstance(calleeOrResults, FuncOp):
            if not isinstance(argumentsOrCallee, list):
                raise ValueError(
                    "when constructing a call to a function, expected "
                    + "the second argument to be a list of call arguments, "
                    + f"got {type(argumentsOrCallee)}"
                )
            if arguments is not None:
                raise ValueError(
                    "unexpected third argument when constructing a call"
                    + "to a function"
                )

            super().__init__(
                calleeOrResults.type.results,
                FlatSymbolRefAttr.get(
                    calleeOrResults.name.value, context=_get_default_loc_context(loc)
                ),
                argumentsOrCallee,
                loc=loc,
                ip=ip,
            )
            return

        if isinstance(argumentsOrCallee, list):
            raise ValueError(
                "when constructing a call to a function by name, "
                + "expected the second argument to be a string or a "
                + f"FlatSymbolRefAttr, got {type(argumentsOrCallee)}"
            )

        if isinstance(argumentsOrCallee, FlatSymbolRefAttr):
            super().__init__(
                calleeOrResults, argumentsOrCallee, arguments, loc=loc, ip=ip
            )
        elif isinstance(argumentsOrCallee, str):
            super().__init__(
                calleeOrResults,
                FlatSymbolRefAttr.get(
                    argumentsOrCallee, context=_get_default_loc_context(loc)
                ),
                arguments,
                loc=loc,
                ip=ip,
            )
