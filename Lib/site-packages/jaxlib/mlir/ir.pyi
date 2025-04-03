# Originally imported via:
#   pybind11-stubgen --print-invalid-expressions-as-is mlir._mlir_libs._mlir.ir
# but with the following diff (in order to remove pipes from types,
# which we won't support until bumping minimum python to 3.10)
#
# --------------------- diff begins ------------------------------------
#
# diff --git a/pybind11_stubgen/printer.py b/pybind11_stubgen/printer.py
# index 1f755aa..4924927 100644
# --- a/pybind11_stubgen/printer.py
# +++ b/pybind11_stubgen/printer.py
# @@ -283,14 +283,6 @@ class Printer:
#              return split[0] + "..."
#
#      def print_type(self, type_: ResolvedType) -> str:
# -        if (
# -            str(type_.name) == "typing.Optional"
# -            and type_.parameters is not None
# -            and len(type_.parameters) == 1
# -        ):
# -            return f"{self.print_annotation(type_.parameters[0])} | None"
# -        if str(type_.name) == "typing.Union" and type_.parameters is not None:
# -            return " | ".join(self.print_annotation(p) for p in type_.parameters)
#          if type_.parameters:
#              param_str = (
#                  "["
#
# --------------------- diff ends ------------------------------------
#
# Local modifications:
#   * Rewrite references to 'mlir.ir' to local types.
#   * Drop `typing.` everywhere (top-level import instead).
#   * List -> List, dict -> Dict, Tuple -> Tuple.
#   * copy-paste Buffer type from typing_extensions.
#   * Shuffle _OperationBase, AffineExpr, Attribute, Type, Value to the top.
#   * Patch raw C++ types (like "PyAsmState") with a regex like `Py(.*)`.
#   * _BaseContext -> Context, MlirType -> Type, MlirTypeID -> TypeID, MlirAttribute -> Attribute.
#   * Local edits to signatures and types that pybind11-stubgen did not auto detect (or detected incorrectly).
#   * Add MLIRError, _GlobalDebug, _OperationBase to __all__ by hand.
#   * Fill in `Any`s where possible.
#   * black formatting.

from __future__ import annotations

import abc
import collections
from collections.abc import Callable, Sequence
import io
from typing import Any, ClassVar, TypeVar, overload

__all__ = [
    "AffineAddExpr",
    "AffineBinaryExpr",
    "AffineCeilDivExpr",
    "AffineConstantExpr",
    "AffineDimExpr",
    "AffineExpr",
    "AffineExprList",
    "AffineFloorDivExpr",
    "AffineMap",
    "AffineMapAttr",
    "AffineModExpr",
    "AffineMulExpr",
    "AffineSymbolExpr",
    "ArrayAttr",
    "ArrayAttributeIterator",
    "AsmState",
    "AttrBuilder",
    "Attribute",
    "BF16Type",
    "Block",
    "BlockArgument",
    "BlockArgumentList",
    "BlockIterator",
    "BlockList",
    "BoolAttr",
    "ComplexType",
    "Context",
    "DenseBoolArrayAttr",
    "DenseBoolArrayIterator",
    "DenseElementsAttr",
    "DenseF32ArrayAttr",
    "DenseF32ArrayIterator",
    "DenseF64ArrayAttr",
    "DenseF64ArrayIterator",
    "DenseFPElementsAttr",
    "DenseI16ArrayAttr",
    "DenseI16ArrayIterator",
    "DenseI32ArrayAttr",
    "DenseI32ArrayIterator",
    "DenseI64ArrayAttr",
    "DenseI64ArrayIterator",
    "DenseI8ArrayAttr",
    "DenseI8ArrayIterator",
    "DenseIntElementsAttr",
    "DenseResourceElementsAttr",
    "Diagnostic",
    "DiagnosticHandler",
    "DiagnosticInfo",
    "DiagnosticSeverity",
    "Dialect",
    "DialectDescriptor",
    "DialectRegistry",
    "Dialects",
    "DictAttr",
    "F16Type",
    "F32Type",
    "F64Type",
    "FlatSymbolRefAttr",
    "Float4E2M1FNType",
    "Float6E2M3FNType",
    "Float6E3M2FNType",
    "Float8E3M4Type",
    "Float8E4M3B11FNUZType",
    "Float8E4M3FNType",
    "Float8E4M3FNUZType",
    "Float8E4M3Type",
    "Float8E5M2FNUZType",
    "Float8E5M2Type",
    "Float8E8M0FNUType",
    "FloatAttr",
    "FloatTF32Type",
    "FloatType",
    "FunctionType",
    "IndexType",
    "InferShapedTypeOpInterface",
    "InferTypeOpInterface",
    "InsertionPoint",
    "IntegerAttr",
    "IntegerSet",
    "IntegerSetAttr",
    "IntegerSetConstraint",
    "IntegerSetConstraintList",
    "IntegerType",
    "Location",
    "MemRefType",
    "Module",
    "NamedAttribute",
    "NoneType",
    "OpAttributeMap",
    "OpOperand",
    "OpOperandIterator",
    "OpOperandList",
    "OpResult",
    "OpResultList",
    "OpSuccessors",
    "OpView",
    "OpaqueAttr",
    "OpaqueType",
    "Operation",
    "OperationIterator",
    "OperationList",
    "RankedTensorType",
    "Region",
    "RegionIterator",
    "RegionSequence",
    "ShapedType",
    "ShapedTypeComponents",
    "StridedLayoutAttr",
    "StringAttr",
    "SymbolRefAttr",
    "SymbolTable",
    "TupleType",
    "Type",
    "TypeAttr",
    "TypeID",
    "UnitAttr",
    "UnrankedMemRefType",
    "UnrankedTensorType",
    "Value",
    "VectorType",
    "_GlobalDebug",
    "_OperationBase",
]

if hasattr(collections.abc, "Buffer"):
    Buffer = collections.abc.Buffer
else:
    class Buffer(abc.ABC):
        pass

class _OperationBase:
    @overload
    def __eq__(self, arg0: _OperationBase) -> bool: ...
    @overload
    def __eq__(self, arg0: _OperationBase) -> bool: ...
    def __hash__(self) -> int: ...
    def __str__(self) -> str:
        """
        Returns the assembly form of the operation.
        """
    def clone(self, ip: InsertionPoint = None) -> OpView: ...
    def detach_from_parent(self) -> OpView:
        """
        Detaches the operation from its parent block.
        """
    def erase(self) -> None: ...
    def get_asm(
        self,
        binary: bool = False,
        large_elements_limit: int | None = None,
        enable_debug_info: bool = False,
        pretty_debug_info: bool = False,
        print_generic_op_form: bool = False,
        use_local_scope: bool = False,
        assume_verified: bool = False,
        skip_regions: bool = False,
    ) -> io.BytesIO | io.StringIO:
        """
        Gets the assembly form of the operation with all options available.

        Args:
          binary: Whether to return a bytes (True) or str (False) object. Defaults to
            False.
          ... others ...: See the print() method for common keyword arguments for
            configuring the printout.
        Returns:
          Either a bytes or str object, depending on the setting of the 'binary'
          argument.
        """
    def move_after(self, other: _OperationBase) -> None:
        """
        Puts self immediately after the other operation in its parent block.
        """
    def move_before(self, other: _OperationBase) -> None:
        """
        Puts self immediately before the other operation in its parent block.
        """
    @overload
    def print(
        self,
        state: AsmState,
        file: Any | None = None,
        binary: bool = False,
    ) -> None:
        """
        Prints the assembly form of the operation to a file like object.

        Args:
          file: The file like object to write to. Defaults to sys.stdout.
          binary: Whether to write bytes (True) or str (False). Defaults to False.
          state: AsmState capturing the operation numbering and flags.
        """
    @overload
    def print(
        self,
        large_elements_limit: int | None = None,
        enable_debug_info: bool = False,
        pretty_debug_info: bool = False,
        print_generic_op_form: bool = False,
        use_local_scope: bool = False,
        assume_verified: bool = False,
        file: Any | None = None,
        binary: bool = False,
        skip_regions: bool = False,
    ) -> None:
        """
        Prints the assembly form of the operation to a file like object.

        Args:
          file: The file like object to write to. Defaults to sys.stdout.
          binary: Whether to write bytes (True) or str (False). Defaults to False.
          large_elements_limit: Whether to elide elements attributes above this
            number of elements. Defaults to None (no limit).
          enable_debug_info: Whether to print debug/location information. Defaults
            to False.
          pretty_debug_info: Whether to format debug information for easier reading
            by a human (warning: the result is unparseable).
          print_generic_op_form: Whether to print the generic assembly forms of all
            ops. Defaults to False.
          use_local_Scope: Whether to print in a way that is more optimized for
            multi-threaded access but may not be consistent with how the overall
            module prints.
          assume_verified: By default, if not printing generic form, the verifier
            will be run and if it fails, generic form will be printed with a comment
            about failed verification. While a reasonable default for interactive use,
            for systematic use, it is often better for the caller to verify explicitly
            and report failures in a more robust fashion. Set this to True if doing this
            in order to avoid running a redundant verification. If the IR is actually
            invalid, behavior is undefined.
          skip_regions: Whether to skip printing regions. Defaults to False.
        """
    def verify(self) -> bool:
        """
        Verify the operation. Raises MLIRError if verification fails, and returns true otherwise.
        """
    def write_bytecode(self, file: Any, desired_version: int | None = None) -> None:
        """
        Write the bytecode form of the operation to a file like object.

        Args:
          file: The file like object to write to.
          desired_version: The version of bytecode to emit.
        Returns:
          The bytecode writer status.
        """
    @property
    def _CAPIPtr(self) -> object: ...
    @property
    def attributes(self) -> OpAttributeMap: ...
    @property
    def context(self) -> Context:
        """
        Context that owns the Operation
        """
    @property
    def location(self) -> Location:
        """
        Returns the source location the operation was defined or derived from.
        """
    @property
    def name(self) -> str: ...
    @property
    def operands(self) -> OpOperandList: ...
    @property
    def parent(self) -> _OperationBase | None: ...
    @property
    def regions(self) -> RegionSequence: ...
    @property
    def result(self) -> OpResult:
        """
        Shortcut to get an op result if it has only one (throws an error otherwise).
        """
    @property
    def results(self) -> OpResultList:
        """
        Returns the List of Operation results.
        """

_TOperation = TypeVar("_TOperation", bound=_OperationBase)

class AffineExpr:
    @staticmethod
    @overload
    def get_add(arg0: AffineExpr, arg1: AffineExpr) -> AffineAddExpr:
        """
        Gets an affine expression containing a sum of two expressions.
        """
    @staticmethod
    @overload
    def get_add(arg0: int, arg1: AffineExpr) -> AffineAddExpr:
        """
        Gets an affine expression containing a sum of a constant and another expression.
        """
    @staticmethod
    @overload
    def get_add(arg0: AffineExpr, arg1: int) -> AffineAddExpr:
        """
        Gets an affine expression containing a sum of an expression and a constant.
        """
    @staticmethod
    @overload
    def get_ceil_div(arg0: AffineExpr, arg1: AffineExpr) -> AffineCeilDivExpr:
        """
        Gets an affine expression containing the rounded-up result of dividing one expression by another.
        """
    @staticmethod
    @overload
    def get_ceil_div(arg0: int, arg1: AffineExpr) -> AffineCeilDivExpr:
        """
        Gets a semi-affine expression containing the rounded-up result of dividing a constant by an expression.
        """
    @staticmethod
    @overload
    def get_ceil_div(arg0: AffineExpr, arg1: int) -> AffineCeilDivExpr:
        """
        Gets an affine expression containing the rounded-up result of dividing an expression by a constant.
        """
    @staticmethod
    def get_constant(
        value: int, context: Context | None = None
    ) -> AffineConstantExpr:
        """
        Gets a constant affine expression with the given value.
        """
    @staticmethod
    def get_dim(position: int, context: Context | None = None) -> AffineDimExpr:
        """
        Gets an affine expression of a dimension at the given position.
        """
    @staticmethod
    @overload
    def get_floor_div(arg0: AffineExpr, arg1: AffineExpr) -> AffineFloorDivExpr:
        """
        Gets an affine expression containing the rounded-down result of dividing one expression by another.
        """
    @staticmethod
    @overload
    def get_floor_div(arg0: int, arg1: AffineExpr) -> AffineFloorDivExpr:
        """
        Gets a semi-affine expression containing the rounded-down result of dividing a constant by an expression.
        """
    @staticmethod
    @overload
    def get_floor_div(arg0: AffineExpr, arg1: int) -> AffineFloorDivExpr:
        """
        Gets an affine expression containing the rounded-down result of dividing an expression by a constant.
        """
    @staticmethod
    @overload
    def get_mod(arg0: AffineExpr, arg1: AffineExpr) -> AffineModExpr:
        """
        Gets an affine expression containing the modulo of dividing one expression by another.
        """
    @staticmethod
    @overload
    def get_mod(arg0: int, arg1: AffineExpr) -> AffineModExpr:
        """
        Gets a semi-affine expression containing the modulo of dividing a constant by an expression.
        """
    @staticmethod
    @overload
    def get_mod(arg0: AffineExpr, arg1: int) -> AffineModExpr:
        """
        Gets an affine expression containing the module of dividingan expression by a constant.
        """
    @staticmethod
    @overload
    def get_mul(arg0: AffineExpr, arg1: AffineExpr) -> AffineMulExpr:
        """
        Gets an affine expression containing a product of two expressions.
        """
    @staticmethod
    @overload
    def get_mul(arg0: int, arg1: AffineExpr) -> AffineMulExpr:
        """
        Gets an affine expression containing a product of a constant and another expression.
        """
    @staticmethod
    @overload
    def get_mul(arg0: AffineExpr, arg1: int) -> AffineMulExpr:
        """
        Gets an affine expression containing a product of an expression and a constant.
        """
    @staticmethod
    def get_symbol(
        position: int, context: Context | None = None
    ) -> AffineSymbolExpr:
        """
        Gets an affine expression of a symbol at the given position.
        """
    def _CAPICreate(self) -> AffineExpr: ...
    @overload
    def __add__(self, arg0: AffineExpr) -> AffineAddExpr: ...
    @overload
    def __add__(self, arg0: int) -> AffineAddExpr: ...
    @overload
    def __eq__(self, arg0: AffineExpr) -> bool: ...
    @overload
    def __eq__(self, arg0: Any) -> bool: ...
    def __hash__(self) -> int: ...
    @overload
    def __mod__(self, arg0: AffineExpr) -> AffineModExpr: ...
    @overload
    def __mod__(self, arg0: int) -> AffineModExpr: ...
    @overload
    def __mul__(self, arg0: AffineExpr) -> AffineMulExpr: ...
    @overload
    def __mul__(self, arg0: int) -> AffineMulExpr: ...
    def __radd__(self, arg0: int) -> AffineAddExpr: ...
    def __rmod__(self, arg0: int) -> AffineModExpr: ...
    def __rmul__(self, arg0: int) -> AffineMulExpr: ...
    def __rsub__(self, arg0: int) -> AffineAddExpr: ...
    @overload
    def __sub__(self, arg0: AffineExpr) -> AffineAddExpr: ...
    @overload
    def __sub__(self, arg0: int) -> AffineAddExpr: ...
    def compose(self, arg0: AffineMap) -> AffineExpr: ...
    def dump(self) -> None:
        """
        Dumps a debug representation of the object to stderr.
        """
    @property
    def _CAPIPtr(self) -> object: ...
    @property
    def context(self) -> Context: ...

class Attribute:
    @staticmethod
    def parse(asm: str | bytes, context: Context | None = None) -> Attribute:
        """
        Parses an attribute from an assembly form. Raises an MLIRError on failure.
        """
    def _CAPICreate(self) -> Attribute: ...
    @overload
    def __eq__(self, arg0: Attribute) -> bool: ...
    @overload
    def __eq__(self, arg0: object) -> bool: ...
    def __hash__(self) -> int: ...
    def __init__(self, cast_from_type: Attribute) -> None:
        """
        Casts the passed attribute to the generic Attribute
        """
    def __str__(self) -> str:
        """
        Returns the assembly form of the Attribute.
        """
    def dump(self) -> None:
        """
        Dumps a debug representation of the object to stderr.
        """
    def get_named(self, arg0: str) -> NamedAttribute:
        """
        Binds a name to the attribute
        """
    def maybe_downcast(self) -> Any: ...
    @property
    def _CAPIPtr(self) -> object: ...
    @property
    def context(self) -> Context:
        """
        Context that owns the Attribute
        """
    @property
    def type(self) -> Type: ...
    @property
    def typeid(self) -> TypeID: ...

class Type:
    @staticmethod
    def parse(asm: str | bytes, context: Context | None = None) -> Type:
        """
        Parses the assembly form of a type.

        Returns a Type object or raises an MLIRError if the type cannot be parsed.

        See also: https://mlir.llvm.org/docs/LangRef/#type-system
        """
    def _CAPICreate(self) -> Type: ...
    @overload
    def __eq__(self, arg0: Type) -> bool: ...
    @overload
    def __eq__(self, arg0: object) -> bool: ...
    def __hash__(self) -> int: ...
    def __init__(self, cast_from_type: Type) -> None:
        """
        Casts the passed type to the generic Type
        """
    def __str__(self) -> str:
        """
        Returns the assembly form of the type.
        """
    def dump(self) -> None:
        """
        Dumps a debug representation of the object to stderr.
        """
    def maybe_downcast(self) -> Any: ...
    @property
    def _CAPIPtr(self) -> object: ...
    @property
    def context(self) -> Context:
        """
        Context that owns the Type
        """
    @property
    def typeid(self) -> TypeID: ...

class Value:
    def _CAPICreate(self) -> Value: ...
    @overload
    def __eq__(self, arg0: Value) -> bool: ...
    @overload
    def __eq__(self, arg0: object) -> bool: ...
    def __hash__(self) -> int: ...
    def __init__(self, value: Value) -> None: ...
    def __str__(self) -> str:
        """
        Returns the string form of the value.

        If the value is a block argument, this is the assembly form of its type and the
        position in the argument List. If the value is an operation result, this is
        equivalent to printing the operation that produced it.
        """
    def dump(self) -> None:
        """
        Dumps a debug representation of the object to stderr.
        """
    @overload
    def get_name(self, use_local_scope: bool = False) -> str: ...
    @overload
    def get_name(self, state: AsmState) -> str:
        """
        Returns the string form of value as an operand (i.e., the ValueID).
        """
    def maybe_downcast(self) -> Any: ...
    def replace_all_uses_with(self, arg0: Value) -> None:
        """
        Replace all uses of value with the new value, updating anything in
        the IR that uses 'self' to use the other value instead.
        """
    def set_type(self, type: Type) -> None: ...
    @property
    def _CAPIPtr(self) -> object: ...
    @property
    def context(self) -> Context:
        """
        Context in which the value lives.
        """
    @property
    def owner(self) -> _OperationBase: ...
    @property
    def type(self) -> Type: ...
    @property
    def uses(self) -> OpOperandIterator: ...

class AffineAddExpr(AffineBinaryExpr):
    @staticmethod
    def get(arg0: AffineExpr, arg1: AffineExpr) -> AffineAddExpr: ...
    @staticmethod
    def isinstance(other: AffineExpr) -> bool: ...
    def __init__(self, expr: AffineExpr) -> None: ...

class AffineBinaryExpr(AffineExpr):
    @staticmethod
    def isinstance(other: AffineExpr) -> bool: ...
    def __init__(self, expr: AffineExpr) -> None: ...
    @property
    def lhs(self) -> AffineExpr: ...
    @property
    def rhs(self) -> AffineExpr: ...

class AffineCeilDivExpr(AffineBinaryExpr):
    @staticmethod
    def get(arg0: AffineExpr, arg1: AffineExpr) -> AffineCeilDivExpr: ...
    @staticmethod
    def isinstance(other: AffineExpr) -> bool: ...
    def __init__(self, expr: AffineExpr) -> None: ...

class AffineConstantExpr(AffineExpr):
    @staticmethod
    def get(value: int, context: Context | None = None) -> AffineConstantExpr: ...
    @staticmethod
    def isinstance(other: AffineExpr) -> bool: ...
    def __init__(self, expr: AffineExpr) -> None: ...
    @property
    def value(self) -> int: ...

class AffineDimExpr(AffineExpr):
    @staticmethod
    def get(position: int, context: Context | None = None) -> AffineDimExpr: ...
    @staticmethod
    def isinstance(other: AffineExpr) -> bool: ...
    def __init__(self, expr: AffineExpr) -> None: ...
    @property
    def position(self) -> int: ...

class AffineExprList:
    def __add__(self, arg0: AffineExprList) -> list[AffineExpr]: ...

class AffineFloorDivExpr(AffineBinaryExpr):
    @staticmethod
    def get(arg0: AffineExpr, arg1: AffineExpr) -> AffineFloorDivExpr: ...
    @staticmethod
    def isinstance(other: AffineExpr) -> bool: ...
    def __init__(self, expr: AffineExpr) -> None: ...

class AffineMap:
    @staticmethod
    def compress_unused_symbols(
        arg0: list, arg1: Context | None
    ) -> list[AffineMap]: ...
    @staticmethod
    def get(
        dim_count: int,
        symbol_count: int,
        exprs: list,
        context: Context | None = None,
    ) -> AffineMap:
        """
        Gets a map with the given expressions as results.
        """
    @staticmethod
    def get_constant(value: int, context: Context | None = None) -> AffineMap:
        """
        Gets an affine map with a single constant result
        """
    @staticmethod
    def get_empty(context: Context | None = None) -> AffineMap:
        """
        Gets an empty affine map.
        """
    @staticmethod
    def get_identity(n_dims: int, context: Context | None = None) -> AffineMap:
        """
        Gets an identity map with the given number of dimensions.
        """
    @staticmethod
    def get_minor_identity(
        n_dims: int, n_results: int, context: Context | None = None
    ) -> AffineMap:
        """
        Gets a minor identity map with the given number of dimensions and results.
        """
    @staticmethod
    def get_permutation(
        permutation: list[int], context: Context | None = None
    ) -> AffineMap:
        """
        Gets an affine map that permutes its inputs.
        """
    def _CAPICreate(self) -> AffineMap: ...
    @overload
    def __eq__(self, arg0: AffineMap) -> bool: ...
    @overload
    def __eq__(self, arg0: object) -> bool: ...
    def __hash__(self) -> int: ...
    def dump(self) -> None:
        """
        Dumps a debug representation of the object to stderr.
        """
    def get_major_submap(self, n_results: int) -> AffineMap: ...
    def get_minor_submap(self, n_results: int) -> AffineMap: ...
    def get_submap(self, result_positions: list[int]) -> AffineMap: ...
    def replace(
        self,
        expr: AffineExpr,
        replacement: AffineExpr,
        n_result_dims: int,
        n_result_syms: int,
    ) -> AffineMap: ...
    @property
    def _CAPIPtr(self) -> object: ...
    @property
    def context(self) -> Context:
        """
        Context that owns the Affine Map
        """
    @property
    def is_permutation(self) -> bool: ...
    @property
    def is_projected_permutation(self) -> bool: ...
    @property
    def n_dims(self) -> int: ...
    @property
    def n_inputs(self) -> int: ...
    @property
    def n_symbols(self) -> int: ...
    @property
    def results(self) -> AffineMapExprList: ...

class AffineMapAttr(Attribute):
    static_typeid: ClassVar[TypeID]
    @staticmethod
    def get(affine_map: AffineMap) -> AffineMapAttr:
        """
        Gets an attribute wrapping an AffineMap.
        """
    @staticmethod
    def isinstance(other: Attribute) -> bool: ...
    def __init__(self, cast_from_attr: Attribute) -> None: ...
    @property
    def type(self) -> Type: ...
    @property
    def typeid(self) -> TypeID: ...

class AffineModExpr(AffineBinaryExpr):
    @staticmethod
    def get(arg0: AffineExpr, arg1: AffineExpr) -> AffineModExpr: ...
    @staticmethod
    def isinstance(other: AffineExpr) -> bool: ...
    def __init__(self, expr: AffineExpr) -> None: ...

class AffineMulExpr(AffineBinaryExpr):
    @staticmethod
    def get(arg0: AffineExpr, arg1: AffineExpr) -> AffineMulExpr: ...
    @staticmethod
    def isinstance(other: AffineExpr) -> bool: ...
    def __init__(self, expr: AffineExpr) -> None: ...

class AffineSymbolExpr(AffineExpr):
    @staticmethod
    def get(position: int, context: Context | None = None) -> AffineSymbolExpr: ...
    @staticmethod
    def isinstance(other: AffineExpr) -> bool: ...
    def __init__(self, expr: AffineExpr) -> None: ...
    @property
    def position(self) -> int: ...

class ArrayAttr(Attribute):
    static_typeid: ClassVar[TypeID]
    @staticmethod
    def get(attributes: list, context: Context | None = None) -> ArrayAttr:
        """
        Gets a uniqued Array attribute
        """
    @staticmethod
    def isinstance(other: Attribute) -> bool: ...
    def __add__(self, arg0: list) -> ArrayAttr: ...
    def __getitem__(self, arg0: int) -> Attribute: ...
    def __init__(self, cast_from_attr: Attribute) -> None: ...
    def __iter__(
        self,
    ) -> ArrayAttributeIterator: ...
    def __len__(self) -> int: ...
    @property
    def type(self) -> Type: ...
    @property
    def typeid(self) -> TypeID: ...

class ArrayAttributeIterator:
    def __iter__(self) -> ArrayAttributeIterator: ...
    def __next__(self) -> Attribute: ...

class AsmState:
    @overload
    def __init__(self, value: Value, use_local_scope: bool = False) -> None: ...
    @overload
    def __init__(self, op: _OperationBase, use_local_scope: bool = False) -> None: ...

class AttrBuilder:
    @staticmethod
    def contains(arg0: str) -> bool: ...
    @staticmethod
    def get(arg0: str) -> Callable: ...
    @staticmethod
    def insert(
        attribute_kind: str, attr_builder: Callable, replace: bool = False
    ) -> None:
        """
        Register an attribute builder for building MLIR attributes from python values.
        """

class BF16Type(Type):
    static_typeid: ClassVar[TypeID]
    @staticmethod
    def get(context: Context | None = None) -> BF16Type:
        """
        Create a bf16 type.
        """
    @staticmethod
    def isinstance(other: Type) -> bool: ...
    def __init__(self, cast_from_type: Type) -> None: ...
    @property
    def typeid(self) -> TypeID: ...

class Block:
    @staticmethod
    def create_at_start(
        parent: Region,
        arg_types: list[Type],
        arg_locs: Sequence | None = None,
    ) -> Block:
        """
        Creates and returns a new Block at the beginning of the given region (with given argument types and locations).
        """
    @overload
    def __eq__(self, arg0: Block) -> bool: ...
    @overload
    def __eq__(self, arg0: Any) -> bool: ...
    def __hash__(self) -> int: ...
    def __iter__(self) -> OperationIterator:
        """
        Iterates over operations in the block.
        """
    def __str__(self) -> str:
        """
        Returns the assembly form of the block.
        """
    def append(self, operation: _OperationBase) -> None:
        """
        Appends an operation to this block. If the operation is currently in another block, it will be moved.
        """
    def append_to(self, arg0: Region) -> None:
        """
        Append this block to a region, transferring ownership if necessary
        """
    def create_after(self, *args, arg_locs: Sequence | None = None) -> Block:
        """
        Creates and returns a new Block after this block (with given argument types and locations).
        """
    def create_before(self, *args, arg_locs: Sequence | None = None) -> Block:
        """
        Creates and returns a new Block before this block (with given argument types and locations).
        """
    @property
    def _CAPIPtr(self) -> object: ...
    @property
    def arguments(self) -> BlockArgumentList:
        """
        Returns a List of block arguments.
        """
    @property
    def operations(self) -> OperationList:
        """
        Returns a forward-optimized sequence of operations.
        """
    @property
    def owner(self) -> OpView:
        """
        Returns the owning operation of this block.
        """
    @property
    def region(self) -> Region:
        """
        Returns the owning region of this block.
        """

class BlockArgument(Value):
    @staticmethod
    def isinstance(other_value: Value) -> bool: ...
    def __init__(self, value: Value) -> None: ...
    def maybe_downcast(self) -> Any: ...
    def set_type(self, type: Type) -> None: ...
    @property
    def arg_number(self) -> int: ...
    @property
    def owner(self) -> Block: ...

class BlockArgumentList:
    @overload
    def __getitem__(self, arg0: int) -> BlockArgument: ...
    @overload
    def __getitem__(self, arg0: slice) -> BlockArgumentList: ...
    def __len__(self) -> int: ...
    def __add__(self, arg0: BlockArgumentList) -> list[BlockArgument]: ...
    @property
    def types(self) -> list[Type]: ...

class BlockIterator:
    def __iter__(self) -> BlockIterator: ...
    def __next__(self) -> Block: ...

class BlockList:
    def __getitem__(self, arg0: int) -> Block: ...
    def __iter__(self) -> BlockIterator: ...
    def __len__(self) -> int: ...
    def append(self, *args, arg_locs: Sequence | None = None) -> Block:
        """
        Appends a new block, with argument types as positional args.

        Returns:
          The created block.
        """

class BoolAttr(Attribute):
    @staticmethod
    def get(value: bool, context: Context | None = None) -> BoolAttr:
        """
        Gets an uniqued bool attribute
        """
    @staticmethod
    def isinstance(other: Attribute) -> bool: ...
    def __bool__(self: Attribute) -> bool:
        """
        Converts the value of the bool attribute to a Python bool
        """
    def __init__(self, cast_from_attr: Attribute) -> None: ...
    @property
    def static_typeid(self) -> TypeID: ...
    @property
    def type(self) -> Type: ...
    @property
    def typeid(self) -> TypeID: ...
    @property
    def value(self) -> bool:
        """
        Returns the value of the bool attribute
        """

class ComplexType(Type):
    static_typeid: ClassVar[TypeID]
    @staticmethod
    def get(arg0: Type) -> ComplexType:
        """
        Create a complex type
        """
    @staticmethod
    def isinstance(other: Type) -> bool: ...
    def __init__(self, cast_from_type: Type) -> None: ...
    @property
    def element_type(self) -> Type:
        """
        Returns element type.
        """
    @property
    def typeid(self) -> TypeID: ...

class Context:
    current: ClassVar[Context] = ...  # read-only
    allow_unregistered_dialects: bool
    @staticmethod
    def _get_live_count() -> int: ...
    def _CAPICreate(self) -> object: ...
    def __enter__(self) -> Context: ...
    def __exit__(self, arg0: Any, arg1: Any, arg2: Any) -> None: ...
    def __init__(self) -> None: ...
    def _clear_live_operations(self) -> int: ...
    def _get_context_again(self) -> Context: ...
    def _get_live_module_count(self) -> int: ...
    def _get_live_operation_count(self) -> int: ...
    def _get_live_operation_objects(self) -> list[Operation]: ...
    def append_dialect_registry(self, registry: DialectRegistry) -> None: ...
    def attach_diagnostic_handler(
        self, callback: Callable[[Diagnostic], bool]
    ) -> DiagnosticHandler:
        """
        Attaches a diagnostic handler that will receive callbacks
        """
    def enable_multithreading(self, enable: bool) -> None: ...
    def get_dialect_descriptor(self, dialect_name: str) -> DialectDescriptor:
        """
        Gets or loads a dialect by name, returning its descriptor object
        """
    def is_registered_operation(self, operation_name: str) -> bool: ...
    def load_all_available_dialects(self) -> None: ...
    @property
    def _CAPIPtr(self) -> object: ...
    @property
    def d(self) -> Dialects:
        """
        Alias for 'dialect'
        """
    @property
    def dialects(self) -> Dialects:
        """
        Gets a container for accessing dialects by name
        """

class DenseBoolArrayAttr(Attribute):
    @staticmethod
    def get(
        values: Sequence[bool], context: Context | None = None
    ) -> DenseBoolArrayAttr:
        """
        Gets a uniqued dense array attribute
        """
    @staticmethod
    def isinstance(other: Attribute) -> bool: ...
    def __add__(self, arg0: list) -> DenseBoolArrayAttr: ...
    def __getitem__(self, arg0: int) -> bool: ...
    def __init__(self, cast_from_attr: Attribute) -> None: ...
    def __iter__(
        self,
    ) -> DenseBoolArrayIterator: ...
    def __len__(self) -> int: ...
    @property
    def static_typeid(self) -> TypeID: ...
    @property
    def type(self) -> Type: ...
    @property
    def typeid(self) -> TypeID: ...

class DenseBoolArrayIterator:
    def __iter__(self) -> DenseBoolArrayIterator: ...
    def __next__(self) -> bool: ...

class DenseElementsAttr(Attribute):
    @staticmethod
    def get(
        array: Buffer,
        signless: bool = True,
        type: Type | None = None,
        shape: list[int] | None = None,
        context: Context | None = None,
    ) -> DenseElementsAttr:
        """
        Gets a DenseElementsAttr from a Python buffer or array.

        When `type` is not provided, then some limited type inferencing is done based
        on the buffer format. Support presently exists for 8/16/32/64 signed and
        unsigned integers and float16/float32/float64. DenseElementsAttrs of these
        types can also be converted back to a corresponding buffer.

        For conversions outside of these types, a `type=` must be explicitly provided
        and the buffer contents must be bit-castable to the MLIR internal
        representation:

          * Integer types (except for i1): the buffer must be byte aligned to the
            next byte boundary.
          * Floating point types: Must be bit-castable to the given floating point
            size.
          * i1 (bool): Bit packed into 8bit words where the bit pattern matches a
            row major ordering. An arbitrary Numpy `bool_` array can be bit packed to
            this specification with: `np.packbits(ary, axis=None, bitorder='little')`.

        If a single element buffer is passed (or for i1, a single byte with value 0
        or 255), then a splat will be created.

        Args:
          array: The array or buffer to convert.
          signless: If inferring an appropriate MLIR type, use signless types for
            integers (defaults True).
          type: Skips inference of the MLIR element type and uses this instead. The
            storage size must be consistent with the actual contents of the buffer.
          shape: Overrides the shape of the buffer when constructing the MLIR
            shaped type. This is needed when the physical and logical shape differ (as
            for i1).
          context: Explicit context, if not from context manager.

        Returns:
          DenseElementsAttr on success.

        Raises:
          ValueError: If the type of the buffer or array cannot be matched to an MLIR
            type or if the buffer does not meet expectations.
        """
    @staticmethod
    def get_splat(shaped_type: Type, element_attr: Attribute) -> DenseElementsAttr:
        """
        Gets a DenseElementsAttr where all values are the same
        """
    @staticmethod
    def isinstance(other: Attribute) -> bool: ...
    def __init__(self, cast_from_attr: Attribute) -> None: ...
    def __len__(self) -> int: ...
    def get_splat_value(self) -> Attribute: ...
    @property
    def is_splat(self) -> bool: ...
    @property
    def static_typeid(self) -> TypeID: ...
    @property
    def type(self) -> Type: ...
    @property
    def typeid(self) -> TypeID: ...

class DenseF32ArrayAttr(Attribute):
    @staticmethod
    def get(
        values: Sequence[float], context: Context | None = None
    ) -> DenseF32ArrayAttr:
        """
        Gets a uniqued dense array attribute
        """
    @staticmethod
    def isinstance(other: Attribute) -> bool: ...
    def __add__(self, arg0: list) -> DenseF32ArrayAttr: ...
    def __getitem__(self, arg0: int) -> float: ...
    def __init__(self, cast_from_attr: Attribute) -> None: ...
    def __iter__(
        self,
    ) -> DenseF32ArrayIterator: ...
    def __len__(self) -> int: ...
    @property
    def static_typeid(self) -> TypeID: ...
    @property
    def type(self) -> Type: ...
    @property
    def typeid(self) -> TypeID: ...

class DenseF32ArrayIterator:
    def __iter__(self) -> DenseF32ArrayIterator: ...
    def __next__(self) -> float: ...

class DenseF64ArrayAttr(Attribute):
    @staticmethod
    def get(
        values: Sequence[float], context: Context | None = None
    ) -> DenseF64ArrayAttr:
        """
        Gets a uniqued dense array attribute
        """
    @staticmethod
    def isinstance(other: Attribute) -> bool: ...
    def __add__(self, arg0: list) -> DenseF64ArrayAttr: ...
    def __getitem__(self, arg0: int) -> float: ...
    def __init__(self, cast_from_attr: Attribute) -> None: ...
    def __iter__(
        self,
    ) -> DenseF64ArrayIterator: ...
    def __len__(self) -> int: ...
    @property
    def static_typeid(self) -> TypeID: ...
    @property
    def type(self) -> Type: ...
    @property
    def typeid(self) -> TypeID: ...

class DenseF64ArrayIterator:
    def __iter__(self) -> DenseF64ArrayIterator: ...
    def __next__(self) -> float: ...

class DenseFPElementsAttr(DenseElementsAttr):
    @staticmethod
    def get(
        array: Buffer,
        signless: bool = True,
        type: Type | None = None,
        shape: list[int] | None = None,
        context: Context | None = None,
    ) -> DenseFPElementsAttr: ...
    @staticmethod
    def isinstance(other: Attribute) -> bool: ...
    def __getitem__(self, arg0: int) -> float: ...
    def __init__(self, cast_from_attr: Attribute) -> None: ...
    @property
    def static_typeid(self) -> TypeID: ...
    @property
    def type(self) -> Type: ...
    @property
    def typeid(self) -> TypeID: ...

class DenseI16ArrayAttr(Attribute):
    @staticmethod
    def get(values: Sequence[int], context: Context | None = None) -> DenseI16ArrayAttr:
        """
        Gets a uniqued dense array attribute
        """
    @staticmethod
    def isinstance(other: Attribute) -> bool: ...
    def __add__(self, arg0: list) -> DenseI16ArrayAttr: ...
    def __getitem__(self, arg0: int) -> int: ...
    def __init__(self, cast_from_attr: Attribute) -> None: ...
    def __iter__(
        self,
    ) -> DenseI16ArrayIterator: ...
    def __len__(self) -> int: ...
    @property
    def static_typeid(self) -> TypeID: ...
    @property
    def type(self) -> Type: ...
    @property
    def typeid(self) -> TypeID: ...

class DenseI16ArrayIterator:
    def __iter__(self) -> DenseI16ArrayIterator: ...
    def __next__(self) -> int: ...

class DenseI32ArrayAttr(Attribute):
    @staticmethod
    def get(values: Sequence[int], context: Context | None = None) -> DenseI32ArrayAttr:
        """
        Gets a uniqued dense array attribute
        """
    @staticmethod
    def isinstance(other: Attribute) -> bool: ...
    def __add__(self, arg0: list) -> DenseI32ArrayAttr: ...
    def __getitem__(self, arg0: int) -> int: ...
    def __init__(self, cast_from_attr: Attribute) -> None: ...
    def __iter__(
        self,
    ) -> DenseI32ArrayIterator: ...
    def __len__(self) -> int: ...
    @property
    def static_typeid(self) -> TypeID: ...
    @property
    def type(self) -> Type: ...
    @property
    def typeid(self) -> TypeID: ...

class DenseI32ArrayIterator:
    def __iter__(self) -> DenseI32ArrayIterator: ...
    def __next__(self) -> int: ...

class DenseI64ArrayAttr(Attribute):
    @staticmethod
    def get(values: Sequence[int], context: Context | None = None) -> DenseI64ArrayAttr:
        """
        Gets a uniqued dense array attribute
        """
    @staticmethod
    def isinstance(other: Attribute) -> bool: ...
    def __add__(self, arg0: list) -> DenseI64ArrayAttr: ...
    def __getitem__(self, arg0: int) -> int: ...
    def __init__(self, cast_from_attr: Attribute) -> None: ...
    def __iter__(
        self,
    ) -> DenseI16ArrayIterator: ...
    def __len__(self) -> int: ...
    @property
    def static_typeid(self) -> TypeID: ...
    @property
    def type(self) -> Type: ...
    @property
    def typeid(self) -> TypeID: ...

class DenseI64ArrayIterator:
    def __iter__(self) -> DenseI64ArrayIterator: ...
    def __next__(self) -> int: ...

class DenseI8ArrayAttr(Attribute):
    @staticmethod
    def get(values: Sequence[int], context: Context | None = None) -> DenseI8ArrayAttr:
        """
        Gets a uniqued dense array attribute
        """
    @staticmethod
    def isinstance(other: Attribute) -> bool: ...
    def __add__(self, arg0: list) -> DenseI8ArrayAttr: ...
    def __getitem__(self, arg0: int) -> int: ...
    def __init__(self, cast_from_attr: Attribute) -> None: ...
    def __iter__(
        self,
    ) -> DenseI8ArrayIterator: ...
    def __len__(self) -> int: ...
    @property
    def static_typeid(self) -> TypeID: ...
    @property
    def type(self) -> Type: ...
    @property
    def typeid(self) -> TypeID: ...

class DenseI8ArrayIterator:
    def __iter__(self) -> DenseI8ArrayIterator: ...
    def __next__(self) -> int: ...

class DenseIntElementsAttr(DenseElementsAttr):
    @staticmethod
    def get(
        array: Buffer,
        signless: bool = True,
        type: Type | None = None,
        shape: list[int] | None = None,
        context: Context | None = None,
    ) -> DenseIntElementsAttr: ...
    @staticmethod
    def isinstance(other: Attribute) -> bool: ...
    def __getitem__(self, arg0: int) -> int: ...
    def __init__(self, cast_from_attr: Attribute) -> None: ...
    @property
    def static_typeid(self) -> TypeID: ...
    @property
    def type(self) -> Type: ...
    @property
    def typeid(self) -> TypeID: ...

class DenseResourceElementsAttr(Attribute):
    @staticmethod
    def get_from_buffer(
        array: Buffer,
        name: str,
        type: Type,
        alignment: int | None = None,
        is_mutable: bool = False,
        context: Context | None = None,
    ) -> DenseResourceElementsAttr:
        """
        Gets a DenseResourceElementsAttr from a Python buffer or array.

        This function does minimal validation or massaging of the data, and it is
        up to the caller to ensure that the buffer meets the characteristics
        implied by the shape.

        The backing buffer and any user objects will be retained for the lifetime
        of the resource blob. This is typically bounded to the context but the
        resource can have a shorter lifespan depending on how it is used in
        subsequent processing.

        Args:
          buffer: The array or buffer to convert.
          name: Name to provide to the resource (may be changed upon collision).
          type: The explicit ShapedType to construct the attribute with.
          context: Explicit context, if not from context manager.

        Returns:
          DenseResourceElementsAttr on success.

        Raises:
          ValueError: If the type of the buffer or array cannot be matched to an MLIR
            type or if the buffer does not meet expectations.
        """
    @staticmethod
    def isinstance(other: Attribute) -> bool: ...
    def __init__(self, cast_from_attr: Attribute) -> None: ...
    @property
    def static_typeid(self) -> TypeID: ...
    @property
    def type(self) -> Type: ...
    @property
    def typeid(self) -> TypeID: ...

class Diagnostic:
    @property
    def location(self) -> Location: ...
    @property
    def message(self) -> str: ...
    @property
    def notes(self) -> tuple[Diagnostic]: ...
    @property
    def severity(self) -> DiagnosticSeverity: ...

class DiagnosticHandler:
    def __enter__(self) -> DiagnosticHandler: ...
    def __exit__(self, arg0: object, arg1: object, arg2: object) -> None: ...
    def detach(self) -> None: ...
    @property
    def attached(self) -> bool: ...
    @property
    def had_error(self) -> bool: ...

class DiagnosticInfo:
    def __init__(self, arg0: Diagnostic) -> None: ...
    @property
    def location(self) -> Location: ...
    @property
    def message(self) -> str: ...
    @property
    def notes(self) -> list[DiagnosticInfo]: ...
    @property
    def severity(self) -> DiagnosticSeverity: ...

class DiagnosticSeverity:
    """
    Members:

      ERROR

      WARNING

      NOTE

      REMARK
    """

    ERROR: ClassVar[DiagnosticSeverity]  # value = <DiagnosticSeverity.ERROR: 0>
    NOTE: ClassVar[DiagnosticSeverity]  # value = <DiagnosticSeverity.NOTE: 2>
    REMARK: ClassVar[DiagnosticSeverity]  # value = <DiagnosticSeverity.REMARK: 3>
    WARNING: ClassVar[DiagnosticSeverity]  # value = <DiagnosticSeverity.WARNING: 1>
    __members__: ClassVar[
        dict[str, DiagnosticSeverity]
    ]  # value = {'ERROR': <DiagnosticSeverity.ERROR: 0>, 'WARNING': <DiagnosticSeverity.WARNING: 1>, 'NOTE': <DiagnosticSeverity.NOTE: 2>, 'REMARK': <DiagnosticSeverity.REMARK: 3>}
    def __eq__(self, other: Any) -> bool: ...
    def __getstate__(self) -> int: ...
    def __hash__(self) -> int: ...
    def __index__(self) -> int: ...
    def __init__(self, value: int) -> None: ...
    def __int__(self) -> int: ...
    def __ne__(self, other: Any) -> bool: ...
    def __setstate__(self, state: int) -> None: ...
    @property
    def name(self) -> str: ...
    @property
    def value(self) -> int: ...

class Dialect:
    def __init__(self, descriptor: DialectDescriptor) -> None: ...
    @property
    def descriptor(self) -> DialectDescriptor: ...

class DialectDescriptor:
    @property
    def namespace(self) -> str: ...

class DialectRegistry:
    def _CAPICreate(self) -> DialectRegistry: ...
    def __init__(self) -> None: ...
    @property
    def _CAPIPtr(self) -> object: ...

class Dialects:
    def __getattr__(self, arg0: str) -> Dialect: ...
    def __getitem__(self, arg0: str) -> Dialect: ...

class DictAttr(Attribute):
    static_typeid: ClassVar[TypeID]
    @staticmethod
    def get(value: dict = {}, context: Context | None = None) -> DictAttr:
        """
        Gets an uniqued Dict attribute
        """
    @staticmethod
    def isinstance(other: Attribute) -> bool: ...
    def __contains__(self, arg0: str) -> bool: ...
    @overload
    def __getitem__(self, arg0: str) -> Attribute: ...
    @overload
    def __getitem__(self, arg0: int) -> NamedAttribute: ...
    def __init__(self, cast_from_attr: Attribute) -> None: ...
    def __len__(self) -> int: ...
    @property
    def type(self) -> Type: ...
    @property
    def typeid(self) -> TypeID: ...

class FloatType(Type):
    @staticmethod
    def isinstance(other: Type) -> bool: ...
    def __init__(self, cast_from_type: Type) -> None: ...
    @property
    def width(self) -> int:
        """
        Returns the width of the floating-point type.
        """

class F16Type(FloatType):
    static_typeid: ClassVar[TypeID]
    @staticmethod
    def get(context: Context | None = None) -> F16Type:
        """
        Create a f16 type.
        """
    @staticmethod
    def isinstance(other: Type) -> bool: ...
    def __init__(self, cast_from_type: Type) -> None: ...
    @property
    def typeid(self) -> TypeID: ...

class F32Type(FloatType):
    static_typeid: ClassVar[TypeID]
    @staticmethod
    def get(context: Context | None = None) -> F32Type:
        """
        Create a f32 type.
        """
    @staticmethod
    def isinstance(other: Type) -> bool: ...
    def __init__(self, cast_from_type: Type) -> None: ...
    @property
    def typeid(self) -> TypeID: ...

class F64Type(FloatType):
    static_typeid: ClassVar[TypeID]
    @staticmethod
    def get(context: Context | None = None) -> F64Type:
        """
        Create a f64 type.
        """
    @staticmethod
    def isinstance(other: Type) -> bool: ...
    def __init__(self, cast_from_type: Type) -> None: ...
    @property
    def typeid(self) -> TypeID: ...

class FlatSymbolRefAttr(Attribute):
    @staticmethod
    def get(value: str, context: Context | None = None) -> FlatSymbolRefAttr:
        """
        Gets a uniqued FlatSymbolRef attribute
        """
    @staticmethod
    def isinstance(other: Attribute) -> bool: ...
    def __init__(self, cast_from_attr: Attribute) -> None: ...
    @property
    def static_typeid(self) -> TypeID: ...
    @property
    def type(self) -> Type: ...
    @property
    def typeid(self) -> TypeID: ...
    @property
    def value(self) -> str:
        """
        Returns the value of the FlatSymbolRef attribute as a string
        """

class Float4E2M1FNType(FloatType):
    static_typeid: ClassVar[TypeID]
    @staticmethod
    def get(context: Context | None = None) -> Float4E2M1FNType:
        """
        Create a float4_e2m1fn type.
        """
    @staticmethod
    def isinstance(other: Type) -> bool: ...
    def __init__(self, cast_from_type: Type) -> None: ...
    @property
    def typeid(self) -> TypeID: ...

class Float6E2M3FNType(FloatType):
    static_typeid: ClassVar[TypeID]
    @staticmethod
    def get(context: Context | None = None) -> Float6E2M3FNType:
        """
        Create a float6_e2m3fn type.
        """
    @staticmethod
    def isinstance(other: Type) -> bool: ...
    def __init__(self, cast_from_type: Type) -> None: ...
    @property
    def typeid(self) -> TypeID: ...

class Float6E3M2FNType(FloatType):
    static_typeid: ClassVar[TypeID]
    @staticmethod
    def get(context: Context | None = None) -> Float6E3M2FNType:
        """
        Create a float6_e3m2fn type.
        """
    @staticmethod
    def isinstance(other: Type) -> bool: ...
    def __init__(self, cast_from_type: Type) -> None: ...
    @property
    def typeid(self) -> TypeID: ...

class Float8E3M4Type(FloatType):
    static_typeid: ClassVar[TypeID]
    @staticmethod
    def get(context: Context | None = None) -> Float8E3M4Type:
        """
        Create a float8_e3m4 type.
        """
    @staticmethod
    def isinstance(other: Type) -> bool: ...
    def __init__(self, cast_from_type: Type) -> None: ...
    @property
    def typeid(self) -> TypeID: ...

class Float8E4M3B11FNUZType(FloatType):
    static_typeid: ClassVar[TypeID]
    @staticmethod
    def get(context: Context | None = None) -> Float8E4M3B11FNUZType:
        """
        Create a float8_e4m3b11fnuz type.
        """
    @staticmethod
    def isinstance(other: Type) -> bool: ...
    def __init__(self, cast_from_type: Type) -> None: ...
    @property
    def typeid(self) -> TypeID: ...

class Float8E4M3FNType(FloatType):
    static_typeid: ClassVar[TypeID]
    @staticmethod
    def get(context: Context | None = None) -> Float8E4M3FNType:
        """
        Create a float8_e4m3fn type.
        """
    @staticmethod
    def isinstance(other: Type) -> bool: ...
    def __init__(self, cast_from_type: Type) -> None: ...
    @property
    def typeid(self) -> TypeID: ...

class Float8E4M3FNUZType(FloatType):
    static_typeid: ClassVar[TypeID]
    @staticmethod
    def get(context: Context | None = None) -> Float8E4M3FNUZType:
        """
        Create a float8_e4m3fnuz type.
        """
    @staticmethod
    def isinstance(other: Type) -> bool: ...
    def __init__(self, cast_from_type: Type) -> None: ...
    @property
    def typeid(self) -> TypeID: ...

class Float8E4M3Type(FloatType):
    static_typeid: ClassVar[TypeID]
    @staticmethod
    def get(context: Context | None = None) -> Float8E4M3Type:
        """
        Create a float8_e4m3 type.
        """
    @staticmethod
    def isinstance(other: Type) -> bool: ...
    def __init__(self, cast_from_type: Type) -> None: ...
    @property
    def typeid(self) -> TypeID: ...

class Float8E5M2FNUZType(FloatType):
    static_typeid: ClassVar[TypeID]
    @staticmethod
    def get(context: Context | None = None) -> Float8E5M2FNUZType:
        """
        Create a float8_e5m2fnuz type.
        """
    @staticmethod
    def isinstance(other: Type) -> bool: ...
    def __init__(self, cast_from_type: Type) -> None: ...
    @property
    def typeid(self) -> TypeID: ...

class Float8E5M2Type(FloatType):
    static_typeid: ClassVar[TypeID]
    @staticmethod
    def get(context: Context | None = None) -> Float8E5M2Type:
        """
        Create a float8_e5m2 type.
        """
    @staticmethod
    def isinstance(other: Type) -> bool: ...
    def __init__(self, cast_from_type: Type) -> None: ...
    @property
    def typeid(self) -> TypeID: ...

class Float8E8M0FNUType(FloatType):
    static_typeid: ClassVar[TypeID]
    @staticmethod
    def get(context: Context | None = None) -> Float8E8M0FNUType:
        """
        Create a float8_e8m0fnu type.
        """
    @staticmethod
    def isinstance(other: Type) -> bool: ...
    def __init__(self, cast_from_type: Type) -> None: ...
    @property
    def typeid(self) -> TypeID: ...

class FloatAttr(Attribute):
    static_typeid: ClassVar[TypeID]
    @staticmethod
    def get(type: Type, value: float, loc: Location | None = None) -> FloatAttr:
        """
        Gets an uniqued float point attribute associated to a type
        """
    @staticmethod
    def get_f32(value: float, context: Context | None = None) -> FloatAttr:
        """
        Gets an uniqued float point attribute associated to a f32 type
        """
    @staticmethod
    def get_f64(value: float, context: Context | None = None) -> FloatAttr:
        """
        Gets an uniqued float point attribute associated to a f64 type
        """
    @staticmethod
    def isinstance(other: Attribute) -> bool: ...
    def __float__(self: Attribute) -> float:
        """
        Converts the value of the float attribute to a Python float
        """
    def __init__(self, cast_from_attr: Attribute) -> None: ...
    @property
    def type(self) -> Type: ...
    @property
    def typeid(self) -> TypeID: ...
    @property
    def value(self) -> float:
        """
        Returns the value of the float attribute
        """

class FloatTF32Type(FloatType):
    static_typeid: ClassVar[TypeID]
    @staticmethod
    def get(context: Context | None = None) -> FloatTF32Type:
        """
        Create a tf32 type.
        """
    @staticmethod
    def isinstance(other: Type) -> bool: ...
    def __init__(self, cast_from_type: Type) -> None: ...
    @property
    def typeid(self) -> TypeID: ...

class FunctionType(Type):
    static_typeid: ClassVar[TypeID]
    @staticmethod
    def get(
        inputs: list[Type], results: list[Type], context: Context | None = None
    ) -> FunctionType:
        """
        Gets a FunctionType from a List of input and result types
        """
    @staticmethod
    def isinstance(other: Type) -> bool: ...
    def __init__(self, cast_from_type: Type) -> None: ...
    @property
    def inputs(self) -> list:
        """
        Returns the List of input types in the FunctionType.
        """
    @property
    def results(self) -> list:
        """
        Returns the List of result types in the FunctionType.
        """
    @property
    def typeid(self) -> TypeID: ...

class IndexType(Type):
    static_typeid: ClassVar[TypeID]
    @staticmethod
    def get(context: Context | None = None) -> IndexType:
        """
        Create a index type.
        """
    @staticmethod
    def isinstance(other: Type) -> bool: ...
    def __init__(self, cast_from_type: Type) -> None: ...
    @property
    def typeid(self) -> TypeID: ...

class InferShapedTypeOpInterface:
    def __init__(self, object: object, context: Context | None = None) -> None:
        """
        Creates an interface from a given operation/opview object or from a
        subclass of OpView. Raises ValueError if the operation does not implement the
        interface.
        """
    def inferReturnTypeComponents(
        self,
        operands: list | None = None,
        attributes: Attribute | None = None,
        properties=None,
        regions: list[Region] | None = None,
        context: Context | None = None,
        loc: Location | None = None,
    ) -> list[ShapedTypeComponents]:
        """
        Given the arguments required to build an operation, attempts to infer
        its return shaped type components. Raises ValueError on failure.
        """
    @property
    def operation(self) -> Operation:
        """
        Returns an Operation for which the interface was constructed.
        """
    @property
    def opview(self) -> OpView:
        """
        Returns an OpView subclass _instance_ for which the interface was
        constructed
        """

class InferTypeOpInterface:
    def __init__(self, object: object, context: Context | None = None) -> None:
        """
        Creates an interface from a given operation/opview object or from a
        subclass of OpView. Raises ValueError if the operation does not implement the
        interface.
        """
    def inferReturnTypes(
        self,
        operands: list | None = None,
        attributes: Attribute | None = None,
        properties=None,
        regions: list[Region] | None = None,
        context: Context | None = None,
        loc: Location | None = None,
    ) -> list[Type]:
        """
        Given the arguments required to build an operation, attempts to infer
        its return types. Raises ValueError on failure.
        """
    @property
    def operation(self) -> Operation:
        """
        Returns an Operation for which the interface was constructed.
        """
    @property
    def opview(self) -> OpView:
        """
        Returns an OpView subclass _instance_ for which the interface was
        constructed
        """

class InsertionPoint:
    current: ClassVar[InsertionPoint] = ...  # read-only
    @staticmethod
    def at_block_begin(block: Block) -> InsertionPoint:
        """
        Inserts at the beginning of the block.
        """
    @staticmethod
    def at_block_terminator(block: Block) -> InsertionPoint:
        """
        Inserts before the block terminator.
        """
    def __enter__(self) -> InsertionPoint: ...
    def __exit__(self, arg0: Any, arg1: Any, arg2: Any) -> None: ...
    @overload
    def __init__(self, block: Block) -> None:
        """
        Inserts after the last operation but still inside the block.
        """
    @overload
    def __init__(self, beforeOperation: _OperationBase) -> None:
        """
        Inserts before a referenced operation.
        """
    def insert(self, operation: _OperationBase) -> None:
        """
        Inserts an operation.
        """
    @property
    def block(self) -> Block:
        """
        Returns the block that this InsertionPoint points to.
        """
    @property
    def ref_operation(self) -> _OperationBase | None:
        """
        The reference operation before which new operations are inserted, or None if the insertion point is at the end of the block
        """

class IntegerAttr(Attribute):
    static_typeid: ClassVar[TypeID]
    @staticmethod
    def get(type: Type, value: int) -> IntegerAttr:
        """
        Gets an uniqued integer attribute associated to a type
        """
    @staticmethod
    def isinstance(other: Attribute) -> bool: ...
    def __init__(self, cast_from_attr: Attribute) -> None: ...
    def __int__(self) -> int:
        """
        Converts the value of the integer attribute to a Python int
        """
    @property
    def type(self) -> Type: ...
    @property
    def typeid(self) -> TypeID: ...
    @property
    def value(self) -> int:
        """
        Returns the value of the integer attribute
        """

class IntegerSet:
    @staticmethod
    def get(
        num_dims: int,
        num_symbols: int,
        exprs: list,
        eq_flags: list[bool],
        context: Context | None = None,
    ) -> IntegerSet: ...
    @staticmethod
    def get_empty(
        num_dims: int, num_symbols: int, context: Context | None = None
    ) -> IntegerSet: ...
    def _CAPICreate(self) -> IntegerSet: ...
    @overload
    def __eq__(self, arg0: IntegerSet) -> bool: ...
    @overload
    def __eq__(self, arg0: object) -> bool: ...
    def __hash__(self) -> int: ...
    def dump(self) -> None:
        """
        Dumps a debug representation of the object to stderr.
        """
    def get_replaced(
        self,
        dim_exprs: list,
        symbol_exprs: list,
        num_result_dims: int,
        num_result_symbols: int,
    ) -> IntegerSet: ...
    @property
    def _CAPIPtr(self) -> object: ...
    @property
    def constraints(self) -> IntegerSetConstraintList: ...
    @property
    def context(self) -> Context: ...
    @property
    def is_canonical_empty(self) -> bool: ...
    @property
    def n_dims(self) -> int: ...
    @property
    def n_equalities(self) -> int: ...
    @property
    def n_inequalities(self) -> int: ...
    @property
    def n_inputs(self) -> int: ...
    @property
    def n_symbols(self) -> int: ...

class IntegerSetAttr(Attribute):
    static_typeid: ClassVar[TypeID]
    @staticmethod
    def get(integer_set) -> IntegerSetAttr:
        """
        Gets an attribute wrapping an IntegerSet.
        """
    @staticmethod
    def isinstance(other: Attribute) -> bool: ...
    def __init__(self, cast_from_attr: Attribute) -> None: ...
    @property
    def type(self) -> Type: ...
    @property
    def typeid(self) -> TypeID: ...

class IntegerSetConstraint:
    def __init__(self, *args, **kwargs) -> None: ...
    @property
    def expr(self) -> AffineExpr: ...
    @property
    def is_eq(self) -> bool: ...

class IntegerSetConstraintList:
    def __init__(self, *args, **kwargs) -> None: ...
    def __add__(self, arg0: IntegerSetConstraintList) -> list[IntegerSetConstraint]: ...
    @overload
    def __getitem__(self, arg0: int) -> IntegerSetConstraint: ...
    @overload
    def __getitem__(self, arg0: slice) -> IntegerSetConstraintList: ...
    def __len__(self) -> int: ...

class IntegerType(Type):
    static_typeid: ClassVar[TypeID]
    @staticmethod
    def get_signed(width: int, context: Context | None = None) -> IntegerType:
        """
        Create a signed integer type
        """
    @staticmethod
    def get_signless(width: int, context: Context | None = None) -> IntegerType:
        """
        Create a signless integer type
        """
    @staticmethod
    def get_unsigned(width: int, context: Context | None = None) -> IntegerType:
        """
        Create an unsigned integer type
        """
    @staticmethod
    def isinstance(other: Type) -> bool: ...
    def __init__(self, cast_from_type: Type) -> None: ...
    @property
    def is_signed(self) -> bool:
        """
        Returns whether this is a signed integer
        """
    @property
    def is_signless(self) -> bool:
        """
        Returns whether this is a signless integer
        """
    @property
    def is_unsigned(self) -> bool:
        """
        Returns whether this is an unsigned integer
        """
    @property
    def typeid(self) -> TypeID: ...
    @property
    def width(self) -> int:
        """
        Returns the width of the integer type
        """

class Location:
    current: ClassVar[Location] = ...  # read-only
    __hash__: ClassVar[None] = None
    @staticmethod
    def callsite(
        callee: Location, frames: Sequence[Location], context: Context | None = None
    ) -> Location:
        """
        Gets a Location representing a caller and callsite
        """
    @staticmethod
    def file(
        filename: str, line: int, col: int, context: Context | None = None
    ) -> Location:
        """
        Gets a Location representing a file, line and column
        """
    @staticmethod
    def from_attr(attribute: Attribute, context: Context | None = None) -> Location:
        """
        Gets a Location from a LocationAttr
        """
    @staticmethod
    def fused(
        locations: Sequence[Location],
        metadata: Attribute | None = None,
        context: Context | None = None,
    ) -> Location:
        """
        Gets a Location representing a fused location with optional metadata
        """
    @staticmethod
    def name(
        name: str,
        childLoc: Location | None = None,
        context: Context | None = None,
    ) -> Location:
        """
        Gets a Location representing a named location with optional child location
        """
    @staticmethod
    def unknown(context: Context | None = None) -> Location:
        """
        Gets a Location representing an unknown location
        """
    def _CAPICreate(self) -> Location: ...
    def __enter__(self) -> Location: ...
    @overload
    def __eq__(self, arg0: Location) -> bool: ...
    @overload
    def __eq__(self, arg0: Location) -> bool: ...
    def __exit__(self, arg0: object, arg1: object, arg2: object) -> None: ...
    def emit_error(self, message: str) -> None:
        """
        Emits an error at this location
        """
    @property
    def _CAPIPtr(self) -> object: ...
    @property
    def attr(self) -> Attribute:
        """
        Get the underlying LocationAttr
        """
    @property
    def context(self) -> Context:
        """
        Context that owns the Location
        """

class MemRefType(ShapedType):
    static_typeid: ClassVar[TypeID]
    @staticmethod
    def get(
        shape: list[int],
        element_type: Type,
        layout: Attribute = None,
        memory_space: Attribute = None,
        loc: Location | None = None,
    ) -> MemRefType:
        """
        Create a memref type
        """
    @staticmethod
    def isinstance(other: Type) -> bool: ...
    def __init__(self, cast_from_type: Type) -> None: ...
    @property
    def affine_map(self) -> AffineMap:
        """
        The layout of the MemRef type as an affine map.
        """
    @property
    def layout(self) -> Attribute:
        """
        The layout of the MemRef type.
        """
    @property
    def memory_space(self) -> Attribute | None:
        """
        Returns the memory space of the given MemRef type.
        """
    @property
    def typeid(self) -> TypeID: ...
    def get_strides_and_offset(self) -> tuple[list[int], list[int]]:
        """
        The strides and offset of the MemRef type.
        """

class Module:
    @staticmethod
    def create(loc: Location | None = None) -> Module:
        """
        Creates an empty module
        """
    @staticmethod
    def parse(asm: str | bytes, context: Context | None = None) -> Module:
        """
        Parses a module's assembly format from a string.

        Returns a new MlirModule or raises an MLIRError if the parsing fails.

        See also: https://mlir.llvm.org/docs/LangRef/
        """
    def _CAPICreate(self) -> Any: ...
    def __str__(self) -> str:
        """
        Gets the assembly form of the operation with default options.

        If more advanced control over the assembly formatting or I/O options is needed,
        use the dedicated print or get_asm method, which supports keyword arguments to
        customize behavior.
        """
    def dump(self) -> None:
        """
        Dumps a debug representation of the object to stderr.
        """
    @property
    def _CAPIPtr(self) -> object: ...
    @property
    def body(self) -> Block:
        """
        Return the block for this module
        """
    @property
    def context(self) -> Context:
        """
        Context that created the Module
        """
    @property
    def operation(self) -> Operation:
        """
        Accesses the module as an operation
        """

class MLIRError(Exception):
    def __init__(
        self, message: str, error_diagnostics: list[DiagnosticInfo]
    ) -> None: ...

class NamedAttribute:
    @property
    def attr(self) -> Attribute:
        """
        The underlying generic attribute of the NamedAttribute binding
        """
    @property
    def name(self) -> str:
        """
        The name of the NamedAttribute binding
        """

class NoneType(Type):
    static_typeid: ClassVar[TypeID]
    @staticmethod
    def get(context: Context | None = None) -> NoneType:
        """
        Create a none type.
        """
    @staticmethod
    def isinstance(other: Type) -> bool: ...
    def __init__(self, cast_from_type: Type) -> None: ...
    @property
    def typeid(self) -> TypeID: ...

class OpAttributeMap:
    def __contains__(self, arg0: str) -> bool: ...
    def __delitem__(self, arg0: str) -> None: ...
    @overload
    def __getitem__(self, arg0: str) -> Attribute: ...
    @overload
    def __getitem__(self, arg0: int) -> NamedAttribute: ...
    def __len__(self) -> int: ...
    def __setitem__(self, arg0: str, arg1: Attribute) -> None: ...

class OpOperand:
    @property
    def operand_number(self) -> int: ...
    @property
    def owner(self) -> _OperationBase: ...

class OpOperandIterator:
    def __iter__(self) -> OpOperandIterator: ...
    def __next__(self) -> OpOperand: ...

class OpOperandList:
    def __add__(self, arg0: OpOperandList) -> list[Value]: ...
    @overload
    def __getitem__(self, arg0: int) -> Value: ...
    @overload
    def __getitem__(self, arg0: slice) -> OpOperandList: ...
    def __len__(self) -> int: ...
    def __setitem__(self, arg0: int, arg1: Value) -> None: ...

class OpResult(Value):
    @staticmethod
    def isinstance(other_value: Value) -> bool: ...
    def __init__(self, value: Value) -> None: ...
    @staticmethod
    def isinstance(arg: Any) -> bool: ...
    @property
    def owner(self) -> _OperationBase: ...
    @property
    def result_number(self) -> int: ...

class OpResultList:
    def __add__(self, arg0: OpResultList) -> list[OpResult]: ...
    @overload
    def __getitem__(self, arg0: int) -> OpResult: ...
    @overload
    def __getitem__(self, arg0: slice) -> OpResultList: ...
    def __len__(self) -> int: ...
    @property
    def owner(self) -> _OperationBase: ...
    @property
    def types(self) -> list[Type]: ...

class OpSuccessors:
    def __add__(self, arg0: OpSuccessors) -> list[Block]: ...
    @overload
    def __getitem__(self, arg0: int) -> Block: ...
    @overload
    def __getitem__(self, arg0: slice) -> OpSuccessors: ...
    def __setitem__(self, arg0: int, arg1: Block) -> None: ...
    def __len__(self) -> int: ...

class OpView(_OperationBase):
    _ODS_OPERAND_SEGMENTS: ClassVar[None] = ...
    _ODS_REGIONS: ClassVar[tuple] = ...
    _ODS_RESULT_SEGMENTS: ClassVar[None] = ...
    def __init__(self, operation: _OperationBase) -> None: ...
    @classmethod
    def build_generic(
        cls: type[_TOperation],
        results: Sequence[Type] | None = None,
        operands: Sequence[Value] | None = None,
        attributes: dict[str, Attribute] | None = None,
        successors: Sequence[Block] | None = None,
        regions: int | None = None,
        loc: Location | None = None,
        ip: InsertionPoint | None = None,
    ) -> _TOperation:
        """
        Builds a specific, generated OpView based on class level attributes.
        """
    @classmethod
    def parse(
        cls: type[_TOperation],
        source: str | bytes,
        *,
        source_name: str = "",
        context: Context | None = None,
    ) -> _TOperation:
        """
        Parses a specific, generated OpView based on class level attributes
        """
    def __init__(self, operation: _OperationBase) -> None: ...
    @property
    def operation(self) -> _OperationBase: ...
    @property
    def opview(self) -> OpView: ...
    @property
    def successors(self) -> OpSuccessors:
        """
        Returns the List of Operation successors.
        """

class OpaqueAttr(Attribute):
    static_typeid: ClassVar[TypeID]
    @staticmethod
    def get(
        dialect_namespace: str,
        buffer: Buffer,
        type: Type,
        context: Context | None = None,
    ) -> OpaqueAttr:
        """
        Gets an Opaque attribute.
        """
    @staticmethod
    def isinstance(other: Attribute) -> bool: ...
    def __init__(self, cast_from_attr: Attribute) -> None: ...
    @property
    def data(self) -> bytes:
        """
        Returns the data for the Opaqued attributes as `bytes`
        """
    @property
    def dialect_namespace(self) -> str:
        """
        Returns the dialect namespace for the Opaque attribute as a string
        """
    @property
    def type(self) -> Type: ...
    @property
    def typeid(self) -> TypeID: ...

class OpaqueType(Type):
    static_typeid: ClassVar[TypeID]
    @staticmethod
    def get(
        dialect_namespace: str, buffer: str, context: Context | None = None
    ) -> OpaqueType:
        """
        Create an unregistered (opaque) dialect type.
        """
    @staticmethod
    def isinstance(other: Type) -> bool: ...
    def __init__(self, cast_from_type: Type) -> None: ...
    @property
    def data(self) -> str:
        """
        Returns the data for the Opaque type as a string.
        """
    @property
    def dialect_namespace(self) -> str:
        """
        Returns the dialect namespace for the Opaque type as a string.
        """
    @property
    def typeid(self) -> TypeID: ...

class Operation(_OperationBase):
    def _CAPICreate(self) -> object: ...
    @staticmethod
    def create(
        name: str,
        results: Sequence[Type] | None = None,
        operands: Sequence[Value] | None = None,
        attributes: dict[str, Attribute] | None = None,
        successors: Sequence[Block] | None = None,
        regions: int = 0,
        loc: Location | None = None,
        ip: InsertionPoint | None = None,
        infer_type: bool = False,
    ) -> Operation:
        """
        Creates a new operation.

        Args:
          name: Operation name (e.g. "dialect.operation").
          results: Sequence of Type representing op result types.
          attributes: Dict of str:Attribute.
          successors: List of Block for the operation's successors.
          regions: Number of regions to create.
          location: A Location object (defaults to resolve from context manager).
          ip: An InsertionPoint (defaults to resolve from context manager or set to
            False to disable insertion, even with an insertion point set in the
            context manager).
          infer_type: Whether to infer result types.
        Returns:
          A new "detached" Operation object. Detached operations can be added
          to blocks, which causes them to become "attached."
        """
    @staticmethod
    def parse(
        source: str | bytes, *, source_name: str = "", context: Context | None = None
    ) -> Operation:
        """
        Parses an operation. Supports both text assembly format and binary bytecode format.
        """
    def _CAPICreate(self) -> object: ...
    @property
    def _CAPIPtr(self) -> object: ...
    @property
    def operation(self) -> Operation: ...
    @property
    def opview(self) -> OpView: ...
    @property
    def successors(self) -> OpSuccessors:
        """
        Returns the List of Operation successors.
        """

class OperationIterator:
    def __iter__(self) -> OperationIterator: ...
    def __next__(self) -> OpView: ...

class OperationList:
    def __getitem__(self, arg0: int) -> OpView: ...
    def __iter__(self) -> OperationIterator: ...
    def __len__(self) -> int: ...

class RankedTensorType(ShapedType):
    static_typeid: ClassVar[TypeID]
    @staticmethod
    def get(
        shape: list[int],
        element_type: Type,
        encoding: Attribute | None = None,
        loc: Location | None = None,
    ) -> RankedTensorType:
        """
        Create a ranked tensor type
        """
    @staticmethod
    def isinstance(other: Type) -> bool: ...
    def __init__(self, cast_from_type: Type) -> None: ...
    @property
    def encoding(self) -> Attribute | None: ...
    @property
    def typeid(self) -> TypeID: ...

class Region:
    __hash__: ClassVar[None] = None
    @overload
    def __eq__(self, arg0: Region) -> bool: ...
    @overload
    def __eq__(self, arg0: object) -> bool: ...
    def __iter__(self) -> BlockIterator:
        """
        Iterates over blocks in the region.
        """
    @property
    def blocks(self) -> BlockList:
        """
        Returns a forward-optimized sequence of blocks.
        """
    @property
    def owner(self) -> OpView:
        """
        Returns the operation owning this region.
        """

class RegionIterator:
    def __iter__(self) -> RegionIterator: ...
    def __next__(self) -> Region: ...

class RegionSequence:
    def __getitem__(self, arg0: int) -> Region: ...
    def __iter__(self) -> RegionIterator: ...
    def __len__(self) -> int: ...

class ShapedType(Type):
    @staticmethod
    def get_dynamic_size() -> int:
        """
        Returns the value used to indicate dynamic dimensions in shaped types.
        """
    @staticmethod
    def get_dynamic_stride_or_offset() -> int:
        """
        Returns the value used to indicate dynamic strides or offsets in shaped types.
        """
    @staticmethod
    def is_dynamic_size(dim_size: int) -> bool:
        """
        Returns whether the given dimension size indicates a dynamic dimension.
        """
    @staticmethod
    def isinstance(other: Type) -> bool: ...
    def __init__(self, cast_from_type: Type) -> None: ...
    def get_dim_size(self, dim: int) -> int:
        """
        Returns the dim-th dimension of the given ranked shaped type.
        """
    def is_dynamic_dim(self, dim: int) -> bool:
        """
        Returns whether the dim-th dimension of the given shaped type is dynamic.
        """
    def is_dynamic_stride_or_offset(self, dim_size: int) -> bool:
        """
        Returns whether the given value is used as a placeholder for dynamic strides and offsets in shaped types.
        """
    @property
    def element_type(self) -> Type:
        """
        Returns the element type of the shaped type.
        """
    @property
    def has_rank(self) -> bool:
        """
        Returns whether the given shaped type is ranked.
        """
    @property
    def has_static_shape(self) -> bool:
        """
        Returns whether the given shaped type has a static shape.
        """
    @property
    def rank(self) -> int:
        """
        Returns the rank of the given ranked shaped type.
        """
    @property
    def shape(self) -> list[int]:
        """
        Returns the shape of the ranked shaped type as a List of integers.
        """
    @property
    def static_typeid(self) -> TypeID: ...
    @property
    def typeid(self) -> TypeID: ...

class ShapedTypeComponents:
    @staticmethod
    @overload
    def get(element_type: Type) -> ShapedTypeComponents:
        """
        Create an shaped type components object with only the element type.
        """
    @staticmethod
    @overload
    def get(shape: list, element_type: Type) -> ShapedTypeComponents:
        """
        Create a ranked shaped type components object.
        """
    @staticmethod
    @overload
    def get(
        shape: list, element_type: Type, attribute: Attribute
    ) -> ShapedTypeComponents:
        """
        Create a ranked shaped type components object with attribute.
        """
    @property
    def element_type(self) -> Type:
        """
        Returns the element type of the shaped type components.
        """
    @property
    def has_rank(self) -> bool:
        """
        Returns whether the given shaped type component is ranked.
        """
    @property
    def rank(self) -> int:
        """
        Returns the rank of the given ranked shaped type components. If the shaped type components does not have a rank, None is returned.
        """
    @property
    def shape(self) -> list[int]:
        """
        Returns the shape of the ranked shaped type components as a List of integers. Returns none if the shaped type component does not have a rank.
        """

class StridedLayoutAttr(Attribute):
    static_typeid: ClassVar[TypeID]
    @staticmethod
    def get(
        offset: int, strides: list[int], context: Context | None = None
    ) -> StridedLayoutAttr:
        """
        Gets a strided layout attribute.
        """
    @staticmethod
    def get_fully_dynamic(
        rank: int, context: Context | None = None
    ) -> StridedLayoutAttr:
        """
        Gets a strided layout attribute with dynamic offset and strides of a given rank.
        """
    @staticmethod
    def isinstance(other: Attribute) -> bool: ...
    def __init__(self, cast_from_attr: Attribute) -> None: ...
    @property
    def offset(self) -> int:
        """
        Returns the value of the float point attribute
        """
    @property
    def strides(self) -> list[int]:
        """
        Returns the value of the float point attribute
        """
    @property
    def type(self) -> Type: ...
    @property
    def typeid(self) -> TypeID: ...

class StringAttr(Attribute):
    static_typeid: ClassVar[TypeID]
    @staticmethod
    def get(value: str | bytes, context: Context | None = None) -> StringAttr:
        """
        Gets a uniqued string attribute
        """
    @staticmethod
    def get_typed(type: Type, value: str) -> StringAttr:
        """
        Gets a uniqued string attribute associated to a type
        """
    @staticmethod
    def isinstance(other: Attribute) -> bool: ...
    def __init__(self, cast_from_attr: Attribute) -> None: ...
    @property
    def type(self) -> Type: ...
    @property
    def typeid(self) -> TypeID: ...
    @property
    def value(self) -> str:
        """
        Returns the value of the string attribute
        """
    @property
    def value_bytes(self) -> bytes:
        """
        Returns the value of the string attribute as `bytes`
        """

class SymbolRefAttr(Attribute):
    @staticmethod
    def get(symbols: list[str], context: Context | None = None) -> Attribute:
        """
        Gets a uniqued SymbolRef attribute from a List of symbol names
        """
    @staticmethod
    def isinstance(other: Attribute) -> bool: ...
    def __init__(self, cast_from_attr: Attribute) -> None: ...
    @property
    def static_typeid(self) -> TypeID: ...
    @property
    def type(self) -> Type: ...
    @property
    def typeid(self) -> TypeID: ...
    @property
    def value(self) -> list[str]:
        """
        Returns the value of the SymbolRef attribute as a List[str]
        """

class SymbolTable:
    @staticmethod
    def get_symbol_name(symbol: _OperationBase) -> Attribute: ...
    @staticmethod
    def get_visibility(symbol: _OperationBase) -> Attribute: ...
    @staticmethod
    def replace_all_symbol_uses(
        old_symbol: str, new_symbol: str, from_op: _OperationBase
    ) -> None: ...
    @staticmethod
    def set_symbol_name(symbol: _OperationBase, name: str) -> None: ...
    @staticmethod
    def set_visibility(symbol: _OperationBase, visibility: str) -> None: ...
    @staticmethod
    def walk_symbol_tables(
        from_op: _OperationBase,
        all_sym_uses_visible: bool,
        callback: Callable[[_OperationBase, bool], None],
    ) -> None: ...
    def __contains__(self, arg0: str) -> bool: ...
    def __delitem__(self, arg0: str) -> None: ...
    def __getitem__(self, arg0: str) -> OpView: ...
    def __init__(self, arg0: _OperationBase) -> None: ...
    def erase(self, operation: _OperationBase) -> None: ...
    def insert(self, operation: _OperationBase) -> Attribute: ...

class TupleType(Type):
    static_typeid: ClassVar[TypeID]
    @staticmethod
    def get_tuple(elements: list[Type], context: Context | None = None) -> TupleType:
        """
        Create a Tuple type
        """
    @staticmethod
    def isinstance(other: Type) -> bool: ...
    def __init__(self, cast_from_type: Type) -> None: ...
    def get_type(self, pos: int) -> Type:
        """
        Returns the pos-th type in the Tuple type.
        """
    @property
    def num_types(self) -> int:
        """
        Returns the number of types contained in a Tuple.
        """
    @property
    def typeid(self) -> TypeID: ...

class TypeAttr(Attribute):
    static_typeid: ClassVar[TypeID]
    @staticmethod
    def get(value: Type, context: Context | None = None) -> TypeAttr:
        """
        Gets a uniqued Type attribute
        """
    @staticmethod
    def isinstance(other: Attribute) -> bool: ...
    def __init__(self, cast_from_attr: Attribute) -> None: ...
    @property
    def type(self) -> Type: ...
    @property
    def typeid(self) -> TypeID: ...
    @property
    def value(self) -> Type: ...

class TypeID:
    def _CAPICreate(self) -> TypeID: ...
    @overload
    def __eq__(self, arg0: TypeID) -> bool: ...
    @overload
    def __eq__(self, arg0: Any) -> bool: ...
    def __hash__(self) -> int: ...
    @property
    def _CAPIPtr(self) -> object: ...

class UnitAttr(Attribute):
    static_typeid: ClassVar[TypeID]
    @staticmethod
    def get(context: Context | None = None) -> UnitAttr:
        """
        Create a Unit attribute.
        """
    @staticmethod
    def isinstance(other: Attribute) -> bool: ...
    def __init__(self, cast_from_attr: Attribute) -> None: ...
    @property
    def type(self) -> Type: ...
    @property
    def typeid(self) -> TypeID: ...

class UnrankedMemRefType(ShapedType):
    static_typeid: ClassVar[TypeID]
    @staticmethod
    def get(
        element_type: Type, memory_space: Attribute, loc: Location | None = None
    ) -> UnrankedMemRefType:
        """
        Create a unranked memref type
        """
    @staticmethod
    def isinstance(other: Type) -> bool: ...
    def __init__(self, cast_from_type: Type) -> None: ...
    @property
    def memory_space(self) -> Attribute | None:
        """
        Returns the memory space of the given Unranked MemRef type.
        """
    @property
    def typeid(self) -> TypeID: ...

class UnrankedTensorType(ShapedType):
    static_typeid: ClassVar[TypeID]
    @staticmethod
    def get(element_type: Type, loc: Location | None = None) -> UnrankedTensorType:
        """
        Create a unranked tensor type
        """
    @staticmethod
    def isinstance(other: Type) -> bool: ...
    def __init__(self, cast_from_type: Type) -> None: ...
    @property
    def typeid(self) -> TypeID: ...

class VectorType(ShapedType):
    static_typeid: ClassVar[TypeID]
    @staticmethod
    def get(
        shape: list[int],
        element_type: Type,
        *,
        scalable: list | None = None,
        scalable_dims: list[int] | None = None,
        loc: Location | None = None,
    ) -> VectorType:
        """
        Create a vector type
        """
    @staticmethod
    def isinstance(other: Type) -> bool: ...
    def __init__(self, cast_from_type: Type) -> None: ...
    @property
    def scalable(self) -> bool: ...
    @property
    def scalable_dims(self) -> list[bool]: ...
    @property
    def typeid(self) -> TypeID: ...

class _GlobalDebug:
    flag: ClassVar[bool] = False
