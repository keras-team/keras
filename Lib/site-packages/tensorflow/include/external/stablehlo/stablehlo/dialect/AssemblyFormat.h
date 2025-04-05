/* Copyright 2022 The StableHLO Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#ifndef STABLEHLO_DIALECT_ASSEMBLYFORMAT_H
#define STABLEHLO_DIALECT_ASSEMBLYFORMAT_H

#include <cstdint>
#include <functional>

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/Region.h"
#include "mlir/IR/TypeRange.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "stablehlo/dialect/Base.h"

namespace mlir {
namespace hlo {

//===----------------------------------------------------------------------===//
// Generic Type Printers and Parsers
//===----------------------------------------------------------------------===//

// Declarative `custom<SameOperandsAndResultType>(...)` implementation:
// Pretty print for ops with many operands, but one result type, simplifies
// print if all operand types match the result type.
//
// Example:
//   custom<SameOperandsAndResultType>(type($result), type($operand1),
//   type($operand2))
//
//   Generic:
//     %0 = "stablehlo.op"(%0, %1) : (tensor<i1>, tensor<i1>) -> tensor<i1>
//   Custom:
//     %0 = stablehlo.op(%0, %1) : tensor<i1>
//
// Falls back to `printFunctionalType` if all operands do not match result
// type.
//
// Note that `type($result)` is the first argument, this is done because the
// behavior of trailing parameter packs is easily understandable.
namespace detail {
void printSameOperandsAndResultTypeImpl(OpAsmPrinter& p, Operation* op,
                                        TypeRange operands, Type result);

ParseResult parseSameOperandsAndResultTypeImpl(OpAsmParser& parser,
                                               ArrayRef<Type*> operands,
                                               Type& result);
}  // namespace detail

template <class... OpTypes>
void printSameOperandsAndResultType(OpAsmPrinter& p, Operation* op,
                                    OpTypes... types) {
  static_assert(sizeof...(types) > 0);  // Must be non empty, must have result
  SmallVector<Type> typesVec{types...};
  ArrayRef<Type> typesRef = ArrayRef(typesVec);
  return detail::printSameOperandsAndResultTypeImpl(
      p, op, typesRef.drop_back(1), typesRef.back());
}

template <class... OpTypes>
ParseResult parseSameOperandsAndResultType(OpAsmParser& parser,
                                           OpTypes&... types) {
  static_assert(sizeof...(types) > 0);  // Must be non empty, must have result
  SmallVector<Type*> typesVec{&types...};
  ArrayRef<Type*> typesRef = ArrayRef(typesVec);
  return detail::parseSameOperandsAndResultTypeImpl(
      parser, typesRef.drop_back(1), *typesRef.back());
}

void printVariadicSameOperandsAndResultType(OpAsmPrinter& p, Operation* op,
                                            OperandRange operands,
                                            TypeRange opTypes, Type result);

ParseResult parseVariadicSameOperandsAndResultType(
    OpAsmParser& parser,
    SmallVectorImpl<OpAsmParser::UnresolvedOperand>& operands,
    SmallVectorImpl<Type>& opTypes, Type& result);

// Print a `constant` op.
//
// op ::= attr-dict $value
//
// When the `value` and `output` have different type, it just uses the default
// operator assembly format as a fallback.
void printConstantOp(OpAsmPrinter& p, Operation* op, ElementsAttr value);

ParseResult parseConstantOp(OpAsmParser& parser, OperationState& result);

// TuplesOp - only print result type. Operand type is trivially inferrable.
//
// Inferring operand types from tuple type:
//  %3 = stablehlo.tuple %1, %2 : tuple<tensor<i1>, tensor<f32>>
//    %1 : tensor<i1>
//    %2 : tensor<f32>
//    %3 : tuple<tensor<i1>, tensor<f32>>
void printTupleOpType(OpAsmPrinter& p, Operation*, TypeRange operands,
                      Type result);

ParseResult parseTupleOpType(OpAsmParser& parser,
                             SmallVectorImpl<Type>& operands, Type& result);

// PairwiseOps - only print result type. Operand types are trivially
// inferrable.
//
// Inferring operand types for pairwise ops:
//  %3, %4 = stablehlo.operation %1, %2 : tensor<i1>, tensor<f32>
//    %1 : tensor<i1>
//    %2 : tensor<f32>
//    %3 : tensor<i1>
//    %4 : tensor<f32>
void printPairwiseOpType(OpAsmPrinter& p, Operation*, TypeRange operands,
                         TypeRange results);

ParseResult parsePairwiseOpType(OpAsmParser& parser,
                                SmallVectorImpl<Type>& operands,
                                SmallVectorImpl<Type>& results);

// Variadic operands with attributes - Need to provide custom parser since
// the built-in operand list parser parses the attribute expecting an SSA value
// and errors.
//
// %0 = stablehlo.operation %arg0, ..., %argN, attr = value
void printVariadicOperandWithAttribute(OpAsmPrinter& p, Operation*,
                                       OperandRange operands);

ParseResult parseVariadicOperandWithAttribute(
    OpAsmParser& parser,
    SmallVectorImpl<OpAsmParser::UnresolvedOperand>& operands);

//===----------------------------------------------------------------------===//
// Operation Printers and Parsers
//===----------------------------------------------------------------------===//

// ComplexOpType - only print result type if the inferred complex type
// matches all operand types.
//
// Inferring operand types for complex ops:
//  %0 = stablehlo.complex %1, %2 : tensor<4xcomplex<f32>>
//    %0 : tensor<4xcomplex<f32>>
//    %1 : tensor<4xf32>
//    %2 : tensor<4xf32>
void printComplexOpType(OpAsmPrinter& p, Operation* op, ShapedType lhs,
                        ShapedType rhs, ShapedType result);

ParseResult parseComplexOpType(OpAsmParser& parser, Type& lhs, Type& rhs,
                               Type& result);

// Print reduce with or without compact printing
// If the reduce-op is eligible for compact printing, we emit a one-line print.
// See IsEligibleForCompactPrint code comments for criteria.
//
// Compact:
//   stablehlo.reduce(...) applies <inner-op> across dimensions = [...] : <type>
// Not compact:
//   stablehlo.reduce(...) across dimensions = [...] : <type>
//     reducer(...)  { ...}
void printReduceOp(OpAsmPrinter& p, Operation* op, ValueRange inputs,
                   ArrayRef<int64_t> dimensions, Region& body);

// Parse reduce with or without compact parsing
ParseResult parseReduceOp(
    OpAsmParser& parser, OperationState& result,
    std::function<Attribute(OpBuilder&, ArrayRef<int64_t>)> createDimensions);

// SelectOpType - only print the condition and result type when branch types
// match the result type.
//
// Inferring operand types for select ops:
//  %3 = stablehlo.select %0, %1, %2 : tensor<2xi1>, tensor<2xi32>
//    %0 : tensor<2xi1>
//    %1 : tensor<2xi32>
//    %2 : tensor<2xi32>
//    %3 : tensor<2xi32>
void printSelectOpType(OpAsmPrinter& p, Operation* op, ShapedType pred,
                       ShapedType onTrue, ShapedType onFalse,
                       ShapedType result);

ParseResult parseSelectOpType(OpAsmParser& parser, Type& pred, Type& onTrue,
                              Type& onFalse, Type& result);

// Print a `while` op.
//
// op ::= `stablehlo.while` `(` assignment-list `)` `:` types attribute-dict
//         `cond` region
//         `do` region
// assignment-list ::= assignment | assignment `,` assignment-list
// assignment ::= ssa-value `=` ssa-value
void printWhileOp(OpAsmPrinter& p, Operation* op, Region& cond, Region& body);

// Parse while with or without compact parsing
ParseResult parseWhileOp(OpAsmParser& parser, OperationState& result);

//===----------------------------------------------------------------------===//
// Attribute Printers and Parsers
//===----------------------------------------------------------------------===//

// SliceRanges - Used to print multi-dimensional ranges for slice.
void printSliceRanges(OpAsmPrinter& p, Operation* op,
                      ArrayRef<int64_t> startIndices,
                      ArrayRef<int64_t> limitIndices,
                      ArrayRef<int64_t> strides);

ParseResult parseSliceRanges(OpAsmParser& parser,
                             DenseI64ArrayAttr& startIndices,
                             DenseI64ArrayAttr& limitIndices,
                             DenseI64ArrayAttr& strides);

// GenericI64DenseArray - Used to print an attr that can be either
//
//   Dense elements:
//     { dense<[1, 2]> : tensor<2xi64> }
//   Array:
//     { array<i64: 1, 2> }
void printDenseI64Array(OpAsmPrinter& p, Operation* op, Attribute attr);

ParseResult parseDenseI64Array(OpAsmParser& parser, Attribute& attr);

// DimSizes - Print an array of ints. Dynamic dimensions printed as `?`.
//
//   Generic:
//     [1, -1]
//   Custom:
//     [1, ?]
std::string dimSizeToString(int64_t dimSize);
std::string dimSizesToString(ArrayRef<int64_t> dimSize);

void printDimSizes(AsmPrinter& p, ArrayRef<int64_t> dimSizes);

FailureOr<SmallVector<int64_t>> parseDimSizes(AsmParser& parser);
ParseResult parseDimSizes(AsmParser& parser, SmallVector<int64_t>& dimSizes);

// ExponentMantissa - Abbreviated printing of exponent and mantissa as e#m#.
//
//   Generic:
//     {exponent = 5 : i32, mantissa = 10 : i32}
//   Custom:
//     e5m10
void printExponentMantissa(AsmPrinter& p, Operation*, IntegerAttr exponent,
                           IntegerAttr mantissa);

ParseResult parseExponentMantissa(AsmParser& parser, IntegerAttr& exponent,
                                  IntegerAttr& mantissa);

// CustomCallTarget - Print custom call target using upstream SymbolRef
// printing.
//
// Generic:
//    {custom_call_target = "foo"}
//    {custom_call_target = "not-valid-id"}
//
// Custom:
//    @foo
//    @"not-valid-id"
void printCustomCallTarget(AsmPrinter& p, Operation*, StringAttr target);

ParseResult parseCustomCallTarget(AsmParser& parser, StringAttr& target);

// TypeExtensions - Print a shorthand form of TypeExtensionsAttr.
// If TypeExtensionsAttr evolves in the future, the shorthand form may evolve
// as well, or we can also fall back to the autogenerated longer form.
//
// Generic:
//    #stablehlo.type_extensions<bounds = [4, ?]>
//
// Custom:
//    #stablehlo.bounds<4, ?>
void printTypeExtensions(BoundedAttrInterface attr, DialectAsmPrinter& os);

Attribute parseTypeExtensions(HloDialectInterface* dialect,
                              DialectAsmParser& parser);

// DotDimensionNumbers - Abbreviated printing using a ConvDimensionNumbers-
// inspired notation. batching_dims are skipped if empty.
//
// Generic:
//    dot_dimension_numbers = #stablehlo.dot<
//      lhs_batching_dimensions = [],
//      lhs_contracting_dimensions = [1],
//      rhs_batching_dimensions = [],
//      rhs_contracting_dimensions = [0]
//    >
//    dot_dimension_numbers = #stablehlo.dot<
//      lhs_batching_dimensions = [0],
//      lhs_contracting_dimensions = [2],
//      rhs_batching_dimensions = [0],
//      rhs_contracting_dimensions = [1]
//    >
//
// Custom:
//    contracting_dims = [1] x [0]
//    batching_dims = [0] x [0], contracting_dims = [2] x [1]
template <typename AttrTy>
void printDotDimensionNumbers(AsmPrinter& p, Operation* op, AttrTy target) {
  // Print two ArrayRef<int64_t> as `[...] x [...]`
  auto printLhsRhsDims = [&](ArrayRef<int64_t> lhsDims,
                             ArrayRef<int64_t> rhsDims) {
    DenseI64ArrayAttr::get(op->getContext(), lhsDims).print(p);
    p << " x ";
    DenseI64ArrayAttr::get(op->getContext(), rhsDims).print(p);
  };

  // Print the optional `batching_dims = [...] x [...]`.
  if (!target.getLhsBatchingDimensions().empty() ||
      !target.getRhsBatchingDimensions().empty()) {
    p << "batching_dims = ";
    printLhsRhsDims(target.getLhsBatchingDimensions(),
                    target.getRhsBatchingDimensions());
    p << ", ";
  }

  // Print the required `contracting_dims = [...] x [...]`.
  p << "contracting_dims = ";
  printLhsRhsDims(target.getLhsContractingDimensions(),
                  target.getRhsContractingDimensions());
}

template <typename AttrTy>
ParseResult parseDotDimensionNumbers(AsmParser& parser, AttrTy& target) {
  // Parse `[...] x [...]` into two DenseI64ArrayAttr attributes.
  auto parseLhsRhsDims = [&](DenseI64ArrayAttr& lhsDims,
                             DenseI64ArrayAttr& rhsDims) -> ParseResult {
    lhsDims = dyn_cast_or_null<DenseI64ArrayAttr>(
        DenseI64ArrayAttr::parse(parser, Type{}));
    if (!lhsDims) return failure();
    if (failed(parser.parseKeyword("x"))) return failure();
    rhsDims = dyn_cast_or_null<DenseI64ArrayAttr>(
        DenseI64ArrayAttr::parse(parser, Type{}));
    if (!rhsDims) return failure();
    return success();
  };

  // Parse the optional `batching_dims = [...] x [...]`.
  DenseI64ArrayAttr lhsBatchingDims, rhsBatchingDims;
  if (succeeded(parser.parseOptionalKeyword("batching_dims"))) {
    if (failed(parser.parseEqual()) ||
        failed(parseLhsRhsDims(lhsBatchingDims, rhsBatchingDims)) ||
        failed(parser.parseComma()))
      return failure();
  }

  // Parse the required `contracting_dims = [...] x [...]`.
  DenseI64ArrayAttr lhsContractingDims, rhsContractingDims;
  if (failed(parser.parseKeyword("contracting_dims")) ||
      failed(parser.parseEqual()) ||
      failed(parseLhsRhsDims(lhsContractingDims, rhsContractingDims)))
    return failure();

  target = AttrTy::get(
      parser.getBuilder().getContext(),
      lhsBatchingDims ? lhsBatchingDims.asArrayRef() : ArrayRef<int64_t>{},
      rhsBatchingDims ? rhsBatchingDims.asArrayRef() : ArrayRef<int64_t>{},
      lhsContractingDims.asArrayRef(), rhsContractingDims.asArrayRef());
  return success();
}

}  // namespace hlo
}  // namespace mlir

#endif  // STABLEHLO_DIALECT_ASSEMBLYFORMAT_H
