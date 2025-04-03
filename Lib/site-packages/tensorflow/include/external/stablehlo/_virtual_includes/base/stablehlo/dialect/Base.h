/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.
   Copyright 2022 The StableHLO Authors.

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

#ifndef STABLEHLO_DIALECT_BASE_H
#define STABLEHLO_DIALECT_BASE_H

#include <algorithm>
#include <optional>

#include "llvm/ADT/APSInt.h"
#include "llvm/ADT/Sequence.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/Bytecode/BytecodeImplementation.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/DialectInterface.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Support/LogicalResult.h"

// Include order matters
#include "stablehlo/dialect/BaseAttrInterfaces.h.inc"

namespace mlir {
namespace hlo {

// TODO(zhouxin) change to a better name as it's used by both of size and bound
// Check if the dimension size is dynamic.
inline static bool isDynamicDimSize(int64_t val) {
  return ShapedType::isDynamic(val);
}

inline static bool isStaticDimSize(int64_t val) {
  return !isDynamicDimSize(val);
}

// Checks whether every position in the given array contains the given value.
bool isSplatArray(ArrayRef<int64_t> arr, int64_t val);

//  Verifies that the two types have compatible shape with bounds but allows
//  different element types.
LogicalResult verifyCompatibleShapeWithBounds(Type type1, Type type2);

// Returns true if the given element types are compatible for the purposes of
// HLO type inference, accounting for special properties of quantization and
// sparsity.
bool isCompatibleElementTypeForHloTypeInference(Type tp1, Type tp2);

// Returns true if the given types are compatible for the purposes of HLO type
// inference, accounting for special properties of dynamism, quantization and
// sparsity.
bool isCompatibleForHloTypeInference(Type tp1, Type tp2);

// Returns true if the given type ranges are compatible for the purposes of HLO
// type inference, accounting for special properties of dynamism, quantization
// and sparsity.
bool isCompatibleForHloTypeInference(TypeRange tp1, TypeRange tp2);

// Returns true if the given shape, expressed as a runtime value, is compatible
// with the given type for the purposes of HLO type inference.
// If we know that this runtime value is a constant, then we perform the check.
// If we don't, then we return true - because shape mismatches at runtime are
// undefined behavior.
bool isCompatibleForHloTypeInference(Value shape1, Type tp2);

// Returns true if the given shape, expressed as a slice of integers, is
// compatible with the given type for the purposes of HLO type inference.
bool isCompatibleForHloTypeInference(ArrayRef<int64_t> shape1, Type tp2);

// Returns true if the given element-type is a mlir::quant::QuantizedType
// and follow the constraints corresponding to quantization parameters as
// mentioned in the StableHLO specification.
bool isValidStablehloQuantizedElementType(Type elementType);

// Returns true if the given type is a ranked per-axis tensor type
// and follow the constraints corresponding to quantized dimension as
// mentioned in the StableHLO specification.
bool isValidQuantizedDimension(Type type);

// TODO(zhouxin) Move type inference related methods to TypeInference.cpp

std::pair<int64_t, int64_t> inferConcatenatedDimAndBound(int64_t leftSize,
                                                         int64_t rightSize,
                                                         int64_t leftBound,
                                                         int64_t rightBound);

FailureOr<std::pair<int64_t, int64_t>> inferMostSpecificDimAndBound(
    std::optional<Location> location, int64_t dim, int64_t leftSize,
    int64_t rightSize, int64_t leftBound, int64_t rightBound);

FailureOr<std::pair<int64_t, int64_t>> inferLeastSpecificDimAndBound(
    std::optional<Location> location, int64_t dim, int64_t leftSize,
    int64_t rightSize, int64_t leftBound, int64_t rightBound);

// Infer single least specific return type from inputTypes with support for
// bounds. (Size, bound) of each dimension of the return type will be merged
// from corresponding dimensions of every inputType by extracting the least
// specific one. Return unranked tensor if any input is unranked.
FailureOr<Type> inferLeastSpecificType(std::optional<Location> location,
                                       TypeRange inputTypes);

// Infer single most specific return type from inputTypes with support for
// bounds. (Size, bound) of each dimension of the return type will be merged
// from corresponding dimensions of every inputType by extracting the most
// specific one. Return unranked tensor if all inputs are unranked.
FailureOr<Type> inferMostSpecificType(std::optional<Location> location,
                                      TypeRange inputTypes);

LogicalResult inferMostSpecificTypeComponents(
    std::optional<Location> location, TypeRange inputTypes,
    SmallVectorImpl<ShapedTypeComponents> &inferredReturnShapes);

// Matches a constant with integer value into int64_t.
LogicalResult matchInt(Value value, int64_t &result);

// Matches a constant tensor with integer values into a 1-dimensional vector.
// Doesn't preserve the bitness or the signedness of the underlying values,
// extracting them into int64_t.
LogicalResult matchInts(Value value, SmallVector<int64_t> &result);

// Matches a constant tensor with integer values into a 1-dimensional vector.
// Preserves the bitness and the signedness of the underlying values.
LogicalResult matchInts(Value value, SmallVector<APSInt> &result);

// Matches a constant tensor with integer values.
// Unlike the functions above, it doesn't return these values - it just checks
// that the given argument is indeed a constant tensor with integer values.
LogicalResult matchInts(Value value);

// Shape derivation function that computes the shape of the result based on an
// operand. For a 2-dimensional input tensor, this produces IR of the form
//
//  %0 = dim %arg0, 0 : memref<?x?xf32>
//  %1 = index_cast %0 : index to i64
//  %2 = dim %arg0, 1 : memref<?x?xf32>
//  %3 = index_cast %2 : index to i64
//  %4 = "shape.shape_of"(%1, %3)
//    : (i64, i64) -> tensor<2xi64>
//
// and returns %4 as the shape value.
LogicalResult deriveShapeFromOperand(
    OpBuilder *builder, Operation *op, Value operand,
    SmallVectorImpl<Value> *reifiedReturnShapes);

// Type derivation function that returns a tensor type with a new element type.
ShapedType getSameShapeTensorType(ShapedType shapedType, Type elementType);

// Takes a tensor type that may have complex elements and returns a type that
// maintains the shape, but with real numeric data types.
//   Ex: tensor<4xcomplex<f32>>  -->  tensor<4xf32>
ShapedType createRealType(ShapedType type);

// Verify bounds expressed by HLO_BoundedAttrInterface against the provided
// type. See documentation for HLO_BoundedAttrInterface for the list of checks.
LogicalResult verifyBounds(ArrayRef<int64_t> bounds, RankedTensorType type,
                           function_ref<InFlightDiagnostic()> emitError);

// If an encoding attribute conforms to HLO_BoundedAttrInterface, return the
// bounds that it carries. Otherwise, return an empty ArrayRef.
ArrayRef<int64_t> encodingToBounds(Attribute encoding);

// Create an HLO_BoundedAttrInterface encoding attribute that carries the given
// bounds. Requires a prototype - an existing encoding attribute - to obtain
// the underlying dialect that knows how to create these attributes.
Attribute boundsToEncoding(Attribute prototype, ArrayRef<int64_t> bounds);

// Get refinements for return types from an indices_of_shape_operands attribute,
// with tuples types flattened (see `flattenTupleTypes` below).
// If the attribute doesn't exist, returns failure.
// If the attribute exists but is not invalid with respect to the operation,
// reports an optional error and returns failure.
// If the attribute is valid but not all shape operands are constants,
// returns failure.
LogicalResult getShapeRefinements(
    std::optional<Location> location, Operation *operation,
    SmallVector<ShapedTypeComponents> &refinements);

// For each type in `types`, recursively flatten tuple types into `result`.
// Result is populated via in-order traversal of tuple types in `types`, i.e.:
//   * Flattenings of individual types from `types` follow one another in the
//     same order as `types`.
//   * Same for flattenings of element types of tuple types.
void flattenTupleTypes(TypeRange types, SmallVector<Type> &result);

// Does the inverse of `flattenTupleTypes` - takes `types` and recursively
// unflattens it, creating tuple types as needed to exactly match the structure
// of `prototype`.
// Fails if the number of elements in flattened prototype is different from
// the number of elements in types.
LogicalResult unflattenTupleTypes(TypeRange prototype, TypeRange types,
                                  SmallVector<Type> &result);

ShapedType createShapedType(ShapedTypeComponents components);

// This interface is implemented by both StableHLO and MHLO dialects
// and is used as the foundation for sharing verification, type inference and
// prettyprinting logic between them.
class HloDialectInterface : public DialectInterface::Base<HloDialectInterface> {
 public:
  HloDialectInterface(Dialect *dialect) : Base(dialect) {}

  // Creates a TokenType type, specific to this dialect.
  // See docs for the particular type in the corresponding dialect.
  virtual Type createTokenType() const = 0;

  // Check whether the type is of TokenType in the corresponding dialect.
  virtual bool isTokenType(Type type) const = 0;

  // Creates a TypeExtensions attribute, specific to this dialect.
  // See docs for the particular attribute in the corresponding dialect.
  virtual Attribute createTypeExtensions(ArrayRef<int64_t> bounds) const = 0;
};

namespace detail {

// An enum which tracks known supported dot algorithm pairs.
// Note this implementation is a detail for now and the APIs are likely to
// change once HLO broadens support for LHS/RHS components and num primitive
// operations.
//
// It is best to not rely on these values until the API solidifies.
// Instead use `isKnownDotAlgorithm`.
enum class KnownDotAlgorithm {
  ANY_F8_ANY_F8_F32 = 1,
  ANY_F8_ANY_F8_F32_FAST_ACCUM = 2,
  F16_F16_F16 = 3,
  F16_F16_F32 = 4,
  BF16_BF16_BF16 = 5,
  BF16_BF16_F32 = 6,
  BF16_BF16_F32_X3 = 7,
  BF16_BF16_F32_X6 = 8,
  TF32_TF32_F32 = 9,
  TF32_TF32_F32_X3 = 10,
  F32_F32_F32 = 11,
  F64_F64_F64 = 12,
};

FailureOr<KnownDotAlgorithm> getKnownDotAlgorithm(
    Type lhsPrecisionType, Type rhsPrecisionType, Type accumulationType,
    int64_t lhsComponentCount, int64_t rhsComponentCount,
    int64_t numPrimitiveOperations, bool allowImpreciseAccumulation);
}  // namespace detail

// Check if the combination of a dot algorithm struct is known.
bool isKnownDotAlgorithm(Type lhsPrecisionType, Type rhsPrecisionType,
                         Type accumulationType, int64_t lhsComponentCount,
                         int64_t rhsComponentCount,
                         int64_t numPrimitiveOperations,
                         bool allowImpreciseAccumulation);

namespace bytecode {
// Helper methods for bytecode
// Enum reader and writer. Many attrs have a single enum type to serialize.
// Use the attributes underlying type to get the numeric value.
// Note this may cause issues if enums use an int64_t and have a large value.
// All enums in StableHLO and CHLO currently use uint32_t.
template <typename EnumTypeAttr, typename SymbolizeFn>
EnumTypeAttr readEnumAttribute(DialectBytecodeReader &reader,
                               MLIRContext *context, SymbolizeFn symbolizeFn) {
  uint64_t code;
  if (failed(reader.readVarInt(code))) return EnumTypeAttr();

  auto enumOpt = symbolizeFn(static_cast<uint32_t>(code));
  if (!enumOpt.has_value()) return EnumTypeAttr();

  return EnumTypeAttr::get(context, enumOpt.value());
}

template <typename EnumType, typename EnumTypeAttr>
void writeEnumAttribute(EnumTypeAttr val, DialectBytecodeWriter &writer) {
  static_assert(
      std::is_same<typename std::underlying_type<EnumType>::type,
                   uint32_t>::value,
      "writeEnumAttribute is only implemented for uint32_t enum values");

  uint32_t enumVal = static_cast<typename std::underlying_type<EnumType>::type>(
      val.getValue());
  writer.writeVarInt(enumVal);
}
}  // namespace bytecode

// Determines the speculatability for a shaped operation `op` with `shapeCount`
// shape operands. The last `count` operands are assumed to be shape operands.
// To be speculatable, such an op must have only static inputs and constant
// shape operands.
mlir::Speculation::Speculatability getShapedSpeculatability(Operation *op,
                                                            int64_t shapeCount);

namespace OpTrait {

template <typename ConcreteType>
class BroadcastingElementwise
    : public mlir::OpTrait::TraitBase<ConcreteType, BroadcastingElementwise> {};

template <typename ConcreteType>
class IsCommutative
    : public mlir::OpTrait::TraitBase<ConcreteType, IsCommutative> {};

template <typename ConcreteType>
class PairwiseSameOperandAndResultType
    : public mlir::OpTrait::TraitBase<ConcreteType,
                                      PairwiseSameOperandAndResultType> {
 public:
  static LogicalResult verifyTrait(Operation *op) {
    const int numOperands = op->getNumOperands();
    const int numResults = op->getNumResults();
    if (numOperands != numResults) {
      return op->emitOpError()
             << "requires the same number of operands and results";
    }

    for (int idx : llvm::seq<int>(0, numOperands)) {
      if (op->getOperand(idx).getType() != op->getResult(idx).getType()) {
        return op->emitOpError()
               << "requires the same type for operand and result at index "
               << idx;
      }
    }
    return success();
  }
};

template <typename ConcreteType>
class PairwiseSameOperandAndResultElementType
    : public mlir::OpTrait::TraitBase<ConcreteType,
                                      PairwiseSameOperandAndResultElementType> {
 public:
  static LogicalResult verifyTrait(Operation *op) {
    const int numOperands = op->getNumOperands();
    const int numResults = op->getNumResults();
    if (numOperands != numResults) {
      return op->emitOpError()
             << "requires the same number of operands and results";
    }

    for (int idx : llvm::seq<int>(0, numOperands)) {
      if (getElementTypeOrSelf(op->getOperand(idx)) !=
          getElementTypeOrSelf(op->getResult(idx))) {
        return op->emitOpError() << "requires the same element type for "
                                    "operand and result at index "
                                 << idx;
      }
    }
    return success();
  }
};

template <typename ConcreteType>
class CompatibleOperandsAndResultElementType
    : public mlir::OpTrait::TraitBase<ConcreteType,
                                      CompatibleOperandsAndResultElementType> {
 public:
  static LogicalResult verifyTrait(Operation *op) {
    Type expected;
    if (op->getNumResults() != 0) expected = op->getResult(0).getType();
    if (op->getNumOperands() != 0) expected = op->getOperand(0).getType();
    if (!expected) return failure();

    auto typeMatch = [&](Type actual) {
      return isCompatibleElementTypeForHloTypeInference(actual, expected);
    };
    auto allMatch = llvm::all_of(op->getOperandTypes(), typeMatch) &&
                    llvm::all_of(op->getResultTypes(), typeMatch);
    if (!allMatch) {
      return op->emitOpError(
          "requires compatible element types for all operands and results");
    }

    return success(allMatch);
  }
};

template <typename ConcreteType>
class CompatibleOperandsElementType
    : public mlir::OpTrait::TraitBase<ConcreteType,
                                      CompatibleOperandsElementType> {
 public:
  static LogicalResult verifyTrait(Operation *op) {
    if (failed(mlir::OpTrait::impl::verifyAtLeastNOperands(op, 1)))
      return failure();

    Type expected = op->getOperand(0).getType();
    auto typeMatch = [&](Type actual) {
      return isCompatibleElementTypeForHloTypeInference(actual, expected);
    };
    auto allMatch = llvm::all_of(op->getOperandTypes(), typeMatch);
    if (!allMatch) {
      return op->emitOpError(
          "requires compatible element types for all operands");
    }

    return success();
  }
};

template <typename ConcreteType>
class CompatibleOperandsAndResultType
    : public mlir::OpTrait::TraitBase<ConcreteType,
                                      CompatibleOperandsAndResultType> {
 public:
  static LogicalResult verifyTrait(Operation *op) {
    Type expected;
    if (op->getNumResults() != 0) expected = op->getResult(0).getType();
    if (op->getNumOperands() != 0) expected = op->getOperand(0).getType();
    if (!expected) return failure();

    auto typeMatch = [&](Type actual) {
      return isCompatibleForHloTypeInference(actual, expected);
    };
    auto allMatch = llvm::all_of(op->getOperandTypes(), typeMatch) &&
                    llvm::all_of(op->getResultTypes(), typeMatch);
    if (!allMatch) {
      return op->emitOpError(
          "requires compatible types for all operands and results");
    }

    return success(allMatch);
  }

  static LogicalResult inferReturnTypes(
      MLIRContext * /*context*/, std::optional<Location> location,
      ValueRange operands, DictionaryAttr /*attributes*/,
      OpaqueProperties /*properties*/, RegionRange /*regions*/,
      SmallVectorImpl<Type> &inferredReturnTypes) {
    // TODO(b/231358795): Review the use of InferTypeOpInterface for ops that
    // support quantization or sparsity.
    if (operands.empty())
      return emitOptionalError(
          location,
          "Expected non-empty operands for [CompatibleOperandsAndResultType]");

    auto inferredTypeOrErr =
        inferMostSpecificType(location, operands.getTypes());
    if (failed(inferredTypeOrErr)) return failure();
    inferredReturnTypes.emplace_back(*inferredTypeOrErr);
    return success();
  }

  // This function is not going to be called automatically.
  // It needs to be paired with INFER_RETURN_TYPE_COMPONENTS_FROM_OPERANDS
  // (see examples in StablehloOps.cpp).
  static LogicalResult inferReturnTypeComponentsFromOperands(
      MLIRContext *context, std::optional<Location> location,
      ValueShapeRange operands, DictionaryAttr attributes,
      OpaqueProperties properties, RegionRange regions,
      SmallVectorImpl<ShapedTypeComponents> &inferredReturnShapes) {
    SmallVector<Type> inferredReturnTypes;
    if (failed(inferReturnTypes(context, location, operands.getValues(),
                                attributes, properties, regions,
                                inferredReturnTypes)))
      return failure();
    if (inferredReturnTypes.size() != 1) return failure();
    auto inferredReturnType = dyn_cast<ShapedType>(inferredReturnTypes[0]);
    if (!inferredReturnType) return failure();
    inferredReturnShapes.push_back(inferredReturnType);
    return success();
  }
};

template <typename ConcreteType>
struct SpeculatableIfStaticDimInOutputIsStaticInInputImplTrait
    : public mlir::OpTrait::TraitBase<
          ConcreteType,
          SpeculatableIfStaticDimInOutputIsStaticInInputImplTrait> {
  // A unary elementwise op is not speculatable if a dimension of the result
  // type is static while the corresponding dimension in the input type is
  // dynamic. Indeed, the input dimension could differ at runtime.
  // If the output dimension is dynamic, there is no expectation, so there
  // cannot be a mismatch.
  // If the input dimension is static, the output dimension can be inferred from
  // it, so there cannot be a mismatch.
  mlir::Speculation::Speculatability getSpeculatability() {
    auto op = this->getOperation();
    auto inputType = cast<RankedTensorType>(op->getOperand(0).getType());
    auto resultType = cast<RankedTensorType>(op->getResult(0).getType());
    for (size_t i : llvm::seq(resultType.getRank())) {
      if (!resultType.isDynamicDim(i) && inputType.isDynamicDim(i))
        return mlir::Speculation::NotSpeculatable;
    }
    return mlir::Speculation::Speculatable;
  }
};

template <typename ConcreteType>
struct RecursivelySpeculatableIfStaticDimInOutputIsStaticInInputImplTrait
    : public mlir::OpTrait::TraitBase<
          ConcreteType,
          RecursivelySpeculatableIfStaticDimInOutputIsStaticInInputImplTrait> {
  mlir::Speculation::Speculatability getSpeculatability() {
    auto op = this->getOperation();
    auto inputType = cast<RankedTensorType>(op->getOperand(0).getType());
    auto resultType = cast<RankedTensorType>(op->getResult(0).getType());
    for (size_t i : llvm::seq(resultType.getRank())) {
      if (!resultType.isDynamicDim(i) && inputType.isDynamicDim(i))
        return mlir::Speculation::NotSpeculatable;
    }
    return mlir::Speculation::RecursivelySpeculatable;
  }
};

template <typename ConcreteType>
struct SpeculatableIfAllInputsStaticImplTrait
    : public mlir::OpTrait::TraitBase<ConcreteType,
                                      SpeculatableIfAllInputsStaticImplTrait> {
  mlir::Speculation::Speculatability getSpeculatability() {
    return llvm::all_of(this->getOperation()->getOperandTypes(),
                        [](Type t) {
                          return cast<RankedTensorType>(t).hasStaticShape();
                        })
               ? mlir::Speculation::Speculatable
               : mlir::Speculation::NotSpeculatable;
  }
};

template <typename ConcreteType>
struct RecursivelySpeculatableIfAllInputsStaticImplTrait
    : public mlir::OpTrait::TraitBase<
          ConcreteType, RecursivelySpeculatableIfAllInputsStaticImplTrait> {
  mlir::Speculation::Speculatability getSpeculatability() {
    return llvm::all_of(this->getOperation()->getOperandTypes(),
                        [](Type t) {
                          return cast<RankedTensorType>(t).hasStaticShape();
                        })
               ? mlir::Speculation::RecursivelySpeculatable
               : mlir::Speculation::NotSpeculatable;
  }
};

template <typename ConcreteType>
struct SpeculatableIfAllInputsStaticAndShapeConstantImplTrait
    : public mlir::OpTrait::TraitBase<
          ConcreteType,
          SpeculatableIfAllInputsStaticAndShapeConstantImplTrait> {
  mlir::Speculation::Speculatability getSpeculatability() {
    return getShapedSpeculatability(this->getOperation(), 1);
  }
};

}  // namespace OpTrait
}  // namespace hlo
}  // namespace mlir

#endif  // STABLEHLO_DIALECT_BASE_H
