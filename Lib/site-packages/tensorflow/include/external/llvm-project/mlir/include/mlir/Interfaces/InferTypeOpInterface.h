//===- InferTypeOpInterface.h - Infer Type Interfaces -----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains the definitions of the infer op interfaces defined in
// `InferTypeOpInterface.td`.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_INTERFACES_INFERTYPEOPINTERFACE_H_
#define MLIR_INTERFACES_INFERTYPEOPINTERFACE_H_

#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/PointerUnion.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir {

class ShapedTypeComponents;
using ReifiedRankedShapedTypeDims = SmallVector<SmallVector<OpFoldResult>>;

/// Reify the shape of the result of an operation (typically in terms of the
/// shape of its operands).
LogicalResult
reifyResultShapes(OpBuilder &b, Operation *op,
                  ReifiedRankedShapedTypeDims &reifiedReturnShapes);

/// Adaptor class to abstract the differences between whether value is from
/// a ShapedType or ShapedTypeComponents or DenseIntElementsAttribute.
class ShapeAdaptor {
public:
  ShapeAdaptor(Type t) {
    if (auto st = dyn_cast<ShapedType>(t))
      val = st;
  }
  ShapeAdaptor(Attribute t) {
    if (auto da = dyn_cast<DenseIntElementsAttr>(t))
      val = da;
  }
  ShapeAdaptor(ShapedTypeComponents *components) : val(components) {}
  ShapeAdaptor(ShapedTypeComponents &components) : val(&components) {}

  /// Returns whether the shape has a rank.
  bool hasRank() const;

  /// Returns the element type.
  Type getElementType() const;

  /// Populates the dimensions from shape referenced.
  /// Requires: shape is ranked.
  void getDims(SmallVectorImpl<int64_t> &res) const;

  /// Populates the dimensions of the ShapeTypeComponents.
  /// Requires: shape is ranked.
  void getDims(ShapedTypeComponents &res) const;

  /// Returns the size of the index'th dimension.
  /// Requires: shape is ranked.
  int64_t getDimSize(int index) const;

  /// Returns whether the index'th dimension is dynamic.
  /// Requires: shape is ranked.
  bool isDynamicDim(int index) const {
    return ShapedType::isDynamic(getDimSize(index));
  }

  /// Returns whether the shape is fully static.
  bool hasStaticShape() const;

  /// Returns the rank of the shape.
  /// Requires: shape is ranked.
  int64_t getRank() const;

  /// Returns the number of elements in the shape.
  /// Requires: hasStaticShape
  int64_t getNumElements() const;

  /// Returns whether valid (non-null) shape.
  explicit operator bool() const { return !val.isNull(); }

  /// Dumps textual repesentation to stderr.
  void dump() const;

private:
  // Union storing either ShapedTypeComponents, ShapedType (stored as Type and
  // casted), or DenseIntElementsAttribute (stored as Atrtribute).
  PointerUnion<ShapedTypeComponents *, Type, Attribute> val = nullptr;
};

/// ShapedTypeComponents that represents the components of a ShapedType.
/// The components consist of
///  - A ranked or unranked shape with the dimension specification match those
///    of ShapeType's getShape() (e.g., dynamic dimension represented using
///    ShapedType::kDynamic)
///  - A element type, may be unset (nullptr)
///  - A attribute, may be unset (nullptr)
/// Used by ShapedType type inferences.
class ShapedTypeComponents {
  /// Internal storage type for shape.
  using ShapeStorageT = SmallVector<int64_t, 3>;

public:
  /// Default construction is an unranked shape.
  ShapedTypeComponents() : elementType(nullptr), attr(nullptr) {};
  ShapedTypeComponents(Type elementType)
      : elementType(elementType), attr(nullptr), ranked(false) {}
  ShapedTypeComponents(ShapedType shapedType) : attr(nullptr) {
    ranked = shapedType.hasRank();
    elementType = shapedType.getElementType();
    if (ranked)
      dims = llvm::to_vector<4>(shapedType.getShape());
  }
  ShapedTypeComponents(ShapeAdaptor adaptor) : attr(nullptr) {
    ranked = adaptor.hasRank();
    elementType = adaptor.getElementType();
    if (ranked)
      adaptor.getDims(*this);
  }
  template <typename Arg, typename = std::enable_if_t<
                              std::is_constructible<ShapeStorageT, Arg>::value>>
  ShapedTypeComponents(Arg &&arg, Type elementType = nullptr,
                       Attribute attr = nullptr)
      : dims(std::forward<Arg>(arg)), elementType(elementType), attr(attr),
        ranked(true) {}
  ShapedTypeComponents(ArrayRef<int64_t> vec, Type elementType = nullptr,
                       Attribute attr = nullptr)
      : dims(vec.begin(), vec.end()), elementType(elementType), attr(attr),
        ranked(true) {}

  /// Return the dimensions of the shape.
  /// Requires: shape is ranked.
  ArrayRef<int64_t> getDims() const {
    assert(ranked && "requires ranked shape");
    return dims;
  }

  /// Return whether the shape has a rank.
  bool hasRank() const { return ranked; };

  /// Return the element type component.
  Type getElementType() const { return elementType; };

  /// Return the raw attribute component.
  Attribute getAttribute() const { return attr; };

private:
  friend class ShapeAdaptor;

  ShapeStorageT dims;
  Type elementType;
  Attribute attr;
  bool ranked{false};
};

/// Range of values and shapes (corresponding effectively to Shapes dialect's
/// ValueShape type concept).
// Currently this exposes the Value (of operands) and Type of the Value. This is
// not ideal as then one can accidentally reference an out of date shape. This
// is done to both enable gradual switch and also as OpAdaptor doesn't currently
// allow returning anything other than Value.
class ValueShapeRange : public ValueRange::RangeBaseT {
public:
  using ValueShapeMapFn = function_ref<ShapeAdaptor(Value)>;

  ValueShapeRange(ValueRange values, ValueShapeMapFn operandShape = nullptr,
                  ValueShapeMapFn valueToShape = nullptr)
      : RangeBaseT(values), operandShape(operandShape),
        valueToShape(valueToShape) {}
  ValueShapeRange(const std::initializer_list<Value> &values)
      : ValueShapeRange(ValueRange(values)) {}

  ValueShapeRange(const ValueShapeRange &) = default;

  /// Sets the Value to ShapeAdaptor mapping function and returns this.
  ValueShapeRange &setValueToShapeMapping(ValueShapeMapFn fn) {
    valueToShape = fn;
    return *this;
  }

  ValueShapeRange &setOperandShapeMapping(ValueShapeMapFn fn) {
    operandShape = fn;
    return *this;
  }

  /// Returns the set Value to ShapeAdaptor mapping function.
  ValueShapeMapFn getValueToShapeMapping() const { return valueToShape; }
  ValueShapeMapFn getOperandShapeMapping() const { return operandShape; }

  // Accessors.

  /// Returns the types of the values within this range.
  /// Note: This returns only the types of Values in the ValueRange and not a
  /// more refined type.
  using type_iterator = ValueTypeIterator<iterator>;
  using type_range = ValueTypeRange<ValueRange>;
  type_range getTypes() const { return {begin(), end()}; }
  auto getType() const { return getTypes(); }

  /// Returns the Values in the ValueRange.
  /// To query the most up to date shape of a Value, query the shape
  /// using getShape below rather than using the type of the Value.
  ValueRange getValues() const { return ValueRange(begin(), end()); };

  /// Returns an argument as shape. If the argument is not constant or not a
  /// shape, then the function returns a nullptr.
  /// This will first query the valueToShape mapping (if set), before querying
  /// the ValueRange.
  ShapeAdaptor getValueAsShape(int index);

  /// Returns the shape of index'th operand.
  // TODO: Update so that operator[] references these instead to avoid
  // accidentally refering to less refined shape.
  ShapeAdaptor getShape(int index) const;

  /// Returns the shape of the given Value.
  ShapeAdaptor getShape(Value val) const;

private:
  // Mapping from Value to ShapedTypeComponents corresponding to shape of type
  // of Value.
  ValueShapeMapFn operandShape;

  // Mapping from Value to ShapedTypeComponents corresponding to constant Value
  // if interpreted as shape.
  ValueShapeMapFn valueToShape;
};

namespace detail {
// Helper function to infer return tensor returns types given element and
// shape inference function.
LogicalResult
inferReturnTensorTypes(ArrayRef<ShapedTypeComponents> retComponents,
                       SmallVectorImpl<Type> &inferredReturnTypes);

/// Verifies that the inferred result types match the actual result types for
/// the op. Precondition: op implements InferTypeOpInterface.
LogicalResult verifyInferredResultTypes(Operation *op);
} // namespace detail

namespace OpTrait {
template <typename ConcreteType>
class InferTensorType;
} // namespace OpTrait
} // namespace mlir

/// Include the generated interface declarations.
#include "mlir/Interfaces/InferTypeOpInterface.h.inc"

namespace mlir {
namespace OpTrait {

template <typename ConcreteType>
class InferTypeOpAdaptor : public TraitBase<ConcreteType, InferTypeOpAdaptor> {
};

template <typename ConcreteType>
class InferShapedTypeOpAdaptor
    : public TraitBase<ConcreteType, InferShapedTypeOpAdaptor> {};

/// Tensor type inference trait that constructs a tensor from the inferred
/// shape and elemental types.
/// Requires: Op implements InferShapedTypeOpInterface and InferTypeOpInterface.
///   Less strict is possible (e.g., implements inferReturnTypeComponents and
///   these always populates all element types and shapes or fails, but this
///   trait is currently only used where the interfaces are, so keep it
///   restricted for now).
template <typename ConcreteType>
class InferTensorType : public TraitBase<ConcreteType, InferTensorType> {};

} // namespace OpTrait
} // namespace mlir

#endif // MLIR_INTERFACES_INFERTYPEOPINTERFACE_H_
