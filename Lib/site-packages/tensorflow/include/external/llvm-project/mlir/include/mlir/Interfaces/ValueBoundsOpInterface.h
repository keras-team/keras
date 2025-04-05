//===- ValueBoundsOpInterface.h - Value Bounds ------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_INTERFACES_VALUEBOUNDSOPINTERFACE_H_
#define MLIR_INTERFACES_VALUEBOUNDSOPINTERFACE_H_

#include "mlir/Analysis/FlatLinearValueConstraints.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Value.h"
#include "mlir/Interfaces/DestinationStyleOpInterface.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/ExtensibleRTTI.h"

#include <queue>

namespace mlir {
class OffsetSizeAndStrideOpInterface;

/// A hyperrectangular slice, represented as a list of offsets, sizes and
/// strides.
class HyperrectangularSlice {
public:
  HyperrectangularSlice(ArrayRef<OpFoldResult> offsets,
                        ArrayRef<OpFoldResult> sizes,
                        ArrayRef<OpFoldResult> strides);

  /// Create a hyperrectangular slice with unit strides.
  HyperrectangularSlice(ArrayRef<OpFoldResult> offsets,
                        ArrayRef<OpFoldResult> sizes);

  /// Infer a hyperrectangular slice from `OffsetSizeAndStrideOpInterface`.
  HyperrectangularSlice(OffsetSizeAndStrideOpInterface op);

  ArrayRef<OpFoldResult> getMixedOffsets() const { return mixedOffsets; }
  ArrayRef<OpFoldResult> getMixedSizes() const { return mixedSizes; }
  ArrayRef<OpFoldResult> getMixedStrides() const { return mixedStrides; }

private:
  SmallVector<OpFoldResult> mixedOffsets;
  SmallVector<OpFoldResult> mixedSizes;
  SmallVector<OpFoldResult> mixedStrides;
};

using ValueDimList = SmallVector<std::pair<Value, std::optional<int64_t>>>;

/// A helper class to be used with `ValueBoundsOpInterface`. This class stores a
/// constraint system and mapping of constrained variables to index-typed
/// values or dimension sizes of shaped values.
///
/// Interface implementations of `ValueBoundsOpInterface` use `addBounds` to
/// insert constraints about their results and/or region block arguments into
/// the constraint set in the form of an AffineExpr. When a bound should be
/// expressed in terms of another value/dimension, `getExpr` can be used to
/// retrieve an AffineExpr that represents the specified value/dimension.
///
/// When a value/dimension is retrieved for the first time through `getExpr`,
/// it is added to an internal worklist. See `computeBound` for more details.
///
/// Note: Any modification of existing IR invalides the data stored in this
/// class. Adding new operations is allowed.
class ValueBoundsConstraintSet
    : public llvm::RTTIExtends<ValueBoundsConstraintSet, llvm::RTTIRoot> {
protected:
  /// Helper class that builds a bound for a shaped value dimension or
  /// index-typed value.
  class BoundBuilder {
  public:
    /// Specify a dimension, assuming that the underlying value is a shaped
    /// value.
    BoundBuilder &operator[](int64_t dim);

    // These overloaded operators add lower/upper/equality bounds.
    void operator<(AffineExpr expr);
    void operator<=(AffineExpr expr);
    void operator>(AffineExpr expr);
    void operator>=(AffineExpr expr);
    void operator==(AffineExpr expr);
    void operator<(OpFoldResult ofr);
    void operator<=(OpFoldResult ofr);
    void operator>(OpFoldResult ofr);
    void operator>=(OpFoldResult ofr);
    void operator==(OpFoldResult ofr);
    void operator<(int64_t i);
    void operator<=(int64_t i);
    void operator>(int64_t i);
    void operator>=(int64_t i);
    void operator==(int64_t i);

  protected:
    friend class ValueBoundsConstraintSet;
    BoundBuilder(ValueBoundsConstraintSet &cstr, Value value)
        : cstr(cstr), value(value) {}

  private:
    BoundBuilder(const BoundBuilder &) = delete;
    BoundBuilder &operator=(const BoundBuilder &) = delete;
    bool operator==(const BoundBuilder &) = delete;
    bool operator!=(const BoundBuilder &) = delete;

    ValueBoundsConstraintSet &cstr;
    Value value;
    std::optional<int64_t> dim;
  };

public:
  static char ID;

  /// A variable that can be added to the constraint set as a "column". The
  /// value bounds infrastructure can compute bounds for variables and compare
  /// two variables.
  ///
  /// Internally, a variable is represented as an affine map and operands.
  class Variable {
  public:
    /// Construct a variable for an index-typed attribute or SSA value.
    Variable(OpFoldResult ofr);

    /// Construct a variable for an index-typed SSA value.
    Variable(Value indexValue);

    /// Construct a variable for a dimension of a shaped value.
    Variable(Value shapedValue, int64_t dim);

    /// Construct a variable for an index-typed attribute/SSA value or for a
    /// dimension of a shaped value. A non-null dimension must be provided if
    /// and only if `ofr` is a shaped value.
    Variable(OpFoldResult ofr, std::optional<int64_t> dim);

    /// Construct a variable for a map and its operands.
    Variable(AffineMap map, ArrayRef<Variable> mapOperands);
    Variable(AffineMap map, ArrayRef<Value> mapOperands);

    MLIRContext *getContext() const { return map.getContext(); }

  private:
    friend class ValueBoundsConstraintSet;
    AffineMap map;
    ValueDimList mapOperands;
  };

  /// The stop condition when traversing the backward slice of a shaped value/
  /// index-type value. The traversal continues until the stop condition
  /// evaluates to "true" for a value.
  ///
  /// The first parameter of the function is the shaped value/index-typed
  /// value. The second parameter is the dimension in case of a shaped value.
  /// The third parameter is this constraint set.
  using StopConditionFn = std::function<bool(
      Value, std::optional<int64_t> /*dim*/, ValueBoundsConstraintSet &cstr)>;

  /// Compute a bound for the given variable. The computed bound is stored in
  /// `resultMap`. The operands of the bound are stored in `mapOperands`. An
  /// operand is either an index-type SSA value or a shaped value and a
  /// dimension.
  ///
  /// The bound is computed in terms of values/dimensions for which
  /// `stopCondition` evaluates to "true". To that end, the backward slice
  /// (reverse use-def chain) of the given value is visited in a worklist-driven
  /// manner and the constraint set is populated according to
  /// `ValueBoundsOpInterface` for each visited value.
  ///
  /// By default, lower/equal bounds are closed and upper bounds are open. If
  /// `closedUB` is set to "true", upper bounds are also closed.
  static LogicalResult
  computeBound(AffineMap &resultMap, ValueDimList &mapOperands,
               presburger::BoundType type, const Variable &var,
               StopConditionFn stopCondition, bool closedUB = false);

  /// Compute a bound in terms of the values/dimensions in `dependencies`. The
  /// computed bound consists of only constant terms and dependent values (or
  /// dimension sizes thereof).
  static LogicalResult
  computeDependentBound(AffineMap &resultMap, ValueDimList &mapOperands,
                        presburger::BoundType type, const Variable &var,
                        ValueDimList dependencies, bool closedUB = false);

  /// Compute a bound in that is independent of all values in `independencies`.
  ///
  /// Independencies are the opposite of dependencies. The computed bound does
  /// not contain any SSA values that are part of `independencies`. E.g., this
  /// function can be used to make ops hoistable from loops. To that end, ops
  /// must be made independent of loop induction variables (in the case of "for"
  /// loops). Loop induction variables are the independencies; they may not
  /// appear in the computed bound.
  static LogicalResult
  computeIndependentBound(AffineMap &resultMap, ValueDimList &mapOperands,
                          presburger::BoundType type, const Variable &var,
                          ValueRange independencies, bool closedUB = false);

  /// Compute a constant bound for the given variable.
  ///
  /// This function traverses the backward slice of the given operands in a
  /// worklist-driven manner until `stopCondition` evaluates to "true". The
  /// constraint set is populated according to `ValueBoundsOpInterface` for each
  /// visited value. (No constraints are added for values for which the stop
  /// condition evaluates to "true".)
  ///
  /// The stop condition is optional: If none is specified, the backward slice
  /// is traversed in a breadth-first manner until a constant bound could be
  /// computed.
  ///
  /// By default, lower/equal bounds are closed and upper bounds are open. If
  /// `closedUB` is set to "true", upper bounds are also closed.
  static FailureOr<int64_t>
  computeConstantBound(presburger::BoundType type, const Variable &var,
                       StopConditionFn stopCondition = nullptr,
                       bool closedUB = false);

  /// Compute a constant delta between the given two values. Return "failure"
  /// if a constant delta could not be determined.
  ///
  /// `dim1`/`dim2` must be `nullopt` if and only if `value1`/`value2` are
  /// index-typed.
  static FailureOr<int64_t>
  computeConstantDelta(Value value1, Value value2,
                       std::optional<int64_t> dim1 = std::nullopt,
                       std::optional<int64_t> dim2 = std::nullopt);

  /// Traverse the IR starting from the given value/dim and populate constraints
  /// as long as the stop condition holds. Also process all values/dims that are
  /// already on the worklist.
  void populateConstraints(Value value, std::optional<int64_t> dim);

  /// Comparison operator for `ValueBoundsConstraintSet::compare`.
  enum ComparisonOperator { LT, LE, EQ, GT, GE };

  /// Populate constraints for lhs/rhs (until the stop condition is met). Then,
  /// try to prove that, based on the current state of this constraint set
  /// (i.e., without analyzing additional IR or adding new constraints), the
  /// "lhs" value/dim is LE/LT/EQ/GT/GE than the "rhs" value/dim.
  ///
  /// Return "true" if the specified relation between the two values/dims was
  /// proven to hold. Return "false" if the specified relation could not be
  /// proven. This could be because the specified relation does in fact not hold
  /// or because there is not enough information in the constraint set. In other
  /// words, if we do not know for sure, this function returns "false".
  bool populateAndCompare(const Variable &lhs, ComparisonOperator cmp,
                          const Variable &rhs);

  /// Return "true" if "lhs cmp rhs" was proven to hold. Return "false" if the
  /// specified relation could not be proven. This could be because the
  /// specified relation does in fact not hold or because there is not enough
  /// information in the constraint set. In other words, if we do not know for
  /// sure, this function returns "false".
  ///
  /// This function keeps traversing the backward slice of lhs/rhs until could
  /// prove the relation or until it ran out of IR.
  static bool compare(const Variable &lhs, ComparisonOperator cmp,
                      const Variable &rhs);

  /// Compute whether the given variables are equal. Return "failure" if
  /// equality could not be determined.
  static FailureOr<bool> areEqual(const Variable &var1, const Variable &var2);

  /// Return "true" if the given slices are guaranteed to be overlapping.
  /// Return "false" if the given slices are guaranteed to be non-overlapping.
  /// Return "failure" if unknown.
  ///
  /// Slices are overlapping if for all dimensions:
  /// *      offset1 + size1 * stride1 <= offset2
  /// * and  offset2 + size2 * stride2 <= offset1
  ///
  /// Slice are non-overlapping if the above constraint is not satisfied for
  /// at least one dimension.
  static FailureOr<bool> areOverlappingSlices(MLIRContext *ctx,
                                              HyperrectangularSlice slice1,
                                              HyperrectangularSlice slice2);

  /// Return "true" if the given slices are guaranteed to be equivalent.
  /// Return "false" if the given slices are guaranteed to be non-equivalent.
  /// Return "failure" if unknown.
  ///
  /// Slices are equivalent if their offsets, sizes and strices are equal.
  static FailureOr<bool> areEquivalentSlices(MLIRContext *ctx,
                                             HyperrectangularSlice slice1,
                                             HyperrectangularSlice slice2);

  /// Add a bound for the given index-typed value or shaped value. This function
  /// returns a builder that adds the bound.
  BoundBuilder bound(Value value) { return BoundBuilder(*this, value); }

  /// Return an expression that represents the given index-typed value or shaped
  /// value dimension. If this value/dimension was not used so far, it is added
  /// to the worklist.
  ///
  /// `dim` must be `nullopt` if and only if the given value is of index type.
  AffineExpr getExpr(Value value, std::optional<int64_t> dim = std::nullopt);

  /// Return an expression that represents a constant or index-typed SSA value.
  /// In case of a value, if this value was not used so far, it is added to the
  /// worklist.
  AffineExpr getExpr(OpFoldResult ofr);

  /// Return an expression that represents a constant.
  AffineExpr getExpr(int64_t constant);

  /// Debugging only: Dump the constraint set and the column-to-value/dim
  /// mapping to llvm::errs.
  void dump() const;

protected:
  /// Dimension identifier to indicate a value is index-typed. This is used for
  /// internal data structures/API only.
  static constexpr int64_t kIndexValue = -1;

  /// An index-typed value or the dimension of a shaped-type value.
  using ValueDim = std::pair<Value, int64_t>;

  ValueBoundsConstraintSet(MLIRContext *ctx, StopConditionFn stopCondition,
                           bool addConservativeSemiAffineBounds = false);

  /// Return "true" if, based on the current state of the constraint system,
  /// "lhs cmp rhs" was proven to hold. Return "false" if the specified relation
  /// could not be proven. This could be because the specified relation does in
  /// fact not hold or because there is not enough information in the constraint
  /// set. In other words, if we do not know for sure, this function returns
  /// "false".
  ///
  /// This function does not analyze any IR and does not populate any additional
  /// constraints.
  bool comparePos(int64_t lhsPos, ComparisonOperator cmp, int64_t rhsPos);

  /// Given an affine map with a single result (and map operands), add a new
  /// column to the constraint set that represents the result of the map.
  /// Traverse additional IR starting from the map operands as needed (as long
  /// as the stop condition is not satisfied). Also process all values/dims that
  /// are already on the worklist. Return the position of the newly added
  /// column.
  int64_t populateConstraints(AffineMap map, ValueDimList mapOperands);

  /// Iteratively process all elements on the worklist until an index-typed
  /// value or shaped value meets `stopCondition`. Such values are not processed
  /// any further.
  void processWorklist();

  /// Bound the given column in the underlying constraint set by the given
  /// expression.
  void addBound(presburger::BoundType type, int64_t pos, AffineExpr expr);

  /// Return the column position of the given value/dimension. Asserts that the
  /// value/dimension exists in the constraint set.
  int64_t getPos(Value value, std::optional<int64_t> dim = std::nullopt) const;

  /// Return an affine expression that represents column `pos` in the constraint
  /// set.
  AffineExpr getPosExpr(int64_t pos);

  /// Return "true" if the given value/dim is mapped (i.e., has a corresponding
  /// column in the constraint system).
  bool isMapped(Value value, std::optional<int64_t> dim = std::nullopt) const;

  /// Insert a value/dimension into the constraint set. If `isSymbol` is set to
  /// "false", a dimension is added. The value/dimension is added to the
  /// worklist if `addToWorklist` is set.
  ///
  /// Note: There are certain affine restrictions wrt. dimensions. E.g., they
  /// cannot be multiplied. Furthermore, bounds can only be queried for
  /// dimensions but not for symbols.
  int64_t insert(Value value, std::optional<int64_t> dim, bool isSymbol = true,
                 bool addToWorklist = true);

  /// Insert an anonymous column into the constraint set. The column is not
  /// bound to any value/dimension. If `isSymbol` is set to "false", a dimension
  /// is added.
  ///
  /// Note: There are certain affine restrictions wrt. dimensions. E.g., they
  /// cannot be multiplied. Furthermore, bounds can only be queried for
  /// dimensions but not for symbols.
  int64_t insert(bool isSymbol = true);

  /// Insert the given affine map and its bound operands as a new column in the
  /// constraint system. Return the position of the new column. Any operands
  /// that were not analyzed yet are put on the worklist.
  int64_t insert(AffineMap map, ValueDimList operands, bool isSymbol = true);
  int64_t insert(const Variable &var, bool isSymbol = true);

  /// Project out the given column in the constraint set.
  void projectOut(int64_t pos);

  /// Project out all columns for which the condition holds.
  void projectOut(function_ref<bool(ValueDim)> condition);

  void projectOutAnonymous(std::optional<int64_t> except = std::nullopt);

  /// Mapping of columns to values/shape dimensions.
  SmallVector<std::optional<ValueDim>> positionToValueDim;
  /// Reverse mapping of values/shape dimensions to columns.
  DenseMap<ValueDim, int64_t> valueDimToPosition;

  /// Worklist of values/shape dimensions that have not been processed yet.
  std::queue<int64_t> worklist;

  /// Constraint system of equalities and inequalities.
  FlatLinearConstraints cstr;

  /// Builder for constructing affine expressions.
  Builder builder;

  /// The current stop condition function.
  StopConditionFn stopCondition = nullptr;

  /// Should conservative bounds be added for semi-affine expressions.
  bool addConservativeSemiAffineBounds = false;
};

} // namespace mlir

#include "mlir/Interfaces/ValueBoundsOpInterface.h.inc"

#endif // MLIR_INTERFACES_VALUEBOUNDSOPINTERFACE_H_
