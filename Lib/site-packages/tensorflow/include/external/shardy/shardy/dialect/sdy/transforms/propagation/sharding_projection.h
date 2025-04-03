/* Copyright 2024 The Shardy Authors.

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

#ifndef SHARDY_DIALECT_SDY_TRANSFORMS_PROPAGATION_SHARDING_PROJECTION_H_
#define SHARDY_DIALECT_SDY_TRANSFORMS_PROPAGATION_SHARDING_PROJECTION_H_

#include <cstdint>

#include "llvm/ADT/DenseMap.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Operation.h"
#include "mlir/Support/LLVM.h"
#include "shardy/dialect/sdy/ir/dialect.h"

namespace mlir {
namespace sdy {

// Returns true if the `oldAxes` is a strict prefix of `newAxes`,
bool shouldUpdate(ArrayRef<AxisRefAttr> oldAxes, ArrayRef<AxisRefAttr> newAxes);

// The axes along which a factor is sharded, and whether the factor can be
// further sharded (unless it's fully sharded already).
struct FactorSharding {
  SmallVector<AxisRefAttr> axisRefs;
  bool isClosed = false;
  bool isMinorMost = false;
  // Additional axes in the dimension sharding that was projected to this
  // `FactorSharding`, such that the size of the first overflow axis doesn't
  // divide the factor size, and the factor is non-minor-most.
  //
  // We need to store these axes so that we can add them when projecting back to
  // dimension shardings.
  SmallVector<AxisRefAttr> overflowAxes;

  bool operator==(const FactorSharding& other) const {
    return axisRefs == other.axisRefs && isClosed == other.isClosed &&
           isMinorMost == other.isMinorMost &&
           overflowAxes == other.overflowAxes;
  }

  bool operator!=(const FactorSharding& other) const {
    return !(*this == other);
  }
};

using FactorIndexToSharding = llvm::DenseMap<int64_t, FactorSharding>;

// Holds the factor shardings and replicated axes of a tensor.
struct TensorFactorShardings {
  // A mapping between factor index to the sharding of that factor.
  // TODO(tomnatan): consider using a vector with null for unmapped factors.
  FactorIndexToSharding factorIndexToSharding;
  SmallVector<AxisRefAttr> replicatedAxes;

  bool operator==(const TensorFactorShardings& other) const {
    return factorIndexToSharding == other.factorIndexToSharding &&
           replicatedAxes == other.replicatedAxes;
  }

  bool operator!=(const TensorFactorShardings& other) const {
    return !(*this == other);
  }

  // Updates the sharding axes of the given `factorIndex` to `newAxes` if
  // 1. this tensor is associated with that factor, and
  // 2. `newAxes` strictly contains existing axes. For example, ["a", "b"]
  //    strictly contains ["a"] and ["a", "b":(1)2]. We assume that `newAxes`
  //    and the existing axes share the same prefix. `newAxes` being ["a", "b"]
  //    is illegal if the existing axes are ["b", "a"] or "["a":(2)2]".
  // Returns if the sharding axes have been updated.
  bool updateShardingAxes(int64_t factorIndex, ArrayRef<AxisRefAttr> newAxes);

  // Creates a `TensorShardingAttr` by projecting the factor shardings in
  // this `TensorFactorShardings` to dimension shardings w.r.t. to
  // `tensorMapping`.
  //
  // Ignores sharding of any factor that needs strided view.
  TensorShardingAttr createTensorShardingAttr(MLIRContext* ctx,
                                              TensorMappingAttr tensorMapping,
                                              ArrayRef<int64_t> factorSizes,
                                              StringRef meshName,
                                              MeshAttr mesh) const;
};

// A struct that specifies which operands and results are updated.
struct UpdateTensorShardings {
  BitVector updateOperands;
  BitVector updateResults;

  UpdateTensorShardings(int64_t numOperands, int64_t numResults)
      : updateOperands(BitVector(numOperands)),
        updateResults(BitVector(numResults)) {}
};

// The sharding projection holds information about how factors (rather than
// dimensions), defined by an `OpShardingRuleAttr`, are sharded.
//
// It provides a view of the sharding axes from the perspective of factors. A
// typical workflow is
//   1. Project dimension shardings to factor shardings via a
//      `OpShardingRuleAttr`. This step may split axes.
//   2. Manipulate the shardings through `updateSharding`.
//   3. Project the updated factor shardings back to dimension shardings. This
//      step may merge sub-axes and require strided view.
//
// `ShardingProjection` holds additional information such as replicated axes
// or whether a factor is closed, so that a `TensorShardingAttr` can be
// reconstructed without losing information in Step 3.
//
// For example:
//
// Sharding rule - ([ij])->([i, j]) {i=2, j=4}
// Operand dimension sharding - [{"x", "y"}]
// Result dimension sharding - [{?}, {?}]
// Mesh - <"x"=4, "y"=2>
//
// The projection will produce the following factor shardings:
//   Operand - {i: ["x":(1)2], j: ["x":(2)2, "y"]}
//   Result - {i: [?], j: [?]}
//
// Assume that we update the result's factor sharding to be the same as the
// operand's, i.e., {i: ["x":(1)2], j: ["x":(2)2, "y"]}.
//
// Finally project back to dimension shardings to get the following
//   Operand dimension sharding - [{"x", "y"}]
//   Result dimension sharding - [{"x":(1)2}, {"x":(2)2, "y"}]
class ShardingProjection {
 public:
  ShardingProjection() = default;

  ShardingProjection(SmallVector<TensorFactorShardings> operands,
                     SmallVector<TensorFactorShardings> results);

  int64_t getNumOperands() const { return operands.size(); }
  int64_t getNumResults() const { return results.size(); }
  int64_t getNumTensors() const { return getNumOperands() + getNumResults(); }

  ArrayRef<TensorFactorShardings> getOperands() const { return operands; }
  ArrayRef<TensorFactorShardings> getResults() const { return results; }

  const TensorFactorShardings& getOperand(int64_t operandNum) const {
    return operands[operandNum];
  }
  const TensorFactorShardings& getResult(int64_t resultNum) const {
    return results[resultNum];
  }

  bool updateOperandSharding(int64_t operandIndex, int64_t factorIndex,
                             ArrayRef<AxisRefAttr> newAxes) {
    return operands[operandIndex].updateShardingAxes(factorIndex, newAxes);
  }
  bool updateResultSharding(int64_t resultIndex, int64_t factorIndex,
                            ArrayRef<AxisRefAttr> newAxes) {
    return results[resultIndex].updateShardingAxes(factorIndex, newAxes);
  }

  // Updates the shardings of all tensors that are associated with
  // `factorIndex` to be `newAxes` for that factor. Returns two BitVectors
  // indicating whether the operands and results have been updated.
  UpdateTensorShardings updateSharding(int64_t factorIndex,
                                       ArrayRef<AxisRefAttr> newAxes);

  // Builds a `ShardingProjection` for the given operand and result shardings,
  // w.r.t. the given `shardingRule`.
  static ShardingProjection build(ArrayRef<TensorShardingAttr> operandShardings,
                                  ArrayRef<TensorShardingAttr> resultShardings,
                                  OpShardingRuleAttr shardingRule,
                                  MeshAttr mesh);

  // Builds a `ShardingProjection` for the operand and result shardings of the
  // given `op`, w.r.t. the given `shardingRule`.
  static ShardingProjection build(Operation* op,
                                  OpShardingRuleAttr shardingRule,
                                  MeshAttr mesh);

  bool operator==(const ShardingProjection& other) const {
    return operands == other.operands && results == other.results;
  }

  bool operator!=(const ShardingProjection& other) const {
    return !(*this == other);
  }

 private:
  SmallVector<TensorFactorShardings> operands;
  SmallVector<TensorFactorShardings> results;
};

}  // namespace sdy
}  // namespace mlir

#endif  // SHARDY_DIALECT_SDY_TRANSFORMS_PROPAGATION_SHARDING_PROJECTION_H_
