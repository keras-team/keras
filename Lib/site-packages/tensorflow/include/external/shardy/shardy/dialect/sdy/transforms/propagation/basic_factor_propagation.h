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

#ifndef SHARDY_DIALECT_SDY_TRANSFORMS_PROPAGATION_BASIC_FACTOR_PROPAGATION_H_
#define SHARDY_DIALECT_SDY_TRANSFORMS_PROPAGATION_BASIC_FACTOR_PROPAGATION_H_

#include <cstdint>
#include <functional>
#include <optional>

#include "mlir/IR/Operation.h"
#include "mlir/Support/LLVM.h"
#include "shardy/dialect/sdy/ir/dialect.h"
#include "shardy/dialect/sdy/transforms/propagation/factor_propagation.h"
#include "shardy/dialect/sdy/transforms/propagation/sharding_projection.h"

namespace mlir {
namespace sdy {

// A conservative strategy of propagating sharding axes along the factor.
//
// Refer to the documentation of `getCompatibleMajorShardingAxes` for the
// conflicts this strategy considers.
//
// Aggressive strategies should extend this class, and override one or more of
// the virtual methods that this class provides.
class BasicFactorPropagation : public FactorPropagation {
 public:
  virtual ~BasicFactorPropagation() = default;

  // Propagates the factor shardings in `projection`.
  UpdateTensorShardings propagateFactorShardings(
      ShardingProjection& projection, PropagationDirection direction,
      ArrayRef<int64_t> factorSizes, MeshAttr mesh, Operation* op,
      bool conservativePropagation) const override;

 protected:
  // Finds all compatible major axes that can shard the given factor for all
  // tensors.
  //
  // We start from `getCompatibleMajorAxes()` to get the longest prefix of axes
  // that shard the given factor. We then truncate the list of axes by removing
  // conflicts.
  // 1. No conflicts within the factor. If a tensor is mapped to the given
  //    factor, the result cannot overlap with the tensor's replicated axes or
  //    the factor's overflow axes. If the tensor isn't already sharded along
  //    that factor and axis (or is only sharded along a sub-axis of it), all of
  //    the following must hold for its factor sharding:
  //   * it's open.
  //   * it has no overflow axes.
  //   * it's either minor-most or the sharded size up to and including this
  //     axis must divide the factor size.
  // 2. No conflicts across factors. The result cannot overlap with the sharded
  //    axes or overflow axes related to all other factors.
  //
  // The direction specifies whether to include axes that further shard the
  // given factor in just the operands (BACKWARD), just the results (FORWARD),
  // or both (BOTH). If there is a conflicting axis, then it's still taken into
  // consideration regardless of the direction. If `NONE` is passed in, then
  // no axes are returned.
  //
  // `conservativePropagation` specifies whether to disallow sub axes when
  // calculating the compatible major axes. If the projection contains a
  // sub-axis, then the axes (and any axes further sharding the factor) is
  // excluded from the result.
  //
  // For example (assuming compatibility with other factors and replicated) if
  // `direction` is `BOTH` and `conservativePropagation` is `false`.
  //   - Given factor shardings ["a", "b"] and ["a", "c"], returns ["a"].
  //   - Given factor shardings ["a"], [], ["a", "b", "c"], and ["a", "b", "d"],
  //     returns ["a","b"].
  //   - Given factor shardings ["a":(1)2] and ["a":(1)4], returns ["a":(1)4].
  //   - Given factor shardings ["a":(1)2, "b"] and ["a":(1)4], returns
  //     ["a":(1)2].
  SmallVector<AxisRefAttr> getCompatibleMajorShardingAxes(
      const ShardingProjection& projection, int64_t factorIndex,
      PropagationDirection direction, int64_t factorSize, MeshAttr mesh,
      Operation* op, bool conservativePropagation) const;

  // Finds the longest prefix of axes that shard the given factor, such that all
  // tensors either:
  // - Have the same prefix of axes sharding the given factor.
  // - Have a prefix of the longest prefix sharding the given factor, and aren't
  //   sharded further along that factor.
  // - Aren't mapped to the given factor.
  // This method does not resolve conflicts across factors or replicated axes.
  SmallVector<AxisRefAttr> getCompatibleMajorAxes(
      const ShardingProjection& projection, int64_t factorIndex,
      PropagationDirection direction, Operation* op) const;

  // Returns the largest prefix of `axisRef`, which does not overlap with
  // sharding axes and overflow axes for all other factors.
  //
  // This function does not consider the conflicts within the factor itself,
  // which are considered in `compatiblePrefixNoConflictsWithinFactor`. The
  // returned prefix can be overlapped with sharding axes and overflow axes of
  // the factor itself.
  //
  // Returns std::nullopt if the prefix does not exist.
  std::optional<AxisRefAttr> compatiblePrefixNoConflictsAcrossFactors(
      AxisRefAttr axisRef, const FactorIndexToSharding& factorIndexToSharding,
      int64_t factorIndex) const;

  // Returns the largest compatible prefix of `axisRef` by removing conflicts
  // with `replicatedAxes` and `factorSharding`.
  //
  // The returned prefix is not explicitly replicated, and it either:
  // 1. is already in the `factorSharding.axisRefs`
  // 2. is not in the `factorSharding.axisRefs`, and the factor satisfies
  //    * it is open.
  //    * it has no overflow axes.
  //    * it is minor-most or `factorSize` is divisible by `shardedSize`.
  //
  // Returns std::nullopt if the compatible prefix does not exist.
  std::optional<AxisRefAttr> compatiblePrefixNoConflictsWithinFactor(
      AxisRefAttr axisRef, ArrayRef<AxisRefAttr> replicatedAxes,
      const FactorSharding& factorSharding, int64_t shardedSize,
      int64_t factorSize) const;

  // For each axis in `axes`, call `removeConflicts` to get the compatible
  // prefix.
  // 1. If (1) `removeConflicts` returns `std::nullopt`, or (2)
  //    `conservativePropagation` is true and `removeConflicts` returns a
  //    sub-axis, remove the current axis and the following ones.
  // 2. If `removeConflicts` returns a prefix that is different from the current
  //    axis, replace the current axis with the returned one and remove the
  //    following axes.
  // 3. If `removeConflicts` returns the same axis, proceed with the next one.
  void truncateAxesByRemovingConflicts(
      SmallVector<AxisRefAttr>& axes,
      std::function<std::optional<AxisRefAttr>(AxisRefAttr curAxis,
                                               int64_t shardedSize)>
          removeConflicts,
      MeshAttr mesh, bool conservativePropagation) const;

 private:
  // Returns the largest compatible prefix of `axisRef` by removing conflicts in
  // `tensorFactorSharding`.
  //
  // If this tensor is not mapped to `factorIndex`, returns the prefix of
  // `axisRef` by removing conflicts with other factors.
  //
  // If this tensor is mapped to `factorIndex`, returns the prefix of `axisRef`
  // by removing conflicts with other factors and within the factor itself.
  std::optional<AxisRefAttr> compatiblePrefix(
      AxisRefAttr axisRef, const TensorFactorShardings& tensorFactorSharding,
      int64_t factorIndex, int64_t shardedSize, int64_t factorSize) const;

  // Returns the largest compatible prefix of `axisRef` by removing conflicts
  // with every `TensorFactorShardings` in `projection`.
  std::optional<AxisRefAttr> compatiblePrefix(
      AxisRefAttr axisRef, const ShardingProjection& projection,
      int64_t factorIndex, int64_t shardedSize, int64_t factorSize) const;
};

}  // namespace sdy
}  // namespace mlir

#endif  // SHARDY_DIALECT_SDY_TRANSFORMS_PROPAGATION_BASIC_FACTOR_PROPAGATION_H_
