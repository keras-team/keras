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

#ifndef SHARDY_DIALECT_SDY_TRANSFORMS_PROPAGATION_FACTOR_PROPAGATION_H_
#define SHARDY_DIALECT_SDY_TRANSFORMS_PROPAGATION_FACTOR_PROPAGATION_H_

#include <cstdint>

#include "mlir/IR/Operation.h"
#include "mlir/Support/LLVM.h"
#include "shardy/dialect/sdy/ir/dialect.h"
#include "shardy/dialect/sdy/transforms/propagation/sharding_projection.h"

namespace mlir {
namespace sdy {

// An interface for propagating factor shardings.
class FactorPropagation {
 public:
  virtual ~FactorPropagation() = default;

  // Propagates the factor shardings in `projection`.
  //
  // * `direction` specifies the direction of propagation.
  // * `factorSizes` is the size of each factor.
  // * `mesh` is the mesh that the factors are sharded over.
  // * `op` is the operation that the factor shardings are propagated through.
  //
  // `conservativePropagation` specifies whether to disallow sub axes when
  // calculating the compatible major axes. If the projection contains a
  // sub-axis, then the axes (and any axes further sharding the factor) is
  // excluded from the result.
  virtual UpdateTensorShardings propagateFactorShardings(
      ShardingProjection& projection, PropagationDirection direction,
      ArrayRef<int64_t> factorSizes, MeshAttr mesh, Operation* op,
      bool conservativePropagation) const = 0;
};

}  // namespace sdy
}  // namespace mlir

#endif  // SHARDY_DIALECT_SDY_TRANSFORMS_PROPAGATION_FACTOR_PROPAGATION_H_
