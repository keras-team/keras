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

#ifndef SHARDY_DIALECT_SDY_TRANSFORMS_PROPAGATION_UTILS_H_
#define SHARDY_DIALECT_SDY_TRANSFORMS_PROPAGATION_UTILS_H_

#include "llvm/ADT/SmallVector.h"
#include "mlir/Support/LLVM.h"
#include "shardy/dialect/sdy/ir/dialect.h"

namespace mlir {
namespace sdy {

// Returns a vector with all indices that are set to true in `bitVector`.
SmallVector<int> toSetBitsVector(const BitVector& bitVector);

// Determines the union of two propagation directions.
// - (NONE, Any)         -> Any
// - (FORWARD, BACKWARD) -> BOTH
// - (BACKWARD, FORWARD) -> BOTH
// - (Any, BOTH)         -> BOTH
PropagationDirection unionOfPropagationDirections(PropagationDirection d1,
                                                  PropagationDirection d2);

// Determines the intersection of two propagation directions.
// - (NONE, Any)         -> NONE
// - (FORWARD, BACKWARD) -> NONE
// - (BACKWARD, FORWARD) -> NONE
// - (Any, BOTH)         -> Any
PropagationDirection intersectionOfPropagationDirections(
    PropagationDirection d1, PropagationDirection d2);

}  // namespace sdy
}  // namespace mlir

#endif  // SHARDY_DIALECT_SDY_TRANSFORMS_PROPAGATION_UTILS_H_
