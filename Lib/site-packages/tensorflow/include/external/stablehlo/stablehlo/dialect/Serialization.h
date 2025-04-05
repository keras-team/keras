/* Copyright 2023 The StableHLO Authors.

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

#ifndef STABLEHLO_DIALECT_SERIALIZATION_H
#define STABLEHLO_DIALECT_SERIALIZATION_H

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Support/LogicalResult.h"

namespace mlir {
namespace stablehlo {

// Write a StableHLO program to a portable artifact
// Writes a stable payload for `module` to `os`. If compatibility with a
// previous version of StableHLO is required, provide the required version
// string `#.#.#` for `targetVersion`.
//
// Can fail if `module` cannot be expressed in the `targetVersion` version of
// StableHLO, e.g. if it's using new or removed features, or if it involves
// unsupported dialects.
LogicalResult serializePortableArtifact(ModuleOp module,
                                        StringRef targetVersion,
                                        raw_ostream& os);

// Read StableHLO portable artifact
//
// Can fail if `sourceStr` cannot be expressed in the current version of
// StableHLO, e.g. if it's using incompatible features. Returns nullptr if
// `sourceStr` is invalid or fails to deserialize.
OwningOpRef<ModuleOp> deserializePortableArtifact(StringRef sourceStr,
                                                  MLIRContext* context);

}  // namespace stablehlo
}  // namespace mlir

#endif  // STABLEHLO_DIALECT_SERIALIZATION_H
