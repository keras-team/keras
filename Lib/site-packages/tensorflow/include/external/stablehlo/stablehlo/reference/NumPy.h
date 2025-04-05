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

#ifndef STABLEHLO_REFERENCE_NUMPY_H
#define STABLEHLO_REFERENCE_NUMPY_H

#include "llvm/Support/ErrorOr.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "stablehlo/reference/Tensor.h"

namespace mlir {
namespace stablehlo {
namespace numpy {

// Filename to use for the metadata CSV file associating probeId values to
// serialized NumPy files.
constexpr char kInstrumentationMetadataFilename[] = "index.csv";

// Read a NumPy serialized tensor from disk stored at `filename` with the given
// `type`.
llvm::ErrorOr<Tensor> deserializeTensor(StringRef filename, ShapedType type);

// Store a tensor using the NumPy file format with the given `type` to the given
// `filename`.
llvm::Error serializeTensor(StringRef filename, ShapedType type,
                            const char* data);

}  // namespace numpy
}  // namespace stablehlo
}  // namespace mlir

#endif  // STABLEHLO_REFERENCE_NUMPY_H
