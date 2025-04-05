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

#ifndef STABLEHLO_REFERENCE_TYPES_H
#define STABLEHLO_REFERENCE_TYPES_H

#include "mlir/IR/Types.h"

namespace mlir {
namespace stablehlo {

/// Check if the type 'type' is a supported unsigned integer type per the
/// StableHLO specification. Such types are: ui4, ui8, ui16, ui32 and ui64.
bool isSupportedUnsignedIntegerType(Type type);

/// Check if the type 'type' is a supported signed integer type per the
/// StableHLO specification. Such types are: i4, i8, i16, i32 and i64.
bool isSupportedSignedIntegerType(Type type);

/// Check if the type 'type' is a supported boolean type per the StableHLO
/// specification. Supported type is: i1.
bool isSupportedBooleanType(Type type);

/// Check if the type 'type' is a supported integer type per the
/// StableHLO specification. See `isSupportedUnsignedIntegerType` and
/// `isSupportedSignedIntegerType` for a comprehensive list of supported types.
bool isSupportedIntegerType(Type type);

/// Check if the type 'type' is a supported floating-point type per the
/// StableHLO specification. Such types are: bf16, f16, f32, f64.
bool isSupportedFloatType(Type type);

/// Check if the type 'type' is a supported complex type per the
/// StableHLO specification. Such types are: complex<f32> and complex<f64>.
bool isSupportedComplexType(Type type);

/// Returns the number of bits in the representation of an element type.
///   * For boolean type: 1.
///   * For integer types: bit width (e.g. 32 for si32).
///   * For floating-point types: bit width (e.g. 32 for f32).
///   * For complex types: 2x the bit width of element type (e.g. 64 for
///     complex<f32>).
int64_t numBits(Type type);

}  // namespace stablehlo
}  // namespace mlir

#endif  // STABLEHLO_REFERENCE_TYPES_H
