/* Copyright 2024 The StableHLO Authors. All Rights Reserved.
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

#ifndef THIRD_PARTY_STABLEHLO_STABLEHLO_TRANSFORMS_PASS_UTILS_H_
#define THIRD_PARTY_STABLEHLO_STABLEHLO_TRANSFORMS_PASS_UTILS_H_

#include "llvm/Support/ErrorHandling.h"
#include "mlir/Dialect/Complex/IR/Complex.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributeInterfaces.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LLVM.h"
#include "stablehlo/dialect/ChloOps.h"

namespace mlir {
namespace stablehlo {
// Add utility functions common across passes.

// Creates a chlo::ConstantLikeOp using a splat `constant` of the same shape
// as `val`.
template <typename T>
Value getConstantLike(OpBuilder &b, Location loc, T constant, Value val) {
  Type ty = getElementTypeOrSelf(val.getType());
  auto getAttr = [&]() -> Attribute {
    if (isa<IntegerType>(ty)) return b.getIntegerAttr(ty, constant);
    if (isa<FloatType>(ty)) return b.getFloatAttr(ty, constant);
    if (auto complexTy = dyn_cast<ComplexType>(ty)) {
      return complex::NumberAttr::get(complexTy, constant, 0);
    }
    llvm_unreachable("unhandled element type");
  };
  return b.create<mlir::chlo::ConstantLikeOp>(loc, cast<TypedAttr>(getAttr()),
                                              val);
}

// Creates a chlo::ConstantLikeOp using a APFloat splat `constant` of the
// same shape as `val`.
Value getConstantLike(OpBuilder &b, Location loc, const APFloat &constant,
                      Value val);

// Check if any of the given types are mlir::quant::QuantizedType.
bool isAnyQuantizedTypes(TypeRange types);

}  // namespace stablehlo
}  // namespace mlir

#endif  // THIRD_PARTY_STABLEHLO_STABLEHLO_TRANSFORMS_PASS_UTILS_H_
