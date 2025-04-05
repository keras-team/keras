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

#ifndef STABLEHLO_REFERENCE_AXES_H
#define STABLEHLO_REFERENCE_AXES_H

#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/IR/BuiltinAttributes.h"

namespace mlir {
namespace stablehlo {

using Axis = int64_t;

/// Represents axes of a tensor.
class Axes : public SmallVector<int64_t> {
 public:
  Axes() = default;
  Axes(const Axes &other) = default;
  Axes &operator=(const Axes &other) = default;

  Axes(std::initializer_list<int64_t> list) : SmallVector(list) {}
  explicit Axes(size_t size, int64_t element = 0)
      : SmallVector(size, element) {}
  explicit Axes(ArrayRef<int64_t> array) : SmallVector(array) {}
  explicit Axes(DenseIntElementsAttr attr)
      : SmallVector(attr.getValues<int64_t>()) {}
};

raw_ostream &operator<<(raw_ostream &os, const Axes &x);

}  // namespace stablehlo
}  // namespace mlir

#endif  // STABLEHLO_REFERENCE_AXES_H
