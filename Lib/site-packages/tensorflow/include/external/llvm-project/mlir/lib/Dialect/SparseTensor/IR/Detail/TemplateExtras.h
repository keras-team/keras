//===- TemplateExtras.h -----------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_SPARSETENSOR_IR_DETAIL_TEMPLATEEXTRAS_H
#define MLIR_DIALECT_SPARSETENSOR_IR_DETAIL_TEMPLATEEXTRAS_H

#include <utility>

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/raw_ostream.h"

namespace mlir {
namespace sparse_tensor {
namespace ir_detail {

//===----------------------------------------------------------------------===//
template <typename T>
using has_print_method =
    decltype(std::declval<T>().print(std::declval<llvm::raw_ostream &>()));
template <typename T>
using detect_has_print_method = llvm::is_detected<has_print_method, T>;
template <typename T, typename R = void>
using enable_if_has_print_method =
    std::enable_if_t<detect_has_print_method<T>::value, R>;

/// Generic template for defining `operator<<` overloads which delegate
/// to `T::print(raw_ostream&) const`.
template <typename T>
inline enable_if_has_print_method<T, llvm::raw_ostream &>
operator<<(llvm::raw_ostream &os, T const &t) {
  t.print(os);
  return os;
}

//===----------------------------------------------------------------------===//
template <typename T>
static constexpr bool IsZeroCostAbstraction =
    // These two predicates license the compiler to make optimizations.
    std::is_trivially_copyable_v<T> && std::is_trivially_destructible_v<T> &&
    // This helps ensure ABI compatibility (e.g., padding and alignment).
    std::is_standard_layout_v<T> &&
    // These two are what SmallVector uses to determine whether it can
    // use memcpy.
    std::is_trivially_copy_constructible<T>::value &&
    std::is_trivially_move_constructible<T>::value;

//===----------------------------------------------------------------------===//

} // namespace ir_detail
} // namespace sparse_tensor
} // namespace mlir

#endif // MLIR_DIALECT_SPARSETENSOR_IR_DETAIL_TEMPLATEEXTRAS_H
