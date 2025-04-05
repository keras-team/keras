/*
 * Copyright 2020 The TensorFlow Runtime Authors
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

// Helpers for defining sync kernels
//
// This file declares simple helper routines to make it easier to write
// synchronous kernels.

#ifndef TFRT_HOST_CONTEXT_SYNC_KERNEL_UTILS_H_
#define TFRT_HOST_CONTEXT_SYNC_KERNEL_UTILS_H_

#include <cstdint>
#include <type_traits>

#include "llvm/Support/Error.h"
#include "tfrt/host_context/attribute_utils.h"
#include "tfrt/host_context/host_context.h"
#include "tfrt/host_context/kernel_registry.h"
#include "tfrt/host_context/location.h"
#include "tfrt/host_context/sync_kernel_frame.h"
#include "tfrt/host_context/value.h"
#include "tfrt/support/ranges.h"
#include "tfrt/support/type_traits.h"

namespace tfrt {

// Kernels should use this so we know the kernel has an argument.
template <typename T>
class SyncArgument {
 public:
  explicit SyncArgument(Value* value) : value_(value) {}

  Value* value() const { return value_; }

  T& get() const { return value_->template get<T>(); }
  T* operator->() const { return &get(); }
  T& operator*() const { return get(); }

 private:
  Value* value_;
};

// RemainingSyncArguments collects all remaining arguments in an ArrayRef. There
// can be at most one RemainingSyncArguments, and it must appear after all other
// Arguments.
class RemainingSyncArguments {
 public:
  RemainingSyncArguments(ArrayRef<uint32_t> reg_indices,
                         ArrayRef<Value*> registers)
      : reg_indices_{reg_indices}, registers_{registers} {}

  size_t size() const { return reg_indices_.size(); }
  Value* operator[](size_t i) const { return registers_[reg_indices_[i]]; }

 private:
  ArrayRef<uint32_t> reg_indices_;
  ArrayRef<Value*> registers_;
};

namespace internal {
// For use by RepeatedSyncArguments below.
struct RepeatedSyncArgumentsBase {
  const uint32_t* reg_indices;
  Value* const* registers;
  bool operator==(const RepeatedSyncArgumentsBase& b) const {
    return reg_indices == b.reg_indices && registers == b.registers;
  }
};
}  // namespace internal

// RepeatedArguments collects all remaining arguments of the same type in an
// ArrayRef. There can be at most one RemainingArguments/RepeatedArguments, and
// it must appear after all other Arguments.
template <typename T>
class RepeatedSyncArguments
    : public IndexedAccessorRangeBase<RepeatedSyncArguments<T>,
                                      internal::RepeatedSyncArgumentsBase, T> {
  using IndexBaseT = internal::RepeatedSyncArgumentsBase;
  using RangeBaseT =
      IndexedAccessorRangeBase<RepeatedSyncArguments<T>,
                               internal::RepeatedSyncArgumentsBase, T>;

 public:
  RepeatedSyncArguments(ArrayRef<uint32_t> reg_indices,
                        ArrayRef<Value*> registers)
      : RangeBaseT(IndexBaseT{reg_indices.data(), registers.data()},
                   reg_indices.size()) {}

 private:
  // See `llvm::detail::indexed_accessor_range_base` for details.
  static IndexBaseT offset_base(const IndexBaseT& base, ptrdiff_t index) {
    return IndexBaseT{base.reg_indices + index, base.registers};
  }
  // See `llvm::detail::indexed_accessor_range_base` for details.
  static T& dereference_iterator(const IndexBaseT& base, ptrdiff_t index) {
    return base.registers[base.reg_indices[index]]->get<T>();
  }

  // Allow access to `offset_base` and `dereference_iterator`.
  friend RangeBaseT;
};

}  // namespace tfrt

#endif  // TFRT_HOST_CONTEXT_SYNC_KERNEL_UTILS_H_
