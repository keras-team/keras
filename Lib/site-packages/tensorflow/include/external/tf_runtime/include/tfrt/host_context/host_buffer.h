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

// Reference counted host buffer
//
// This file declares HostBuffer.

#ifndef TFRT_HOST_CONTEXT_HOST_BUFFER_H_
#define TFRT_HOST_CONTEXT_HOST_BUFFER_H_

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/FunctionExtras.h"
#include "llvm/Support/raw_ostream.h"
#include "tfrt/support/forward_decls.h"
#include "tfrt/support/ref_count.h"
#include "tfrt/support/type_traits.h"

namespace tfrt {
class HostAllocator;

// HostBuffer is a reference counted chunk of untyped memory on the host, whose
// memory may be managed by HostContext or allocated by the user.
class HostBuffer : public ReferenceCounted<HostBuffer> {
 public:
  // Create an uninitialized HostBuffer of the specified size and alignment.
  // This returns a null RCReference on allocation failure.
  // `allocator` will be used to allocate the memory and to deallocate it
  // when the returned buffer is destroyed.
  static RCReference<HostBuffer> CreateUninitialized(size_t size,
                                                     size_t alignment,
                                                     HostAllocator *allocator);

  using Deallocator = llvm::unique_function<void(void *ptr, size_t size)>;
  // Create a HostBuffer by taking ownership of an externally allocated buffer.
  // `deallocator` is called with `ptr` and `size` as arguments when we destroy
  // this buffer.
  static RCReference<HostBuffer> CreateFromExternal(void *ptr, size_t size,
                                                    Deallocator deallocator);

  // Create a HostBuffer by creating a reference to an externally allocated
  // HostBuffer. Manages the data in the original buffer from `offset` to
  // `offset + size`.
  // This returns a null RCReference if `parent_buffer` is null. Asserts that
  // `offset + size` is within the `parent_buffer`.
  static RCReference<HostBuffer> CreateFromExternal(
      RCReference<HostBuffer> parent_buffer, size_t offset, size_t size);

  // Returns the address of the data. If the buffer is empty, the behavior is
  // undefined to dereference the returned address.
  void *data() { return data_; }

  // Returns the address of the data. If the buffer is empty, the behavior is
  // undefined to dereference the returned address.
  const void *data() const { return data_; }

  size_t size() const { return size_; }

  template <typename T>
  ArrayRef<T> CastAs() const {
    assert((size() % sizeof(T) == 0) &&
           "Invalid size for HostBuffer::CastAs<T>");
    assert(IsAlignedPtr<T>(data()) &&
           "HostBuffer is not aligned for the target type");
    return {static_cast<const T *>(data()), size() / sizeof(T)};
  }

  template <typename T>
  MutableArrayRef<T> CastAs() {
    auto array_ref = static_cast<const HostBuffer *>(this)->CastAs<T>();
    return {const_cast<T *>(array_ref.data()), array_ref.size()};
  }

  // Returns `true` iff `*this` is an exclusive owner of the underlying data.
  bool IsExclusiveDataOwner() {
    // We don't know anything about the custom deallocator and can't guarantee
    // that we have an exclusive access to the data.
    if (mode_ == Mode::kOutOfLine) return false;

    // There are multiple references to this buffer, and we can't claim
    // that we have an exclusive access to the data.
    if (!IsUnique()) return false;

    // The last reference to inlined buffer has exclusive access to the data.
    if (mode_ == Mode::kInlined) return true;

    // Otherwise check if the parent buffer has an exclusive access.
    if (mode_ == Mode::kSliced) {
      return sliced_parent_buffer_->IsExclusiveDataOwner();
    }

    return false;
  }

 private:
  // For access to Destroy().
  friend class ReferenceCounted<HostBuffer>;

  HostBuffer(size_t size, size_t allocated_size, HostAllocator *allocator)
      : size_(size),
        mode_{Mode::kInlined},
        inlined_{allocator, allocated_size} {}

  HostBuffer(void *ptr, size_t size, Deallocator deallocator)
      : data_(ptr),
        size_(size),
        mode_{Mode::kOutOfLine},
        out_of_line_deallocator_{std::move(deallocator)} {}

  HostBuffer(void *ptr, size_t size, RCReference<HostBuffer> parent_buffer)
      : data_(ptr),
        size_(size),
        mode_{Mode::kSliced},
        sliced_parent_buffer_{std::move(parent_buffer)} {}

  ~HostBuffer();

  void Destroy();

  void *data_;
  size_t size_ : 62;

  enum class Mode : uint8_t {
    kInlined,
    kOutOfLine,
    kSliced,
  };

  // The number of bits in `mode_` should be adjusted to represent the number of
  // enum values in `Mode`. The total number of bits for `mode_` and `size_`
  // should be 64 bits.
  Mode mode_ : 2;

  // TODO(zhangqiaorjc): Use variant instead of union.
  union {
    struct {
      HostAllocator *allocator;
      size_t allocated_size;
    } inlined_;

    Deallocator out_of_line_deallocator_;

    RCReference<HostBuffer> sliced_parent_buffer_;
  };

  // The inlined data is allocated in the flexible memory array. This needs to
  // be the last member of the class.
  alignas(alignof(std::max_align_t)) char unaligned_data_[];
};

llvm::raw_ostream &operator<<(llvm::raw_ostream &os, const HostBuffer &buffer);

}  // namespace tfrt

#endif  // TFRT_HOST_CONTEXT_HOST_BUFFER_H_
