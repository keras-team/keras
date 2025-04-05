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

// Host Memory Allocator Abstraction
//
// This file declares the generic interface for host memory allocators.

#ifndef TFRT_HOST_CONTEXT_HOST_ALLOCATOR_H_
#define TFRT_HOST_CONTEXT_HOST_ALLOCATOR_H_

#include <memory>

#include "llvm/ADT/ArrayRef.h"
#include "tfrt/support/forward_decls.h"

namespace tfrt {

// This is a pure virtual base class for memory allocator implementations. This
// abstraction allows clients of the runtime to plug in their own memory
// allocation policies, e.g. to cordon off allocations for one HostContext to
// happen in a region of memory known to be pinned to a specific NUMA zone.
//
// The major allocations in the runtime get funnelled through this interface,
// but this is just an optimization - small node allocations and other
// non-critical allocations can just use malloc/free or new/delete.
//
// This interface is just for host memory, it has nothing to with device memory
// allocation.
class HostAllocator {
 public:
  virtual ~HostAllocator() = default;

  // Allocate memory for one or more entries of type T.
  template <typename T>
  T* Allocate(size_t num_elements = 1) {
    return static_cast<T*>(AllocateBytes(sizeof(T) * num_elements, alignof(T)));
  }

  // Deallocate the memory for one or more entries of type T.
  template <typename T>
  void Deallocate(T* ptr, size_t num_elements = 1) {
    DeallocateBytes(ptr, sizeof(T) * num_elements);
  }

  // Allocate the specified number of bytes with the specified alignment.
  virtual void* AllocateBytes(size_t size, size_t alignment) = 0;

  // Deallocate the specified pointer that has the specified size.
  virtual void DeallocateBytes(void* ptr, size_t size) = 0;

 protected:
  friend class HostContext;
  HostAllocator() = default;
  HostAllocator(const HostAllocator&) = delete;
  HostAllocator& operator=(const HostAllocator&) = delete;

 private:
  virtual void VtableAnchor();
};

// Create an allocator that just calls malloc/free.
std::unique_ptr<HostAllocator> CreateMallocAllocator();

// Create an allocator of fixed size for testing.
std::unique_ptr<HostAllocator> CreateFixedSizeAllocator(size_t capacity = 1024);

// An RAII-based abstraction that manages an array of objects via HostAllocator.
template <typename ObjectT>
class HostArray {
 public:
  HostArray() {}
  HostArray(size_t num_objects, HostAllocator* host_allocator)
      : array_(MutableArrayRef<ObjectT>(
            host_allocator->Allocate<ObjectT>(num_objects), num_objects)),
        host_allocator_(host_allocator) {}

  ~HostArray() {
    // Destroy all of the objects.
    for (auto& object : array_) object.~ObjectT();
    if (host_allocator_ != nullptr)
      host_allocator_->Deallocate(array_.data(), array_.size());
  }

  // HostArray is move-only.
  HostArray(HostArray&& other)
      : array_(other.array_), host_allocator_(other.host_allocator_) {
    other.array_ = {};
    other.host_allocator_ = nullptr;
  }
  HostArray& operator=(HostArray&& other) {
    array_ = other.array_;
    other.array_ = {};

    host_allocator_ = other.host_allocator_;
    other.host_allocator_ = nullptr;
    return *this;
  }
  HostArray(const HostArray&) = delete;
  HostArray& operator=(const HostArray&) = delete;

  ArrayRef<ObjectT> array() const { return array_; }
  MutableArrayRef<ObjectT>& mutable_array() { return array_; }
  size_t size() const { return array_.size(); }
  ObjectT& operator[](size_t index) { return array_[index]; }

 private:
  MutableArrayRef<ObjectT> array_;
  HostAllocator* host_allocator_ = nullptr;
};

}  // namespace tfrt

#endif  // TFRT_HOST_CONTEXT_HOST_ALLOCATOR_H_
