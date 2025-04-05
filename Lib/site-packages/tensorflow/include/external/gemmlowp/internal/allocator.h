// Copyright 2015 The Gemmlowp Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// allocator.h: a buffer allocator that allows avoiding most of the
// malloc/free overhead, by:
// 1. Requiring all N allocations to be reserved in advance, and
//    then commited at once, turning N allocations into 1.
// 2. Being persistent, the allocated storage is reused across commits,
//    and only reallocated as needed when the commit size gets larger.
//
// This is driven by Android-specific needs:
// 1. On Android, the default (Bionic) allocator tends to aggressively
// unmap pages, which means that malloc/free can be surprisingly expensive.
// 2. On Android, stack allocations with alloca() can't be as large as on
// desktop platforms.
//
// General usage:
// 1. Reserve blocks by calling Reserve(), which returns a Handle.
// 2. Call Commit() once.
// 3. Now it is possible to get pointers to allocated buffers by calling
//    GetPointer().
// 4. Call Decommit() once.
// 5. The allocator is now reverted to its original state, except that
//    it retained its allocated storage, so the next Commit() will be faster.
//    The allocated storage is only freed when the Allocator object is
//    destroyed.

#ifndef GEMMLOWP_INTERNAL_ALLOCATOR_H_
#define GEMMLOWP_INTERNAL_ALLOCATOR_H_

#include "common.h"

namespace gemmlowp {

enum class TypeId : std::uint8_t { Uint8, Int8, Uint16, Int16, Uint32, Int32 };

template <typename T>
struct GetTypeIdImpl {};

template <typename T>
inline TypeId GetTypeId() {
  return GetTypeIdImpl<T>::Value;
}

template <typename T>
struct GetTypeIdImpl<const T> : GetTypeIdImpl<T> {};

#define GEMMLOWP_REGISTER_TYPEID(type_, id) \
  template <>                               \
  struct GetTypeIdImpl<type_> {             \
    static const TypeId Value = TypeId::id; \
  };

GEMMLOWP_REGISTER_TYPEID(std::uint8_t, Uint8)
GEMMLOWP_REGISTER_TYPEID(std::int8_t, Int8)
GEMMLOWP_REGISTER_TYPEID(std::uint16_t, Uint16)
GEMMLOWP_REGISTER_TYPEID(std::int16_t, Int16)
GEMMLOWP_REGISTER_TYPEID(std::uint32_t, Uint32)
GEMMLOWP_REGISTER_TYPEID(std::int32_t, Int32)

class Allocator {
 public:
  Allocator()
      : committed_(false),
        storage_size_(0),
        storage_(nullptr),
        reserved_blocks_(0),
        reserved_bytes_(0),
        generation_(0) {}

  ~Allocator() {
    assert(!committed_);
    assert(!reserved_blocks_);
    DeallocateStorage();
  }

  // Alignment of allocated blocks.
  static constexpr std::size_t kAlignment = kDefaultCacheLineSize;

  // This is all we need so far, and since the usage pattern is fixed,
  // there is no point in allowing more until we need to.
  static constexpr std::size_t kMaxBlocks = 5;

  void Commit() {
    assert(!committed_);

    if (reserved_bytes_ > storage_size_) {
      DeallocateStorage();
      storage_size_ = RoundUpToPowerOfTwo(reserved_bytes_);
      storage_ = aligned_alloc(kAlignment, storage_size_);
    }

    ReleaseBuildAssertion(!storage_size_ || storage_, "allocation failure");
    committed_ = true;
  }

  void Decommit() {
    assert(committed_);
    committed_ = false;
    generation_++;

    reserved_blocks_ = 0;
    reserved_bytes_ = 0;
  }

  // See generation_
  typedef std::size_t generation_t;

  // A handle on a reserved block. The user obtains
  // one by calling Reserve() and, after committing,
  // passes it to GetPointer().
  class Handle {
    std::uint8_t index_;
    generation_t generation_;
    TypeId type_;

    friend class Allocator;
  };

  // Reserves a block sized for n elements of type T, and
  // returns a handle to it. Must be called before committing.
  template <typename T>
  Handle Reserve(std::size_t n) {
    assert(!committed_ && "can't reserve blocks while committed");
    assert(reserved_blocks_ < kMaxBlocks &&
           "didn't expect to allocate this many blocks");
    const std::size_t bytes = RoundUp<kAlignment>(n * sizeof(T));
    const std::size_t offset = reserved_bytes_;
    const std::size_t index = reserved_blocks_;

    reserved_blocks_offsets_[index] = offset;
    Handle h;
    h.index_ = index;
    h.generation_ = generation_;
    h.type_ = GetTypeId<T>();

    reserved_blocks_++;
    reserved_bytes_ += bytes;

    return h;
  }

  // Returns the pointer to the allocated buffer for the given handle.
  // Must be called after committing.
  template <typename T>
  T* GetPointer(const Handle& h) const {
    assert(committed_ && "can't get block pointers unless committed");
    assert(h.index_ < reserved_blocks_ &&
           "bad handle, points to inexistant block");
    assert(h.generation_ == generation_ &&
           "handle from earlier generation, have decommitted since");
    assert(h.type_ == GetTypeId<T>() && "type mismatch");
    std::size_t offset = reserved_blocks_offsets_[h.index_];
    std::uintptr_t addr = reinterpret_cast<std::uintptr_t>(storage_) + offset;
    return reinterpret_cast<T*>(addr);
  }

 private:
  void DeallocateStorage() {
    assert(!committed_);
    aligned_free(storage_);
    storage_size_ = 0;
  }

  // Set to true by Commit() and to false by Decommit(). Initially false.
  bool committed_;

  // The actually allocated storage size and buffer pointer.
  std::size_t storage_size_;
  mutable void* storage_;

  // The number of blocks that have been reserved by Reserve().
  std::size_t reserved_blocks_;
  // The number of bytes that have been reserved by Reserve().
  std::size_t reserved_bytes_;
  // The offsets of reserved blocks into the storage buffer.
  std::size_t reserved_blocks_offsets_[kMaxBlocks];

  // The 'generation' is incremented on Decommit() and allows catching
  // bad GetPointer() calls still referring to a previous commit.
  generation_t generation_;
};

}  // namespace gemmlowp

#endif  // GEMMLOWP_INTERNAL_ALLOCATOR_H_
