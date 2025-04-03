/*
 * Copyright 2016 gRPC authors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef GRPC_CORE_LIB_SLICE_SLICE_WEAK_HASH_TABLE_H
#define GRPC_CORE_LIB_SLICE_SLICE_WEAK_HASH_TABLE_H

#include <grpc/support/port_platform.h>

#include "src/core/lib/gprpp/memory.h"
#include "src/core/lib/gprpp/ref_counted.h"
#include "src/core/lib/gprpp/ref_counted_ptr.h"
#include "src/core/lib/slice/slice_internal.h"

/// Weak hash table implementation.
///
/// This entries in this table are weak: an entry may be removed at any time due
/// to a number of reasons: memory pressure, hash collisions, etc.
///
/// The keys are \a grpc_slice objects. The values are of arbitrary type.
///
/// This class is thread unsafe. It's the caller's responsibility to ensure
/// proper locking when accessing its methods.

namespace grpc_core {

template <typename T, size_t Size>
class SliceWeakHashTable : public RefCounted<SliceWeakHashTable<T, Size>> {
 public:
  /// Creates a new table of at most \a size entries.
  static RefCountedPtr<SliceWeakHashTable> Create() {
    return MakeRefCounted<SliceWeakHashTable<T, Size>>();
  }

  /// Use Create function instead of using this directly.
  SliceWeakHashTable() = default;
  ~SliceWeakHashTable() = default;

  /// Add a mapping from \a key to \a value, taking ownership of \a key. This
  /// operation will always succeed. It may discard older entries.
  void Add(const grpc_slice& key, T value) {
    const size_t idx = grpc_slice_hash_internal(key) % Size;
    entries_[idx].Set(key, std::move(value));
    return;
  }

  /// Returns the value from the table associated with / \a key or null if not
  /// found.
  const T* Get(const grpc_slice& key) const {
    const size_t idx = grpc_slice_hash_internal(key) % Size;
    const auto& entry = entries_[idx];
    return grpc_slice_eq(entry.key(), key) ? entry.value() : nullptr;
  }

 private:
  /// The type of the table "rows".
  class Entry {
   public:
    Entry() = default;
    ~Entry() {
      if (is_set_) grpc_slice_unref_internal(key_);
    }
    const grpc_slice& key() const { return key_; }

    /// Return the entry's value, or null if unset.
    const T* value() const {
      if (!is_set_) return nullptr;
      return &value_;
    }

    /// Set the \a key and \a value (which is moved) for the entry.
    void Set(const grpc_slice& key, T&& value) {
      if (is_set_) grpc_slice_unref_internal(key_);
      key_ = key;
      value_ = std::move(value);
      is_set_ = true;
    }

   private:
    grpc_slice key_;
    T value_;
    bool is_set_ = false;
  };

  Entry entries_[Size];
};

}  // namespace grpc_core

#endif /* GRPC_CORE_LIB_SLICE_SLICE_WEAK_HASH_TABLE_H */
