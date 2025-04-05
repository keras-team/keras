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

#ifndef GRPC_CORE_LIB_SLICE_SLICE_HASH_TABLE_H
#define GRPC_CORE_LIB_SLICE_SLICE_HASH_TABLE_H

#include <grpc/support/port_platform.h>

#include <string.h>

#include <grpc/support/alloc.h>
#include <grpc/support/log.h>

#include "src/core/lib/gpr/useful.h"
#include "src/core/lib/gprpp/memory.h"
#include "src/core/lib/gprpp/ref_counted.h"
#include "src/core/lib/gprpp/ref_counted_ptr.h"
#include "src/core/lib/slice/slice_internal.h"

/// Hash table implementation.
///
/// This implementation uses open addressing
/// (https://en.wikipedia.org/wiki/Open_addressing) with linear
/// probing (https://en.wikipedia.org/wiki/Linear_probing).
///
/// The keys are \a grpc_slice objects.  The values can be any type.
///
/// Hash tables are intentionally immutable, to avoid the need for locking.

namespace grpc_core {

template <typename T>
class SliceHashTable : public RefCounted<SliceHashTable<T>> {
 public:
  struct Entry {
    grpc_slice key;
    T value;
    bool is_set;
  };

  // Function for comparing values.
  // TODO(roth): Eliminate this and the Cmp() method from this API once
  // grpc_channel_args is redesigned to require that keys are unique.
  typedef int (*ValueCmp)(const T&, const T&);

  /// Creates a new hash table containing \a entries, which is an array
  /// of length \a num_entries.  Takes ownership of all keys and values in \a
  /// entries.  If not null, \a value_cmp will be used to compare values in
  /// the context of \a Cmp(). If null, raw pointer (\a GPR_ICMP) comparison
  /// will be used.
  static RefCountedPtr<SliceHashTable> Create(size_t num_entries,
                                              Entry* entries,
                                              ValueCmp value_cmp);

  // Use Create function instead of using this directly.
  SliceHashTable(size_t num_entries, Entry* entries, ValueCmp value_cmp);
  virtual ~SliceHashTable();

  /// Returns the value from the table associated with \a key.
  /// Returns null if \a key is not found.
  const T* Get(const grpc_slice& key) const;

  /// Compares \a a vs. \a b.
  /// A table is considered "smaller" (resp. "greater") if:
  ///  - GPR_ICMP(a->value_cmp, b->value_cmp) < 1 (resp. > 1),
  ///  - else, it contains fewer (resp. more) entries,
  ///  - else, if strcmp(a_key, b_key) < 1 (resp. > 1),
  ///  - else, if value_cmp(a_value, b_value) < 1 (resp. > 1).
  static int Cmp(const SliceHashTable& a, const SliceHashTable& b);

 private:
  void Add(const grpc_slice& key, T& value);

  // Default value comparison function, if none specified by caller.
  static int DefaultValueCmp(const T& a, const T& b) { return GPR_ICMP(a, b); }

  const ValueCmp value_cmp_;
  const size_t size_;
  size_t max_num_probes_;
  Entry* entries_;
};

//
// implementation -- no user-serviceable parts below
//

template <typename T>
RefCountedPtr<SliceHashTable<T>> SliceHashTable<T>::Create(size_t num_entries,
                                                           Entry* entries,
                                                           ValueCmp value_cmp) {
  return MakeRefCounted<SliceHashTable<T>>(num_entries, entries, value_cmp);
}

template <typename T>
SliceHashTable<T>::SliceHashTable(size_t num_entries, Entry* entries,
                                  ValueCmp value_cmp)
    : value_cmp_(value_cmp),
      // Keep load factor low to improve performance of lookups.
      size_(num_entries * 2),
      max_num_probes_(0) {
  entries_ = static_cast<Entry*>(gpr_zalloc(sizeof(Entry) * size_));
  for (size_t i = 0; i < num_entries; ++i) {
    Entry* entry = &entries[i];
    Add(entry->key, entry->value);
  }
}

template <typename T>
SliceHashTable<T>::~SliceHashTable() {
  for (size_t i = 0; i < size_; ++i) {
    Entry& entry = entries_[i];
    if (entry.is_set) {
      grpc_slice_unref_internal(entry.key);
      entry.value.~T();
    }
  }
  gpr_free(entries_);
}

template <typename T>
void SliceHashTable<T>::Add(const grpc_slice& key, T& value) {
  const size_t hash = grpc_slice_hash_internal(key);
  for (size_t offset = 0; offset < size_; ++offset) {
    const size_t idx = (hash + offset) % size_;
    if (!entries_[idx].is_set) {
      entries_[idx].is_set = true;
      entries_[idx].key = key;
      entries_[idx].value = std::move(value);
      // Keep track of the maximum number of probes needed, since this
      // provides an upper bound for lookups.
      if (offset > max_num_probes_) max_num_probes_ = offset;
      return;
    }
  }
  GPR_ASSERT(false);  // Table should never be full.
}

template <typename T>
const T* SliceHashTable<T>::Get(const grpc_slice& key) const {
  const size_t hash = grpc_slice_hash_internal(key);
  // We cap the number of probes at the max number recorded when
  // populating the table.
  for (size_t offset = 0; offset <= max_num_probes_; ++offset) {
    const size_t idx = (hash + offset) % size_;
    if (!entries_[idx].is_set) break;
    if (grpc_slice_eq(entries_[idx].key, key)) {
      return &entries_[idx].value;
    }
  }
  return nullptr;  // Not found.
}

template <typename T>
int SliceHashTable<T>::Cmp(const SliceHashTable& a, const SliceHashTable& b) {
  ValueCmp value_cmp_a =
      a.value_cmp_ != nullptr ? a.value_cmp_ : DefaultValueCmp;
  ValueCmp value_cmp_b =
      b.value_cmp_ != nullptr ? b.value_cmp_ : DefaultValueCmp;
  // Compare value_fns
  const int value_fns_cmp = GPR_ICMP((void*)value_cmp_a, (void*)value_cmp_b);
  if (value_fns_cmp != 0) return value_fns_cmp;
  // Compare sizes
  if (a.size_ < b.size_) return -1;
  if (a.size_ > b.size_) return 1;
  // Compare rows.
  for (size_t i = 0; i < a.size_; ++i) {
    if (!a.entries_[i].is_set) {
      if (b.entries_[i].is_set) {
        return -1;  // a empty but b non-empty
      }
      continue;  // both empty, no need to check key or value
    } else if (!b.entries_[i].is_set) {
      return 1;  // a non-empty but b empty
    }
    // neither entry is empty
    const int key_cmp = grpc_slice_cmp(a.entries_[i].key, b.entries_[i].key);
    if (key_cmp != 0) return key_cmp;
    const int value_cmp = value_cmp_a(a.entries_[i].value, b.entries_[i].value);
    if (value_cmp != 0) return value_cmp;
  }
  return 0;
}

}  // namespace grpc_core

#endif /* GRPC_CORE_LIB_SLICE_SLICE_HASH_TABLE_H */
