/*
 *
 * Copyright 2019 gRPC authors.
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
 *
 */

#ifndef GRPC_CORE_LIB_SLICE_SLICE_UTILS_H
#define GRPC_CORE_LIB_SLICE_SLICE_UTILS_H

#include <grpc/support/port_platform.h>

#include <cstring>

#include <grpc/slice.h>

#include "src/core/lib/gpr/murmur_hash.h"

namespace grpc_core {
extern uint32_t g_hash_seed;
}  // namespace grpc_core

// When we compare two slices, and we know the latter is not inlined, we can
// short circuit our comparison operator. We specifically use differs()
// semantics instead of equals() semantics due to more favourable code
// generation when using differs(). Specifically, we may use the output of
// grpc_slice_differs_refcounted for control flow. If we use differs()
// semantics, we end with a tailcall to memcmp(). If we use equals() semantics,
// we need to invert the result that memcmp provides us, which costs several
// instructions to do so. If we're using the result for control flow (i.e.
// branching based on the output) then we're just performing the extra
// operations to invert the result pointlessly. Concretely, we save 6 ops on
// x86-64/clang with differs().
int grpc_slice_differs_refcounted(const grpc_slice& a,
                                  const grpc_slice& b_not_inline);

// When we compare two slices, and we *know* that one of them is static or
// interned, we can short circuit our slice equality function. The second slice
// here must be static or interned; slice a can be any slice, inlined or not.
inline bool grpc_slice_eq_static_interned(const grpc_slice& a,
                                          const grpc_slice& b_static_interned) {
  if (a.refcount == b_static_interned.refcount) {
    return true;
  }
  return !grpc_slice_differs_refcounted(a, b_static_interned);
}

// TODO(arjunroy): These type declarations ought to be in
// src/core/lib/slice/slice_internal.h instead; they are here due to a circular
// header depedency between slice_internal.h and
// src/core/lib/transport/metadata.h. We need to fix this circular reference and
// when we do, move these type declarations.
//
// Internal slice type declarations.
// Externally, a grpc_slice is a grpc_slice is a grpc_slice.
// Internally, we may have heap allocated slices, static slices, interned
// slices, and inlined slices. If we know the specific type of slice
// we're dealing with, we can save cycles (e.g. fast-paths when we know we don't
// need to take a reference on a slice). Rather than introducing new methods
// ad-hoc in these cases, we rely on type-system backed overloads to keep
// internal APIs clean.
//
// For each overload, the definition and layout of the underlying slice does not
// change; this is purely type-system information.
namespace grpc_core {

// There are two main types of slices: those that have their memory
// managed by the slice library and those that do not.
//
// The following types of slices are not managed:
// - inlined slices (i.e., refcount is null)
// - slices that have a custom refcount type (i.e., not STATIC or INTERNED)
// - slices where the memory is managed by some external agent. The slice is not
//   ref-counted by grpc, and the programmer is responsible for ensuring the
//   data is valid for the duration of the period that grpc may access it.
//
// The following types of slices are managed:
// - static metadata slices (i.e., refcount type is STATIC)
// - interned slices (i.e., refcount type is INTERNED)
//
// This categorization is reflected in the following hierarchy:
//
// - grpc_slice
// > - UnmanagedMemorySlice
//   > - ExternallyManagedSlice
//   - ManagedMemorySlice
//   > - InternedSlice
//     - StaticMetadataSlice
//
struct ManagedMemorySlice : public grpc_slice {
  ManagedMemorySlice() {
    refcount = nullptr;
    data.refcounted.bytes = nullptr;
    data.refcounted.length = 0;
  }
  explicit ManagedMemorySlice(const char* string);
  ManagedMemorySlice(const char* buf, size_t len);
  explicit ManagedMemorySlice(const grpc_slice* slice);
  bool operator==(const grpc_slice& other) const {
    if (refcount == other.refcount) {
      return true;
    }
    return !grpc_slice_differs_refcounted(other, *this);
  }
  bool operator!=(const grpc_slice& other) const { return !(*this == other); }
  bool operator==(std::pair<const char*, size_t> buflen) const {
    return data.refcounted.length == buflen.second && buflen.first != nullptr &&
           memcmp(buflen.first, data.refcounted.bytes, buflen.second) == 0;
  }
};
struct UnmanagedMemorySlice : public grpc_slice {
  // TODO(arjunroy): Can we use a default=false param instead of this enum?
  enum class ForceHeapAllocation {};
  UnmanagedMemorySlice() {
    refcount = nullptr;
    data.inlined.length = 0;
  }
  explicit UnmanagedMemorySlice(const char* source);
  UnmanagedMemorySlice(const char* source, size_t length);
  // The first constructor creates a slice that may be heap allocated, or
  // inlined in the slice structure if length is small enough
  // (< GRPC_SLICE_INLINED_SIZE). The second constructor forces heap alloc.
  explicit UnmanagedMemorySlice(size_t length);
  explicit UnmanagedMemorySlice(size_t length, const ForceHeapAllocation&) {
    HeapInit(length);
  }

 private:
  void HeapInit(size_t length);
};

extern grpc_slice_refcount kNoopRefcount;

struct ExternallyManagedSlice : public UnmanagedMemorySlice {
  ExternallyManagedSlice()
      : ExternallyManagedSlice(&kNoopRefcount, 0, nullptr) {}
  explicit ExternallyManagedSlice(const char* s)
      : ExternallyManagedSlice(s, strlen(s)) {}
  ExternallyManagedSlice(const void* s, size_t len)
      : ExternallyManagedSlice(
            &kNoopRefcount, len,
            reinterpret_cast<uint8_t*>(const_cast<void*>(s))) {}
  ExternallyManagedSlice(grpc_slice_refcount* ref, size_t length,
                         uint8_t* bytes) {
    refcount = ref;
    data.refcounted.length = length;
    data.refcounted.bytes = bytes;
  }
  bool operator==(const grpc_slice& other) const {
    return data.refcounted.length == GRPC_SLICE_LENGTH(other) &&
           memcmp(data.refcounted.bytes, GRPC_SLICE_START_PTR(other),
                  data.refcounted.length) == 0;
  }
  bool operator!=(const grpc_slice& other) const { return !(*this == other); }
  uint32_t Hash() {
    return gpr_murmur_hash3(data.refcounted.bytes, data.refcounted.length,
                            g_hash_seed);
  }
};

struct StaticMetadataSlice : public ManagedMemorySlice {
  StaticMetadataSlice(grpc_slice_refcount* ref, size_t length,
                      const uint8_t* bytes) {
    refcount = ref;
    data.refcounted.length = length;
    // NB: grpc_slice may or may not point to a static slice, but we are
    // definitely pointing to static data here. Since we are not changing
    // the underlying C-type, we need a const_cast here.
    data.refcounted.bytes = const_cast<uint8_t*>(bytes);
  }
};

struct InternedSliceRefcount;
struct InternedSlice : public ManagedMemorySlice {
  explicit InternedSlice(InternedSliceRefcount* s);
};

}  // namespace grpc_core

#endif /* GRPC_CORE_LIB_SLICE_SLICE_UTILS_H */
