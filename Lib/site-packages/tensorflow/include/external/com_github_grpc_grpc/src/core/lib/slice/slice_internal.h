/*
 *
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
 *
 */

#ifndef GRPC_CORE_LIB_SLICE_SLICE_INTERNAL_H
#define GRPC_CORE_LIB_SLICE_SLICE_INTERNAL_H

#include <grpc/support/port_platform.h>

#include <grpc/support/log.h>

#include <grpc/slice.h>
#include <grpc/slice_buffer.h>
#include <string.h>

#include "src/core/lib/gpr/murmur_hash.h"
#include "src/core/lib/gprpp/memory.h"
#include "src/core/lib/gprpp/ref_counted.h"
#include "src/core/lib/slice/slice_utils.h"
#include "src/core/lib/transport/static_metadata.h"

// Interned slices have specific fast-path operations for hashing. To inline
// these operations, we need to forward declare them here.
extern uint32_t grpc_static_metadata_hash_values[GRPC_STATIC_MDSTR_COUNT];

// grpc_slice_refcount : A reference count for grpc_slice.
//
// Non-inlined grpc_slice objects are refcounted. Historically this was
// implemented via grpc_slice_refcount, a C-style polymorphic class using a
// manually managed vtable of operations. Subclasses would define their own
// vtable; the 'virtual' methods (ref, unref, equals and hash) would simply call
// the function pointers in the vtable as necessary.
//
// Unfortunately, this leads to some inefficiencies in the generated code that
// can be improved upon. For example, equality checking for interned slices is a
// simple equality check on the refcount pointer. With the vtable approach, this
// would translate to roughly the following (high-level) instructions:
//
// grpc_slice_equals(slice1, slice2):
//   load vtable->eq -> eq_func
//   call eq_func(slice1, slice2)
//
// interned_slice_equals(slice1, slice2)
//   load slice1.ref -> r1
//   load slice2.ref -> r2
//   cmp r1, r2 -> retval
//   ret retval
//
// This leads to a function call for a function defined in another translation
// unit, which imposes memory barriers, which reduces the compiler's ability to
// optimize (in addition to the added overhead of call/ret). Additionally, it
// may be harder to reason about branch prediction when we're jumping to
// essentially arbitrarily provided function pointers.
//
// In addition, it is arguable that while virtualization was helpful for
// Equals()/Hash() methods, that it was fundamentally unnecessary for
// Ref()/Unref().
//
// Instead, grpc_slice_refcount provides the same functionality as the C-style
// virtual class, but in a de-virtualized manner - Eq(), Hash(), Ref() and
// Unref() are provided within this header file. Fastpaths for Eq()/Hash()
// (interned and static metadata slices), as well as the Ref() operation, can
// all be inlined without any memory barriers.
//
// It does this by:
// 1. Using grpc_core::RefCount<> (header-only) for Ref/Unref. Two special cases
//    need support: No-op ref/unref (eg. static metadata slices) and stream
//    slice references (where all the slices share the streamref). This is in
//    addition to the normal case of '1 slice, 1 ref'.
//    To support these cases, we explicitly track a nullable pointer to the
//    underlying RefCount<>. No-op ref/unref is used by checking the pointer for
//    null, and doing nothing if it is. Both stream slice refs and 'normal'
//    slices use the same path for Ref/Unref (by targeting the non-null
//    pointer).
//
// 2. introducing the notion of grpc_slice_refcount::Type. This describes if a
//    slice ref is used by a static metadata slice, an interned slice, or other
//    slices. We switch on the slice ref type in order to provide fastpaths for
//    Equals() and Hash().
//
// In total, this saves us roughly 1-2% latency for unary calls, with smaller
// calls benefitting. The effect is present, but not as useful, for larger calls
// where the cost of sending the data dominates.
// TODO(arjunroy): Investigate if this can be removed with strongly typed
// grpc_slices.
struct grpc_slice_refcount {
 public:
  enum class Type {
    STATIC,    // Refcount for a static metadata slice.
    INTERNED,  // Refcount for an interned slice.
    NOP,       // No-Op
    REGULAR    // Refcount for non-static-metadata, non-interned slices.
  };
  typedef void (*DestroyerFn)(void*);

  grpc_slice_refcount() = default;

  explicit grpc_slice_refcount(Type t) : ref_type_(t) {}

  explicit grpc_slice_refcount(grpc_slice_refcount* sub) : sub_refcount_(sub) {}
  // Regular constructor for grpc_slice_refcount.
  //
  // Parameters:
  //  1. grpc_slice_refcount::Type type
  //  Whether we are the refcount for a static
  //  metadata slice, an interned slice, or any other kind of slice.
  //
  //  2. RefCount* ref
  //  The pointer to the actual underlying grpc_core::RefCount. Rather than
  //  performing struct offset computations as in the original implementation to
  //  get to the refcount, which requires a virtual method, we devirtualize by
  //  using a nullable pointer to allow a single pair of Ref/Unref methods.
  //
  //  3. DestroyerFn destroyer_fn
  //  Called when the refcount goes to 0, with destroyer_arg as parameter.
  //
  //  4. void* destroyer_arg
  //  Argument for the virtualized destructor.
  //
  //  5. grpc_slice_refcount* sub
  //  Argument used for interned slices.
  grpc_slice_refcount(grpc_slice_refcount::Type type, grpc_core::RefCount* ref,
                      DestroyerFn destroyer_fn, void* destroyer_arg,
                      grpc_slice_refcount* sub)
      : ref_(ref),
        ref_type_(type),
        sub_refcount_(sub),
        dest_fn_(destroyer_fn),
        destroy_fn_arg_(destroyer_arg) {}
  // Initializer for static refcounts.
  grpc_slice_refcount(grpc_slice_refcount* sub, Type type)
      : ref_type_(type), sub_refcount_(sub) {}

  Type GetType() const { return ref_type_; }

  int Eq(const grpc_slice& a, const grpc_slice& b);

  uint32_t Hash(const grpc_slice& slice);
  void Ref() {
    if (ref_ == nullptr) return;
    ref_->RefNonZero();
  }
  void Unref() {
    if (ref_ == nullptr) return;
    if (ref_->Unref()) {
      dest_fn_(destroy_fn_arg_);
    }
  }

  grpc_slice_refcount* sub_refcount() const { return sub_refcount_; }

 private:
  grpc_core::RefCount* ref_ = nullptr;
  const Type ref_type_ = Type::REGULAR;
  grpc_slice_refcount* sub_refcount_ = this;
  DestroyerFn dest_fn_ = nullptr;
  void* destroy_fn_arg_ = nullptr;
};

namespace grpc_core {

struct StaticSliceRefcount {
  static grpc_slice_refcount kStaticSubRefcount;

  StaticSliceRefcount(uint32_t index)
      : base(&kStaticSubRefcount, grpc_slice_refcount::Type::STATIC),
        index(index) {}

  grpc_slice_refcount base;
  const uint32_t index;
};

extern grpc_slice_refcount kNoopRefcount;

struct InternedSliceRefcount {
  static void Destroy(void* arg) {
    auto* rc = static_cast<InternedSliceRefcount*>(arg);
    rc->~InternedSliceRefcount();
    gpr_free(rc);
  }

  InternedSliceRefcount(size_t length, uint32_t hash,
                        InternedSliceRefcount* bucket_next)
      : base(grpc_slice_refcount::Type::INTERNED, &refcnt, Destroy, this, &sub),
        sub(grpc_slice_refcount::Type::REGULAR, &refcnt, Destroy, this, &sub),
        length(length),
        hash(hash),
        bucket_next(bucket_next) {}

  ~InternedSliceRefcount();

  grpc_slice_refcount base;
  grpc_slice_refcount sub;
  const size_t length;
  RefCount refcnt;
  const uint32_t hash;
  InternedSliceRefcount* bucket_next;
};

}  // namespace grpc_core

inline size_t grpc_refcounted_slice_length(const grpc_slice& slice) {
  GPR_DEBUG_ASSERT(slice.refcount != nullptr);
  return slice.data.refcounted.length;
}

inline const uint8_t* grpc_refcounted_slice_data(const grpc_slice& slice) {
  GPR_DEBUG_ASSERT(slice.refcount != nullptr);
  return slice.data.refcounted.bytes;
}

inline int grpc_slice_refcount::Eq(const grpc_slice& a, const grpc_slice& b) {
  GPR_DEBUG_ASSERT(a.refcount != nullptr);
  GPR_DEBUG_ASSERT(a.refcount == this);
  switch (ref_type_) {
    case Type::STATIC:
      GPR_DEBUG_ASSERT(
          (GRPC_STATIC_METADATA_INDEX(a) == GRPC_STATIC_METADATA_INDEX(b)) ==
          (a.refcount == b.refcount));
    case Type::INTERNED:
      return a.refcount == b.refcount;
    case Type::NOP:
    case Type::REGULAR:
      break;
  }
  if (grpc_refcounted_slice_length(a) != GRPC_SLICE_LENGTH(b)) return false;
  if (grpc_refcounted_slice_length(a) == 0) return true;
  return 0 == memcmp(grpc_refcounted_slice_data(a), GRPC_SLICE_START_PTR(b),
                     grpc_refcounted_slice_length(a));
}

inline uint32_t grpc_slice_refcount::Hash(const grpc_slice& slice) {
  GPR_DEBUG_ASSERT(slice.refcount != nullptr);
  GPR_DEBUG_ASSERT(slice.refcount == this);
  switch (ref_type_) {
    case Type::STATIC:
      return ::grpc_static_metadata_hash_values[GRPC_STATIC_METADATA_INDEX(
          slice)];
    case Type::INTERNED:
      return reinterpret_cast<grpc_core::InternedSliceRefcount*>(slice.refcount)
          ->hash;
    case Type::NOP:
    case Type::REGULAR:
      break;
  }
  return gpr_murmur_hash3(grpc_refcounted_slice_data(slice),
                          grpc_refcounted_slice_length(slice),
                          grpc_core::g_hash_seed);
}

inline const grpc_slice& grpc_slice_ref_internal(const grpc_slice& slice) {
  if (slice.refcount) {
    slice.refcount->Ref();
  }
  return slice;
}

inline void grpc_slice_unref_internal(const grpc_slice& slice) {
  if (slice.refcount) {
    slice.refcount->Unref();
  }
}

void grpc_slice_buffer_reset_and_unref_internal(grpc_slice_buffer* sb);
void grpc_slice_buffer_partial_unref_internal(grpc_slice_buffer* sb,
                                              size_t idx);
void grpc_slice_buffer_destroy_internal(grpc_slice_buffer* sb);

// Returns a pointer to the first slice in the slice buffer without giving
// ownership to or a reference count on that slice.
inline grpc_slice* grpc_slice_buffer_peek_first(grpc_slice_buffer* sb) {
  GPR_DEBUG_ASSERT(sb->count > 0);
  return &sb->slices[0];
}

// Removes the first slice from the slice buffer.
void grpc_slice_buffer_remove_first(grpc_slice_buffer* sb);

// Calls grpc_slice_sub with the given parameters on the first slice.
void grpc_slice_buffer_sub_first(grpc_slice_buffer* sb, size_t begin,
                                 size_t end);

/* Check if a slice is interned */
bool grpc_slice_is_interned(const grpc_slice& slice);
inline bool grpc_slice_is_interned(const grpc_slice& slice) {
  return (slice.refcount &&
          (slice.refcount->GetType() == grpc_slice_refcount::Type::INTERNED ||
           slice.refcount->GetType() == grpc_slice_refcount::Type::STATIC));
}

inline bool grpc_slice_static_interned_equal(const grpc_slice& a,
                                             const grpc_slice& b) {
  GPR_DEBUG_ASSERT(grpc_slice_is_interned(a) && grpc_slice_is_interned(b));
  return a.refcount == b.refcount;
}

void grpc_slice_intern_init(void);
void grpc_slice_intern_shutdown(void);
void grpc_test_only_set_slice_hash_seed(uint32_t key);
// if slice matches a static slice, returns the static slice
// otherwise returns the passed in slice (without reffing it)
// used for surface boundaries where we might receive an un-interned static
// string
grpc_slice grpc_slice_maybe_static_intern(grpc_slice slice,
                                          bool* returned_slice_is_different);
uint32_t grpc_static_slice_hash(grpc_slice s);
int grpc_static_slice_eq(grpc_slice a, grpc_slice b);

inline uint32_t grpc_slice_hash_refcounted(const grpc_slice& s) {
  GPR_DEBUG_ASSERT(s.refcount != nullptr);
  return s.refcount->Hash(s);
}

inline uint32_t grpc_slice_default_hash_internal(const grpc_slice& s) {
  return gpr_murmur_hash3(GRPC_SLICE_START_PTR(s), GRPC_SLICE_LENGTH(s),
                          grpc_core::g_hash_seed);
}

inline uint32_t grpc_slice_hash_internal(const grpc_slice& s) {
  return s.refcount == nullptr ? grpc_slice_default_hash_internal(s)
                               : grpc_slice_hash_refcounted(s);
}

grpc_slice grpc_slice_from_moved_buffer(grpc_core::UniquePtr<char> p,
                                        size_t len);
grpc_slice grpc_slice_from_moved_string(grpc_core::UniquePtr<char> p);

// Returns the memory used by this slice, not counting the slice structure
// itself. This means that inlined and slices from static strings will return
// 0. All other slices will return the size of the allocated chars.
size_t grpc_slice_memory_usage(grpc_slice s);

grpc_core::UnmanagedMemorySlice grpc_slice_sub_no_ref(
    const grpc_core::UnmanagedMemorySlice& source, size_t begin, size_t end);

#endif /* GRPC_CORE_LIB_SLICE_SLICE_INTERNAL_H */
