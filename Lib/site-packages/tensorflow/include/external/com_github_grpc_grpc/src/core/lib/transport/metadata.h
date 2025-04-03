/*
 *
 * Copyright 2015 gRPC authors.
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

#ifndef GRPC_CORE_LIB_TRANSPORT_METADATA_H
#define GRPC_CORE_LIB_TRANSPORT_METADATA_H

#include <grpc/support/port_platform.h>

#include <grpc/impl/codegen/log.h>

#include <grpc/grpc.h>
#include <grpc/slice.h>

#include "src/core/lib/debug/trace.h"
#include "src/core/lib/gpr/useful.h"
#include "src/core/lib/gprpp/atomic.h"
#include "src/core/lib/gprpp/sync.h"
#include "src/core/lib/slice/slice_utils.h"

extern grpc_core::DebugOnlyTraceFlag grpc_trace_metadata;

/* This file provides a mechanism for tracking metadata through the grpc stack.
   It's not intended for consumption outside of the library.

   Metadata is tracked in the context of a sharded global grpc_mdctx. The
   context tracks unique strings (grpc_mdstr) and pairs of strings
   (grpc_mdelem). Any of these objects can be checked for equality by comparing
   their pointers. These objects are reference counted.

   grpc_mdelem can additionally store a (non-NULL) user data pointer. This
   pointer is intended to be used to cache semantic meaning of a metadata
   element. For example, an OAuth token may cache the credentials it represents
   and the time at which it expires in the mdelem user data.

   Combining this metadata cache and the hpack compression table allows us to
   simply lookup complete preparsed objects quickly, incurring a few atomic
   ops per metadata element on the fast path.

   grpc_mdelem instances MAY live longer than their refcount implies, and are
   garbage collected periodically, meaning cached data can easily outlive a
   single request.

   STATIC METADATA: in static_metadata.h we declare a set of static metadata.
   These mdelems and mdstrs are available via pre-declared code generated macros
   and are available to code anywhere between grpc_init() and grpc_shutdown().
   They are not refcounted, but can be passed to _ref and _unref functions
   declared here - in which case those functions are effectively no-ops. */

/* Forward declarations */
typedef struct grpc_mdelem grpc_mdelem;

/* if changing this, make identical changes in:
   - grpc_core::{InternedMetadata, AllocatedMetadata}
   - grpc_metadata in grpc_types.h */
typedef struct grpc_mdelem_data {
  const grpc_slice key;
  const grpc_slice value;
  /* there is a private part to this in metadata.c */
} grpc_mdelem_data;

/* GRPC_MDELEM_STORAGE_* enum values that can be treated as interned always have
   this bit set in their integer value */
#define GRPC_MDELEM_STORAGE_INTERNED_BIT 1

/* External and static storage metadata has no refcount to ref/unref. Allocated
 * and interned metadata do have a refcount. Metadata ref and unref methods use
 * a switch statement on this enum to determine which behaviour to execute.
 * Keeping the no-ref cases together and the ref-cases together leads to
 * slightly better code generation (9 inlined instructions rather than 10). */
typedef enum {
  /* memory pointed to by grpc_mdelem::payload is owned by an external system */
  GRPC_MDELEM_STORAGE_EXTERNAL = 0,
  /* memory is in the static metadata table */
  GRPC_MDELEM_STORAGE_STATIC = GRPC_MDELEM_STORAGE_INTERNED_BIT,
  /* memory pointed to by grpc_mdelem::payload is allocated by the metadata
     system */
  GRPC_MDELEM_STORAGE_ALLOCATED = 2,
  /* memory pointed to by grpc_mdelem::payload is interned by the metadata
     system */
  GRPC_MDELEM_STORAGE_INTERNED = 2 | GRPC_MDELEM_STORAGE_INTERNED_BIT,
} grpc_mdelem_data_storage;

struct grpc_mdelem {
  /* a grpc_mdelem_data* generally, with the two lower bits signalling memory
     ownership as per grpc_mdelem_data_storage */
  uintptr_t payload;
};

#define GRPC_MDELEM_DATA(md) ((grpc_mdelem_data*)((md).payload & ~(uintptr_t)3))
#define GRPC_MDELEM_STORAGE(md) \
  ((grpc_mdelem_data_storage)((md).payload & (uintptr_t)3))
#ifdef __cplusplus
#define GRPC_MAKE_MDELEM(data, storage) \
  (grpc_mdelem{((uintptr_t)(data)) | ((uintptr_t)storage)})
#else
#define GRPC_MAKE_MDELEM(data, storage) \
  ((grpc_mdelem){((uintptr_t)(data)) | ((uintptr_t)storage)})
#endif
#define GRPC_MDELEM_IS_INTERNED(md)          \
  ((grpc_mdelem_data_storage)((md).payload & \
                              (uintptr_t)GRPC_MDELEM_STORAGE_INTERNED_BIT))

/* Given arbitrary input slices, create a grpc_mdelem object. The caller refs
 * the input slices; we unref them. This method is always safe to call; however,
 * if we know data about the slices in question (e.g. if we knew our key was
 * static) we can call specializations that save on cycle count. */
grpc_mdelem grpc_mdelem_from_slices(const grpc_slice& key,
                                    const grpc_slice& value);

/* Like grpc_mdelem_from_slices, but we know that key is a static slice. This
   saves us a few branches and a no-op call to md_unref() for the key. */
grpc_mdelem grpc_mdelem_from_slices(const grpc_core::StaticMetadataSlice& key,
                                    const grpc_slice& value);

/* Like grpc_mdelem_from_slices, but key is static and val is static. */
grpc_mdelem grpc_mdelem_from_slices(
    const grpc_core::StaticMetadataSlice& key,
    const grpc_core::StaticMetadataSlice& value);

/* Like grpc_mdelem_from_slices, but key is static and val is interned. */
grpc_mdelem grpc_mdelem_from_slices(const grpc_core::StaticMetadataSlice& key,
                                    const grpc_core::ManagedMemorySlice& value);

/* Like grpc_mdelem_from_slices, but key and val are interned. */
grpc_mdelem grpc_mdelem_from_slices(const grpc_core::ManagedMemorySlice& key,
                                    const grpc_core::ManagedMemorySlice& value);

/* Cheaply convert a grpc_metadata to a grpc_mdelem; may use the grpc_metadata
   object as backing storage (so lifetimes should align) */
grpc_mdelem grpc_mdelem_from_grpc_metadata(grpc_metadata* metadata);

/* Does not unref the slices; if a new non-interned mdelem is needed, allocates
   one if compatible_external_backing_store is NULL, or uses
   compatible_external_backing_store if it is non-NULL (in which case it's the
   users responsibility to ensure that it outlives usage) */
grpc_mdelem grpc_mdelem_create(
    const grpc_slice& key, const grpc_slice& value,
    grpc_mdelem_data* compatible_external_backing_store);

/* Like grpc_mdelem_create, but we know that key is static. */
grpc_mdelem grpc_mdelem_create(
    const grpc_core::StaticMetadataSlice& key, const grpc_slice& value,
    grpc_mdelem_data* compatible_external_backing_store);

#define GRPC_MDKEY(md) (GRPC_MDELEM_DATA(md)->key)
#define GRPC_MDVALUE(md) (GRPC_MDELEM_DATA(md)->value)

bool grpc_mdelem_eq(grpc_mdelem a, grpc_mdelem b);
/* Often we compare metadata where we know a-priori that the second parameter is
 * static, and that the keys match. This most commonly happens when processing
 * metadata batch callouts in initial/trailing filters. In this case, fastpath
 * grpc_mdelem_eq and remove unnecessary checks. */
inline bool grpc_mdelem_static_value_eq(grpc_mdelem a, grpc_mdelem b_static) {
  if (a.payload == b_static.payload) return true;
  return grpc_slice_eq_static_interned(GRPC_MDVALUE(a), GRPC_MDVALUE(b_static));
}
#define GRPC_MDISNULL(md) (GRPC_MDELEM_DATA(md) == NULL)

inline bool grpc_mdelem_both_interned_eq(grpc_mdelem a_interned,
                                         grpc_mdelem b_interned) {
  GPR_DEBUG_ASSERT(GRPC_MDELEM_IS_INTERNED(a_interned) ||
                   GRPC_MDISNULL(a_interned));
  GPR_DEBUG_ASSERT(GRPC_MDELEM_IS_INTERNED(b_interned) ||
                   GRPC_MDISNULL(b_interned));
  return a_interned.payload == b_interned.payload;
}

/* Mutator and accessor for grpc_mdelem user data. The destructor function
   is used as a type tag and is checked during user_data fetch. */
void* grpc_mdelem_get_user_data(grpc_mdelem md, void (*if_destroy_func)(void*));
void* grpc_mdelem_set_user_data(grpc_mdelem md, void (*destroy_func)(void*),
                                void* data);

// Defined in metadata.cc.
struct mdtab_shard;

#ifndef NDEBUG
void grpc_mdelem_trace_ref(void* md, const grpc_slice& key,
                           const grpc_slice& value, intptr_t refcnt,
                           const char* file, int line);
void grpc_mdelem_trace_unref(void* md, const grpc_slice& key,
                             const grpc_slice& value, intptr_t refcnt,
                             const char* file, int line);
#endif
namespace grpc_core {

typedef void (*destroy_user_data_func)(void* data);

struct UserData {
  Mutex mu_user_data;
  grpc_core::Atomic<destroy_user_data_func> destroy_user_data;
  grpc_core::Atomic<void*> data;
};

class StaticMetadata {
 public:
  StaticMetadata(const grpc_slice& key, const grpc_slice& value, uintptr_t idx)
      : kv_({key, value}), hash_(0), static_idx_(idx) {}

  const grpc_mdelem_data& data() const { return kv_; }

  void HashInit();
  uint32_t hash() { return hash_; }
  uintptr_t StaticIndex() { return static_idx_; }

 private:
  grpc_mdelem_data kv_;

  /* private only data */
  uint32_t hash_;
  uintptr_t static_idx_;
};

class RefcountedMdBase {
 public:
  RefcountedMdBase(const grpc_slice& key, const grpc_slice& value)
      : key_(key), value_(value), refcnt_(1) {}
  RefcountedMdBase(const grpc_slice& key, const grpc_slice& value,
                   uint32_t hash)
      : key_(key), value_(value), refcnt_(1), hash_(hash) {}

  const grpc_slice& key() const { return key_; }
  const grpc_slice& value() const { return value_; }
  uint32_t hash() { return hash_; }

#ifndef NDEBUG
  void Ref(const char* file, int line) {
    grpc_mdelem_trace_ref(this, key_, value_, RefValue(), file, line);
    const intptr_t prior = refcnt_.FetchAdd(1, MemoryOrder::RELAXED);
    GPR_ASSERT(prior > 0);
  }
  bool Unref(const char* file, int line) {
    grpc_mdelem_trace_unref(this, key_, value_, RefValue(), file, line);
    return Unref();
  }
#endif
  void Ref() {
    /* we can assume the ref count is >= 1 as the application is calling
       this function - meaning that no adjustment to mdtab_free is necessary,
       simplifying the logic here to be just an atomic increment */
    refcnt_.FetchAdd(1, MemoryOrder::RELAXED);
  }
  bool Unref() {
    const intptr_t prior = refcnt_.FetchSub(1, MemoryOrder::ACQ_REL);
    GPR_DEBUG_ASSERT(prior > 0);
    return prior == 1;
  }

 protected:
#ifndef NDEBUG
  void TraceAtStart(const char* tag);
#endif

  intptr_t RefValue() { return refcnt_.Load(MemoryOrder::RELAXED); }
  bool AllRefsDropped() { return refcnt_.Load(MemoryOrder::ACQUIRE) == 0; }
  bool FirstRef() { return refcnt_.FetchAdd(1, MemoryOrder::RELAXED) == 0; }

 private:
  /* must be byte compatible with grpc_mdelem_data */
  grpc_slice key_;
  grpc_slice value_;
  grpc_core::Atomic<intptr_t> refcnt_;
  uint32_t hash_ = 0;
};

class InternedMetadata : public RefcountedMdBase {
 public:
  // TODO(arjunroy): Change to use strongly typed slices instead.
  struct NoRefKey {};
  struct BucketLink {
    explicit BucketLink(InternedMetadata* md) : next(md) {}

    InternedMetadata* next = nullptr;
  };
  InternedMetadata(const grpc_slice& key, const grpc_slice& value,
                   uint32_t hash, InternedMetadata* next);
  InternedMetadata(const grpc_slice& key, const grpc_slice& value,
                   uint32_t hash, InternedMetadata* next, const NoRefKey*);

  ~InternedMetadata();
  void RefWithShardLocked(mdtab_shard* shard);
  UserData* user_data() { return &user_data_; }
  InternedMetadata* bucket_next() { return link_.next; }
  void set_bucket_next(InternedMetadata* md) { link_.next = md; }

  static size_t CleanupLinkedMetadata(BucketLink* head);

 private:
  UserData user_data_;
  BucketLink link_;
};

/* Shadow structure for grpc_mdelem_data for allocated elements */
class AllocatedMetadata : public RefcountedMdBase {
 public:
  // TODO(arjunroy): Change to use strongly typed slices instead.
  struct NoRefKey {};
  AllocatedMetadata(const grpc_slice& key, const grpc_slice& value);
  AllocatedMetadata(const grpc_core::ManagedMemorySlice& key,
                    const grpc_core::UnmanagedMemorySlice& value);
  AllocatedMetadata(const grpc_core::ExternallyManagedSlice& key,
                    const grpc_core::UnmanagedMemorySlice& value);
  AllocatedMetadata(const grpc_slice& key, const grpc_slice& value,
                    const NoRefKey*);
  ~AllocatedMetadata();

  UserData* user_data() { return &user_data_; }

 private:
  UserData user_data_;
};

}  // namespace grpc_core

#ifndef NDEBUG
#define GRPC_MDELEM_REF(s) grpc_mdelem_ref((s), __FILE__, __LINE__)
inline grpc_mdelem grpc_mdelem_ref(grpc_mdelem gmd, const char* file,
                                   int line) {
#else  // ifndef NDEBUG
#define GRPC_MDELEM_REF(s) grpc_mdelem_ref((s))
inline grpc_mdelem grpc_mdelem_ref(grpc_mdelem gmd) {
#endif  // ifndef NDEBUG
  switch (GRPC_MDELEM_STORAGE(gmd)) {
    case GRPC_MDELEM_STORAGE_EXTERNAL:
    case GRPC_MDELEM_STORAGE_STATIC:
      break;
    case GRPC_MDELEM_STORAGE_INTERNED: {
      auto* md =
          reinterpret_cast<grpc_core::InternedMetadata*> GRPC_MDELEM_DATA(gmd);
      /* use C assert to have this removed in opt builds */
#ifndef NDEBUG
      md->Ref(file, line);
#else
      md->Ref();
#endif
      break;
    }
    case GRPC_MDELEM_STORAGE_ALLOCATED: {
      auto* md =
          reinterpret_cast<grpc_core::AllocatedMetadata*> GRPC_MDELEM_DATA(gmd);
#ifndef NDEBUG
      md->Ref(file, line);
#else
      md->Ref();
#endif
      break;
    }
  }
  return gmd;
}

#ifndef NDEBUG
#define GRPC_MDELEM_UNREF(s) grpc_mdelem_unref((s), __FILE__, __LINE__)
void grpc_mdelem_on_final_unref(grpc_mdelem_data_storage storage, void* ptr,
                                uint32_t hash, const char* file, int line);
inline void grpc_mdelem_unref(grpc_mdelem gmd, const char* file, int line) {
#else
#define GRPC_MDELEM_UNREF(s) grpc_mdelem_unref((s))
void grpc_mdelem_on_final_unref(grpc_mdelem_data_storage storage, void* ptr,
                                uint32_t hash);
inline void grpc_mdelem_unref(grpc_mdelem gmd) {
#endif
  const grpc_mdelem_data_storage storage = GRPC_MDELEM_STORAGE(gmd);
  switch (storage) {
    case GRPC_MDELEM_STORAGE_EXTERNAL:
    case GRPC_MDELEM_STORAGE_STATIC:
      return;
    case GRPC_MDELEM_STORAGE_INTERNED:
    case GRPC_MDELEM_STORAGE_ALLOCATED:
      auto* md =
          reinterpret_cast<grpc_core::RefcountedMdBase*> GRPC_MDELEM_DATA(gmd);
      /* once the refcount hits zero, some other thread can come along and
         free an interned md at any time: it's unsafe from this point on to
         access it so we read the hash now. */
      uint32_t hash = md->hash();
#ifndef NDEBUG
      if (GPR_UNLIKELY(md->Unref(file, line))) {
        grpc_mdelem_on_final_unref(storage, md, hash, file, line);
#else
      if (GPR_UNLIKELY(md->Unref())) {
        grpc_mdelem_on_final_unref(storage, md, hash);
#endif
      }
      return;
  }
}

#define GRPC_MDNULL GRPC_MAKE_MDELEM(NULL, GRPC_MDELEM_STORAGE_EXTERNAL)

/* We add 32 bytes of padding as per RFC-7540 section 6.5.2. */
#define GRPC_MDELEM_LENGTH(e)                                                  \
  (GRPC_SLICE_LENGTH(GRPC_MDKEY((e))) + GRPC_SLICE_LENGTH(GRPC_MDVALUE((e))) + \
   32)

#define GRPC_MDSTR_KV_HASH(k_hash, v_hash) (GPR_ROTL((k_hash), 2) ^ (v_hash))

void grpc_mdctx_global_init(void);
void grpc_mdctx_global_shutdown();

/* Like grpc_mdelem_from_slices, but we know that key is a static or interned
   slice and value is not static or interned. This gives us an inlinable
   fastpath - we know we must allocate metadata now, and that we do not need to
   unref the value (rather, we just transfer the ref). We can avoid a ref since:
   1) the key slice is passed in already ref'd
   2) We're guaranteed to create a new Allocated slice, thus meaning the
      ref can be considered 'transferred'.*/
inline grpc_mdelem grpc_mdelem_from_slices(
    const grpc_core::ManagedMemorySlice& key,
    const grpc_core::UnmanagedMemorySlice& value) {
  using grpc_core::AllocatedMetadata;
  return GRPC_MAKE_MDELEM(new AllocatedMetadata(key, value),
                          GRPC_MDELEM_STORAGE_ALLOCATED);
}

inline grpc_mdelem grpc_mdelem_from_slices(
    const grpc_core::ExternallyManagedSlice& key,
    const grpc_core::UnmanagedMemorySlice& value) {
  using grpc_core::AllocatedMetadata;
  return GRPC_MAKE_MDELEM(new AllocatedMetadata(key, value),
                          GRPC_MDELEM_STORAGE_ALLOCATED);
}

inline grpc_mdelem grpc_mdelem_from_slices(
    const grpc_core::StaticMetadataSlice& key,
    const grpc_core::UnmanagedMemorySlice& value) {
  using grpc_core::AllocatedMetadata;
  return GRPC_MAKE_MDELEM(new AllocatedMetadata(key, value),
                          GRPC_MDELEM_STORAGE_ALLOCATED);
}

#endif /* GRPC_CORE_LIB_TRANSPORT_METADATA_H */
