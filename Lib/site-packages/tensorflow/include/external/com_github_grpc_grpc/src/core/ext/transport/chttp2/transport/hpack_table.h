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

#ifndef GRPC_CORE_EXT_TRANSPORT_CHTTP2_TRANSPORT_HPACK_TABLE_H
#define GRPC_CORE_EXT_TRANSPORT_CHTTP2_TRANSPORT_HPACK_TABLE_H

#include <grpc/support/port_platform.h>

#include <grpc/slice.h>
#include "src/core/lib/gprpp/memory.h"
#include "src/core/lib/iomgr/error.h"
#include "src/core/lib/transport/metadata.h"
#include "src/core/lib/transport/static_metadata.h"

/* HPACK header table */

/* last index in the static table */
#define GRPC_CHTTP2_LAST_STATIC_ENTRY 61

/* Initial table size as per the spec */
#define GRPC_CHTTP2_INITIAL_HPACK_TABLE_SIZE 4096
/* Maximum table size that we'll use */
#define GRPC_CHTTP2_MAX_HPACK_TABLE_SIZE GRPC_CHTTP2_INITIAL_HPACK_TABLE_SIZE
/* Per entry overhead bytes as per the spec */
#define GRPC_CHTTP2_HPACK_ENTRY_OVERHEAD 32
#if 0
/* Maximum number of entries we could possibly fit in the table, given defined
   overheads */
#define GRPC_CHTTP2_MAX_TABLE_COUNT                                            \
  ((GRPC_CHTTP2_MAX_HPACK_TABLE_SIZE + GRPC_CHTTP2_HPACK_ENTRY_OVERHEAD - 1) / \
   GRPC_CHTTP2_HPACK_ENTRY_OVERHEAD)
#endif

/* hpack decoder table */
struct grpc_chttp2_hptbl {
  static uint32_t entries_for_bytes(uint32_t bytes) {
    return (bytes + GRPC_CHTTP2_HPACK_ENTRY_OVERHEAD - 1) /
           GRPC_CHTTP2_HPACK_ENTRY_OVERHEAD;
  }
  static constexpr uint32_t kInitialCapacity =
      (GRPC_CHTTP2_INITIAL_HPACK_TABLE_SIZE + GRPC_CHTTP2_HPACK_ENTRY_OVERHEAD -
       1) /
      GRPC_CHTTP2_HPACK_ENTRY_OVERHEAD;

  grpc_chttp2_hptbl() {
    GPR_DEBUG_ASSERT(!ents);
    constexpr uint32_t AllocSize = sizeof(*ents) * kInitialCapacity;
    ents = static_cast<grpc_mdelem*>(gpr_malloc(AllocSize));
    memset(ents, 0, AllocSize);
  }

  /* the first used entry in ents */
  uint32_t first_ent = 0;
  /* how many entries are in the table */
  uint32_t num_ents = 0;
  /* the amount of memory used by the table, according to the hpack algorithm */
  uint32_t mem_used = 0;
  /* the max memory allowed to be used by the table, according to the hpack
     algorithm */
  uint32_t max_bytes = GRPC_CHTTP2_INITIAL_HPACK_TABLE_SIZE;
  /* the currently agreed size of the table, according to the hpack algorithm */
  uint32_t current_table_bytes = GRPC_CHTTP2_INITIAL_HPACK_TABLE_SIZE;
  /* Maximum number of entries we could possibly fit in the table, given defined
     overheads */
  uint32_t max_entries = kInitialCapacity;
  /* Number of entries allocated in ents */
  uint32_t cap_entries = kInitialCapacity;
  /* a circular buffer of headers - this is stored in the opposite order to
     what hpack specifies, in order to simplify table management a little...
     meaning lookups need to SUBTRACT from the end position */
  grpc_mdelem* ents = nullptr;
};

void grpc_chttp2_hptbl_destroy(grpc_chttp2_hptbl* tbl);
void grpc_chttp2_hptbl_set_max_bytes(grpc_chttp2_hptbl* tbl,
                                     uint32_t max_bytes);
grpc_error* grpc_chttp2_hptbl_set_current_table_size(grpc_chttp2_hptbl* tbl,
                                                     uint32_t bytes);

/* lookup a table entry based on its hpack index */
grpc_mdelem grpc_chttp2_hptbl_lookup_dynamic_index(const grpc_chttp2_hptbl* tbl,
                                                   uint32_t tbl_index);
grpc_mdelem grpc_chttp2_hptbl_lookup_ref_dynamic_index(
    const grpc_chttp2_hptbl* tbl, uint32_t tbl_index);
template <bool take_ref = false>
inline grpc_mdelem grpc_chttp2_hptbl_lookup(const grpc_chttp2_hptbl* tbl,
                                            uint32_t index) {
  /* Static table comes first, just return an entry from it.
     NB: This imposes the constraint that the first
     GRPC_CHTTP2_LAST_STATIC_ENTRY entries in the core static metadata table
     must follow the hpack standard. If that changes, we *must* not rely on
     reading the core static metadata table here; at that point we'd need our
     own singleton static metadata in the correct order. */
  if (index <= GRPC_CHTTP2_LAST_STATIC_ENTRY) {
    return grpc_static_mdelem_manifested()[index - 1];
  } else {
    if (take_ref) {
      return grpc_chttp2_hptbl_lookup_ref_dynamic_index(tbl, index);
    } else {
      return grpc_chttp2_hptbl_lookup_dynamic_index(tbl, index);
    }
  }
}
/* add a table entry to the index */
grpc_error* grpc_chttp2_hptbl_add(grpc_chttp2_hptbl* tbl,
                                  grpc_mdelem md) GRPC_MUST_USE_RESULT;

size_t grpc_chttp2_get_size_in_hpack_table(grpc_mdelem elem,
                                           bool use_true_binary_metadata);

/* Returns the static hpack table index that corresponds to /a elem. Returns 0
  if /a elem is not statically stored or if it is not in the static hpack
  table */
inline uintptr_t grpc_chttp2_get_static_hpack_table_index(grpc_mdelem md) {
  uintptr_t index =
      reinterpret_cast<grpc_core::StaticMetadata*>(GRPC_MDELEM_DATA(md)) -
      grpc_static_mdelem_table();
  if (index < GRPC_CHTTP2_LAST_STATIC_ENTRY) {
    return index + 1;  // Hpack static metadata element indices start at 1
  }
  return 0;
}

/* Find a key/value pair in the table... returns the index in the table of the
   most similar entry, or 0 if the value was not found */
typedef struct {
  uint32_t index;
  int has_value;
} grpc_chttp2_hptbl_find_result;
grpc_chttp2_hptbl_find_result grpc_chttp2_hptbl_find(
    const grpc_chttp2_hptbl* tbl, grpc_mdelem md);

#endif /* GRPC_CORE_EXT_TRANSPORT_CHTTP2_TRANSPORT_HPACK_TABLE_H */
