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

#ifndef GRPC_CORE_EXT_TRANSPORT_CHTTP2_TRANSPORT_HPACK_PARSER_H
#define GRPC_CORE_EXT_TRANSPORT_CHTTP2_TRANSPORT_HPACK_PARSER_H

#include <grpc/support/port_platform.h>

#include <stddef.h>

#include "src/core/ext/transport/chttp2/transport/frame.h"
#include "src/core/ext/transport/chttp2/transport/hpack_table.h"
#include "src/core/lib/transport/metadata.h"

typedef struct grpc_chttp2_hpack_parser grpc_chttp2_hpack_parser;

typedef grpc_error* (*grpc_chttp2_hpack_parser_state)(
    grpc_chttp2_hpack_parser* p, const uint8_t* beg, const uint8_t* end);

typedef struct {
  bool copied;
  struct {
    grpc_slice referenced;
    struct {
      char* str;
      uint32_t length;
      uint32_t capacity;
    } copied;
  } data;
} grpc_chttp2_hpack_parser_string;

struct grpc_chttp2_hpack_parser {
  /* user specified callback for each header output */
  grpc_error* (*on_header)(void* user_data, grpc_mdelem md);
  void* on_header_user_data;

  grpc_error* last_error;

  /* current parse state - or a function that implements it */
  grpc_chttp2_hpack_parser_state state;
  /* future states dependent on the opening op code */
  const grpc_chttp2_hpack_parser_state* next_state;
  /* what to do after skipping prioritization data */
  grpc_chttp2_hpack_parser_state after_prioritization;
  /* the refcount of the slice that we're currently parsing */
  grpc_slice_refcount* current_slice_refcount;
  /* the value we're currently parsing */
  union {
    uint32_t* value;
    grpc_chttp2_hpack_parser_string* str;
  } parsing;
  /* string parameters for each chunk */
  grpc_chttp2_hpack_parser_string key;
  grpc_chttp2_hpack_parser_string value;
  /* parsed index */
  uint32_t index;
  /* When we parse a value string, we determine the metadata element for a
     specific index, which we need again when we're finishing up with that
     header. To avoid calculating the metadata element for that index a second
     time at that stage, we cache (and invalidate) the element here. */
  grpc_mdelem md_for_index;
#ifndef NDEBUG
  int64_t precomputed_md_index;
#endif
  /* length of source bytes for the currently parsing string */
  uint32_t strlen;
  /* number of source bytes read for the currently parsing string */
  uint32_t strgot;
  /* huffman decoding state */
  int16_t huff_state;
  /* is the string being decoded binary? */
  uint8_t binary;
  /* is the current string huffman encoded? */
  uint8_t huff;
  /* is a dynamic table update allowed? */
  uint8_t dynamic_table_update_allowed;
  /* set by higher layers, used by grpc_chttp2_header_parser_parse to signal
     it should append a metadata boundary at the end of frame */
  uint8_t is_boundary;
  uint8_t is_eof;
  uint32_t base64_buffer;

  /* hpack table */
  grpc_chttp2_hptbl table;
};

void grpc_chttp2_hpack_parser_init(grpc_chttp2_hpack_parser* p);
void grpc_chttp2_hpack_parser_destroy(grpc_chttp2_hpack_parser* p);

void grpc_chttp2_hpack_parser_set_has_priority(grpc_chttp2_hpack_parser* p);

grpc_error* grpc_chttp2_hpack_parser_parse(grpc_chttp2_hpack_parser* p,
                                           const grpc_slice& slice);

/* wraps grpc_chttp2_hpack_parser_parse to provide a frame level parser for
   the transport */
grpc_error* grpc_chttp2_header_parser_parse(void* hpack_parser,
                                            grpc_chttp2_transport* t,
                                            grpc_chttp2_stream* s,
                                            const grpc_slice& slice,
                                            int is_last);

#endif /* GRPC_CORE_EXT_TRANSPORT_CHTTP2_TRANSPORT_HPACK_PARSER_H */
