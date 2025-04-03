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

#ifndef GRPC_CORE_EXT_TRANSPORT_CHTTP2_TRANSPORT_FRAME_DATA_H
#define GRPC_CORE_EXT_TRANSPORT_CHTTP2_TRANSPORT_FRAME_DATA_H

/* Parser for GRPC streams embedded in DATA frames */

#include <grpc/support/port_platform.h>

#include <grpc/slice.h>
#include <grpc/slice_buffer.h>
#include "src/core/ext/transport/chttp2/transport/frame.h"
#include "src/core/lib/transport/byte_stream.h"
#include "src/core/lib/transport/transport.h"

typedef enum {
  GRPC_CHTTP2_DATA_FH_0,
  GRPC_CHTTP2_DATA_FH_1,
  GRPC_CHTTP2_DATA_FH_2,
  GRPC_CHTTP2_DATA_FH_3,
  GRPC_CHTTP2_DATA_FH_4,
  GRPC_CHTTP2_DATA_FRAME,
  GRPC_CHTTP2_DATA_ERROR
} grpc_chttp2_stream_state;

namespace grpc_core {
class Chttp2IncomingByteStream;
}  // namespace grpc_core

struct grpc_chttp2_data_parser {
  grpc_chttp2_data_parser() = default;
  ~grpc_chttp2_data_parser();

  grpc_chttp2_stream_state state = GRPC_CHTTP2_DATA_FH_0;
  uint8_t frame_type = 0;
  uint32_t frame_size = 0;
  grpc_error* error = GRPC_ERROR_NONE;

  bool is_frame_compressed = false;
  grpc_core::Chttp2IncomingByteStream* parsing_frame = nullptr;
};

/* start processing a new data frame */
grpc_error* grpc_chttp2_data_parser_begin_frame(grpc_chttp2_data_parser* parser,
                                                uint8_t flags,
                                                uint32_t stream_id,
                                                grpc_chttp2_stream* s);

/* handle a slice of a data frame - is_last indicates the last slice of a
   frame */
grpc_error* grpc_chttp2_data_parser_parse(void* parser,
                                          grpc_chttp2_transport* t,
                                          grpc_chttp2_stream* s,
                                          const grpc_slice& slice, int is_last);

void grpc_chttp2_encode_data(uint32_t id, grpc_slice_buffer* inbuf,
                             uint32_t write_bytes, int is_eof,
                             grpc_transport_one_way_stats* stats,
                             grpc_slice_buffer* outbuf);

grpc_error* grpc_deframe_unprocessed_incoming_frames(
    grpc_chttp2_data_parser* p, grpc_chttp2_stream* s,
    grpc_slice_buffer* slices, grpc_slice* slice_out,
    grpc_core::OrphanablePtr<grpc_core::ByteStream>* stream_out);

#endif /* GRPC_CORE_EXT_TRANSPORT_CHTTP2_TRANSPORT_FRAME_DATA_H */
