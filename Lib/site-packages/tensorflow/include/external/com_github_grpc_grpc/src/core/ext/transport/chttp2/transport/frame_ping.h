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

#ifndef GRPC_CORE_EXT_TRANSPORT_CHTTP2_TRANSPORT_FRAME_PING_H
#define GRPC_CORE_EXT_TRANSPORT_CHTTP2_TRANSPORT_FRAME_PING_H

#include <grpc/support/port_platform.h>

#include <grpc/slice.h>
#include "src/core/ext/transport/chttp2/transport/frame.h"

typedef struct {
  uint8_t byte;
  uint8_t is_ack;
  uint64_t opaque_8bytes;
} grpc_chttp2_ping_parser;

grpc_slice grpc_chttp2_ping_create(uint8_t ack, uint64_t opaque_8bytes);

grpc_error* grpc_chttp2_ping_parser_begin_frame(grpc_chttp2_ping_parser* parser,
                                                uint32_t length, uint8_t flags);
grpc_error* grpc_chttp2_ping_parser_parse(void* parser,
                                          grpc_chttp2_transport* t,
                                          grpc_chttp2_stream* s,
                                          const grpc_slice& slice, int is_last);

/* Test-only function for disabling ping ack */
void grpc_set_disable_ping_ack(bool disable_ping_ack);

#endif /* GRPC_CORE_EXT_TRANSPORT_CHTTP2_TRANSPORT_FRAME_PING_H */
