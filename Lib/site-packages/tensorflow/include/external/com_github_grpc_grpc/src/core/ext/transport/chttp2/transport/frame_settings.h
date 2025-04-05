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

#ifndef GRPC_CORE_EXT_TRANSPORT_CHTTP2_TRANSPORT_FRAME_SETTINGS_H
#define GRPC_CORE_EXT_TRANSPORT_CHTTP2_TRANSPORT_FRAME_SETTINGS_H

#include <grpc/support/port_platform.h>

#include <grpc/slice.h>
#include "src/core/ext/transport/chttp2/transport/frame.h"
#include "src/core/ext/transport/chttp2/transport/http2_settings.h"

typedef enum {
  GRPC_CHTTP2_SPS_ID0,
  GRPC_CHTTP2_SPS_ID1,
  GRPC_CHTTP2_SPS_VAL0,
  GRPC_CHTTP2_SPS_VAL1,
  GRPC_CHTTP2_SPS_VAL2,
  GRPC_CHTTP2_SPS_VAL3
} grpc_chttp2_settings_parse_state;

typedef struct {
  grpc_chttp2_settings_parse_state state;
  uint32_t* target_settings;
  uint8_t is_ack;
  uint16_t id;
  uint32_t value;
  uint32_t incoming_settings[GRPC_CHTTP2_NUM_SETTINGS];
} grpc_chttp2_settings_parser;

/* Create a settings frame by diffing old & new, and updating old to be new */
grpc_slice grpc_chttp2_settings_create(uint32_t* old, const uint32_t* newval,
                                       uint32_t force_mask, size_t count);
/* Create an ack settings frame */
grpc_slice grpc_chttp2_settings_ack_create(void);

grpc_error* grpc_chttp2_settings_parser_begin_frame(
    grpc_chttp2_settings_parser* parser, uint32_t length, uint8_t flags,
    uint32_t* settings);
grpc_error* grpc_chttp2_settings_parser_parse(void* parser,
                                              grpc_chttp2_transport* t,
                                              grpc_chttp2_stream* s,
                                              const grpc_slice& slice,
                                              int is_last);

#endif /* GRPC_CORE_EXT_TRANSPORT_CHTTP2_TRANSPORT_FRAME_SETTINGS_H */
