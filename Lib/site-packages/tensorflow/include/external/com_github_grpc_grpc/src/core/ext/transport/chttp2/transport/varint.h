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

#ifndef GRPC_CORE_EXT_TRANSPORT_CHTTP2_TRANSPORT_VARINT_H
#define GRPC_CORE_EXT_TRANSPORT_CHTTP2_TRANSPORT_VARINT_H

#include <grpc/support/port_platform.h>

/* Helpers for hpack varint encoding */

/* length of a value that needs varint tail encoding (it's bigger than can be
   bitpacked into the opcode byte) - returned value includes the length of the
   opcode byte */
uint32_t grpc_chttp2_hpack_varint_length(uint32_t tail_value);

void grpc_chttp2_hpack_write_varint_tail(uint32_t tail_value, uint8_t* target,
                                         uint32_t tail_length);

/* maximum value that can be bitpacked with the opcode if the opcode has a
   prefix
   of length prefix_bits */
#define GRPC_CHTTP2_MAX_IN_PREFIX(prefix_bits) \
  ((uint32_t)((1 << (8 - (prefix_bits))) - 1))

/* length required to bitpack a value */
#define GRPC_CHTTP2_VARINT_LENGTH(n, prefix_bits) \
  ((n) < GRPC_CHTTP2_MAX_IN_PREFIX(prefix_bits)   \
       ? 1u                                       \
       : grpc_chttp2_hpack_varint_length(         \
             (n)-GRPC_CHTTP2_MAX_IN_PREFIX(prefix_bits)))

#define GRPC_CHTTP2_WRITE_VARINT(n, prefix_bits, prefix_or, target, length)   \
  do {                                                                        \
    uint8_t* tgt = target;                                                    \
    if ((length) == 1u) {                                                     \
      (tgt)[0] = (uint8_t)((prefix_or) | (n));                                \
    } else {                                                                  \
      (tgt)[0] =                                                              \
          (prefix_or) | (uint8_t)GRPC_CHTTP2_MAX_IN_PREFIX(prefix_bits);      \
      grpc_chttp2_hpack_write_varint_tail(                                    \
          (n)-GRPC_CHTTP2_MAX_IN_PREFIX(prefix_bits), (tgt) + 1, (length)-1); \
    }                                                                         \
  } while (0)

#endif /* GRPC_CORE_EXT_TRANSPORT_CHTTP2_TRANSPORT_VARINT_H */
