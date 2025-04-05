/*
 *
 * Copyright 2017 gRPC authors.
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

#ifndef GRPC_CORE_LIB_COMPRESSION_COMPRESSION_INTERNAL_H
#define GRPC_CORE_LIB_COMPRESSION_COMPRESSION_INTERNAL_H

#include <grpc/support/port_platform.h>

#include <grpc/impl/codegen/compression_types.h>

#include "src/core/lib/gpr/useful.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
  GRPC_MESSAGE_COMPRESS_NONE = 0,
  GRPC_MESSAGE_COMPRESS_DEFLATE,
  GRPC_MESSAGE_COMPRESS_GZIP,
  /* TODO(ctiller): snappy */
  GRPC_MESSAGE_COMPRESS_ALGORITHMS_COUNT
} grpc_message_compression_algorithm;

/** Stream compression algorithms supported by gRPC */
typedef enum {
  GRPC_STREAM_COMPRESS_NONE = 0,
  GRPC_STREAM_COMPRESS_GZIP,
  GRPC_STREAM_COMPRESS_ALGORITHMS_COUNT
} grpc_stream_compression_algorithm;

/* Interfaces performing transformation between compression algorithms and
 * levels. */

grpc_message_compression_algorithm
grpc_compression_algorithm_to_message_compression_algorithm(
    grpc_compression_algorithm algo);

grpc_stream_compression_algorithm
grpc_compression_algorithm_to_stream_compression_algorithm(
    grpc_compression_algorithm algo);

uint32_t grpc_compression_bitset_to_message_bitset(uint32_t bitset);

uint32_t grpc_compression_bitset_to_stream_bitset(uint32_t bitset);

uint32_t grpc_compression_bitset_from_message_stream_compression_bitset(
    uint32_t message_bitset, uint32_t stream_bitset);

int grpc_compression_algorithm_from_message_stream_compression_algorithm(
    grpc_compression_algorithm* algorithm,
    grpc_message_compression_algorithm message_algorithm,
    grpc_stream_compression_algorithm stream_algorithm);

/* Interfaces for message compression. */

int grpc_message_compression_algorithm_name(
    grpc_message_compression_algorithm algorithm, const char** name);

grpc_message_compression_algorithm grpc_message_compression_algorithm_for_level(
    grpc_compression_level level, uint32_t accepted_encodings);

int grpc_message_compression_algorithm_parse(
    grpc_slice value, grpc_message_compression_algorithm* algorithm);

/* Interfaces for stream compression. */

int grpc_stream_compression_algorithm_parse(
    grpc_slice value, grpc_stream_compression_algorithm* algorithm);

#ifdef __cplusplus
}
#endif

inline int grpc_compression_options_is_algorithm_enabled_internal(
    const grpc_compression_options* opts,
    grpc_compression_algorithm algorithm) {
  return GPR_BITGET(opts->enabled_algorithms_bitset, algorithm);
}

#endif /* GRPC_CORE_LIB_COMPRESSION_COMPRESSION_INTERNAL_H */
