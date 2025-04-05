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

#ifndef GRPC_CORE_LIB_COMPRESSION_COMPRESSION_ARGS_H
#define GRPC_CORE_LIB_COMPRESSION_COMPRESSION_ARGS_H

#include <grpc/support/port_platform.h>

#include <grpc/compression.h>
#include <grpc/impl/codegen/grpc_types.h>

/** Returns the compression algorithm set in \a a. */
grpc_compression_algorithm
grpc_channel_args_get_channel_default_compression_algorithm(
    const grpc_channel_args* a);

/** Returns a channel arg instance with compression enabled. If \a a is
 * non-NULL, its args are copied. N.B. GRPC_COMPRESS_NONE disables compression
 * for the channel. */
grpc_channel_args* grpc_channel_args_set_channel_default_compression_algorithm(
    grpc_channel_args* a, grpc_compression_algorithm algorithm);

/** Sets the support for the given compression algorithm. By default, all
 * compression algorithms are enabled. It's an error to disable an algorithm set
 * by grpc_channel_args_set_compression_algorithm.
 *
 * Returns an instance with the updated algorithm states. The \a a pointer is
 * modified to point to the returned instance (which may be different from the
 * input value of \a a). */
grpc_channel_args* grpc_channel_args_compression_algorithm_set_state(
    grpc_channel_args** a, grpc_compression_algorithm algorithm, int state);

/** Returns the bitset representing the support state (true for enabled, false
 * for disabled) for compression algorithms.
 *
 * The i-th bit of the returned bitset corresponds to the i-th entry in the
 * grpc_compression_algorithm enum. */
uint32_t grpc_channel_args_compression_algorithm_get_states(
    const grpc_channel_args* a);

#endif /* GRPC_CORE_LIB_COMPRESSION_COMPRESSION_ARGS_H */
