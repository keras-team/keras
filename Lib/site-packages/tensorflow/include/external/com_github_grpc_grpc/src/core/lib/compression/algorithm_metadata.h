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

#ifndef GRPC_CORE_LIB_COMPRESSION_ALGORITHM_METADATA_H
#define GRPC_CORE_LIB_COMPRESSION_ALGORITHM_METADATA_H

#include <grpc/support/port_platform.h>

#include <grpc/compression.h>
#include "src/core/lib/compression/compression_internal.h"
#include "src/core/lib/transport/metadata.h"

/** Return compression algorithm based metadata value */
grpc_slice grpc_compression_algorithm_slice(
    grpc_compression_algorithm algorithm);

/** Find compression algorithm based on passed in mdstr - returns
 *  GRPC_COMPRESS_ALGORITHM_COUNT on failure */
grpc_compression_algorithm grpc_compression_algorithm_from_slice(
    const grpc_slice& str);

/** Return compression algorithm based metadata element */
grpc_mdelem grpc_compression_encoding_mdelem(
    grpc_compression_algorithm algorithm);

/** Return message compression algorithm based metadata element (grpc-encoding:
 * xxx) */
grpc_mdelem grpc_message_compression_encoding_mdelem(
    grpc_message_compression_algorithm algorithm);

/** Return stream compression algorithm based metadata element
 * (content-encoding: xxx) */
grpc_mdelem grpc_stream_compression_encoding_mdelem(
    grpc_stream_compression_algorithm algorithm);

/** Find compression algorithm based on passed in mdstr - returns
 * GRPC_COMPRESS_ALGORITHM_COUNT on failure */
grpc_message_compression_algorithm
grpc_message_compression_algorithm_from_slice(const grpc_slice& str);

/** Find stream compression algorithm based on passed in mdstr - returns
 * GRPC_STREAM_COMPRESS_ALGORITHM_COUNT on failure */
grpc_stream_compression_algorithm grpc_stream_compression_algorithm_from_slice(
    const grpc_slice& str);

#endif /* GRPC_CORE_LIB_COMPRESSION_ALGORITHM_METADATA_H */
