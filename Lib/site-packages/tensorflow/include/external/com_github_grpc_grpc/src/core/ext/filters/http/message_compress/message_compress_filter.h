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

#ifndef GRPC_CORE_EXT_FILTERS_HTTP_MESSAGE_COMPRESS_MESSAGE_COMPRESS_FILTER_H
#define GRPC_CORE_EXT_FILTERS_HTTP_MESSAGE_COMPRESS_MESSAGE_COMPRESS_FILTER_H

#include <grpc/support/port_platform.h>

#include <grpc/impl/codegen/compression_types.h>

#include "src/core/lib/channel/channel_stack.h"

/** Compression filter for outgoing data.
 *
 * See <grpc/compression.h> for the available compression settings.
 *
 * Compression settings may come from:
 *  - Channel configuration, as established at channel creation time.
 *  - The metadata accompanying the outgoing data to be compressed. This is
 *    taken as a request only. We may choose not to honor it. The metadata key
 *    is given by \a GRPC_COMPRESSION_REQUEST_ALGORITHM_MD_KEY.
 *
 * Compression can be disabled for concrete messages (for instance in order to
 * prevent CRIME/BEAST type attacks) by having the GRPC_WRITE_NO_COMPRESS set in
 * the BEGIN_MESSAGE flags.
 *
 * The attempted compression mechanism is added to the resulting initial
 * metadata under the'grpc-encoding' key.
 *
 * If compression is actually performed, BEGIN_MESSAGE's flag is modified to
 * incorporate GRPC_WRITE_INTERNAL_COMPRESS. Otherwise, and regardless of the
 * aforementioned 'grpc-encoding' metadata value, data will pass through
 * uncompressed. */

extern const grpc_channel_filter grpc_message_compress_filter;

#endif /* GRPC_CORE_EXT_FILTERS_HTTP_MESSAGE_COMPRESS_MESSAGE_COMPRESS_FILTER_H \
        */
