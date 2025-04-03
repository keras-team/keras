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

#ifndef GRPC_CORE_EXT_TRANSPORT_CHTTP2_TRANSPORT_BIN_ENCODER_H
#define GRPC_CORE_EXT_TRANSPORT_CHTTP2_TRANSPORT_BIN_ENCODER_H

#include <grpc/support/port_platform.h>

#include <grpc/slice.h>

/* base64 encode a slice. Returns a new slice, does not take ownership of the
   input */
grpc_slice grpc_chttp2_base64_encode(const grpc_slice& input);

/* Compress a slice with the static huffman encoder detailed in the hpack
   standard. Returns a new slice, does not take ownership of the input */
grpc_slice grpc_chttp2_huffman_compress(const grpc_slice& input);

/* equivalent to:
   grpc_slice x = grpc_chttp2_base64_encode(input);
   grpc_slice y = grpc_chttp2_huffman_compress(x);
   grpc_slice_unref_internal( x);
   return y; */
grpc_slice grpc_chttp2_base64_encode_and_huffman_compress(
    const grpc_slice& input);

#endif /* GRPC_CORE_EXT_TRANSPORT_CHTTP2_TRANSPORT_BIN_ENCODER_H */
