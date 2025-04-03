/*
 *
 * Copyright 2016 gRPC authors.
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

#ifndef GRPC_CORE_LIB_SLICE_PERCENT_ENCODING_H
#define GRPC_CORE_LIB_SLICE_PERCENT_ENCODING_H

/* Percent encoding and decoding of slices.
   Transforms arbitrary strings into safe-for-transmission strings by using
   variants of percent encoding (RFC 3986).
   Two major variants are supplied: one that strictly matches URL encoding,
     and another which applies percent encoding only to non-http2 header
     bytes (the 'compatible' variant) */

#include <grpc/support/port_platform.h>

#include <stdbool.h>

#include <grpc/slice.h>

/* URL percent encoding spec bitfield (usabel as 'unreserved_bytes' in
   grpc_percent_encode_slice, grpc_strict_percent_decode_slice).
   Flags [A-Za-z0-9-_.~] as unreserved bytes for the percent encoding routines
   */
extern const uint8_t grpc_url_percent_encoding_unreserved_bytes[256 / 8];
/* URL percent encoding spec bitfield (usabel as 'unreserved_bytes' in
   grpc_percent_encode_slice, grpc_strict_percent_decode_slice).
   Flags ascii7 non-control characters excluding '%' as unreserved bytes for the
   percent encoding routines */
extern const uint8_t grpc_compatible_percent_encoding_unreserved_bytes[256 / 8];

/* Percent-encode a slice, returning the new slice (this cannot fail):
   unreserved_bytes is a bitfield indicating which bytes are considered
   unreserved and thus do not need percent encoding */
grpc_slice grpc_percent_encode_slice(const grpc_slice& slice,
                                     const uint8_t* unreserved_bytes);
/* Percent-decode a slice, strictly.
   If the input is legal (contains no unreserved bytes, and legal % encodings),
   returns true and sets *slice_out to the decoded slice.
   If the input is not legal, returns false and leaves *slice_out untouched.
   unreserved_bytes is a bitfield indicating which bytes are considered
   unreserved and thus do not need percent encoding */
bool grpc_strict_percent_decode_slice(const grpc_slice& slice_in,
                                      const uint8_t* unreserved_bytes,
                                      grpc_slice* slice_out);
/* Percent-decode a slice, permissively.
   If a % triplet can not be decoded, pass it through verbatim.
   This cannot fail. */
grpc_slice grpc_permissive_percent_decode_slice(const grpc_slice& slice_in);

#endif /* GRPC_CORE_LIB_SLICE_PERCENT_ENCODING_H */
