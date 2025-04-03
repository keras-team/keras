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

#ifndef GRPC_CORE_LIB_SLICE_B64_H
#define GRPC_CORE_LIB_SLICE_B64_H

#include <grpc/support/port_platform.h>

#include <grpc/slice.h>

/* Encodes data using base64. It is the caller's responsibility to free
   the returned char * using gpr_free. Returns NULL on NULL input.
   TODO(makdharma) : change the flags to bool from int */
char* grpc_base64_encode(const void* data, size_t data_size, int url_safe,
                         int multiline);

/* estimate the upper bound on size of base64 encoded data. The actual size
 * is guaranteed to be less than or equal to the size returned here. */
size_t grpc_base64_estimate_encoded_size(size_t data_size, int multiline);

/* Encodes data using base64 and write it to memory pointed to by result. It is
 * the caller's responsibility to allocate enough memory in |result| to fit the
 * encoded data. */
void grpc_base64_encode_core(char* result, const void* vdata, size_t data_size,
                             int url_safe, int multiline);

/* Decodes data according to the base64 specification. Returns an empty
   slice in case of failure. */
grpc_slice grpc_base64_decode(const char* b64, int url_safe);

/* Same as above except that the length is provided by the caller. */
grpc_slice grpc_base64_decode_with_len(const char* b64, size_t b64_len,
                                       int url_safe);

#endif /* GRPC_CORE_LIB_SLICE_B64_H */
