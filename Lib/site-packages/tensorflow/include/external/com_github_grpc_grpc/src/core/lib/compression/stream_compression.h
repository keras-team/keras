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

#ifndef GRPC_CORE_LIB_COMPRESSION_STREAM_COMPRESSION_H
#define GRPC_CORE_LIB_COMPRESSION_STREAM_COMPRESSION_H

#include <grpc/support/port_platform.h>

#include <stdbool.h>

#include <grpc/slice_buffer.h>
#include <zlib.h>

#include "src/core/lib/transport/static_metadata.h"

typedef struct grpc_stream_compression_vtable grpc_stream_compression_vtable;

/* Stream compression/decompression context */
typedef struct grpc_stream_compression_context {
  const grpc_stream_compression_vtable* vtable;
} grpc_stream_compression_context;

typedef enum grpc_stream_compression_method {
  GRPC_STREAM_COMPRESSION_IDENTITY_COMPRESS = 0,
  GRPC_STREAM_COMPRESSION_IDENTITY_DECOMPRESS,
  GRPC_STREAM_COMPRESSION_GZIP_COMPRESS,
  GRPC_STREAM_COMPRESSION_GZIP_DECOMPRESS,
  GRPC_STREAM_COMPRESSION_METHOD_COUNT
} grpc_stream_compression_method;

typedef enum grpc_stream_compression_flush {
  GRPC_STREAM_COMPRESSION_FLUSH_NONE = 0,
  GRPC_STREAM_COMPRESSION_FLUSH_SYNC,
  GRPC_STREAM_COMPRESSION_FLUSH_FINISH,
  GRPC_STREAM_COMPRESSION_FLUSH_COUNT
} grpc_stream_compression_flush;

struct grpc_stream_compression_vtable {
  bool (*compress)(grpc_stream_compression_context* ctx, grpc_slice_buffer* in,
                   grpc_slice_buffer* out, size_t* output_size,
                   size_t max_output_size, grpc_stream_compression_flush flush);
  bool (*decompress)(grpc_stream_compression_context* ctx,
                     grpc_slice_buffer* in, grpc_slice_buffer* out,
                     size_t* output_size, size_t max_output_size,
                     bool* end_of_context);
  grpc_stream_compression_context* (*context_create)(
      grpc_stream_compression_method method);
  void (*context_destroy)(grpc_stream_compression_context* ctx);
};

/**
 * Compress bytes provided in \a in with a given context, with an optional flush
 * at the end of compression. Emits at most \a max_output_size compressed bytes
 * into \a out. If all the bytes in input buffer \a in are depleted and \a flush
 * is not GRPC_STREAM_COMPRESSION_FLUSH_NONE, the corresponding flush method is
 * executed. The total number of bytes emitted is outputted in \a output_size.
 *
 * A SYNC flush indicates that the entire messages in \a in can be decompressed
 * from \a out. A FINISH flush implies a SYNC flush, and that any further
 * compression will not be dependent on the state of the current context and any
 * previous compressed bytes. It allows corresponding decompression context to
 * be dropped when reaching this boundary.
 */
bool grpc_stream_compress(grpc_stream_compression_context* ctx,
                          grpc_slice_buffer* in, grpc_slice_buffer* out,
                          size_t* output_size, size_t max_output_size,
                          grpc_stream_compression_flush flush);

/**
 * Decompress bytes provided in \a in with a given context. Emits at most \a
 * max_output_size decompressed bytes into \a out. If decompression process
 * reached the end of a gzip stream, \a end_of_context is set to true; otherwise
 * it is set to false. The total number of bytes emitted is outputted in \a
 * output_size.
 */
bool grpc_stream_decompress(grpc_stream_compression_context* ctx,
                            grpc_slice_buffer* in, grpc_slice_buffer* out,
                            size_t* output_size, size_t max_output_size,
                            bool* end_of_context);

/**
 * Creates a stream compression context. \a pending_bytes_buffer is the input
 * buffer for compression/decompression operations. \a method specifies whether
 * the context is for compression or decompression.
 */
grpc_stream_compression_context* grpc_stream_compression_context_create(
    grpc_stream_compression_method method);

/**
 * Destroys a stream compression context.
 */
void grpc_stream_compression_context_destroy(
    grpc_stream_compression_context* ctx);

/**
 * Parse stream compression method based on algorithm name
 */
int grpc_stream_compression_method_parse(
    grpc_slice value, bool is_compress, grpc_stream_compression_method* method);

#endif
