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

#ifndef GRPC_CORE_LIB_TRANSPORT_BYTE_STREAM_H
#define GRPC_CORE_LIB_TRANSPORT_BYTE_STREAM_H

#include <grpc/support/port_platform.h>

#include <grpc/slice_buffer.h>
#include "src/core/lib/gprpp/orphanable.h"
#include "src/core/lib/iomgr/closure.h"

/** Internal bit flag for grpc_begin_message's \a flags signaling the use of
 * compression for the message */
#define GRPC_WRITE_INTERNAL_COMPRESS (0x80000000u)
/** Mask of all valid internal flags. */
#define GRPC_WRITE_INTERNAL_USED_MASK (GRPC_WRITE_INTERNAL_COMPRESS)

namespace grpc_core {

class ByteStream : public Orphanable {
 public:
  virtual ~ByteStream() {}

  // Returns true if the bytes are available immediately (in which case
  // on_complete will not be called), or false if the bytes will be available
  // asynchronously (in which case on_complete will be called when they
  // are available). Should not be called if there is no data left on the
  // stream.
  //
  // max_size_hint can be set as a hint as to the maximum number
  // of bytes that would be acceptable to read.
  virtual bool Next(size_t max_size_hint, grpc_closure* on_complete) = 0;

  // Returns the next slice in the byte stream when it is available, as
  // indicated by Next().
  //
  // Once a slice is returned into *slice, it is owned by the caller.
  virtual grpc_error* Pull(grpc_slice* slice) = 0;

  // Shuts down the byte stream.
  //
  // If there is a pending call to on_complete from Next(), it will be
  // invoked with the error passed to Shutdown().
  //
  // The next call to Pull() (if any) will return the error passed to
  // Shutdown().
  virtual void Shutdown(grpc_error* error) = 0;

  uint32_t length() const { return length_; }
  uint32_t flags() const { return flags_; }

  void set_flags(uint32_t flags) { flags_ = flags; }

 protected:
  ByteStream(uint32_t length, uint32_t flags)
      : length_(length), flags_(flags) {}

 private:
  const uint32_t length_;
  uint32_t flags_;
};

//
// SliceBufferByteStream
//
// A ByteStream that wraps a slice buffer.
//

class SliceBufferByteStream : public ByteStream {
 public:
  // Removes all slices in slice_buffer, leaving it empty.
  SliceBufferByteStream(grpc_slice_buffer* slice_buffer, uint32_t flags);

  ~SliceBufferByteStream();

  void Orphan() override;

  bool Next(size_t max_size_hint, grpc_closure* on_complete) override;
  grpc_error* Pull(grpc_slice* slice) override;
  void Shutdown(grpc_error* error) override;

 private:
  grpc_error* shutdown_error_ = GRPC_ERROR_NONE;
  grpc_slice_buffer backing_buffer_;
};

//
// CachingByteStream
//
// A ByteStream that that wraps an underlying byte stream but caches
// the resulting slices in a slice buffer.  If an initial attempt fails
// without fully draining the underlying stream, a new caching stream
// can be created from the same underlying cache, in which case it will
// return whatever is in the backing buffer before continuing to read the
// underlying stream.
//
// NOTE: No synchronization is done, so it is not safe to have multiple
// CachingByteStreams simultaneously drawing from the same underlying
// ByteStreamCache at the same time.
//

class ByteStreamCache {
 public:
  class CachingByteStream : public ByteStream {
   public:
    explicit CachingByteStream(ByteStreamCache* cache);

    ~CachingByteStream();

    void Orphan() override;

    bool Next(size_t max_size_hint, grpc_closure* on_complete) override;
    grpc_error* Pull(grpc_slice* slice) override;
    void Shutdown(grpc_error* error) override;

    // Resets the byte stream to the start of the underlying stream.
    void Reset();

   private:
    ByteStreamCache* cache_;
    size_t cursor_ = 0;
    size_t offset_ = 0;
    grpc_error* shutdown_error_ = GRPC_ERROR_NONE;
  };

  explicit ByteStreamCache(OrphanablePtr<ByteStream> underlying_stream);

  ~ByteStreamCache();

  // Must not be destroyed while still in use by a CachingByteStream.
  void Destroy();

  grpc_slice_buffer* cache_buffer() { return &cache_buffer_; }

 private:
  OrphanablePtr<ByteStream> underlying_stream_;
  uint32_t length_;
  uint32_t flags_;
  grpc_slice_buffer cache_buffer_;
};

}  // namespace grpc_core

#endif /* GRPC_CORE_LIB_TRANSPORT_BYTE_STREAM_H */
