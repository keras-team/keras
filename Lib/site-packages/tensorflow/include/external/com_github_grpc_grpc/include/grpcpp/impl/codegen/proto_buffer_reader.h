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

#ifndef GRPCPP_IMPL_CODEGEN_PROTO_BUFFER_READER_H
#define GRPCPP_IMPL_CODEGEN_PROTO_BUFFER_READER_H

#include <type_traits>

#include <grpc/impl/codegen/byte_buffer_reader.h>
#include <grpc/impl/codegen/grpc_types.h>
#include <grpc/impl/codegen/slice.h>
#include <grpcpp/impl/codegen/byte_buffer.h>
#include <grpcpp/impl/codegen/config_protobuf.h>
#include <grpcpp/impl/codegen/core_codegen_interface.h>
#include <grpcpp/impl/codegen/serialization_traits.h>
#include <grpcpp/impl/codegen/status.h>

/// This header provides an object that reads bytes directly from a
/// grpc::ByteBuffer, via the ZeroCopyInputStream interface

namespace grpc {

extern CoreCodegenInterface* g_core_codegen_interface;

/// This is a specialization of the protobuf class ZeroCopyInputStream
/// The principle is to get one chunk of data at a time from the proto layer,
/// with options to backup (re-see some bytes) or skip (forward past some bytes)
///
/// Read more about ZeroCopyInputStream interface here:
/// https://developers.google.com/protocol-buffers/docs/reference/cpp/google.protobuf.io.zero_copy_stream#ZeroCopyInputStream
class ProtoBufferReader : public ::grpc::protobuf::io::ZeroCopyInputStream {
 public:
  /// Constructs buffer reader from \a buffer. Will set \a status() to non ok
  /// if \a buffer is invalid (the internal buffer has not been initialized).
  explicit ProtoBufferReader(ByteBuffer* buffer)
      : byte_count_(0), backup_count_(0), status_() {
    /// Implemented through a grpc_byte_buffer_reader which iterates
    /// over the slices that make up a byte buffer
    if (!buffer->Valid() ||
        !g_core_codegen_interface->grpc_byte_buffer_reader_init(
            &reader_, buffer->c_buffer())) {
      status_ = Status(StatusCode::INTERNAL,
                       "Couldn't initialize byte buffer reader");
    }
  }

  ~ProtoBufferReader() {
    if (status_.ok()) {
      g_core_codegen_interface->grpc_byte_buffer_reader_destroy(&reader_);
    }
  }

  /// Give the proto library a chunk of data from the stream. The caller
  /// may safely read from data[0, size - 1].
  bool Next(const void** data, int* size) override {
    if (!status_.ok()) {
      return false;
    }
    /// If we have backed up previously, we need to return the backed-up slice
    if (backup_count_ > 0) {
      *data = GRPC_SLICE_START_PTR(*slice_) + GRPC_SLICE_LENGTH(*slice_) -
              backup_count_;
      GPR_CODEGEN_ASSERT(backup_count_ <= INT_MAX);
      *size = (int)backup_count_;
      backup_count_ = 0;
      return true;
    }
    /// Otherwise get the next slice from the byte buffer reader
    if (!g_core_codegen_interface->grpc_byte_buffer_reader_peek(&reader_,
                                                                &slice_)) {
      return false;
    }
    *data = GRPC_SLICE_START_PTR(*slice_);
    // On win x64, int is only 32bit
    GPR_CODEGEN_ASSERT(GRPC_SLICE_LENGTH(*slice_) <= INT_MAX);
    byte_count_ += * size = (int)GRPC_SLICE_LENGTH(*slice_);
    return true;
  }

  /// Returns the status of the buffer reader.
  Status status() const { return status_; }

  /// The proto library calls this to indicate that we should back up \a count
  /// bytes that have already been returned by the last call of Next.
  /// So do the backup and have that ready for a later Next.
  void BackUp(int count) override {
    GPR_CODEGEN_ASSERT(count <= static_cast<int>(GRPC_SLICE_LENGTH(*slice_)));
    backup_count_ = count;
  }

  /// The proto library calls this to skip over \a count bytes. Implement this
  /// using Next and BackUp combined.
  bool Skip(int count) override {
    const void* data;
    int size;
    while (Next(&data, &size)) {
      if (size >= count) {
        BackUp(size - count);
        return true;
      }
      // size < count;
      count -= size;
    }
    // error or we have too large count;
    return false;
  }

  /// Returns the total number of bytes read since this object was created.
  int64_t ByteCount() const override { return byte_count_ - backup_count_; }

  // These protected members are needed to support internal optimizations.
  // they expose internal bits of grpc core that are NOT stable. If you have
  // a use case needs to use one of these functions, please send an email to
  // https://groups.google.com/forum/#!forum/grpc-io.
 protected:
  void set_byte_count(int64_t byte_count) { byte_count_ = byte_count; }
  int64_t backup_count() { return backup_count_; }
  void set_backup_count(int64_t backup_count) { backup_count_ = backup_count; }
  grpc_byte_buffer_reader* reader() { return &reader_; }
  grpc_slice* slice() { return slice_; }
  grpc_slice** mutable_slice_ptr() { return &slice_; }

 private:
  int64_t byte_count_;              ///< total bytes read since object creation
  int64_t backup_count_;            ///< how far backed up in the stream we are
  grpc_byte_buffer_reader reader_;  ///< internal object to read \a grpc_slice
                                    ///< from the \a grpc_byte_buffer
  grpc_slice* slice_;               ///< current slice passed back to the caller
  Status status_;                   ///< status of the entire object
};

}  // namespace grpc

#endif  // GRPCPP_IMPL_CODEGEN_PROTO_BUFFER_READER_H
