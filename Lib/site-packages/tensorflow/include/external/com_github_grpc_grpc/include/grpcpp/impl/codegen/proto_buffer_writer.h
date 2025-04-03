/*
 *
 * Copyright 2018 gRPC authors.
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

#ifndef GRPCPP_IMPL_CODEGEN_PROTO_BUFFER_WRITER_H
#define GRPCPP_IMPL_CODEGEN_PROTO_BUFFER_WRITER_H

#include <type_traits>

#include <grpc/impl/codegen/grpc_types.h>
#include <grpc/impl/codegen/slice.h>
#include <grpcpp/impl/codegen/byte_buffer.h>
#include <grpcpp/impl/codegen/config_protobuf.h>
#include <grpcpp/impl/codegen/core_codegen_interface.h>
#include <grpcpp/impl/codegen/serialization_traits.h>
#include <grpcpp/impl/codegen/status.h>

/// This header provides an object that writes bytes directly into a
/// grpc::ByteBuffer, via the ZeroCopyOutputStream interface

namespace grpc {

extern CoreCodegenInterface* g_core_codegen_interface;

// Forward declaration for testing use only
namespace internal {
class ProtoBufferWriterPeer;
}  // namespace internal

const int kProtoBufferWriterMaxBufferLength = 1024 * 1024;

/// This is a specialization of the protobuf class ZeroCopyOutputStream.
/// The principle is to give the proto layer one buffer of bytes at a time
/// that it can use to serialize the next portion of the message, with the
/// option to "backup" if more buffer is given than required at the last buffer.
///
/// Read more about ZeroCopyOutputStream interface here:
/// https://developers.google.com/protocol-buffers/docs/reference/cpp/google.protobuf.io.zero_copy_stream#ZeroCopyOutputStream
class ProtoBufferWriter : public ::grpc::protobuf::io::ZeroCopyOutputStream {
 public:
  /// Constructor for this derived class
  ///
  /// \param[out] byte_buffer A pointer to the grpc::ByteBuffer created
  /// \param block_size How big are the chunks to allocate at a time
  /// \param total_size How many total bytes are required for this proto
  ProtoBufferWriter(ByteBuffer* byte_buffer, int block_size, int total_size)
      : block_size_(block_size),
        total_size_(total_size),
        byte_count_(0),
        have_backup_(false) {
    GPR_CODEGEN_ASSERT(!byte_buffer->Valid());
    /// Create an empty raw byte buffer and look at its underlying slice buffer
    grpc_byte_buffer* bp =
        g_core_codegen_interface->grpc_raw_byte_buffer_create(NULL, 0);
    byte_buffer->set_buffer(bp);
    slice_buffer_ = &bp->data.raw.slice_buffer;
  }

  ~ProtoBufferWriter() {
    if (have_backup_) {
      g_core_codegen_interface->grpc_slice_unref(backup_slice_);
    }
  }

  /// Give the proto library the next buffer of bytes and its size. It is
  /// safe for the caller to write from data[0, size - 1].
  bool Next(void** data, int* size) override {
    // Protobuf should not ask for more memory than total_size_.
    GPR_CODEGEN_ASSERT(byte_count_ < total_size_);
    // 1. Use the remaining backup slice if we have one
    // 2. Otherwise allocate a slice, up to the remaining length needed
    //    or our maximum allocation size
    // 3. Provide the slice start and size available
    // 4. Add the slice being returned to the slice buffer
    size_t remain = static_cast<size_t>(total_size_ - byte_count_);
    if (have_backup_) {
      /// If we have a backup slice, we should use it first
      slice_ = backup_slice_;
      have_backup_ = false;
      if (GRPC_SLICE_LENGTH(slice_) > remain) {
        GRPC_SLICE_SET_LENGTH(slice_, remain);
      }
    } else {
      // When less than a whole block is needed, only allocate that much.
      // But make sure the allocated slice is not inlined.
      size_t allocate_length =
          remain > static_cast<size_t>(block_size_) ? block_size_ : remain;
      slice_ = g_core_codegen_interface->grpc_slice_malloc(
          allocate_length > GRPC_SLICE_INLINED_SIZE
              ? allocate_length
              : GRPC_SLICE_INLINED_SIZE + 1);
    }
    *data = GRPC_SLICE_START_PTR(slice_);
    // On win x64, int is only 32bit
    GPR_CODEGEN_ASSERT(GRPC_SLICE_LENGTH(slice_) <= INT_MAX);
    byte_count_ += * size = (int)GRPC_SLICE_LENGTH(slice_);
    g_core_codegen_interface->grpc_slice_buffer_add(slice_buffer_, slice_);
    return true;
  }

  /// Backup by \a count bytes because Next returned more bytes than needed
  /// (only used in the last buffer). \a count must be less than or equal too
  /// the last buffer returned from next.
  void BackUp(int count) override {
    /// 1. Remove the partially-used last slice from the slice buffer
    /// 2. Split it into the needed (if any) and unneeded part
    /// 3. Add the needed part back to the slice buffer
    /// 4. Mark that we still have the remaining part (for later use/unref)
    GPR_CODEGEN_ASSERT(count <= static_cast<int>(GRPC_SLICE_LENGTH(slice_)));
    g_core_codegen_interface->grpc_slice_buffer_pop(slice_buffer_);
    if ((size_t)count == GRPC_SLICE_LENGTH(slice_)) {
      backup_slice_ = slice_;
    } else {
      backup_slice_ = g_core_codegen_interface->grpc_slice_split_tail(
          &slice_, GRPC_SLICE_LENGTH(slice_) - count);
      g_core_codegen_interface->grpc_slice_buffer_add(slice_buffer_, slice_);
    }
    // It's dangerous to keep an inlined grpc_slice as the backup slice, since
    // on a following Next() call, a reference will be returned to this slice
    // via GRPC_SLICE_START_PTR, which will not be an address held by
    // slice_buffer_.
    have_backup_ = backup_slice_.refcount != NULL;
    byte_count_ -= count;
  }

  /// Returns the total number of bytes written since this object was created.
  int64_t ByteCount() const override { return byte_count_; }

  // These protected members are needed to support internal optimizations.
  // they expose internal bits of grpc core that are NOT stable. If you have
  // a use case needs to use one of these functions, please send an email to
  // https://groups.google.com/forum/#!forum/grpc-io.
 protected:
  grpc_slice_buffer* slice_buffer() { return slice_buffer_; }
  void set_byte_count(int64_t byte_count) { byte_count_ = byte_count; }

 private:
  // friend for testing purposes only
  friend class internal::ProtoBufferWriterPeer;
  const int block_size_;  ///< size to alloc for each new \a grpc_slice needed
  const int total_size_;  ///< byte size of proto being serialized
  int64_t byte_count_;    ///< bytes written since this object was created
  grpc_slice_buffer*
      slice_buffer_;  ///< internal buffer of slices holding the serialized data
  bool have_backup_;  ///< if we are holding a backup slice or not
  grpc_slice backup_slice_;  ///< holds space we can still write to, if the
                             ///< caller has called BackUp
  grpc_slice slice_;         ///< current slice passed back to the caller
};

}  // namespace grpc

#endif  // GRPCPP_IMPL_CODEGEN_PROTO_BUFFER_WRITER_H
