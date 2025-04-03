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

#ifndef GRPCPP_IMPL_CODEGEN_PROTO_UTILS_H
#define GRPCPP_IMPL_CODEGEN_PROTO_UTILS_H

#include <type_traits>

#include <grpc/impl/codegen/byte_buffer_reader.h>
#include <grpc/impl/codegen/grpc_types.h>
#include <grpc/impl/codegen/slice.h>
#include <grpcpp/impl/codegen/byte_buffer.h>
#include <grpcpp/impl/codegen/config_protobuf.h>
#include <grpcpp/impl/codegen/core_codegen_interface.h>
#include <grpcpp/impl/codegen/proto_buffer_reader.h>
#include <grpcpp/impl/codegen/proto_buffer_writer.h>
#include <grpcpp/impl/codegen/serialization_traits.h>
#include <grpcpp/impl/codegen/slice.h>
#include <grpcpp/impl/codegen/status.h>

/// This header provides serialization and deserialization between gRPC
/// messages serialized using protobuf and the C++ objects they represent.

namespace grpc {

extern CoreCodegenInterface* g_core_codegen_interface;

// ProtoBufferWriter must be a subclass of ::protobuf::io::ZeroCopyOutputStream.
template <class ProtoBufferWriter, class T>
Status GenericSerialize(const grpc::protobuf::MessageLite& msg, ByteBuffer* bb,
                        bool* own_buffer) {
  static_assert(std::is_base_of<protobuf::io::ZeroCopyOutputStream,
                                ProtoBufferWriter>::value,
                "ProtoBufferWriter must be a subclass of "
                "::protobuf::io::ZeroCopyOutputStream");
  *own_buffer = true;
  int byte_size = msg.ByteSize();
  if ((size_t)byte_size <= GRPC_SLICE_INLINED_SIZE) {
    Slice slice(byte_size);
    // We serialize directly into the allocated slices memory
    GPR_CODEGEN_ASSERT(slice.end() == msg.SerializeWithCachedSizesToArray(
                                          const_cast<uint8_t*>(slice.begin())));
    ByteBuffer tmp(&slice, 1);
    bb->Swap(&tmp);

    return g_core_codegen_interface->ok();
  }
  ProtoBufferWriter writer(bb, kProtoBufferWriterMaxBufferLength, byte_size);
  return msg.SerializeToZeroCopyStream(&writer)
             ? g_core_codegen_interface->ok()
             : Status(StatusCode::INTERNAL, "Failed to serialize message");
}

// BufferReader must be a subclass of ::protobuf::io::ZeroCopyInputStream.
template <class ProtoBufferReader, class T>
Status GenericDeserialize(ByteBuffer* buffer,
                          grpc::protobuf::MessageLite* msg) {
  static_assert(std::is_base_of<protobuf::io::ZeroCopyInputStream,
                                ProtoBufferReader>::value,
                "ProtoBufferReader must be a subclass of "
                "::protobuf::io::ZeroCopyInputStream");
  if (buffer == nullptr) {
    return Status(StatusCode::INTERNAL, "No payload");
  }
  Status result = g_core_codegen_interface->ok();
  {
    ProtoBufferReader reader(buffer);
    if (!reader.status().ok()) {
      return reader.status();
    }
    if (!msg->ParseFromZeroCopyStream(&reader)) {
      result = Status(StatusCode::INTERNAL, msg->InitializationErrorString());
    }
  }
  buffer->Clear();
  return result;
}

// this is needed so the following class does not conflict with protobuf
// serializers that utilize internal-only tools.
#ifdef GRPC_OPEN_SOURCE_PROTO
// This class provides a protobuf serializer. It translates between protobuf
// objects and grpc_byte_buffers. More information about SerializationTraits can
// be found in include/grpcpp/impl/codegen/serialization_traits.h.
template <class T>
class SerializationTraits<
    T, typename std::enable_if<
           std::is_base_of<grpc::protobuf::MessageLite, T>::value>::type> {
 public:
  static Status Serialize(const grpc::protobuf::MessageLite& msg,
                          ByteBuffer* bb, bool* own_buffer) {
    return GenericSerialize<ProtoBufferWriter, T>(msg, bb, own_buffer);
  }

  static Status Deserialize(ByteBuffer* buffer,
                            grpc::protobuf::MessageLite* msg) {
    return GenericDeserialize<ProtoBufferReader, T>(buffer, msg);
  }
};
#endif

}  // namespace grpc

#endif  // GRPCPP_IMPL_CODEGEN_PROTO_UTILS_H
