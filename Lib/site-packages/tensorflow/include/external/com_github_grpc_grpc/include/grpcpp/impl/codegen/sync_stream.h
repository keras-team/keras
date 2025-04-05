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

#ifndef GRPCPP_IMPL_CODEGEN_SYNC_STREAM_H
#define GRPCPP_IMPL_CODEGEN_SYNC_STREAM_H

#include <grpcpp/impl/codegen/sync_stream_impl.h>

namespace grpc {

namespace internal {

typedef ::grpc_impl::internal::ClientStreamingInterface
    ClientStreamingInterface;

typedef ::grpc_impl::internal::ServerStreamingInterface
    ServerStreamingInterface;

template <class R>
using ReaderInterface = ::grpc_impl::internal::ReaderInterface<R>;

template <class W>
using WriterInterface = ::grpc_impl::internal::WriterInterface<W>;

template <class R>
using ClientReaderFactory = ::grpc_impl::internal::ClientReaderFactory<R>;

template <class W>
using ClientWriterFactory = ::grpc_impl::internal::ClientWriterFactory<W>;

template <class W, class R>
using ClientReaderWriterFactory =
    ::grpc_impl::internal::ClientReaderWriterFactory<W, R>;

}  // namespace internal

template <class R>
using ClientReaderInterface = ::grpc_impl::ClientReaderInterface<R>;

template <class R>
using ClientReader = ::grpc_impl::ClientReader<R>;

template <class W>
using ClientWriterInterface = ::grpc_impl::ClientWriterInterface<W>;

template <class W>
using ClientWriter = ::grpc_impl::ClientWriter<W>;

template <class W, class R>
using ClientReaderWriterInterface =
    ::grpc_impl::ClientReaderWriterInterface<W, R>;

template <class W, class R>
using ClientReaderWriter = ::grpc_impl::ClientReaderWriter<W, R>;

template <class R>
using ServerReaderInterface = ::grpc_impl::ServerReaderInterface<R>;

template <class R>
using ServerReader = ::grpc_impl::ServerReader<R>;

template <class W>
using ServerWriterInterface = ::grpc_impl::ServerWriterInterface<W>;

template <class W>
using ServerWriter = ::grpc_impl::ServerWriter<W>;

template <class W, class R>
using ServerReaderWriterInterface =
    ::grpc_impl::ServerReaderWriterInterface<W, R>;

template <class W, class R>
using ServerReaderWriter = ::grpc_impl::ServerReaderWriter<W, R>;

template <class RequestType, class ResponseType>
using ServerUnaryStreamer =
    ::grpc_impl::ServerUnaryStreamer<RequestType, ResponseType>;

template <class RequestType, class ResponseType>
using ServerSplitStreamer =
    ::grpc_impl::ServerSplitStreamer<RequestType, ResponseType>;

}  // namespace grpc

#endif  // GRPCPP_IMPL_CODEGEN_SYNC_STREAM_H
