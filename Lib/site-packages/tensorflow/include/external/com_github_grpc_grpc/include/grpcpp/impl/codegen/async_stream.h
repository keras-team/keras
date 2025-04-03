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

#ifndef GRPCPP_IMPL_CODEGEN_ASYNC_STREAM_H
#define GRPCPP_IMPL_CODEGEN_ASYNC_STREAM_H

#include <grpcpp/impl/codegen/async_stream_impl.h>

namespace grpc {

namespace internal {

typedef ::grpc_impl::internal::ClientAsyncStreamingInterface
    ClientAsyncStreamingInterface;

template <class R>
using AsyncReaderInterface = ::grpc_impl::internal::AsyncReaderInterface<R>;

template <class W>
using AsyncWriterInterface = ::grpc_impl::internal::AsyncWriterInterface<W>;

}  // namespace internal

template <class R>
using ClientAsyncReaderInterface = ::grpc_impl::ClientAsyncReaderInterface<R>;

template <class R>
using ClientAsyncReader = ::grpc_impl::ClientAsyncReader<R>;

template <class W>
using ClientAsyncWriterInterface = ::grpc_impl::ClientAsyncWriterInterface<W>;

template <class W>
using ClientAsyncWriter = ::grpc_impl::ClientAsyncWriter<W>;

template <class W, class R>
using ClientAsyncReaderWriterInterface =
    ::grpc_impl::ClientAsyncReaderWriterInterface<W, R>;

template <class W, class R>
using ClientAsyncReaderWriter = ::grpc_impl::ClientAsyncReaderWriter<W, R>;

template <class W, class R>
using ServerAsyncReaderInterface =
    ::grpc_impl::ServerAsyncReaderInterface<W, R>;

template <class W, class R>
using ServerAsyncReader = ::grpc_impl::ServerAsyncReader<W, R>;

template <class W>
using ServerAsyncWriterInterface = ::grpc_impl::ServerAsyncWriterInterface<W>;

template <class W>
using ServerAsyncWriter = ::grpc_impl::ServerAsyncWriter<W>;

template <class W, class R>
using ServerAsyncReaderWriterInterface =
    ::grpc_impl::ServerAsyncReaderWriterInterface<W, R>;

template <class W, class R>
using ServerAsyncReaderWriter = ::grpc_impl::ServerAsyncReaderWriter<W, R>;

namespace internal {
template <class R>
using ClientAsyncReaderFactory =
    ::grpc_impl::internal::ClientAsyncReaderFactory<R>;

template <class W>
using ClientAsyncWriterFactory =
    ::grpc_impl::internal::ClientAsyncWriterFactory<W>;

template <class W, class R>
using ClientAsyncReaderWriterFactory =
    ::grpc_impl::internal::ClientAsyncReaderWriterFactory<W, R>;

}  // namespace internal

}  // namespace grpc

#endif  // GRPCPP_IMPL_CODEGEN_ASYNC_STREAM_H
