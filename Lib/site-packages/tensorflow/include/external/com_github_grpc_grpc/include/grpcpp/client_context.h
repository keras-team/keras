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

/// A ClientContext allows the person implementing a service client to:
///
/// - Add custom metadata key-value pairs that will propagated to the server
/// side.
/// - Control call settings such as compression and authentication.
/// - Initial and trailing metadata coming from the server.
/// - Get performance metrics (ie, census).
///
/// Context settings are only relevant to the call they are invoked with, that
/// is to say, they aren't sticky. Some of these settings, such as the
/// compression options, can be made persistent at channel construction time
/// (see \a grpc::CreateCustomChannel).
///
/// \warning ClientContext instances should \em not be reused across rpcs.

#ifndef GRPCPP_CLIENT_CONTEXT_H
#define GRPCPP_CLIENT_CONTEXT_H

#include <grpcpp/impl/codegen/client_context.h>

#endif  // GRPCPP_CLIENT_CONTEXT_H
