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

#ifndef GRPCPP_IMPL_CODEGEN_GRPC_LIBRARY_H
#define GRPCPP_IMPL_CODEGEN_GRPC_LIBRARY_H

#include <grpcpp/impl/codegen/core_codegen_interface.h>

namespace grpc {

class GrpcLibraryInterface {
 public:
  virtual ~GrpcLibraryInterface() = default;
  virtual void init() = 0;
  virtual void shutdown() = 0;
};

/// Initialized by \a grpc::GrpcLibraryInitializer from
/// <grpcpp/impl/grpc_library.h>
extern GrpcLibraryInterface* g_glip;

/// Classes that require gRPC to be initialized should inherit from this class.
class GrpcLibraryCodegen {
 public:
  GrpcLibraryCodegen(bool call_grpc_init = true) : grpc_init_called_(false) {
    if (call_grpc_init) {
      GPR_CODEGEN_ASSERT(g_glip &&
                         "gRPC library not initialized. See "
                         "grpc::internal::GrpcLibraryInitializer.");
      g_glip->init();
      grpc_init_called_ = true;
    }
  }
  virtual ~GrpcLibraryCodegen() {
    if (grpc_init_called_) {
      GPR_CODEGEN_ASSERT(g_glip &&
                         "gRPC library not initialized. See "
                         "grpc::internal::GrpcLibraryInitializer.");
      g_glip->shutdown();
    }
  }

 private:
  bool grpc_init_called_;
};

}  // namespace grpc

#endif  // GRPCPP_IMPL_CODEGEN_GRPC_LIBRARY_H
