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

#ifndef GRPC_CORE_LIB_GPRPP_MEMORY_H
#define GRPC_CORE_LIB_GPRPP_MEMORY_H

#include <grpc/support/port_platform.h>

#include <grpc/support/alloc.h>
#include <grpc/support/log.h>

#include <limits>
#include <memory>
#include <utility>

namespace grpc_core {

class DefaultDeleteChar {
 public:
  void operator()(char* p) {
    if (p == nullptr) return;
    gpr_free(p);
  }
};

// UniquePtr<T> is only allowed for char and UniquePtr<char> is deprecated
// in favor of std::string. UniquePtr<char> is equivalent std::unique_ptr
// except that it uses gpr_free for deleter.
template <typename T>
using UniquePtr = std::unique_ptr<T, DefaultDeleteChar>;

// TODO(veblush): Replace this with absl::make_unique once abseil is added.
template <typename T, typename... Args>
inline std::unique_ptr<T> MakeUnique(Args&&... args) {
  return std::unique_ptr<T>(new T(std::forward<Args>(args)...));
}

}  // namespace grpc_core

#endif /* GRPC_CORE_LIB_GPRPP_MEMORY_H */
