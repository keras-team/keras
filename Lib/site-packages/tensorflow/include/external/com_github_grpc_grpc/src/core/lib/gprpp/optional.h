/*
 *
 * Copyright 2019 gRPC authors.
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

#ifndef GRPC_CORE_LIB_GPRPP_OPTIONAL_H
#define GRPC_CORE_LIB_GPRPP_OPTIONAL_H

#include <grpc/support/port_platform.h>

// TODO(yashykt): Remove false once migration to abseil is done.
#if false && GRPC_USE_ABSL

#include "absl/types/optional.h"

namespace grpc_core {

template <typename T>
using Optional = absl::optional<T>;

}  // namespace grpc_core

#else

#include <utility>

namespace grpc_core {

/* A make-shift alternative for absl::Optional. This can be removed in favor of
 * that once absl dependencies can be introduced. */
template <typename T>
class Optional {
 public:
  Optional() : value_() {}

  void set(const T& val) {
    value_ = val;
    set_ = true;
  }

  void set(T&& val) {
    value_ = std::move(val);
    set_ = true;
  }

  bool has_value() const { return set_; }

  void reset() { set_ = false; }

  T value() const { return value_; }

 private:
  T value_;
  bool set_ = false;
};

} /* namespace grpc_core */

#endif

#endif /* GRPC_CORE_LIB_GPRPP_OPTIONAL_H */
