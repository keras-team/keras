/*
 * Copyright 2020 The TensorFlow Runtime Authors
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

// This file declares TypeName.

#ifndef TFRT_HOST_CONTEXT_TYPE_NAME_H_
#define TFRT_HOST_CONTEXT_TYPE_NAME_H_

#include "llvm/ADT/StringRef.h"
#include "tfrt/support/forward_decls.h"

namespace tfrt {

// TypeName contains the name of a type uniqued into KernelRegistry, allowing
// name based reflection and debugging capabilities.
class TypeName {
 public:
  TypeName() : name_(nullptr) {}
  TypeName(const TypeName&) = default;
  TypeName& operator=(const TypeName&) = default;

  string_view GetName() const { return name_; }

  bool operator==(TypeName rhs) const {
    return name_ == rhs.name_ || strcmp(name_, rhs.name_) == 0;
  }

  bool operator!=(TypeName rhs) const { return !(*this == rhs); }

 private:
  friend class KernelRegistry;
  explicit TypeName(const char* name) : name_(name) {}
  const char* name_;
};

}  // namespace tfrt

#endif  // TFRT_HOST_CONTEXT_TYPE_NAME_H_
