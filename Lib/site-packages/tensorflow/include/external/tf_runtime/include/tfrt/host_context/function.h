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

// This file declares the Function type.

#ifndef TFRT_HOST_CONTEXT_FUNCTION_H_
#define TFRT_HOST_CONTEXT_FUNCTION_H_

#include <vector>

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "tfrt/bef/bef_encoding.h"
#include "tfrt/host_context/type_name.h"
#include "tfrt/support/forward_decls.h"
#include "tfrt/support/ref_count.h"

namespace tfrt {

class ExecutionContext;

class Function {
 public:
  virtual ~Function() {}
  Function(const Function&) = delete;
  Function& operator=(const Function&) = delete;

  string_view name() const { return name_; }

  ArrayRef<TypeName> argument_types() const {
    return ArrayRef<TypeName>(argument_result_types_).take_front(num_argument_);
  }

  ArrayRef<TypeName> result_types() const {
    return ArrayRef<TypeName>(argument_result_types_).drop_front(num_argument_);
  }

  size_t num_arguments() const { return num_argument_; }
  size_t num_results() const {
    return argument_result_types_.size() - num_argument_;
  }

  // Execute this function on the specified ExecutionContext, passing the
  // specified arguments. This returns one AsyncValue for each result.
  virtual void Execute(
      const ExecutionContext& exec_ctx, ArrayRef<AsyncValue*> arguments,
      MutableArrayRef<RCReference<AsyncValue>> results) const = 0;

  // Similar to Execute() but passes the arguments by moving the references.
  virtual void ExecuteByValue(
      const ExecutionContext& exec_ctx,
      std::vector<RCReference<AsyncValue>> arguments,
      MutableArrayRef<RCReference<AsyncValue>> results) const {
    std::vector<AsyncValue*> args;
    args.reserve(arguments.size());
    for (auto& arg : arguments) {
      args.push_back(arg.get());
    }
    Execute(exec_ctx, args, results);
  }

  // Similar to the above method, but return immediately after setting up the
  // arguments and results. `results` will be populated with unavailable
  // AsyncValues that serve as futures to the clients (i.e. emplace() or
  // SetError must not be called on these async values by the client).
  virtual void ExecuteAsync(
      const ExecutionContext& exec_ctx,
      std::vector<RCReference<AsyncValue>> arguments,
      MutableArrayRef<RCReference<AsyncValue>> results) const = 0;

  // Reference counting operations, used by async kernels to keep the underlying
  // storage for a function alive.
  virtual void AddRef() const = 0;
  virtual void DropRef() const = 0;

  FunctionKind function_kind() const { return function_kind_; }

 protected:
  Function(string_view name, FunctionKind function_kind,
           ArrayRef<TypeName> argument_types, ArrayRef<TypeName> result_types)
      : name_(name),
        function_kind_(function_kind),
        num_argument_(argument_types.size()) {
    argument_result_types_.reserve(argument_types.size() + result_types.size());
    argument_result_types_.insert(argument_result_types_.end(),
                                  argument_types.begin(), argument_types.end());
    argument_result_types_.insert(argument_result_types_.end(),
                                  result_types.begin(), result_types.end());
  }

  Function(Function&& other) = default;

 private:
  virtual void VtableAnchor();

  // This is the name of the function, or empty if anonymous.
  string_view name_;
  FunctionKind function_kind_;

  size_t num_argument_;
  llvm::SmallVector<TypeName, 8> argument_result_types_;
};

}  // namespace tfrt

#endif  // TFRT_HOST_CONTEXT_FUNCTION_H_
