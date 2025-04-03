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

// This file declares class NativeFunction.

#ifndef TFRT_HOST_CONTEXT_NATIVE_FUNCTION_H_
#define TFRT_HOST_CONTEXT_NATIVE_FUNCTION_H_

#include <vector>

#include "llvm/ADT/StringMap.h"
#include "tfrt/host_context/function.h"
#include "tfrt/support/mutex.h"

namespace tfrt {

class HostContext;

// Use C-style arrays to pass arugments and results for ABI compatiblity. The
// callee is responsible for allocating results if there are any. All results
// should be set to errors to signal that an error occurred.
//
// TODO(tfrt-devs): Consider removing AsyncValue from this signature so
// that external functions do not have to depend on AsyncValue.
using NativeCallable = void (*)(AsyncValue* const* arguments, int num_arguments,
                                RCReference<AsyncValue>* results,
                                int num_results, HostContext* host);

class NativeFunctionRegistry {
 public:
  static NativeFunctionRegistry& GetGlobalRegistry() {
    static auto* const global_registry = new NativeFunctionRegistry();
    return *global_registry;
  }

  void Add(string_view name, NativeCallable callable) {
    mutex_lock lock(m_);
    auto r = callables_.try_emplace(name, callable);
    assert(r.second && "native function already exists");
    (void)r;
  }

  NativeCallable Get(string_view name) const {
    mutex_lock lock(m_);
    return callables_.lookup(name);
  }

 private:
  mutable mutex m_;
  llvm::StringMap<NativeCallable> callables_ TFRT_GUARDED_BY(m_);
};

// NativeFunction provides an interface for BEF Executor to run native
// executables.
class NativeFunction : public Function {
 public:
  NativeFunction(string_view name, ArrayRef<TypeName> argument_types,
                 ArrayRef<TypeName> result_types, NativeCallable callable)
      : Function(name, FunctionKind::kNativeFunction, argument_types,
                 result_types),
        callable_(callable) {}

  void Execute(const ExecutionContext& exec_ctx,
               ArrayRef<AsyncValue*> arguments,
               MutableArrayRef<RCReference<AsyncValue>> results) const final;

  void ExecuteAsync(
      const ExecutionContext& exec_ctx,
      std::vector<RCReference<AsyncValue>> arguments,
      MutableArrayRef<RCReference<AsyncValue>> results) const override {
    assert(false && "not implemented");
  }

  // Do nothing with reference counting as a native function should be always
  // available.
  void AddRef() const final {}
  void DropRef() const final {}

 private:
  NativeCallable callable_;
};

}  // namespace tfrt

#endif  // TFRT_HOST_CONTEXT_NATIVE_FUNCTION_H_
