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

// Data for sync kernel invocation
//
// This file implements SyncKernelFrame which captures argument, result, and
// other related information provided to synchronous kernels on kernel
// invocation.

#ifndef TFRT_HOST_CONTEXT_SYNC_KERNEL_FRAME_H_
#define TFRT_HOST_CONTEXT_SYNC_KERNEL_FRAME_H_

#include <utility>

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Error.h"
#include "tfrt/host_context/attribute_utils.h"
#include "tfrt/host_context/execution_context.h"
#include "tfrt/host_context/host_context.h"
#include "tfrt/host_context/location.h"
#include "tfrt/host_context/value.h"
#include "tfrt/support/forward_decls.h"
#include "tfrt/support/string_util.h"

namespace tfrt {

// SyncKernelFrame captures the states associated with a kernel invocation,
// including the input arguments, attributes, result values, and the execution
// context. SyncKernelFrame is constructed by the kernel caller (currently only
// BEFInterpreter) using the SyncKernelFrameBuilder subclass. The kernel
// implementation is passed a pointer to a SyncKernelFrame object for them to
// access the inputs and attributes, and return result values.
class SyncKernelFrame {
 public:
  const ExecutionContext& GetExecutionContext() const { return exec_ctx_; }
  HostContext* GetHostContext() const { return exec_ctx_.host(); }

  // Get the location.
  Location GetLocation() const { return exec_ctx_.location(); }

  ArrayRef<Value*> GetRegisters() const { return registers_; }

  // Get the number of arguments.
  int GetNumArgs() const { return argument_indices_.size(); }

  // Get the argument at the given index as type T.
  template <typename T>
  T& GetArgAt(int index) const {
    return GetArgAt(index)->get<T>();
  }

  // Get the argument at the given index as Value*.
  Value* GetArgAt(int index) const {
    assert(index < GetNumArgs());
    return registers_[argument_indices_[index]];
  }

  // Get all arguments.
  ArrayRef<uint32_t> GetArguments() const { return argument_indices_; }

  // Get the number of attributes.
  int GetNumAttributes() const { return attributes_.size(); }

  const void* GetAttributeAt(int index) const {
    assert(index < GetNumAttributes());
    return attributes_[index];
  }

  // Get the attribute at the given index as type T.
  // TODO(jingdong): Disable const char*.
  template <typename T>
  Attribute<T> GetAttributeAt(int index) const {
    return Attribute<T>(GetAttributeAt(index));
  }

  AggregateAttr GetAggregateAttr(int index) const {
    assert(index < GetNumAttributes());
    return AggregateAttr(GetAttributeAt(index));
  }

  // Get the array attribute at the given index as type T.
  template <typename T>
  ArrayAttribute<T> GetArrayAttributeAt(int index) const {
    assert(index < GetNumAttributes());
    return ArrayAttribute<T>(GetAttributeAt(index));
  }

  // Get array attribute as a string. Equivalent to
  // GetArrayAttributeAt<char>, except that this returns StringRef instead
  // of ArrayRef<char>.
  StringAttribute GetStringAttribute(int index) const {
    return StringAttribute(GetAttributeAt(index));
  }

  // Get the number of results.
  int GetNumResults() const { return result_indices_.size(); }

  // Emplace construct the result at given index.
  template <typename T, typename... Args>
  void EmplaceResultAt(int index, Args&&... args) {
    assert(index < GetNumResults() && "Invalid result index");
    Value* result = GetResultAt(index);
    assert(!result->HasValue() && "Result value is non-empty.");
    result->emplace<T>(std::forward<Args>(args)...);
  }

  // Get result at the given index.
  Value* GetResultAt(int index) const {
    assert(index < result_indices_.size());
    return registers_[result_indices_[index]];
  }

  // Report error from the kernel execution.
  void SetError(Error error) {
    assert(!error_ && "Error is already set.");
    error_ = std::move(error);
  }

  // This should only be called once.
  Error TakeError() { return std::move(error_); }

 protected:
  // `exec_ctx` must out-live the SyncKernelFrame object, as SyncKernelFrame
  // only keeps a reference to `exec_ctx`.
  SyncKernelFrame(ArrayRef<Value*> registers, const ExecutionContext& exec_ctx)
      : registers_{registers}, exec_ctx_{exec_ctx} {}

  // These are indices into `registers_`.
  ArrayRef<uint32_t> argument_indices_;
  ArrayRef<const void*> attributes_;
  // These are indices into `registers_`.
  ArrayRef<uint32_t> result_indices_;

  const ArrayRef<Value*> registers_;

  const ExecutionContext& exec_ctx_;
  Error error_ = Error::success();
};

// SyncKernelFrameBuilder is used by the kernel caller to construct a
// SyncKernelFrame object without exposing the builder methods to the kernel
// implementation.
//
// As an optimization, SyncKernelFrame stores arguments, attributes, and results
// in a single SmallVector. As a result, to initialize a SyncKernelFrame, this
// class requires that the client performs the following actions in order:
// 1. Adds the arguments (using AddArg())
// 2. Add the attributes (using AddAttribute())
// 3. Add the results (using AddResult())
class SyncKernelFrameBuilder : public SyncKernelFrame {
 public:
  // `exec_ctx` must out-live the SyncKernelFrameBuilder object, as
  // SyncKernelFrameBuilder only keeps a reference to `exec_ctx`.
  explicit SyncKernelFrameBuilder(ArrayRef<Value*> registers,
                                  const ExecutionContext& exec_ctx)
      : SyncKernelFrame{registers, exec_ctx} {}

  void SetArguments(ArrayRef<uint32_t> argument_indices) {
    argument_indices_ = argument_indices;
  }
  void SetAttributes(ArrayRef<const void*> attributes) {
    attributes_ = attributes;
  }
  void SetResults(ArrayRef<uint32_t> result_indices) {
    result_indices_ = result_indices;
  }
};

// Implementation details

}  // namespace tfrt

#endif  // TFRT_HOST_CONTEXT_KERNEL_FRAME_H_
