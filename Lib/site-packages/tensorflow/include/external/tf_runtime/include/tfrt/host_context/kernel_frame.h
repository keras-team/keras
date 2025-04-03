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

// Information for kernel invocation
//
// This file implements AsyncKernelFrame which captures argument, result, and
// other related information provided to kernels on kernel invocation.

#ifndef TFRT_HOST_CONTEXT_KERNEL_CONTEXT_H_
#define TFRT_HOST_CONTEXT_KERNEL_CONTEXT_H_

#include <string>
#include <utility>

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "tfrt/host_context/async_value.h"
#include "tfrt/host_context/attribute_utils.h"
#include "tfrt/host_context/diagnostic.h"
#include "tfrt/host_context/execution_context.h"
#include "tfrt/host_context/host_context.h"
#include "tfrt/host_context/location.h"
#include "tfrt/support/forward_decls.h"
#include "tfrt/support/string_util.h"

namespace tfrt {

class Function;

// AsyncKernelFrame captures the states associated with a kernel invocation,
// including the input arguments, attributes, result values, location and host
// context. AsyncKernelFrame is constructed by the kernel caller (currently only
// BEFExecutor) using the KernelFrameBuilder subclass. The kernel implementation
// is passed a pointer to a AsyncKernelFrame object for them to access the
// inputs and attributes, and return result values.
//
// The result AsyncValue pointers are not initialized when a kernel is called.
// The Kernel implementation is responsible for creating AsyncValue objects and
// setting the result AsyncValue pointers.
class AsyncKernelFrame {
 public:
  explicit AsyncKernelFrame(ExecutionContext exec_ctx)
      : exec_ctx_{std::move(exec_ctx)} {}

  AsyncKernelFrame(const AsyncKernelFrame& other) : exec_ctx_(other.exec_ctx_) {
    AssignFields(other);
  }
  AsyncKernelFrame& operator=(const AsyncKernelFrame& other) {
    exec_ctx_ = other.exec_ctx_;
    AssignFields(other);
    return *this;
  }
  AsyncKernelFrame(AsyncKernelFrame&& other)
      : exec_ctx_(std::move(other.exec_ctx_)) {
    AssignFields(std::move(other));
  }
  AsyncKernelFrame& operator=(AsyncKernelFrame&& other) {
    exec_ctx_ = std::move(other.exec_ctx_);
    AssignFields(std::move(other));
    return *this;
  }

  ~AsyncKernelFrame() { ResetArguments(); }

  const ExecutionContext& GetExecutionContext() const { return exec_ctx_; }
  HostContext* GetHostContext() const { return exec_ctx_.host(); }

  // Get the location.
  Location GetLocation() const { return exec_ctx_.location(); }

  ArrayRef<uint8_t> GetAttributeSection() const { return attribute_section_; }

  // Get the number of arguments.
  int GetNumArgs() const { return arguments_.size(); }

  // Get the argument at the given index as type T.
  template <typename T>
  T& GetArgAt(int index) const {
    return GetArgAt(index)->get<T>();
  }

  // Get the argument at the given index as AsyncValue*.
  AsyncValue* GetArgAt(int index) const {
    assert(index < GetNumArgs());
    return arguments_[index];
  }

  // Get all arguments.
  ArrayRef<AsyncValue*> GetArguments() const { return arguments_; }

  // Get the attribute at the given index.
  const void* GetAttribute(int index) const {
    return attribute_section_.data() + attribute_offsets_[index];
  }

  // Get the number of attributes.
  int GetNumAttributes() const { return attribute_offsets_.size(); }

  // Get the function at the given index.
  const Function* GetFunction(int index) const {
    return functions_[function_indices_[index]].get();
  }

  // Get the number of functions.
  int GetNumFunctions() const { return function_indices_.size(); }

  // Get the attribute at the given index as type T.
  // TODO(jingdong): Disable const char*.
  template <typename T>
  Attribute<T> GetAttributeAt(int index) const {
    assert(index < GetNumAttributes());
    return Attribute<T>(GetAttribute(index));
  }

  AggregateAttr GetAggregateAttr(int index) const {
    assert(index < GetNumAttributes());
    return AggregateAttr(GetAttribute(index));
  }

  DenseAttr GetDenseAttr(int index) const {
    assert(index < GetNumAttributes());
    return DenseAttr(GetAttribute(index));
  }

  ShapeAttr GetShapeAttr(int index) const {
    assert(index < GetNumAttributes());
    return ShapeAttr(GetAttribute(index));
  }

  ArrayAttr GetArrayAttr(int index) const {
    assert(index < GetNumAttributes());
    return ArrayAttr(GetAttribute(index));
  }

  // Get the array attribute at the given index as type T.
  template <typename T>
  ArrayAttribute<T> GetArrayAttributeAt(int index) const {
    assert(index < GetNumAttributes());
    return ArrayAttribute<T>(GetAttribute(index));
  }

  // Get array attribute as a string. Equivalent to
  // GetArrayAttributeAt<char>, except that this returns StringRef instead
  // of ArrayRef<char>.
  StringAttribute GetStringAttribute(int index) const {
    return StringAttribute(GetAttribute(index));
  }

  CompilationUnitAttribute GetCompilationUnitAttribute(int index) const {
    return CompilationUnitAttribute(GetAttribute(index));
  }

  // Get the number of results.
  int GetNumResults() const { return results_.size(); }

  // Emplace construct the result at index 0.
  template <typename T, typename... Args>
  void EmplaceResult(Args&&... args) {
    EmplaceResultAt<T>(0, std::forward<Args>(args)...);
  }

  // Emplace construct the result at given index.
  template <typename T, typename... Args>
  void EmplaceResultAt(int index, Args&&... args) {
    SetResultAt(index,
                MakeAvailableAsyncValueRef<T>(std::forward<Args>(args)...));
  }

  // Allocate an AsyncValue with uninitialized payload as the result at the
  // index 0 and return the allocated AsyncValue.
  template <typename T>
  AsyncValueRef<T> AllocateResult() {
    return AllocateResultAt<T>(0);
  }

  // Allocate an AsyncValue with uninitialized payload as the result at the
  // given index and return the allocated AsyncValue.
  template <typename T>
  AsyncValueRef<T> AllocateResultAt(int index) {
    auto result = MakeUnconstructedAsyncValueRef<T>();
    SetResultAt(index, result.CopyRef());
    return result;
  }

  // Set the result at the given index with the given AsyncValue.
  void SetResultAt(int index, RCReference<AsyncValue> value) {
    assert(index < results_.size() && "Invalid result index");
    RCReference<AsyncValue>& result = results_[index];
    assert(!result && "Result is not nullptr");
    result = std::move(value);
  }

  template <typename T>
  void SetResultAt(int index, AsyncValueRef<T> value) {
    SetResultAt(index, value.ReleaseRCRef());
  }

  // Allocate an AsyncValue with uninitialized payload as the result at the
  // given index and return the allocated AsyncValue.
  RCReference<IndirectAsyncValue> AllocateIndirectResultAt(int index) {
    auto result = MakeIndirectAsyncValue();
    SetResultAt(index, result);
    return result;
  }

  // Get all results as an immutable ArrayRef
  ArrayRef<RCReference<AsyncValue>> GetResults() const { return results_; }

  // Get all results as MutableArrayRef.
  MutableArrayRef<RCReference<AsyncValue>> GetResults() { return results_; }

  // Example usage:
  //
  // kernel_handler.ReportError("This is an error message");
  // int i = 2;
  // TensorShape shape = ...
  // kernel_handler.ReportError("Error: i is ", i, ", shape is ", shape);
  template <typename... Args>
  void ReportError(Args&&... args) {
    ReportError(string_view(StrCat(std::forward<Args>(args)...)));
  }
  // Report error and set any unset results with an error AsyncValue.
  void ReportError(string_view msg);

  template <typename... Args>
  RCReference<AsyncValue> EmitError(Args&&... args) {
    return EmitError(string_view(StrCat(std::forward<Args>(args)...)));
  }

  // Emit an AsyncValue that contains an error using the kernel location.
  // For consistency, the error message should start with a lower case letter
  // and not end with a period.
  RCReference<AsyncValue> EmitError(string_view msg) {
    return EmitErrorAsync(exec_ctx_, absl::InternalError(msg));
  }

  // Assert the size of arguments, attributes, and results are as expected.
  void AssertArity(int num_arguments, int num_attributes,
                   int num_results) const;

  // Clear arguments.
  void ResetArguments() {
    for (auto* arg : arguments_) arg->DropRef();
    arguments_.clear();
  }

 protected:
  // Assign each member except ExecutionContext.
  void AssignFields(const AsyncKernelFrame& other);
  void AssignFields(AsyncKernelFrame&& other);

  // AsyncValues of `arguments_` are owned by AsyncKernelFrame.
  //
  // TODO(tfrt-devs): Use RCReference<AsyncValue> instead of AsyncValue* so
  // the ownership is clearer.
  llvm::SmallVector<AsyncValue*, 8> arguments_;
  llvm::SmallVector<RCReference<AsyncValue>, 8> results_;

  ArrayRef<uint8_t> attribute_section_;
  ArrayRef<uint32_t> attribute_offsets_;
  ArrayRef<uint32_t> function_indices_;
  ArrayRef<std::unique_ptr<Function>> functions_;
  ExecutionContext exec_ctx_;
};

inline void AsyncKernelFrame::AssignFields(const AsyncKernelFrame& other) {
  for (auto* arg : arguments_) arg->DropRef();
  arguments_ = other.arguments_;
  for (auto* arg : arguments_) arg->AddRef();

  assert(results_.empty());
  results_.reserve(other.results_.size());
  for (auto& result : other.results_) results_.push_back(result);

  attribute_section_ = other.attribute_section_;
  attribute_offsets_ = other.attribute_offsets_;
  function_indices_ = other.function_indices_;
  functions_ = other.functions_;
}

inline void AsyncKernelFrame::AssignFields(AsyncKernelFrame&& other) {
  for (auto* arg : arguments_) arg->DropRef();
  arguments_ = std::move(other.arguments_);
  results_ = std::move(other.results_);

  attribute_section_ = other.attribute_section_;
  attribute_offsets_ = other.attribute_offsets_;
  function_indices_ = other.function_indices_;
  functions_ = other.functions_;
}

// KernelFrameBuilder is used by the kernel caller to construct a
// AsyncKernelFrame object without exposing the builder methods to the kernel
// implementation.
//
// This class requires that the client performs the following in order:
// 1. Add args (using AddArg()) and set the number of results (using
//    SetNumResults()),
// 2. call the kernel,
// 3. reset arguments (using ResetArguments()) and release results (using
//    ReleaseResultAt()).
class KernelFrameBuilder : public AsyncKernelFrame {
 public:
  explicit KernelFrameBuilder(ExecutionContext exec_ctx)
      : AsyncKernelFrame{std::move(exec_ctx)} {}

  // Get result AsyncValue at the given index.
  const RCReference<AsyncValue>& GetResultAt(int index) const {
    return results_[index];
  }

  RCReference<AsyncValue> ReleaseResultAt(int index) {
    return std::move(results_[index]);
  }

  // TODO(tfrt-devs): Consider keeping BEFFile* in the kernel frame directly
  // instead of keeping individual fields.
  void SetAttributeSection(ArrayRef<uint8_t> attribute_section) {
    attribute_section_ = attribute_section;
  }
  void SetFunctions(ArrayRef<std::unique_ptr<Function>> functions) {
    functions_ = functions;
  }

  // Add a new argument to the AsyncKernelFrame.
  void AddArg(RCReference<AsyncValue> async_value) {
    arguments_.push_back(async_value.release());
  }

  // Add all attributes to the AsyncKernelFrame.
  void SetAttributes(ArrayRef<uint32_t> attribute_offsets) {
    attribute_offsets_ = attribute_offsets;
  }

  // Add all functions to the AsyncKernelFrame.
  void SetFunctionIndices(ArrayRef<uint32_t> function_indices) {
    function_indices_ = function_indices;
  }

  // Set the number of results expected.
  void SetNumResults(size_t n) { results_.resize(n); }

  // Set the location.
  void SetLocation(const Location& location) {
    exec_ctx_.set_location(location);
  }
};

// Implementation details

inline void AsyncKernelFrame::AssertArity(int num_arguments, int num_attributes,
                                          int num_results) const {
  assert(arguments_.size() == num_arguments);
  assert(GetNumAttributes() == num_attributes);
  assert(GetNumResults() == num_results);
}

}  // namespace tfrt

#endif  // TFRT_HOST_CONTEXT_KERNEL_CONTEXT_H_
