//===- PassInstrumentation.h ------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_PASS_PASSINSTRUMENTATION_H_
#define MLIR_PASS_PASSINSTRUMENTATION_H_

#include "mlir/Support/LLVM.h"
#include "mlir/Support/TypeID.h"
#include <optional>

namespace mlir {
class OperationName;
class Operation;
class Pass;

namespace detail {
struct PassInstrumentorImpl;
} // namespace detail

/// PassInstrumentation provides several entry points into the pass manager
/// infrastructure. Instrumentations should be added directly to a PassManager
/// before running a pipeline.
class PassInstrumentation {
public:
  /// This struct represents information related to the parent pass of pipeline.
  /// It includes information that allows for effectively linking pipelines that
  /// run on different threads.
  struct PipelineParentInfo {
    /// The thread of the parent pass that the current pipeline was spawned
    /// from. Note: This is acquired from llvm::get_threadid().
    uint64_t parentThreadID;

    /// The pass that spawned this pipeline.
    Pass *parentPass;
  };

  virtual ~PassInstrumentation() = 0;

  /// A callback to run before a pass pipeline is executed. This function takes
  /// the name of the operation type being operated on, or std::nullopt if the
  /// pipeline is op-agnostic, and information related to the parent that
  /// spawned this pipeline.
  virtual void runBeforePipeline(std::optional<OperationName> name,
                                 const PipelineParentInfo &parentInfo);

  /// A callback to run after a pass pipeline has executed. This function takes
  /// the name of the operation type being operated on, or std::nullopt if the
  /// pipeline is op-agnostic, and information related to the parent that
  /// spawned this pipeline.
  virtual void runAfterPipeline(std::optional<OperationName> name,
                                const PipelineParentInfo &parentInfo);

  /// A callback to run before a pass is executed. This function takes a pointer
  /// to the pass to be executed, as well as the current operation being
  /// operated on.
  virtual void runBeforePass(Pass *pass, Operation *op) {}

  /// A callback to run after a pass is successfully executed. This function
  /// takes a pointer to the pass to be executed, as well as the current
  /// operation being operated on.
  virtual void runAfterPass(Pass *pass, Operation *op) {}

  /// A callback to run when a pass execution fails. This function takes a
  /// pointer to the pass that was being executed, as well as the current
  /// operation being operated on. Note that the operation may be in an invalid
  /// state.
  virtual void runAfterPassFailed(Pass *pass, Operation *op) {}

  /// A callback to run before an analysis is computed. This function takes the
  /// name of the analysis to be computed, its TypeID, as well as the
  /// current operation being analyzed.
  virtual void runBeforeAnalysis(StringRef name, TypeID id, Operation *op) {}

  /// A callback to run before an analysis is computed. This function takes the
  /// name of the analysis that was computed, its TypeID, as well as the
  /// current operation being analyzed.
  virtual void runAfterAnalysis(StringRef name, TypeID id, Operation *op) {}
};

/// This class holds a collection of PassInstrumentation objects, and invokes
/// their respective call backs.
class PassInstrumentor {
public:
  PassInstrumentor();
  PassInstrumentor(PassInstrumentor &&) = delete;
  PassInstrumentor(const PassInstrumentor &) = delete;
  ~PassInstrumentor();

  /// See PassInstrumentation::runBeforePipeline for details.
  void
  runBeforePipeline(std::optional<OperationName> name,
                    const PassInstrumentation::PipelineParentInfo &parentInfo);

  /// See PassInstrumentation::runAfterPipeline for details.
  void
  runAfterPipeline(std::optional<OperationName> name,
                   const PassInstrumentation::PipelineParentInfo &parentInfo);

  /// See PassInstrumentation::runBeforePass for details.
  void runBeforePass(Pass *pass, Operation *op);

  /// See PassInstrumentation::runAfterPass for details.
  void runAfterPass(Pass *pass, Operation *op);

  /// See PassInstrumentation::runAfterPassFailed for details.
  void runAfterPassFailed(Pass *pass, Operation *op);

  /// See PassInstrumentation::runBeforeAnalysis for details.
  void runBeforeAnalysis(StringRef name, TypeID id, Operation *op);

  /// See PassInstrumentation::runAfterAnalysis for details.
  void runAfterAnalysis(StringRef name, TypeID id, Operation *op);

  /// Add the given instrumentation to the collection.
  void addInstrumentation(std::unique_ptr<PassInstrumentation> pi);

private:
  std::unique_ptr<detail::PassInstrumentorImpl> impl;
};

} // namespace mlir

namespace llvm {
template <>
struct DenseMapInfo<mlir::PassInstrumentation::PipelineParentInfo> {
  using T = mlir::PassInstrumentation::PipelineParentInfo;
  using PairInfo = DenseMapInfo<std::pair<uint64_t, void *>>;

  static T getEmptyKey() {
    auto pair = PairInfo::getEmptyKey();
    return {pair.first, reinterpret_cast<mlir::Pass *>(pair.second)};
  }
  static T getTombstoneKey() {
    auto pair = PairInfo::getTombstoneKey();
    return {pair.first, reinterpret_cast<mlir::Pass *>(pair.second)};
  }
  static unsigned getHashValue(T val) {
    return PairInfo::getHashValue({val.parentThreadID, val.parentPass});
  }
  static bool isEqual(T lhs, T rhs) {
    return lhs.parentThreadID == rhs.parentThreadID &&
           lhs.parentPass == rhs.parentPass;
  }
};
} // namespace llvm

#endif // MLIR_PASS_PASSINSTRUMENTATION_H_
