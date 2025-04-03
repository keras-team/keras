//===- PassDetail.h - MLIR Pass details -------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#ifndef MLIR_PASS_PASSDETAIL_H_
#define MLIR_PASS_PASSDETAIL_H_

#include "mlir/IR/Action.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/FormatVariadic.h"

namespace mlir {
namespace detail {

//===----------------------------------------------------------------------===//
// OpToOpPassAdaptor
//===----------------------------------------------------------------------===//

/// An adaptor pass used to run operation passes over nested operations.
class OpToOpPassAdaptor
    : public PassWrapper<OpToOpPassAdaptor, OperationPass<>> {
public:
  OpToOpPassAdaptor(OpPassManager &&mgr);
  OpToOpPassAdaptor(const OpToOpPassAdaptor &rhs) = default;

  /// Run the held pipeline over all operations.
  void runOnOperation(bool verifyPasses);
  void runOnOperation() override;

  /// Try to merge the current pass adaptor into 'rhs'. This will try to append
  /// the pass managers of this adaptor into those within `rhs`, or return
  /// failure if merging isn't possible. The main situation in which merging is
  /// not possible is if one of the adaptors has an `any` pipeline that is not
  /// compatible with a pass manager in the other adaptor. For example, if this
  /// adaptor has a `func.func` pipeline and `rhs` has an `any` pipeline that
  /// operates on FunctionOpInterface. In this situation the pipelines have a
  /// conflict (they both want to run on the same operations), so we can't
  /// merge.
  LogicalResult tryMergeInto(MLIRContext *ctx, OpToOpPassAdaptor &rhs);

  /// Returns the pass managers held by this adaptor.
  MutableArrayRef<OpPassManager> getPassManagers() { return mgrs; }

  /// Populate the set of dependent dialects for the passes in the current
  /// adaptor.
  void getDependentDialects(DialectRegistry &dialects) const override;

  /// Return the async pass managers held by this parallel adaptor.
  MutableArrayRef<SmallVector<OpPassManager, 1>> getParallelPassManagers() {
    return asyncExecutors;
  }

  /// Returns the adaptor pass name.
  std::string getAdaptorName();

private:
  /// Run this pass adaptor synchronously.
  void runOnOperationImpl(bool verifyPasses);

  /// Run this pass adaptor asynchronously.
  void runOnOperationAsyncImpl(bool verifyPasses);

  /// Run the given operation and analysis manager on a single pass.
  /// `parentInitGeneration` is the initialization generation of the parent pass
  /// manager, and is used to initialize any dynamic pass pipelines run by the
  /// given pass.
  static LogicalResult run(Pass *pass, Operation *op, AnalysisManager am,
                           bool verifyPasses, unsigned parentInitGeneration);

  /// Run the given operation and analysis manager on a provided op pass
  /// manager. `parentInitGeneration` is the initialization generation of the
  /// parent pass manager, and is used to initialize any dynamic pass pipelines
  /// run by the given passes.
  static LogicalResult runPipeline(
      OpPassManager &pm, Operation *op, AnalysisManager am, bool verifyPasses,
      unsigned parentInitGeneration, PassInstrumentor *instrumentor = nullptr,
      const PassInstrumentation::PipelineParentInfo *parentInfo = nullptr);

  /// A set of adaptors to run.
  SmallVector<OpPassManager, 1> mgrs;

  /// A set of executors, cloned from the main executor, that run asynchronously
  /// on different threads. This is used when threading is enabled.
  SmallVector<SmallVector<OpPassManager, 1>, 8> asyncExecutors;

  // For accessing "runPipeline".
  friend class mlir::PassManager;
};

//===----------------------------------------------------------------------===//
// PassCrashReproducerGenerator
//===----------------------------------------------------------------------===//

class PassCrashReproducerGenerator {
public:
  PassCrashReproducerGenerator(ReproducerStreamFactory &streamFactory,
                               bool localReproducer);
  ~PassCrashReproducerGenerator();

  /// Initialize the generator in preparation for reproducer generation. The
  /// generator should be reinitialized before each run of the pass manager.
  void initialize(iterator_range<PassManager::pass_iterator> passes,
                  Operation *op, bool pmFlagVerifyPasses);
  /// Finalize the current run of the generator, generating any necessary
  /// reproducers if the provided execution result is a failure.
  void finalize(Operation *rootOp, LogicalResult executionResult);

  /// Prepare a new reproducer for the given pass, operating on `op`.
  void prepareReproducerFor(Pass *pass, Operation *op);

  /// Prepare a new reproducer for the given passes, operating on `op`.
  void prepareReproducerFor(iterator_range<PassManager::pass_iterator> passes,
                            Operation *op);

  /// Remove the last recorded reproducer anchored at the given pass and
  /// operation.
  void removeLastReproducerFor(Pass *pass, Operation *op);

private:
  struct Impl;

  /// The internal implementation of the crash reproducer.
  std::unique_ptr<Impl> impl;
};

} // namespace detail
} // namespace mlir
#endif // MLIR_PASS_PASSDETAIL_H_
