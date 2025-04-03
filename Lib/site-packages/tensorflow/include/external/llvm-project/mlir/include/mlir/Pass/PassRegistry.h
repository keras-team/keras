//===- PassRegistry.h - Pass Registration Utilities -------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains utilities for registering information about compiler
// passes.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_PASS_PASSREGISTRY_H_
#define MLIR_PASS_PASSREGISTRY_H_

#include "mlir/Pass/PassOptions.h"
#include "mlir/Support/TypeID.h"
#include <functional>
#include <utility>
#include <optional>

namespace mlir {
class OpPassManager;
class ParserConfig;
class Pass;
class PassManager;

namespace detail {
class PassOptions;
} // namespace detail

/// A registry function that adds passes to the given pass manager. This should
/// also parse options and return success() if parsing succeeded.
/// `errorHandler` is a functor used to emit errors during parsing.
/// parameter corresponds to the raw location within the pipeline string. This
/// should always return failure.
using PassRegistryFunction = std::function<LogicalResult(
    OpPassManager &, StringRef options,
    function_ref<LogicalResult(const Twine &)> errorHandler)>;
using PassAllocatorFunction = std::function<std::unique_ptr<Pass>()>;

//===----------------------------------------------------------------------===//
// PassRegistry
//===----------------------------------------------------------------------===//

/// Prints the passes that were previously registered and stored in passRegistry
void printRegisteredPasses();

/// Structure to group information about a passes and pass pipelines (argument
/// to invoke via mlir-opt, description, pass pipeline builder).
class PassRegistryEntry {
public:
  /// Adds this pass registry entry to the given pass manager. `options` is
  /// an opaque string that will be parsed by the builder. The success of
  /// parsing will be returned.
  LogicalResult
  addToPipeline(OpPassManager &pm, StringRef options,
                function_ref<LogicalResult(const Twine &)> errorHandler) const {
    assert(builder &&
           "cannot call addToPipeline on PassRegistryEntry without builder");
    return builder(pm, options, errorHandler);
  }

  /// Returns the command line option that may be passed to 'mlir-opt' that will
  /// cause this pass to run or null if there is no such argument.
  StringRef getPassArgument() const { return arg; }

  /// Returns a description for the pass, this never returns null.
  StringRef getPassDescription() const { return description; }

  /// Print the help information for this pass. This includes the argument,
  /// description, and any pass options. `descIndent` is the indent that the
  /// descriptions should be aligned.
  void printHelpStr(size_t indent, size_t descIndent) const;

  /// Return the maximum width required when printing the options of this entry.
  size_t getOptionWidth() const;

protected:
  PassRegistryEntry(
      StringRef arg, StringRef description, const PassRegistryFunction &builder,
      std::function<void(function_ref<void(const detail::PassOptions &)>)>
          optHandler)
      : arg(arg), description(description), builder(builder),
        optHandler(std::move(optHandler)) {}

private:
  /// The argument with which to invoke the pass via mlir-opt.
  std::string arg;

  /// Description of the pass.
  std::string description;

  /// Function to register this entry to a pass manager pipeline.
  PassRegistryFunction builder;

  /// Function to invoke a handler for a pass options instance.
  std::function<void(function_ref<void(const detail::PassOptions &)>)>
      optHandler;
};

/// A structure to represent the information of a registered pass pipeline.
class PassPipelineInfo : public PassRegistryEntry {
public:
  PassPipelineInfo(
      StringRef arg, StringRef description, const PassRegistryFunction &builder,
      std::function<void(function_ref<void(const detail::PassOptions &)>)>
          optHandler)
      : PassRegistryEntry(arg, description, builder, std::move(optHandler)) {}

  /// Returns the pass pipeline info for the specified pass pipeline or null if
  /// unknown.
  static const PassPipelineInfo *lookup(StringRef pipelineArg);
};

/// A structure to represent the information for a derived pass class.
class PassInfo : public PassRegistryEntry {
public:
  /// PassInfo constructor should not be invoked directly, instead use
  /// PassRegistration or registerPass.
  PassInfo(StringRef arg, StringRef description,
           const PassAllocatorFunction &allocator);

  /// Returns the pass info for the specified pass class or null if unknown.
  static const PassInfo *lookup(StringRef passArg);
};

//===----------------------------------------------------------------------===//
// PassRegistration
//===----------------------------------------------------------------------===//

/// Register a specific dialect pipeline registry function with the system,
/// typically used through the PassPipelineRegistration template.
void registerPassPipeline(
    StringRef arg, StringRef description, const PassRegistryFunction &function,
    std::function<void(function_ref<void(const detail::PassOptions &)>)>
        optHandler);

/// Register a specific dialect pass allocator function with the system,
/// typically used through the PassRegistration template.
void registerPass(const PassAllocatorFunction &function);

/// PassRegistration provides a global initializer that registers a Pass
/// allocation routine for a concrete pass instance. The argument is
/// optional and provides a callback to construct a pass that does not have
/// a default constructor.
///
/// Usage:
///
///   /// At namespace scope.
///   static PassRegistration<MyPass> reg;
///
template <typename ConcretePass>
struct PassRegistration {
  PassRegistration(const PassAllocatorFunction &constructor) {
    registerPass(constructor);
  }
  PassRegistration()
      : PassRegistration([] { return std::make_unique<ConcretePass>(); }) {}
};

/// PassPipelineRegistration provides a global initializer that registers a Pass
/// pipeline builder routine.
///
/// Usage:
///
///   // At namespace scope.
///   void pipelineBuilder(OpPassManager &pm) {
///      pm.addPass(new MyPass());
///      pm.addPass(new MyOtherPass());
///   }
///
///   static PassPipelineRegistration Unused("unused", "Unused pass",
///                                          pipelineBuilder);
template <typename Options = EmptyPipelineOptions>
struct PassPipelineRegistration {
  PassPipelineRegistration(
      StringRef arg, StringRef description,
      std::function<void(OpPassManager &, const Options &options)> builder) {
    registerPassPipeline(
        arg, description,
        [builder](OpPassManager &pm, StringRef optionsStr,
                  function_ref<LogicalResult(const Twine &)> errorHandler) {
          Options options;
          if (failed(options.parseFromString(optionsStr)))
            return failure();
          builder(pm, options);
          return success();
        },
        [](function_ref<void(const detail::PassOptions &)> optHandler) {
          optHandler(Options());
        });
  }
};

/// Convenience specialization of PassPipelineRegistration for EmptyPassOptions
/// that does not pass an empty options struct to the pass builder function.
template <>
struct PassPipelineRegistration<EmptyPipelineOptions> {
  PassPipelineRegistration(
      StringRef arg, StringRef description,
      const std::function<void(OpPassManager &)> &builder) {
    registerPassPipeline(
        arg, description,
        [builder](OpPassManager &pm, StringRef optionsStr,
                  function_ref<LogicalResult(const Twine &)> errorHandler) {
          if (!optionsStr.empty())
            return failure();
          builder(pm);
          return success();
        },
        [](function_ref<void(const detail::PassOptions &)>) {});
  }
};

/// Parse the textual representation of a pass pipeline, adding the result to
/// 'pm' on success. Returns failure if the given pipeline was invalid.
/// 'errorStream' is the output stream used to emit errors found during parsing.
LogicalResult parsePassPipeline(StringRef pipeline, OpPassManager &pm,
                                raw_ostream &errorStream = llvm::errs());

/// Parse the given textual representation of a pass pipeline, and return the
/// parsed pipeline on success. The given pipeline string should be wrapped with
/// the desired type of operation to root the created operation, i.e.
/// `builtin.module(cse)` over `cse`. Returns failure if the given pipeline was
/// invalid. 'errorStream' is the output stream used to emit errors found during
/// parsing.
FailureOr<OpPassManager>
parsePassPipeline(StringRef pipeline, raw_ostream &errorStream = llvm::errs());

//===----------------------------------------------------------------------===//
// PassPipelineCLParser
//===----------------------------------------------------------------------===//

namespace detail {
struct PassPipelineCLParserImpl;
} // namespace detail

/// This class implements a command-line parser for MLIR passes. It registers a
/// cl option with a given argument and description. This parser will register
/// options for each of the passes and pipelines that have been registered with
/// the pass registry; Meaning that `-cse` will refer to the CSE pass in MLIR.
/// It also registers an argument, `pass-pipeline`, that supports parsing a
/// textual description of a pipeline. This option is mutually exclusive with
/// the individual pass options.
class PassPipelineCLParser {
public:
  /// Construct a pass pipeline parser with the given command line description.
  /// Optionally registers an alias for the `pass-pipeline` option.
  PassPipelineCLParser(StringRef arg, StringRef description);
  PassPipelineCLParser(StringRef arg, StringRef description, StringRef alias);
  ~PassPipelineCLParser();

  /// Returns true if this parser contains any valid options to add.
  bool hasAnyOccurrences() const;

  /// Returns true if the given pass registry entry was registered at the
  /// top-level of the parser, i.e. not within an explicit textual pipeline.
  bool contains(const PassRegistryEntry *entry) const;

  /// Adds the passes defined by this parser entry to the given pass manager.
  /// Returns failure() if the pass could not be properly constructed due
  /// to options parsing.
  LogicalResult
  addToPipeline(OpPassManager &pm,
                function_ref<LogicalResult(const Twine &)> errorHandler) const;

private:
  std::unique_ptr<detail::PassPipelineCLParserImpl> impl;

  llvm::cl::opt<std::string> passPipeline;
  std::optional<llvm::cl::alias> passPipelineAlias;
};

/// This class implements a command-line parser specifically for MLIR pass
/// names. It registers a cl option with a given argument and description that
/// accepts a comma delimited list of pass names.
class PassNameCLParser {
public:
  /// Construct a parser with the given command line description.
  PassNameCLParser(StringRef arg, StringRef description);
  ~PassNameCLParser();

  /// Returns true if this parser contains any valid options to add.
  bool hasAnyOccurrences() const;

  /// Returns true if the given pass registry entry was registered at the
  /// top-level of the parser, i.e. not within an explicit textual pipeline.
  bool contains(const PassRegistryEntry *entry) const;

private:
  std::unique_ptr<detail::PassPipelineCLParserImpl> impl;
};

//===----------------------------------------------------------------------===//
// Pass Reproducer
//===----------------------------------------------------------------------===//

struct PassReproducerOptions {
  /// Attach an assembly resource parser to 'config' that collects the MLIR
  /// reproducer configuration into this instance.
  void attachResourceParser(ParserConfig &config);

  /// Apply the reproducer options to 'pm' and its context.
  LogicalResult apply(PassManager &pm) const;

private:
  std::optional<std::string> pipeline;
  std::optional<bool> verifyEach;
  std::optional<bool> disableThreading;
};

} // namespace mlir

#endif // MLIR_PASS_PASSREGISTRY_H_
