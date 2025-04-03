/* Copyright 2023 The StableHLO Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#ifndef STABLEHLO_REFERENCE_CONFIGURATION_H
#define STABLEHLO_REFERENCE_CONFIGURATION_H

#include "llvm/Support/Error.h"
#include "stablehlo/reference/Process.h"
#include "stablehlo/reference/Scope.h"

namespace mlir {
namespace stablehlo {

/// Base interpreter fallback callback functor to run when no registered kernels
/// are found for a given StableHLO operation. See InterpreterApi for default
/// implementation.
class InterpreterFallback {
 public:
  /// Custom op kernels for any user specified ops not found in the StableHLO
  /// op dialect or StableHLO interpreter dialect.
  virtual llvm::Error operator()(Operation &op, Scope &scope, Process *process);

  virtual ~InterpreterFallback() = default;
};

struct InterpreterConfiguration {
  InterpreterConfiguration()
      : fallback(std::make_unique<InterpreterFallback>()) {}

  /// If specified, the directory to which StableHLO interpreter tensors will
  /// be serialized to disk.
  std::string probeInstrumentationDir = "";

  /// Use the specified named function as the main entrypoint into a module.
  /// Defaults to `main` for modules with multiple functions. If a module only
  /// contains 1 function and the default `main` value is used, the singular
  /// function will be used as the entrypoint (irrespective of a function name
  /// match).
  std::string mainFunction = "main";

  /// If specified, use the callback to run on ops which do not have a
  /// registered kernel.
  std::unique_ptr<InterpreterFallback> fallback;
};

}  // namespace stablehlo
}  // namespace mlir

#endif  // STABLEHLO_REFERENCE_CONFIGURATION_H
