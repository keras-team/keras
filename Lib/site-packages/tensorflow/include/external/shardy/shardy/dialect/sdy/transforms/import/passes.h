/* Copyright 2024 The Shardy Authors.

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

#ifndef SHARDY_DIALECT_SDY_TRANSFORMS_IMPORT_PASSES_H_
#define SHARDY_DIALECT_SDY_TRANSFORMS_IMPORT_PASSES_H_

// IWYU pragma: begin_keep

#include <memory>

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LLVM.h"
#include "shardy/dialect/sdy/ir/dialect.h"

// IWYU pragma: end_keep

namespace mlir {
namespace sdy {

#define GEN_PASS_DECL
#define GEN_PASS_REGISTRATION
#include "shardy/dialect/sdy/transforms/import/passes.h.inc"

// Adds a sequence of import passes needed as a pre-processing step for SDY
// propagation.
void addImportPipeline(OpPassManager& pm, StringRef dumpDirectory = "");

// Register the sdy-import-pipeline.
void registerImportPipeline();

}  // namespace sdy
}  // namespace mlir

#endif  // SHARDY_DIALECT_SDY_TRANSFORMS_IMPORT_PASSES_H_
