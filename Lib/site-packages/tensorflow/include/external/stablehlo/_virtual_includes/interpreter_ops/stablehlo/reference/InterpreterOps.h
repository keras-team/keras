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

#ifndef STABLEHLO_REFERENCE_INTERPRETEROPS_H
#define STABLEHLO_REFERENCE_INTERPRETEROPS_H

#include <queue>

#include "llvm/Support/Error.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Support/LLVM.h"
#include "stablehlo/reference/Value.h"

namespace mlir {
namespace stablehlo {
namespace interpreter {

class InterpreterDialect : public Dialect {
 public:
  explicit InterpreterDialect(MLIRContext *context);
  static StringRef getDialectNamespace() { return "interpreter"; }
};

SmallVector<InterpreterValue> evalRunParallelOp(
    ArrayRef<InterpreterValue> inputs, std::queue<StringAttr> &infeed,
    SmallVector<SmallVector<StringAttr>> programs, SymbolTable &symbolTable);

llvm::Error evalProbeOp(InterpreterValue input, StringRef probeId,
                        StringRef probeOutputDir,
                        int64_t serializedProbeFileId);

}  // namespace interpreter
}  // namespace stablehlo
}  // namespace mlir

#define GET_OP_CLASSES
#include "stablehlo/reference/InterpreterOps.h.inc"

#endif  // STABLEHLO_REFERENCE_INTERPRETEROPS_H
