/* Copyright 2022 The StableHLO Authors.

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

#ifndef STABLEHLO_TRANSFORMS_STABLEHLO_REFINE_SHAPES_H
#define STABLEHLO_TRANSFORMS_STABLEHLO_REFINE_SHAPES_H

#include "llvm/ADT/SmallVector.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Support/LogicalResult.h"
#include "stablehlo/dialect/Base.h"

namespace mlir {
namespace stablehlo {

// Gets a FuncOp that --stablehlo-refine-shapes will run on.
// Returns a nullptr and emits appropriate errors if such a function cannot
// be obtained from the module.
func::FuncOp getStablehloRefineShapesTarget(ModuleOp module);

// Refines the values using the given types.
// Tricky implementation details:
//   1) Need to support partial shape refinements, e.g. if just a single
//      dimension size out of an entire tensor type got refined. This is done
//      via inferMostSpecificType.
//   2) Need to signal propagation of the refined shapes across the
//      StableHLO program. Different callers of this function have different
//      propagation needs, so this function doesn't signal anything on its own
//      and leaves that to the callers.
LogicalResult refineValues(PatternRewriter& rewriter, Operation* op,
                           ValueRange values, TypeRange types);

// Refines the return types of the given operation using the given types.
// This function also signals PatternRewriter that it needs to visit all the
// users of this op if any updates to its results have happened during execution
// of the function.
LogicalResult refineReturnTypes(PatternRewriter& rewriter, Operation* op,
                                ArrayRef<Type> types);

// Refines the return types of the given operation using the given types.
// Tricky implementation details:
//   1) `types` can include non-shaped types. If there are tuple types,
//      then they are first flattened into non-tuple types using in-order
//      traversal, and only then we apply the refinements. If there are other
//      types, then the corresponding refinements must be completely empty.
//   2) Encodings are not supported. In principle, TypeExtensions should be
//      supportable, but this needs careful thinking through. Given that no one
//      asked for support for bounded dynamism in this pass yet, this is left
//      for future work.
// This function also signals PatternRewriter that it needs to visit all the
// users of this op if any updates to its results have happened during execution
// of the function.
LogicalResult refineReturnTypes(PatternRewriter& rewriter, Operation* op,
                                ArrayRef<ShapedTypeComponents> refinements);

// Refines the return type of the given operation using the given shape.
// This function also signals PatternRewriter that it needs to visit all the
// users of this op if any updates to its results have happened during execution
// of the function.
template <typename OpType>
LogicalResult refineReturnShape(PatternRewriter& rewriter, OpType op,
                                ArrayRef<int64_t> shape) {
  return refineReturnTypes(rewriter, op, ShapedTypeComponents(shape));
}

// Refines the return type of the given operation using the given shape.
// This function also signals PatternRewriter that it needs to visit all the
// users of this op if any updates to its results have happened during execution
// of the function.
template <typename OpType>
LogicalResult refineReturnShape(PatternRewriter& rewriter, OpType op,
                                Value shapeValue) {
  // At the moment, we only support refining return types using fully static
  // shape values which serves the current use cases well.
  // Support for partially static shape values is left for future work.
  SmallVector<int64_t> shape;
  if (failed(hlo::matchInts(shapeValue, shape)))
    return rewriter.notifyMatchFailure(op, "expected constant output shape");
  return refineReturnShape(rewriter, op, shape);
}

// Custom call used to buffer operands for shape refinement
// This is a temporary artifact that is introduced by StablehloRefineArguments
// and is washed away during StablehloRefineShapes.
constexpr StringRef kCustomCallOperandBarrierTarget =
    "stablehlo.shape_refinement_operand_wrapper";

}  // namespace stablehlo
}  // namespace mlir

#endif  // STABLEHLO_TRANSFORMS_STABLEHLO_REFINE_SHAPES_H
