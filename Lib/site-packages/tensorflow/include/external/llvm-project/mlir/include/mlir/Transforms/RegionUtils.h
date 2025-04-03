//===- RegionUtils.h - Region-related transformation utilities --*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_TRANSFORMS_REGIONUTILS_H_
#define MLIR_TRANSFORMS_REGIONUTILS_H_

#include "mlir/IR/Region.h"
#include "mlir/IR/Value.h"

#include "llvm/ADT/SetVector.h"

namespace mlir {
class RewriterBase;

/// Check if all values in the provided range are defined above the `limit`
/// region.  That is, if they are defined in a region that is a proper ancestor
/// of `limit`.
template <typename Range>
bool areValuesDefinedAbove(Range values, Region &limit) {
  for (Value v : values)
    if (!v.getParentRegion()->isProperAncestor(&limit))
      return false;
  return true;
}

/// Replace all uses of `orig` within the given region with `replacement`.
void replaceAllUsesInRegionWith(Value orig, Value replacement, Region &region);

/// Calls `callback` for each use of a value within `region` or its descendants
/// that was defined at the ancestors of the `limit`.
void visitUsedValuesDefinedAbove(Region &region, Region &limit,
                                 function_ref<void(OpOperand *)> callback);

/// Calls `callback` for each use of a value within any of the regions provided
/// that was defined in one of the ancestors.
void visitUsedValuesDefinedAbove(MutableArrayRef<Region> regions,
                                 function_ref<void(OpOperand *)> callback);

/// Fill `values` with a list of values defined at the ancestors of the `limit`
/// region and used within `region` or its descendants.
void getUsedValuesDefinedAbove(Region &region, Region &limit,
                               SetVector<Value> &values);

/// Fill `values` with a list of values used within any of the regions provided
/// but defined in one of the ancestors.
void getUsedValuesDefinedAbove(MutableArrayRef<Region> regions,
                               SetVector<Value> &values);

/// Make a region isolated from above
/// - Capture the values that are defined above the region and used within it.
/// - Append to the entry block arguments that represent the captured values
/// (one per captured value).
/// - Replace all uses within the region of the captured values with the
///   newly added arguments.
/// - `cloneOperationIntoRegion` is a callback that allows caller to specify
///   if the operation defining an `OpOperand` needs to be cloned into the
///   region. Then the operands of this operation become part of the captured
///   values set (unless the operations that define the operands themeselves
///   are to be cloned). The cloned operations are added to the entry block
///   of the region.
/// Return the set of captured values for the operation.
SmallVector<Value> makeRegionIsolatedFromAbove(
    RewriterBase &rewriter, Region &region,
    llvm::function_ref<bool(Operation *)> cloneOperationIntoRegion =
        [](Operation *) { return false; });

/// Run a set of structural simplifications over the given regions. This
/// includes transformations like unreachable block elimination, dead argument
/// elimination, as well as some other DCE. This function returns success if any
/// of the regions were simplified, failure otherwise. The provided rewriter is
/// used to notify callers of operation and block deletion.
/// Structurally similar blocks will be merged if the `mergeBlock` argument is
/// true. Note this can lead to merged blocks with extra arguments.
LogicalResult simplifyRegions(RewriterBase &rewriter,
                              MutableArrayRef<Region> regions,
                              bool mergeBlocks = true);

/// Erase the unreachable blocks within the provided regions. Returns success
/// if any blocks were erased, failure otherwise.
LogicalResult eraseUnreachableBlocks(RewriterBase &rewriter,
                                     MutableArrayRef<Region> regions);

/// This function returns success if any operations or arguments were deleted,
/// failure otherwise.
LogicalResult runRegionDCE(RewriterBase &rewriter,
                           MutableArrayRef<Region> regions);

} // namespace mlir

#endif // MLIR_TRANSFORMS_REGIONUTILS_H_
