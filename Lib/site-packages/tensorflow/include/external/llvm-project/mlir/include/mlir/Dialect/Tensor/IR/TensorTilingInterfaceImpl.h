//===- TensorTilingOpInterfaceImpl.h - ------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements Tiling interface for TensorOps with ExternalModel.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_TENSOR_IR_TENSORTILINGINTERFACEIMPL_H_
#define MLIR_DIALECT_TENSOR_IR_TENSORTILINGINTERFACEIMPL_H_

#include "mlir/IR/Dialect.h"

namespace mlir {

struct TilingResult;

namespace tensor {

class PadOp;

/// Bubbles up a slice of this pad by taking the slice first and then performing
/// the padding. `offsets` and `strides` specifies each dimension's start offset
/// and size for the slice. The slice has unit strides along all dimensions.
///
/// Specifically, this function converts:
/// ```
/// %0 = tensor.pad %source low[...] high[...] { linalg.yield %cst }
/// %1 = <extract-slice> %0 offsets=[...], sizes[...]
/// ```
/// into
/// ```
/// %0 = tensor.extract_slice %source ...
/// %0 = tensor.pad %0 low[...] high[...] { linalg.yield %cst }
/// ```
///
/// If `generateZeroSliceGuard` is true, the generated IR will contain logic
/// to guard against the case that we might take a zero-sized slice from the
/// original source. For such cases, we `tensor.generate` to generate the
/// full tensor.
FailureOr<TilingResult> bubbleUpPadSlice(OpBuilder &b, tensor::PadOp padOp,
                                         ArrayRef<OpFoldResult> offsets,
                                         ArrayRef<OpFoldResult> sizes,
                                         bool generateZeroSliceGuard = true);

/// Registers external models for Tiling interface for tensor ops.
/// Currently, it registers:
///
/// * TilingInterface for `tensor.pad`, `tensor.pack`, and `tensor.unpack`.
///
/// Unfortunately, a "normal" internal registration is not possible at the
/// moment, because of the dependency of the interface implementation for these
/// ops on `affine.apply` and Affine dialect already depends on TensorOps. In
/// order to break the cyclic dependency (TensorOps->AffineOps->TensorOps) the
/// implementation is moved to a separate library.
void registerTilingInterfaceExternalModels(mlir::DialectRegistry &registry);

/// Similar to the above registeration, but it is only for `tensor.pack` and
/// `tensor.unpack` ops.
void registerTilingInterfaceExternalModelsForPackUnPackOps(
    DialectRegistry &registry);

} // namespace tensor
} // namespace mlir

#endif // MLIR_DIALECT_TENSOR_IR_TENSORTILINGINTERFACEIMPL_H_
