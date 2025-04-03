//===-- OpenMPClauseOperands.h ----------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the structures defining MLIR operands associated with each
// OpenMP clause, and structures grouping the appropriate operands for each
// construct.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_OPENMP_OPENMPCLAUSEOPERANDS_H_
#define MLIR_DIALECT_OPENMP_OPENMPCLAUSEOPERANDS_H_

#include "mlir/IR/BuiltinAttributes.h"
#include "llvm/ADT/SmallVector.h"

#include "mlir/Dialect/OpenMP/OpenMPOpsEnums.h.inc"

#define GET_ATTRDEF_CLASSES
#include "mlir/Dialect/OpenMP/OpenMPOpsAttributes.h.inc"

#include "mlir/Dialect/OpenMP/OpenMPClauseOps.h.inc"

namespace mlir {
namespace omp {

//===----------------------------------------------------------------------===//
// Extra clause operand structures.
//===----------------------------------------------------------------------===//

struct DeviceTypeClauseOps {
  /// The default capture type.
  DeclareTargetDeviceType deviceType = DeclareTargetDeviceType::any;
};

//===----------------------------------------------------------------------===//
// Extra operation operand structures.
//===----------------------------------------------------------------------===//

// TODO: Add `indirect` clause.
using DeclareTargetOperands = detail::Clauses<DeviceTypeClauseOps>;

/// omp.target_enter_data, omp.target_exit_data and omp.target_update take the
/// same clauses, so we give the structure to be shared by all of them a
/// representative name.
using TargetEnterExitUpdateDataOperands = TargetEnterDataOperands;

} // namespace omp
} // namespace mlir

#endif // MLIR_DIALECT_OPENMP_OPENMPCLAUSEOPERANDS_H_
