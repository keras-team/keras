//===- LocationSnapshot.h - Location Snapshot Utilities ---------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This header file several utility methods for snapshotting the current IR to
// produce new debug locations.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_TRANSFORMS_LOCATIONSNAPSHOT_H
#define MLIR_TRANSFORMS_LOCATIONSNAPSHOT_H

#include "mlir/Support/LLVM.h"
#include "llvm/ADT/StringRef.h"

#include <memory>

namespace mlir {
class Location;
class Operation;
class OpPrintingFlags;
class Pass;

#define GEN_PASS_DECL_LOCATIONSNAPSHOT
#include "mlir/Transforms/Passes.h.inc"

/// This function generates new locations from the given IR by snapshotting the
/// IR to the given stream, and using the printed locations within that stream.
/// The generated locations replace the current operation locations.
void generateLocationsFromIR(raw_ostream &os, StringRef fileName, Operation *op,
                             OpPrintingFlags flags);
/// This function generates new locations from the given IR by snapshotting the
/// IR to the given file, and using the printed locations within that file. If
/// `filename` is empty, a temporary file is generated instead.
LogicalResult generateLocationsFromIR(StringRef fileName, Operation *op,
                                      OpPrintingFlags flags);

/// This function generates new locations from the given IR by snapshotting the
/// IR to the given stream, and using the printed locations within that stream.
/// The generated locations are represented as a NameLoc with the given tag as
/// the name, and then fused with the existing locations.
void generateLocationsFromIR(raw_ostream &os, StringRef fileName, StringRef tag,
                             Operation *op, OpPrintingFlags flags);
/// This function generates new locations from the given IR by snapshotting the
/// IR to the given file, and using the printed locations within that file. If
/// `filename` is empty, a temporary file is generated instead.
LogicalResult generateLocationsFromIR(StringRef fileName, StringRef tag,
                                      Operation *op, OpPrintingFlags flags);

/// Create a pass to generate new locations by snapshotting the IR to the given
/// file, and using the printed locations within that file. If `filename` is
/// empty, a temporary file is generated instead. If a 'tag' is non-empty, the
/// generated locations are represented as a NameLoc with the given tag as the
/// name, and then fused with the existing locations. Otherwise, the existing
/// locations are replaced.
std::unique_ptr<Pass> createLocationSnapshotPass(OpPrintingFlags flags,
                                                 StringRef fileName = "",
                                                 StringRef tag = "");
/// Overload utilizing pass options for initialization.
std::unique_ptr<Pass> createLocationSnapshotPass();

} // namespace mlir

#endif // MLIR_TRANSFORMS_LOCATIONSNAPSHOT_H
