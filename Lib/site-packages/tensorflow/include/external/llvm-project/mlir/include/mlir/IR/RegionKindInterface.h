//===- RegionKindInterface.h - Region Kind Interfaces -----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains the definitions of the infer op interfaces defined in
// `RegionKindInterface.td`.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_IR_REGIONKINDINTERFACE_H_
#define MLIR_IR_REGIONKINDINTERFACE_H_

#include "mlir/IR/OpDefinition.h"

namespace mlir {

/// The kinds of regions contained in an operation. SSACFG regions
/// require the SSA-Dominance property to hold. Graph regions do not
/// require SSA-Dominance. If a registered operation does not implement
/// RegionKindInterface, then any regions it contains are assumed to be
/// SSACFG regions.
enum class RegionKind {
  SSACFG,
  Graph,
};

namespace OpTrait {
/// A trait that specifies that an operation only defines graph regions.
template <typename ConcreteType>
class HasOnlyGraphRegion : public TraitBase<ConcreteType, HasOnlyGraphRegion> {
public:
  static RegionKind getRegionKind(unsigned index) { return RegionKind::Graph; }
  static bool hasSSADominance(unsigned index) { return false; }
};
} // namespace OpTrait

/// Return "true" if the given region may have SSA dominance. This function also
/// returns "true" in case the owner op is an unregistered op or an op that does
/// not implement the RegionKindInterface.
bool mayHaveSSADominance(Region &region);

/// Return "true" if the given region may be a graph region without SSA
/// dominance. This function returns "true" in case the owner op is an
/// unregistered op. It returns "false" if it is a registered op that does not
/// implement the RegionKindInterface.
bool mayBeGraphRegion(Region &region);

} // namespace mlir

#include "mlir/IR/RegionKindInterface.h.inc"

#endif // MLIR_IR_REGIONKINDINTERFACE_H_
