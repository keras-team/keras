//===- CallInterfaces.h - Call Interfaces for MLIR --------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains the definitions of the call interfaces defined in
// `CallInterfaces.td`.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_INTERFACES_CALLINTERFACES_H
#define MLIR_INTERFACES_CALLINTERFACES_H

#include "mlir/IR/SymbolTable.h"
#include "llvm/ADT/PointerUnion.h"

namespace mlir {
/// A callable is either a symbol, or an SSA value, that is referenced by a
/// call-like operation. This represents the destination of the call.
struct CallInterfaceCallable : public PointerUnion<SymbolRefAttr, Value> {
  using PointerUnion<SymbolRefAttr, Value>::PointerUnion;
};

class CallOpInterface;

namespace call_interface_impl {

/// Resolve the callable operation for given callee to a CallableOpInterface, or
/// nullptr if a valid callable was not resolved.  `symbolTable` is an optional
/// parameter that will allow for using a cached symbol table for symbol lookups
/// instead of performing an O(N) scan.
Operation *resolveCallable(CallOpInterface call,
                           SymbolTableCollection *symbolTable = nullptr);

} // namespace call_interface_impl

} // namespace mlir

namespace llvm {

// Allow llvm::cast style functions.
template <typename To>
struct CastInfo<To, mlir::CallInterfaceCallable>
    : public CastInfo<To, mlir::CallInterfaceCallable::PointerUnion> {};

template <typename To>
struct CastInfo<To, const mlir::CallInterfaceCallable>
    : public CastInfo<To, const mlir::CallInterfaceCallable::PointerUnion> {};

} // namespace llvm

/// Include the generated interface declarations.
#include "mlir/Interfaces/CallInterfaces.h.inc"

#endif // MLIR_INTERFACES_CALLINTERFACES_H
