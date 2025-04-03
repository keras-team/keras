//===- AttrKindDetail.h - AttrKind conversion details -----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef ATTRKINDDETAIL_H_
#define ATTRKINDDETAIL_H_

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "llvm/IR/Attributes.h"

namespace mlir {
namespace LLVM {
namespace detail {

/// Returns a list of pairs that each hold a mapping from LLVM attribute kinds
/// to their corresponding string name in LLVM IR dialect.
static llvm::ArrayRef<std::pair<llvm::Attribute::AttrKind, llvm::StringRef>>
getAttrKindToNameMapping() {
  using ElemTy = std::pair<llvm::Attribute::AttrKind, llvm::StringRef>;
  // Mapping from llvm attribute kinds to their corresponding MLIR name.
  static const llvm::SmallVector<ElemTy> kindNamePairs = {
      {llvm::Attribute::AttrKind::Alignment, LLVMDialect::getAlignAttrName()},
      {llvm::Attribute::AttrKind::AllocAlign,
       LLVMDialect::getAllocAlignAttrName()},
      {llvm::Attribute::AttrKind::AllocatedPointer,
       LLVMDialect::getAllocatedPointerAttrName()},
      {llvm::Attribute::AttrKind::ByVal, LLVMDialect::getByValAttrName()},
      {llvm::Attribute::AttrKind::ByRef, LLVMDialect::getByRefAttrName()},
      {llvm::Attribute::AttrKind::NoUndef, LLVMDialect::getNoUndefAttrName()},
      {llvm::Attribute::AttrKind::Dereferenceable,
       LLVMDialect::getDereferenceableAttrName()},
      {llvm::Attribute::AttrKind::DereferenceableOrNull,
       LLVMDialect::getDereferenceableOrNullAttrName()},
      {llvm::Attribute::AttrKind::InAlloca, LLVMDialect::getInAllocaAttrName()},
      {llvm::Attribute::AttrKind::InReg, LLVMDialect::getInRegAttrName()},
      {llvm::Attribute::AttrKind::Nest, LLVMDialect::getNestAttrName()},
      {llvm::Attribute::AttrKind::NoAlias, LLVMDialect::getNoAliasAttrName()},
      {llvm::Attribute::AttrKind::NoCapture,
       LLVMDialect::getNoCaptureAttrName()},
      {llvm::Attribute::AttrKind::NoFree, LLVMDialect::getNoFreeAttrName()},
      {llvm::Attribute::AttrKind::NonNull, LLVMDialect::getNonNullAttrName()},
      {llvm::Attribute::AttrKind::Preallocated,
       LLVMDialect::getPreallocatedAttrName()},
      {llvm::Attribute::AttrKind::ReadOnly, LLVMDialect::getReadonlyAttrName()},
      {llvm::Attribute::AttrKind::ReadNone, LLVMDialect::getReadnoneAttrName()},
      {llvm::Attribute::AttrKind::Returned, LLVMDialect::getReturnedAttrName()},
      {llvm::Attribute::AttrKind::SExt, LLVMDialect::getSExtAttrName()},
      {llvm::Attribute::AttrKind::StackAlignment,
       LLVMDialect::getStackAlignmentAttrName()},
      {llvm::Attribute::AttrKind::StructRet,
       LLVMDialect::getStructRetAttrName()},
      {llvm::Attribute::AttrKind::WriteOnly,
       LLVMDialect::getWriteOnlyAttrName()},
      {llvm::Attribute::AttrKind::ZExt, LLVMDialect::getZExtAttrName()}};
  return kindNamePairs;
}

/// Returns a dense map from LLVM attribute name to their kind in LLVM IR
/// dialect.
[[maybe_unused]] static llvm::DenseMap<llvm::StringRef,
                                       llvm::Attribute::AttrKind>
getAttrNameToKindMapping() {
  static auto attrNameToKindMapping = []() {
    llvm::DenseMap<llvm::StringRef, llvm::Attribute::AttrKind> nameKindMap;
    for (auto kindNamePair : getAttrKindToNameMapping()) {
      nameKindMap.insert({kindNamePair.second, kindNamePair.first});
    }
    return nameKindMap;
  }();
  return attrNameToKindMapping;
}

} // namespace detail
} // namespace LLVM
} // namespace mlir

#endif // ATTRKINDDETAIL_H_
