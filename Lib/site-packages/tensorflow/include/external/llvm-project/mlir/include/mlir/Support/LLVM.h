//===- LLVM.h - Import and forward declare core LLVM types ------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file forward declares and imports various common LLVM datatypes that
// MLIR wants to use unqualified.
//
// Note that most of these are forward declared and then imported into the MLIR
// namespace with using decls, rather than being #included.  This is because we
// want clients to explicitly #include the files they need.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_SUPPORT_LLVM_H
#define MLIR_SUPPORT_LLVM_H

// We include this header because it cannot be practically forward
// declared, and are effectively language features.
#include "llvm/Support/Casting.h"

// We include this header because large portions of mlir would have to include
// it anyway.
#include "llvm/Support/LogicalResult.h"

// Workaround for clang-5 (PR41549)
#if defined(__clang_major__)
#if __clang_major__ <= 5
#include "llvm/ADT/DenseMapInfo.h"
#include "llvm/ADT/SmallVector.h"
#endif
#endif

// Forward declarations.
namespace llvm {
// String types
template <unsigned N>
class SmallString;
class StringRef;
class StringLiteral;
class Twine;

// Containers.
template <typename T>
class ArrayRef;
class BitVector;
namespace detail {
template <typename KeyT, typename ValueT>
struct DenseMapPair;
} // namespace detail
template <typename KeyT, typename ValueT, typename KeyInfoT, typename BucketT>
class DenseMap;
template <typename T, typename Enable>
struct DenseMapInfo;
template <typename ValueT, typename ValueInfoT>
class DenseSet;
class MallocAllocator;
template <typename T>
class MutableArrayRef;
template <typename... PT>
class PointerUnion;
template <typename T, typename Vector, typename Set, unsigned N>
class SetVector;
template <typename T, unsigned N>
class SmallPtrSet;
template <typename T>
class SmallPtrSetImpl;
template <typename T, unsigned N>
class SmallVector;
template <typename T>
class SmallVectorImpl;
template <typename AllocatorTy>
class StringSet;
template <typename T, typename R>
class StringSwitch;
template <typename T>
class TinyPtrVector;
template <typename T, typename ResultT>
class TypeSwitch;

// Other common classes.
class APInt;
class DynamicAPInt;
class APSInt;
class APFloat;
template <typename Fn>
class function_ref;
template <typename IteratorT>
class iterator_range;
class raw_ostream;
class SMLoc;
class SMRange;
} // namespace llvm

namespace mlir {
// Casting operators.
using llvm::cast;
using llvm::cast_if_present;
using llvm::cast_or_null;
using llvm::dyn_cast;
using llvm::dyn_cast_if_present;
using llvm::dyn_cast_or_null;
using llvm::isa;
using llvm::isa_and_nonnull;
using llvm::isa_and_present;

// String types
using llvm::SmallString;
using llvm::StringLiteral;
using llvm::StringRef;
using llvm::Twine;

// Container Related types
//
// Containers.
using llvm::ArrayRef;
using llvm::BitVector;
template <typename T, typename Enable = void>
using DenseMapInfo = llvm::DenseMapInfo<T, Enable>;
template <typename KeyT, typename ValueT,
          typename KeyInfoT = DenseMapInfo<KeyT>,
          typename BucketT = llvm::detail::DenseMapPair<KeyT, ValueT>>
using DenseMap = llvm::DenseMap<KeyT, ValueT, KeyInfoT, BucketT>;
template <typename ValueT, typename ValueInfoT = DenseMapInfo<ValueT>>
using DenseSet = llvm::DenseSet<ValueT, ValueInfoT>;
template <typename T, typename Vector = llvm::SmallVector<T, 0>,
          typename Set = DenseSet<T>, unsigned N = 0>
using SetVector = llvm::SetVector<T, Vector, Set, N>;
template <typename AllocatorTy = llvm::MallocAllocator>
using StringSet = llvm::StringSet<AllocatorTy>;
using llvm::MutableArrayRef;
using llvm::PointerUnion;
using llvm::SmallPtrSet;
using llvm::SmallPtrSetImpl;
using llvm::SmallVector;
using llvm::SmallVectorImpl;
template <typename T, typename R = T>
using StringSwitch = llvm::StringSwitch<T, R>;
using llvm::TinyPtrVector;
template <typename T, typename ResultT = void>
using TypeSwitch = llvm::TypeSwitch<T, ResultT>;

// Other common classes.
using llvm::APFloat;
using llvm::APInt;
using llvm::APSInt;
using llvm::DynamicAPInt;
template <typename Fn>
using function_ref = llvm::function_ref<Fn>;
using llvm::iterator_range;
using llvm::raw_ostream;
using llvm::SMLoc;
using llvm::SMRange;

// LogicalResult.
using llvm::failed;
using llvm::failure;
using llvm::FailureOr;
using llvm::LogicalResult;
using llvm::ParseResult;
using llvm::succeeded;
using llvm::success;
} // namespace mlir

#endif // MLIR_SUPPORT_LLVM_H
