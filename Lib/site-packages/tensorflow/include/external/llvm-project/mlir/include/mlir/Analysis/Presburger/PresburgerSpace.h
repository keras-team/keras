//===- PresburgerSpace.h - MLIR PresburgerSpace Class -----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Classes representing space information like number of variables and kind of
// variables.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_ANALYSIS_PRESBURGER_PRESBURGERSPACE_H
#define MLIR_ANALYSIS_PRESBURGER_PRESBURGERSPACE_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/PointerLikeTypeTraits.h"
#include "llvm/Support/TypeName.h"
#include "llvm/Support/raw_ostream.h"

namespace mlir {
namespace presburger {
using llvm::ArrayRef;
using llvm::SmallVector;

/// Kind of variable. Implementation wise SetDims are treated as Range
/// vars, and spaces with no distinction between dimension vars are treated
/// as relations with zero domain vars.
enum class VarKind { Symbol, Local, Domain, Range, SetDim = Range };

/// An Identifier stores a pointer to an object, such as a Value or an
/// Operation. Identifiers are intended to be attached to a variable in a
/// PresburgerSpace and can be used to check if two variables correspond to the
/// same object.
///
/// Take for example the following code:
///
/// for i = 0 to 100
///   for j = 0 to 100
///     S0: A[j] = 0
///   for k = 0 to 100
///     S1: A[k] = 1
///
/// If we represent the space of iteration variables surrounding S0, S1 we have:
/// space(S0): {d0, d1}
/// space(S1): {d0, d1}
///
/// Since the variables are in different spaces, without an identifier, there
/// is no way to distinguish if the variables in the two spaces correspond to
/// different SSA values in the program. So, we attach an Identifier
/// corresponding to the loop iteration variable to them. Now,
///
/// space(S0) = {d0(id = i), d1(id = j)}
/// space(S1) = {d0(id = i), d1(id = k)}.
///
/// Using the identifier, we can check that the first iteration variable in
/// both the spaces correspond to the same variable in the program, while they
/// are different for second iteration variable.
///
/// The equality of Identifiers is checked by comparing the stored pointers.
/// Checking equality asserts that the type of the equal identifiers is same.
/// Identifiers storing null pointers are treated as having no attachment and
/// are considered unequal to any other identifier, including other identifiers
/// with no attachments.
///
/// The type of the pointer stored must have an `llvm::PointerLikeTypeTraits`
/// specialization.
class Identifier {
public:
  Identifier() = default;

  // Create an identifier from a pointer.
  template <typename T>
  explicit Identifier(T value)
      : value(llvm::PointerLikeTypeTraits<T>::getAsVoidPointer(value)) {
#ifdef LLVM_ENABLE_ABI_BREAKING_CHECKS
    idType = llvm::getTypeName<T>();
#endif
  }

  /// Get the value of the identifier casted to type `T`. `T` here should match
  /// the type of the identifier used to create it.
  template <typename T>
  T getValue() const {
#ifdef LLVM_ENABLE_ABI_BREAKING_CHECKS
    assert(llvm::getTypeName<T>() == idType &&
           "Identifier was initialized with a different type than the one used "
           "to retrieve it.");
#endif
    return llvm::PointerLikeTypeTraits<T>::getFromVoidPointer(value);
  }

  bool hasValue() const { return value != nullptr; }

  /// Check if the two identifiers are equal. Null identifiers are considered
  /// not equal. Asserts if two identifiers are equal but their types are not.
  bool isEqual(const Identifier &other) const;

  bool operator==(const Identifier &other) const { return isEqual(other); }
  bool operator!=(const Identifier &other) const { return !isEqual(other); }

  void print(llvm::raw_ostream &os) const;
  void dump() const;

private:
  /// The value of the identifier.
  void *value = nullptr;

#ifdef LLVM_ENABLE_ABI_BREAKING_CHECKS
  /// TypeID of the identifiers in space. This should be used in asserts only.
  llvm::StringRef idType;
#endif
};

/// PresburgerSpace is the space of all possible values of a tuple of integer
/// valued variables/variables. Each variable has one of the three types:
///
/// Dimension: Ordinary variables over which the space is represented.
///
/// Symbol: Symbol variables correspond to fixed but unknown values.
/// Mathematically, a space with symbolic variables is like a
/// family of spaces indexed by the symbolic variables.
///
/// Local: Local variables correspond to existentially quantified variables.
/// For example, consider the space: `(x, exists q)` where x is a dimension
/// variable and q is a local variable. Let us put the constraints:
///       `1 <= x <= 7, x = 2q`
/// on this space to get the set:
///       `(x) : (exists q : q <= x <= 7, x = 2q)`.
/// An assignment to symbolic and dimension variables is valid if there
/// exists some assignment to the local variable `q` satisfying these
/// constraints. For this example, the set is equivalent to {2, 4, 6}.
/// Mathematically, existential quantification can be thought of as the result
/// of projection. In this example, `q` is existentially quantified. This can be
/// thought of as the result of projecting out `q` from the previous example,
/// i.e. we obtained {2, 4, 6} by projecting out the second dimension from
/// {(2, 1), (4, 2), (6, 2)}.
///
/// Dimension variables are further divided into Domain and Range variables
/// to support building relations.
///
/// Variables are stored in the following order:
///       [Domain, Range, Symbols, Locals]
///
/// A space with no distinction between types of dimension variables can
/// be implemented as a space with zero domain. VarKind::SetDim should be used
/// to refer to dimensions in such spaces.
///
/// Compatibility of two spaces implies that number of variables of each kind
/// other than Locals are equal. Equality of two spaces implies that number of
/// variables of each kind are equal.
///
/// PresburgerSpace optionally also supports attaching an Identifier with each
/// non-local variable in the space. This is disabled by default. `resetIds` is
/// used to enable/reset these identifiers. The user can identify each variable
/// in the space as corresponding to some Identifier. Some example use cases
/// are described in the `Identifier` documentation above. The type attached to
/// the Identifier can be different for different variables in the space.
class PresburgerSpace {
public:
  static PresburgerSpace getRelationSpace(unsigned numDomain = 0,
                                          unsigned numRange = 0,
                                          unsigned numSymbols = 0,
                                          unsigned numLocals = 0) {
    return PresburgerSpace(numDomain, numRange, numSymbols, numLocals);
  }

  static PresburgerSpace getSetSpace(unsigned numDims = 0,
                                     unsigned numSymbols = 0,
                                     unsigned numLocals = 0) {
    return PresburgerSpace(/*numDomain=*/0, /*numRange=*/numDims, numSymbols,
                           numLocals);
  }

  /// Get the domain/range space of this space. The returned space is a set
  /// space.
  PresburgerSpace getDomainSpace() const;
  PresburgerSpace getRangeSpace() const;

  /// Get the space without local variables.
  PresburgerSpace getSpaceWithoutLocals() const;

  unsigned getNumDomainVars() const { return numDomain; }
  unsigned getNumRangeVars() const { return numRange; }
  unsigned getNumSetDimVars() const { return numRange; }
  unsigned getNumSymbolVars() const { return numSymbols; }
  unsigned getNumLocalVars() const { return numLocals; }

  unsigned getNumDimVars() const { return numDomain + numRange; }
  unsigned getNumDimAndSymbolVars() const {
    return numDomain + numRange + numSymbols;
  }
  unsigned getNumVars() const {
    return numDomain + numRange + numSymbols + numLocals;
  }

  /// Get the number of vars of the specified kind.
  unsigned getNumVarKind(VarKind kind) const;

  /// Return the index at which the specified kind of var starts.
  unsigned getVarKindOffset(VarKind kind) const;

  /// Return the index at Which the specified kind of var ends.
  unsigned getVarKindEnd(VarKind kind) const;

  /// Get the number of elements of the specified kind in the range
  /// [varStart, varLimit).
  unsigned getVarKindOverlap(VarKind kind, unsigned varStart,
                             unsigned varLimit) const;

  /// Return the VarKind of the var at the specified position.
  VarKind getVarKindAt(unsigned pos) const;

  /// Insert `num` variables of the specified kind at position `pos`.
  /// Positions are relative to the kind of variable. Return the absolute
  /// column position (i.e., not relative to the kind of variable) of the
  /// first added variable.
  ///
  /// If identifiers are being used, the newly added variables have no
  /// identifiers.
  unsigned insertVar(VarKind kind, unsigned pos, unsigned num = 1);

  /// Removes variables of the specified kind in the column range [varStart,
  /// varLimit). The range is relative to the kind of variable.
  void removeVarRange(VarKind kind, unsigned varStart, unsigned varLimit);

  /// Converts variables of the specified kind in the column range [srcPos,
  /// srcPos + num) to variables of the specified kind at position dstPos. The
  /// ranges are relative to the kind of variable.
  ///
  /// srcKind and dstKind must be different.
  void convertVarKind(VarKind srcKind, unsigned srcPos, unsigned num,
                      VarKind dstKind, unsigned dstPos);

  /// Changes the partition between dimensions and symbols. Depending on the new
  /// symbol count, either a chunk of dimensional variables immediately before
  /// the split become symbols, or some of the symbols immediately after the
  /// split become dimensions.
  void setVarSymbolSeparation(unsigned newSymbolCount);

  /// Swaps the posA^th variable of kindA and posB^th variable of kindB.
  void swapVar(VarKind kindA, VarKind kindB, unsigned posA, unsigned posB);

  /// Returns true if both the spaces are compatible i.e. if both spaces have
  /// the same number of variables of each kind (excluding locals).
  bool isCompatible(const PresburgerSpace &other) const;

  /// Returns true if both the spaces are equal including local variables i.e.
  /// if both spaces have the same number of variables of each kind (including
  /// locals).
  bool isEqual(const PresburgerSpace &other) const;

  /// Get the identifier of pos^th variable of the specified kind.
  Identifier getId(VarKind kind, unsigned pos) const {
    assert(kind != VarKind::Local && "Local variables have no identifiers");
    if (!usingIds)
      return Identifier();
    return identifiers[getVarKindOffset(kind) + pos];
  }

  ArrayRef<Identifier> getIds(VarKind kind) const {
    assert(kind != VarKind::Local && "Local variables have no identifiers");
    assert(usingIds && "Identifiers not enabled for space");
    return {identifiers.data() + getVarKindOffset(kind), getNumVarKind(kind)};
  }

  ArrayRef<Identifier> getIds() const {
    assert(usingIds && "Identifiers not enabled for space");
    return identifiers;
  }

  /// Set the identifier of pos^th variable of the specified kind. Calls
  /// resetIds if identifiers are not enabled.
  void setId(VarKind kind, unsigned pos, Identifier id) {
    assert(kind != VarKind::Local && "Local variables have no identifiers");
    if (!usingIds)
      resetIds();
    identifiers[getVarKindOffset(kind) + pos] = id;
  }

  /// Returns if identifiers are being used.
  bool isUsingIds() const { return usingIds; }

  /// Reset the stored identifiers in the space. Enables `usingIds` if it was
  /// `false` before.
  void resetIds() {
    identifiers.clear();
    identifiers.resize(getNumDimAndSymbolVars());
    usingIds = true;
  }

  /// Disable identifiers being stored in space.
  void disableIds() {
    identifiers.clear();
    usingIds = false;
  }

  /// Check if the spaces are compatible, and the non-local variables having
  /// same identifiers are in the same positions. If the space is not using
  /// Identifiers, this check is same as isCompatible.
  bool isAligned(const PresburgerSpace &other) const;
  /// Same as above but only check the specified VarKind. Useful to check if
  /// the symbols in two spaces are aligned.
  bool isAligned(const PresburgerSpace &other, VarKind kind) const;

  /// Merge and align symbol variables of `this` and `other` with respect to
  /// identifiers. After this operation the symbol variables of both spaces have
  /// the same identifiers in the same order.
  void mergeAndAlignSymbols(PresburgerSpace &other);

  void print(llvm::raw_ostream &os) const;
  void dump() const;

protected:
  PresburgerSpace(unsigned numDomain, unsigned numRange, unsigned numSymbols,
                  unsigned numLocals)
      : numDomain(numDomain), numRange(numRange), numSymbols(numSymbols),
        numLocals(numLocals) {}

private:
  // Number of variables corresponding to domain variables.
  unsigned numDomain;

  // Number of variables corresponding to range variables.
  unsigned numRange;

  /// Number of variables corresponding to symbols (unknown but constant for
  /// analysis).
  unsigned numSymbols;

  /// Number of variables corresponding to locals (variables corresponding
  /// to existentially quantified variables).
  unsigned numLocals;

  /// Stores whether or not identifiers are being used in this space.
  bool usingIds = false;

  /// Stores an identifier for each non-local variable as a `void` pointer.
  SmallVector<Identifier, 0> identifiers;
};

} // namespace presburger
} // namespace mlir

#endif // MLIR_ANALYSIS_PRESBURGER_PRESBURGERSPACE_H
