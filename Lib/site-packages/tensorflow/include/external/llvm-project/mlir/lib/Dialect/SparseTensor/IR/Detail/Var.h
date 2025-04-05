//===- Var.h ----------------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_SPARSETENSOR_IR_DETAIL_VAR_H
#define MLIR_DIALECT_SPARSETENSOR_IR_DETAIL_VAR_H

#include "TemplateExtras.h"

#include "mlir/IR/OpImplementation.h"
#include "llvm/ADT/EnumeratedArray.h"
#include "llvm/ADT/STLForwardCompat.h"
#include "llvm/ADT/SmallBitVector.h"
#include "llvm/ADT/StringMap.h"

namespace mlir {
namespace sparse_tensor {
namespace ir_detail {

//===----------------------------------------------------------------------===//
/// The three kinds of variables that `Var` can be.
///
/// NOTE: The numerical values used to represent this enum should be
/// treated as an implementation detail, not as part of the API.  In the
/// API below we use the canonical ordering `{Symbol,Dimension,Level}` even
/// though that does not agree with the numerical ordering of the numerical
/// representation.
enum class VarKind { Symbol = 1, Dimension = 0, Level = 2 };

[[nodiscard]] constexpr bool isWF(VarKind vk) {
  const auto vk_ = llvm::to_underlying(vk);
  return 0 <= vk_ && vk_ <= 2;
}

/// Gets the ASCII character used as the prefix when printing `Var`.
constexpr char toChar(VarKind vk) {
  // If `isWF(vk)` then this computation's intermediate results are always
  // in the range [-44..126] (where that lower bound is under worst-case
  // rearranging of the expression); and `int_fast8_t` is the fastest type
  // which can support that range without over-/underflow.
  const auto vk_ = static_cast<int_fast8_t>(llvm::to_underlying(vk));
  return static_cast<char>(100 + vk_ * (26 - vk_ * 11));
}
static_assert(toChar(VarKind::Symbol) == 's' &&
              toChar(VarKind::Dimension) == 'd' &&
              toChar(VarKind::Level) == 'l');

//===----------------------------------------------------------------------===//
/// The type of arrays indexed by `VarKind`.
template <typename T>
using VarKindArray = llvm::EnumeratedArray<T, VarKind, VarKind::Level>;

//===----------------------------------------------------------------------===//
/// A concrete variable, to be used in our variant of `AffineExpr`.
/// Client-facing class for `VarKind` + `Var::Num` pairs, with RTTI
/// support for subclasses with a fixed `VarKind`.
class Var {
public:
  /// Typedef for the type of variable numbers.
  using Num = unsigned;

private:
  /// Typedef for the underlying storage of `Var::Impl`.
  using Storage = unsigned;

  /// The largest `Var::Num` supported by `Var`/`Var::Impl`/`Var::Storage`.
  /// Two low-order bits are reserved for storing the `VarKind`,
  /// and one high-order bit is reserved for future use (e.g., to support
  /// `DenseMapInfo<Var>` while maintaining the usual numeric values for
  /// "empty" and "tombstone").
  static constexpr Num kMaxNum =
      static_cast<Num>(std::numeric_limits<Storage>::max() >> 3);

public:
  /// Checks whether the number would be accepted by `Var(VarKind,Var::Num)`.
  //
  // This must be public for `VarInfo` to use it (whereas we don't want
  // to expose the `impl` field via friendship).
  [[nodiscard]] static constexpr bool isWF_Num(Num n) { return n <= kMaxNum; }

protected:
  /// The underlying implementation of `Var`.  Note that this must be kept
  /// distinct from `Var` itself, since we want to ensure that the RTTI
  /// methods will select the `U(Var::Impl)` ctor rather than selecting
  /// the `U(Var::Num)` ctor.
  class Impl final {
    Storage data;

  public:
    constexpr Impl(VarKind vk, Num n)
        : data((static_cast<Storage>(n) << 2) |
               static_cast<Storage>(llvm::to_underlying(vk))) {
      assert(isWF(vk) && "unknown VarKind");
      assert(isWF_Num(n) && "Var::Num is too large");
    }
    constexpr bool operator==(Impl other) const { return data == other.data; }
    constexpr bool operator!=(Impl other) const { return !(*this == other); }
    constexpr VarKind getKind() const { return static_cast<VarKind>(data & 3); }
    constexpr Num getNum() const { return static_cast<Num>(data >> 2); }
  };
  static_assert(IsZeroCostAbstraction<Impl>);

private:
  Impl impl;

protected:
  /// Protected ctor for the RTTI methods to use.
  constexpr explicit Var(Impl impl) : impl(impl) {}

public:
  constexpr Var(VarKind vk, Num n) : impl(Impl(vk, n)) {}
  Var(AffineSymbolExpr sym) : Var(VarKind::Symbol, sym.getPosition()) {}
  Var(VarKind vk, AffineDimExpr var) : Var(vk, var.getPosition()) {
    assert(vk != VarKind::Symbol);
  }

  constexpr bool operator==(Var other) const { return impl == other.impl; }
  constexpr bool operator!=(Var other) const { return !(*this == other); }

  constexpr VarKind getKind() const { return impl.getKind(); }
  constexpr Num getNum() const { return impl.getNum(); }

  template <typename U>
  constexpr bool isa() const;
  template <typename U>
  constexpr U cast() const;
  template <typename U>
  constexpr std::optional<U> dyn_cast() const;

  std::string str() const;
  void print(llvm::raw_ostream &os) const;
  void print(AsmPrinter &printer) const;
  void dump() const;
};
static_assert(IsZeroCostAbstraction<Var>);

class SymVar final : public Var {
  using Var::Var; // inherit `Var(Impl)` ctor for RTTI use.
public:
  static constexpr VarKind Kind = VarKind::Symbol;
  static constexpr bool classof(Var const *var) {
    return var->getKind() == Kind;
  }
  constexpr SymVar(Num sym) : Var(Kind, sym) {}
  SymVar(AffineSymbolExpr symExpr) : Var(symExpr) {}
};
static_assert(IsZeroCostAbstraction<SymVar>);

class DimVar final : public Var {
  using Var::Var; // inherit `Var(Impl)` ctor for RTTI use.
public:
  static constexpr VarKind Kind = VarKind::Dimension;
  static constexpr bool classof(Var const *var) {
    return var->getKind() == Kind;
  }
  constexpr DimVar(Num dim) : Var(Kind, dim) {}
  DimVar(AffineDimExpr dimExpr) : Var(Kind, dimExpr) {}
};
static_assert(IsZeroCostAbstraction<DimVar>);

class LvlVar final : public Var {
  using Var::Var; // inherit `Var(Impl)` ctor for RTTI use.
public:
  static constexpr VarKind Kind = VarKind::Level;
  static constexpr bool classof(Var const *var) {
    return var->getKind() == Kind;
  }
  constexpr LvlVar(Num lvl) : Var(Kind, lvl) {}
  LvlVar(AffineDimExpr lvlExpr) : Var(Kind, lvlExpr) {}
};
static_assert(IsZeroCostAbstraction<LvlVar>);

template <typename U>
constexpr bool Var::isa() const {
  if constexpr (std::is_same_v<U, SymVar>)
    return getKind() == VarKind::Symbol;
  if constexpr (std::is_same_v<U, DimVar>)
    return getKind() == VarKind::Dimension;
  if constexpr (std::is_same_v<U, LvlVar>)
    return getKind() == VarKind::Level;
}

template <typename U>
constexpr U Var::cast() const {
  assert(isa<U>());
  // NOTE: This should select the `U(Var::Impl)` ctor, *not* `U(Var::Num)`
  return U(impl);
}

template <typename U>
constexpr std::optional<U> Var::dyn_cast() const {
  // NOTE: This should select the `U(Var::Impl)` ctor, *not* `U(Var::Num)`
  return isa<U>() ? std::make_optional(U(impl)) : std::nullopt;
}

//===----------------------------------------------------------------------===//
// Forward-decl so that we can declare methods of `Ranks` and `VarSet`.
class DimLvlExpr;

//===----------------------------------------------------------------------===//
class Ranks final {
  // Not using `VarKindArray` since `EnumeratedArray` doesn't support constexpr.
  unsigned impl[3];

  static constexpr unsigned to_index(VarKind vk) {
    assert(isWF(vk) && "unknown VarKind");
    return static_cast<unsigned>(llvm::to_underlying(vk));
  }

public:
  constexpr Ranks(unsigned symRank, unsigned dimRank, unsigned lvlRank)
      : impl() {
    impl[to_index(VarKind::Symbol)] = symRank;
    impl[to_index(VarKind::Dimension)] = dimRank;
    impl[to_index(VarKind::Level)] = lvlRank;
  }
  Ranks(VarKindArray<unsigned> const &ranks)
      : Ranks(ranks[VarKind::Symbol], ranks[VarKind::Dimension],
              ranks[VarKind::Level]) {}

  bool operator==(Ranks const &other) const;
  bool operator!=(Ranks const &other) const { return !(*this == other); }

  constexpr unsigned getRank(VarKind vk) const { return impl[to_index(vk)]; }
  constexpr unsigned getSymRank() const { return getRank(VarKind::Symbol); }
  constexpr unsigned getDimRank() const { return getRank(VarKind::Dimension); }
  constexpr unsigned getLvlRank() const { return getRank(VarKind::Level); }

  [[nodiscard]] constexpr bool isValid(Var var) const {
    return var.getNum() < getRank(var.getKind());
  }
  [[nodiscard]] bool isValid(DimLvlExpr expr) const;
};
static_assert(IsZeroCostAbstraction<Ranks>);

//===----------------------------------------------------------------------===//
/// Efficient representation of a set of `Var`.
class VarSet final {
  VarKindArray<llvm::SmallBitVector> impl;

public:
  explicit VarSet(Ranks const &ranks);

  unsigned getRank(VarKind vk) const { return impl[vk].size(); }
  unsigned getSymRank() const { return getRank(VarKind::Symbol); }
  unsigned getDimRank() const { return getRank(VarKind::Dimension); }
  unsigned getLvlRank() const { return getRank(VarKind::Level); }
  Ranks getRanks() const {
    return Ranks(getSymRank(), getDimRank(), getLvlRank());
  }
  /// For the `contains` method: if variables occurring in
  /// the method parameter are OOB for the `VarSet`, then these methods will
  /// always return false.
  bool contains(Var var) const;

  /// For the `add` methods: OOB parameters cause undefined behavior.
  /// Currently the `add` methods will raise an assertion error.
  void add(Var var);
  void add(VarSet const &vars);
  void add(DimLvlExpr expr);
};

//===----------------------------------------------------------------------===//
/// A record of metadata for/about a variable, used by `VarEnv`.
/// The principal goal of this record is to enable `VarEnv` to be used for
/// incremental parsing; in particular, `VarInfo` allows the `Var::Num` to
/// remain unknown, since each record is instead identified by `VarInfo::ID`.
/// Therefore the `VarEnv` can freely allocate `VarInfo::ID` in whatever
/// order it likes, irrespective of the binding order (`Var::Num`) of the
/// associated variable.
class VarInfo final {
public:
  /// Newtype for unique identifiers of `VarInfo` records, to ensure
  /// they aren't confused with `Var::Num`.
  enum class ID : unsigned {};

private:
  StringRef name;              // The bare-id used in the MLIR source.
  llvm::SMLoc loc;             // The location of the first occurence.
  ID id;                       // The unique `VarInfo`-identifier.
  std::optional<Var::Num> num; // The unique `Var`-identifier (if resolved).
  VarKind kind;                // The kind of variable.

public:
  constexpr VarInfo(ID id, StringRef name, llvm::SMLoc loc, VarKind vk,
                    std::optional<Var::Num> n = {})
      : name(name), loc(loc), id(id), num(n), kind(vk) {
    assert(!name.empty() && "null StringRef");
    assert(loc.isValid() && "null SMLoc");
    assert(isWF(vk) && "unknown VarKind");
    assert((!n || Var::isWF_Num(*n)) && "Var::Num is too large");
  }

  constexpr StringRef getName() const { return name; }
  constexpr llvm::SMLoc getLoc() const { return loc; }
  Location getLocation(AsmParser &parser) const {
    return parser.getEncodedSourceLoc(loc);
  }
  constexpr ID getID() const { return id; }
  constexpr VarKind getKind() const { return kind; }
  constexpr std::optional<Var::Num> getNum() const { return num; }
  constexpr bool hasNum() const { return num.has_value(); }
  void setNum(Var::Num n);
  constexpr Var getVar() const {
    assert(hasNum());
    return Var(kind, *num);
  }
};

//===----------------------------------------------------------------------===//
enum class Policy { MustNot, May, Must };

//===----------------------------------------------------------------------===//
class VarEnv final {
  /// Map from `VarKind` to the next free `Var::Num`; used by `bindVar`.
  VarKindArray<Var::Num> nextNum;
  /// Map from `VarInfo::ID` to shared storage for the actual `VarInfo` objects.
  SmallVector<VarInfo> vars;
  /// Map from variable names to their `VarInfo::ID`.
  llvm::StringMap<VarInfo::ID> ids;

  VarInfo::ID nextID() const { return static_cast<VarInfo::ID>(vars.size()); }

public:
  VarEnv() : nextNum(0) {}

  /// Gets the underlying storage for the `VarInfo` identified by
  /// the `VarInfo::ID`.
  ///
  /// NOTE: The returned reference can become dangling if the `VarEnv`
  /// object is mutated during the lifetime of the pointer.  Therefore,
  /// client code should not store the reference nor otherwise allow it
  /// to live too long.
  VarInfo const &access(VarInfo::ID id) const {
    // `SmallVector::operator[]` already asserts the index is in-bounds.
    return vars[llvm::to_underlying(id)];
  }
  VarInfo const *access(std::optional<VarInfo::ID> oid) const {
    return oid ? &access(*oid) : nullptr;
  }

private:
  VarInfo &access(VarInfo::ID id) {
    return const_cast<VarInfo &>(std::as_const(*this).access(id));
  }
  VarInfo *access(std::optional<VarInfo::ID> oid) {
    return const_cast<VarInfo *>(std::as_const(*this).access(oid));
  }

public:
  /// Looks up the variable with the given name.
  std::optional<VarInfo::ID> lookup(StringRef name) const;

  /// Creates a new currently-unbound variable.  When a variable
  /// of that name already exists: if `verifyUsage` is true, then will assert
  /// that the variable has the same kind and a consistent location; otherwise,
  /// when `verifyUsage` is false, this is a noop.  Returns the identifier
  /// for the variable with the given name, and a bool indicating whether
  /// a new variable was created.
  std::optional<std::pair<VarInfo::ID, bool>>
  create(StringRef name, llvm::SMLoc loc, VarKind vk, bool verifyUsage = false);

  /// Looks up or creates a variable according to the given
  /// `Policy`.  Returns nullopt in one of two circumstances:
  /// (1) the policy says we `Must` create, yet the variable already exists;
  /// (2) the policy says we `MustNot` create, yet no such variable exists.
  /// Otherwise, if the variable already exists then it is validated against
  /// the given kind and location to ensure consistency.
  std::optional<std::pair<VarInfo::ID, bool>>
  lookupOrCreate(Policy creationPolicy, StringRef name, llvm::SMLoc loc,
                 VarKind vk);

  /// Binds the given variable to the next free `Var::Num` for its `VarKind`.
  Var bindVar(VarInfo::ID id);

  /// Creates a new variable of the given kind and immediately binds it.
  /// This should only be used whenever the variable is known to be unused
  /// and therefore does not have a name.
  Var bindUnusedVar(VarKind vk);

  InFlightDiagnostic emitErrorIfAnyUnbound(AsmParser &parser) const;

  /// Returns the current ranks of bound variables.  This method should
  /// only be used after the environment is "finished", since binding new
  /// variables will (semantically) invalidate any previously returned `Ranks`.
  Ranks getRanks() const { return Ranks(nextNum); }

  /// Gets the `Var` identified by the `VarInfo::ID`, raising an assertion
  /// failure if the variable is not bound.
  Var getVar(VarInfo::ID id) const { return access(id).getVar(); }
};

//===----------------------------------------------------------------------===//

} // namespace ir_detail
} // namespace sparse_tensor
} // namespace mlir

#endif // MLIR_DIALECT_SPARSETENSOR_IR_DETAIL_VAR_H
