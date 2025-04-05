//===- DimLvlMap.h ----------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_SPARSETENSOR_IR_DETAIL_DIMLVLMAP_H
#define MLIR_DIALECT_SPARSETENSOR_IR_DETAIL_DIMLVLMAP_H

#include "Var.h"

#include "mlir/Dialect/SparseTensor/IR/SparseTensor.h"
#include "llvm/ADT/STLForwardCompat.h"

namespace mlir {
namespace sparse_tensor {
namespace ir_detail {

//===----------------------------------------------------------------------===//
enum class ExprKind : bool { Dimension = false, Level = true };

constexpr VarKind getVarKindAllowedInExpr(ExprKind ek) {
  using VK = std::underlying_type_t<VarKind>;
  return VarKind{2 * static_cast<VK>(!llvm::to_underlying(ek))};
}
static_assert(getVarKindAllowedInExpr(ExprKind::Dimension) == VarKind::Level &&
              getVarKindAllowedInExpr(ExprKind::Level) == VarKind::Dimension);

//===----------------------------------------------------------------------===//
class DimLvlExpr {
private:
  ExprKind kind;
  AffineExpr expr;

public:
  constexpr DimLvlExpr(ExprKind ek, AffineExpr expr) : kind(ek), expr(expr) {}

  //
  // Boolean operators.
  //
  constexpr bool operator==(DimLvlExpr other) const {
    return kind == other.kind && expr == other.expr;
  }
  constexpr bool operator!=(DimLvlExpr other) const {
    return !(*this == other);
  }
  explicit operator bool() const { return static_cast<bool>(expr); }

  //
  // RTTI support (for the `DimLvlExpr` class itself).
  //
  template <typename U>
  constexpr bool isa() const;
  template <typename U>
  constexpr U cast() const;
  template <typename U>
  constexpr U dyn_cast() const;

  //
  // Simple getters.
  //
  constexpr ExprKind getExprKind() const { return kind; }
  constexpr VarKind getAllowedVarKind() const {
    return getVarKindAllowedInExpr(kind);
  }
  constexpr AffineExpr getAffineExpr() const { return expr; }
  AffineExprKind getAffineKind() const {
    assert(expr);
    return expr.getKind();
  }
  MLIRContext *tryGetContext() const {
    return expr ? expr.getContext() : nullptr;
  }

  //
  // Getters for handling `AffineExpr` subclasses.
  //
  SymVar castSymVar() const;
  std::optional<SymVar> dyn_castSymVar() const;
  Var castDimLvlVar() const;
  std::optional<Var> dyn_castDimLvlVar() const;
  std::tuple<DimLvlExpr, AffineExprKind, DimLvlExpr> unpackBinop() const;

  /// Checks whether the variables bound/used by this spec are valid
  /// with respect to the given ranks.
  [[nodiscard]] bool isValid(Ranks const &ranks) const;

protected:
  // Variant of `mlir::AsmPrinter::Impl::BindingStrength`
  enum class BindingStrength : bool { Weak = false, Strong = true };
};
static_assert(IsZeroCostAbstraction<DimLvlExpr>);

class DimExpr final : public DimLvlExpr {
  friend class DimLvlExpr;
  constexpr explicit DimExpr(DimLvlExpr expr) : DimLvlExpr(expr) {}

public:
  static constexpr ExprKind Kind = ExprKind::Dimension;
  static constexpr bool classof(DimLvlExpr const *expr) {
    return expr->getExprKind() == Kind;
  }
  constexpr explicit DimExpr(AffineExpr expr) : DimLvlExpr(Kind, expr) {}

  LvlVar castLvlVar() const { return castDimLvlVar().cast<LvlVar>(); }
  std::optional<LvlVar> dyn_castLvlVar() const {
    const auto var = dyn_castDimLvlVar();
    return var ? std::make_optional(var->cast<LvlVar>()) : std::nullopt;
  }
};
static_assert(IsZeroCostAbstraction<DimExpr>);

class LvlExpr final : public DimLvlExpr {
  friend class DimLvlExpr;
  constexpr explicit LvlExpr(DimLvlExpr expr) : DimLvlExpr(expr) {}

public:
  static constexpr ExprKind Kind = ExprKind::Level;
  static constexpr bool classof(DimLvlExpr const *expr) {
    return expr->getExprKind() == Kind;
  }
  constexpr explicit LvlExpr(AffineExpr expr) : DimLvlExpr(Kind, expr) {}

  DimVar castDimVar() const { return castDimLvlVar().cast<DimVar>(); }
  std::optional<DimVar> dyn_castDimVar() const {
    const auto var = dyn_castDimLvlVar();
    return var ? std::make_optional(var->cast<DimVar>()) : std::nullopt;
  }
};
static_assert(IsZeroCostAbstraction<LvlExpr>);

template <typename U>
constexpr bool DimLvlExpr::isa() const {
  if constexpr (std::is_same_v<U, DimExpr>)
    return getExprKind() == ExprKind::Dimension;
  if constexpr (std::is_same_v<U, LvlExpr>)
    return getExprKind() == ExprKind::Level;
}

template <typename U>
constexpr U DimLvlExpr::cast() const {
  assert(isa<U>());
  return U(*this);
}

template <typename U>
constexpr U DimLvlExpr::dyn_cast() const {
  return isa<U>() ? U(*this) : U();
}

//===----------------------------------------------------------------------===//
/// The full `dimVar = dimExpr : dimSlice` specification for a given dimension.
class DimSpec final {
  /// The dimension-variable bound by this specification.
  DimVar var;
  /// The dimension-expression.  The `DimSpec` ctor treats this field
  /// as optional; whereas the `DimLvlMap` ctor will fill in (or verify)
  /// the expression via function-inversion inference.
  DimExpr expr;
  /// Can the `expr` be elided when printing? The `DimSpec` ctor assumes
  /// not (though if `expr` is null it will elide printing that); whereas
  /// the `DimLvlMap` ctor will reset it as appropriate.
  bool elideExpr = false;
  /// The dimension-slice; optional, default is null.
  SparseTensorDimSliceAttr slice;

public:
  DimSpec(DimVar var, DimExpr expr, SparseTensorDimSliceAttr slice);

  MLIRContext *tryGetContext() const { return expr.tryGetContext(); }

  constexpr DimVar getBoundVar() const { return var; }
  bool hasExpr() const { return static_cast<bool>(expr); }
  constexpr DimExpr getExpr() const { return expr; }
  void setExpr(DimExpr newExpr) {
    assert(!hasExpr());
    expr = newExpr;
  }
  constexpr bool canElideExpr() const { return elideExpr; }
  void setElideExpr(bool b) { elideExpr = b; }
  constexpr SparseTensorDimSliceAttr getSlice() const { return slice; }

  /// Checks whether the variables bound/used by this spec are valid with
  /// respect to the given ranks.  Note that null `DimExpr` is considered
  /// to be vacuously valid, and therefore calling `setExpr` invalidates
  /// the result of this predicate.
  [[nodiscard]] bool isValid(Ranks const &ranks) const;
};

static_assert(IsZeroCostAbstraction<DimSpec>);

//===----------------------------------------------------------------------===//
/// The full `lvlVar = lvlExpr : lvlType` specification for a given level.
class LvlSpec final {
  /// The level-variable bound by this specification.
  LvlVar var;
  /// Can the `var` be elided when printing?  The `LvlSpec` ctor assumes not;
  /// whereas the `DimLvlMap` ctor will reset this as appropriate.
  bool elideVar = false;
  /// The level-expression.
  LvlExpr expr;
  /// The level-type (== level-format + lvl-properties).
  LevelType type;

public:
  LvlSpec(LvlVar var, LvlExpr expr, LevelType type);

  MLIRContext *getContext() const {
    MLIRContext *ctx = expr.tryGetContext();
    assert(ctx);
    return ctx;
  }

  constexpr LvlVar getBoundVar() const { return var; }
  constexpr bool canElideVar() const { return elideVar; }
  void setElideVar(bool b) { elideVar = b; }
  constexpr LvlExpr getExpr() const { return expr; }
  constexpr LevelType getType() const { return type; }

  /// Checks whether the variables bound/used by this spec are valid
  /// with respect to the given ranks.
  [[nodiscard]] bool isValid(Ranks const &ranks) const;
};

static_assert(IsZeroCostAbstraction<LvlSpec>);

//===----------------------------------------------------------------------===//
class DimLvlMap final {
public:
  DimLvlMap(unsigned symRank, ArrayRef<DimSpec> dimSpecs,
            ArrayRef<LvlSpec> lvlSpecs);

  unsigned getSymRank() const { return symRank; }
  unsigned getDimRank() const { return dimSpecs.size(); }
  unsigned getLvlRank() const { return lvlSpecs.size(); }
  unsigned getRank(VarKind vk) const { return getRanks().getRank(vk); }
  Ranks getRanks() const { return {getSymRank(), getDimRank(), getLvlRank()}; }

  ArrayRef<DimSpec> getDims() const { return dimSpecs; }
  const DimSpec &getDim(Dimension dim) const { return dimSpecs[dim]; }
  SparseTensorDimSliceAttr getDimSlice(Dimension dim) const {
    return getDim(dim).getSlice();
  }

  ArrayRef<LvlSpec> getLvls() const { return lvlSpecs; }
  const LvlSpec &getLvl(Level lvl) const { return lvlSpecs[lvl]; }
  LevelType getLvlType(Level lvl) const { return getLvl(lvl).getType(); }

  AffineMap getDimToLvlMap(MLIRContext *context) const;
  AffineMap getLvlToDimMap(MLIRContext *context) const;

private:
  /// Checks for integrity of variable-binding structure.
  /// This is already called by the ctor.
  [[nodiscard]] bool isWF() const;

  /// Helper function to call `DimSpec::setExpr` while asserting that
  /// the invariant established by `DimLvlMap:isWF` is maintained.
  /// This is used by the ctor.
  void setDimExpr(Dimension dim, DimExpr expr) {
    assert(expr && getRanks().isValid(expr));
    dimSpecs[dim].setExpr(expr);
  }

  // All these fields are const-after-ctor.
  unsigned symRank;
  SmallVector<DimSpec> dimSpecs;
  SmallVector<LvlSpec> lvlSpecs;
  bool mustPrintLvlVars;
};

//===----------------------------------------------------------------------===//

} // namespace ir_detail
} // namespace sparse_tensor
} // namespace mlir

#endif // MLIR_DIALECT_SPARSETENSOR_IR_DETAIL_DIMLVLMAP_H
