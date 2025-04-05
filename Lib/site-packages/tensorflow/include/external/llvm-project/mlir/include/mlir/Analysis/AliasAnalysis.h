//===- AliasAnalysis.h - Alias Analysis in MLIR -----------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This header file defines utilities and analyses for performing alias queries
// and related memory queries in MLIR.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_ANALYSIS_ALIASANALYSIS_H_
#define MLIR_ANALYSIS_ALIASANALYSIS_H_

#include "mlir/IR/Operation.h"

namespace mlir {

//===----------------------------------------------------------------------===//
// AliasResult
//===----------------------------------------------------------------------===//

/// The possible results of an alias query.
class AliasResult {
public:
  enum Kind {
    /// The two locations do not alias at all.
    ///
    /// This value is arranged to convert to false, while all other values
    /// convert to true. This allows a boolean context to convert the result to
    /// a binary flag indicating whether there is the possibility of aliasing.
    NoAlias = 0,
    /// The two locations may or may not alias. This is the least precise
    /// result.
    MayAlias,
    /// The two locations alias, but only due to a partial overlap.
    PartialAlias,
    /// The two locations precisely alias each other.
    MustAlias,
  };

  AliasResult(Kind kind) : kind(kind) {}
  bool operator==(const AliasResult &other) const { return kind == other.kind; }
  bool operator!=(const AliasResult &other) const { return !(*this == other); }

  /// Allow conversion to bool to signal if there is an aliasing or not.
  explicit operator bool() const { return kind != NoAlias; }

  /// Merge this alias result with `other` and return a new result that
  /// represents the conservative merge of both results. If the results
  /// represent a known alias, the stronger alias is chosen (i.e.
  /// Partial+Must=Must). If the two results are conflicting, MayAlias is
  /// returned.
  AliasResult merge(AliasResult other) const;

  /// Returns if this result indicates no possibility of aliasing.
  bool isNo() const { return kind == NoAlias; }

  /// Returns if this result is a may alias.
  bool isMay() const { return kind == MayAlias; }

  /// Returns if this result is a must alias.
  bool isMust() const { return kind == MustAlias; }

  /// Returns if this result is a partial alias.
  bool isPartial() const { return kind == PartialAlias; }

  /// Print this alias result to the provided output stream.
  void print(raw_ostream &os) const;

private:
  /// The internal kind of the result.
  Kind kind;
};

inline raw_ostream &operator<<(raw_ostream &os, const AliasResult &result) {
  result.print(os);
  return os;
}

//===----------------------------------------------------------------------===//
// ModRefResult
//===----------------------------------------------------------------------===//

/// The possible results of whether a memory access modifies or references
/// a memory location. The possible results are: no access at all, a
/// modification, a reference, or both a modification and a reference.
class [[nodiscard]] ModRefResult {
  /// Note: This is a simplified version of the ModRefResult in
  /// `llvm/Analysis/AliasAnalysis.h`, and namely removes the `Must` concept. If
  /// this becomes useful/necessary we should add it here.
  enum class Kind {
    /// The access neither references nor modifies the value stored in memory.
    NoModRef = 0,
    /// The access may reference the value stored in memory.
    Ref = 1,
    /// The access may modify the value stored in memory.
    Mod = 2,
    /// The access may reference and may modify the value stored in memory.
    ModRef = Ref | Mod,
  };

public:
  bool operator==(const ModRefResult &rhs) const { return kind == rhs.kind; }
  bool operator!=(const ModRefResult &rhs) const { return !(*this == rhs); }

  /// Return a new result that indicates that the memory access neither
  /// references nor modifies the value stored in memory.
  static ModRefResult getNoModRef() { return Kind::NoModRef; }

  /// Return a new result that indicates that the memory access may reference
  /// the value stored in memory.
  static ModRefResult getRef() { return Kind::Ref; }

  /// Return a new result that indicates that the memory access may modify the
  /// value stored in memory.
  static ModRefResult getMod() { return Kind::Mod; }

  /// Return a new result that indicates that the memory access may reference
  /// and may modify the value stored in memory.
  static ModRefResult getModAndRef() { return Kind::ModRef; }

  /// Returns if this result does not modify or reference memory.
  [[nodiscard]] bool isNoModRef() const { return kind == Kind::NoModRef; }

  /// Returns if this result modifies memory.
  [[nodiscard]] bool isMod() const {
    return static_cast<int>(kind) & static_cast<int>(Kind::Mod);
  }

  /// Returns if this result references memory.
  [[nodiscard]] bool isRef() const {
    return static_cast<int>(kind) & static_cast<int>(Kind::Ref);
  }

  /// Returns if this result modifies *or* references memory.
  [[nodiscard]] bool isModOrRef() const { return kind != Kind::NoModRef; }

  /// Returns if this result modifies *and* references memory.
  [[nodiscard]] bool isModAndRef() const { return kind == Kind::ModRef; }

  /// Merge this ModRef result with `other` and return the result.
  ModRefResult merge(const ModRefResult &other) {
    return ModRefResult(static_cast<Kind>(static_cast<int>(kind) |
                                          static_cast<int>(other.kind)));
  }
  /// Intersect this ModRef result with `other` and return the result.
  ModRefResult intersect(const ModRefResult &other) {
    return ModRefResult(static_cast<Kind>(static_cast<int>(kind) &
                                          static_cast<int>(other.kind)));
  }

  /// Print this ModRef result to the provided output stream.
  void print(raw_ostream &os) const;

private:
  ModRefResult(Kind kind) : kind(kind) {}

  /// The internal kind of the result.
  Kind kind;
};

inline raw_ostream &operator<<(raw_ostream &os, const ModRefResult &result) {
  result.print(os);
  return os;
}

//===----------------------------------------------------------------------===//
// AliasAnalysisTraits
//===----------------------------------------------------------------------===//

namespace detail {
/// This class contains various internal trait classes used by the main
/// AliasAnalysis class below.
struct AliasAnalysisTraits {
  /// This class represents the `Concept` of an alias analysis implementation.
  /// It is the abstract base class used by the AliasAnalysis class for
  /// querying into derived analysis implementations.
  class Concept {
  public:
    virtual ~Concept() = default;

    /// Given two values, return their aliasing behavior.
    virtual AliasResult alias(Value lhs, Value rhs) = 0;

    /// Return the modify-reference behavior of `op` on `location`.
    virtual ModRefResult getModRef(Operation *op, Value location) = 0;
  };

  /// This class represents the `Model` of an alias analysis implementation
  /// `ImplT`. A model is instantiated for each alias analysis implementation
  /// to implement the `Concept` without the need for the derived
  /// implementation to inherit from the `Concept` class.
  template <typename ImplT>
  class Model final : public Concept {
  public:
    explicit Model(ImplT &&impl) : impl(std::forward<ImplT>(impl)) {}
    ~Model() override = default;

    /// Given two values, return their aliasing behavior.
    AliasResult alias(Value lhs, Value rhs) final {
      return impl.alias(lhs, rhs);
    }

    /// Return the modify-reference behavior of `op` on `location`.
    ModRefResult getModRef(Operation *op, Value location) final {
      return impl.getModRef(op, location);
    }

  private:
    ImplT impl;
  };
};
} // namespace detail

//===----------------------------------------------------------------------===//
// AliasAnalysis
//===----------------------------------------------------------------------===//

/// This class represents the main alias analysis interface in MLIR. It
/// functions as an aggregate of various different alias analysis
/// implementations. This aggregation allows for utilizing the strengths of
/// different alias analysis implementations that either target or have access
/// to different aliasing information. This is especially important for MLIR
/// given the scope of different types of memory models and aliasing behaviors.
/// For users of this analysis that want to perform aliasing queries, see the
/// `Alias Queries` section below for the available methods. For users of this
/// analysis that want to add a new alias analysis implementation to the
/// aggregate, see the `Alias Implementations` section below.
class AliasAnalysis {
  using Concept = detail::AliasAnalysisTraits::Concept;
  template <typename ImplT>
  using Model = detail::AliasAnalysisTraits::Model<ImplT>;

public:
  AliasAnalysis(Operation *op);

  //===--------------------------------------------------------------------===//
  // Alias Implementations
  //===--------------------------------------------------------------------===//

  /// Add a new alias analysis implementation `AnalysisT` to this analysis
  /// aggregate. This allows for users to access this implementation when
  /// performing alias queries. Implementations added here must provide the
  /// following:
  ///   * AnalysisT(AnalysisT &&)
  ///   * AliasResult alias(Value lhs, Value rhs)
  ///     - This method returns an `AliasResult` that corresponds to the
  ///       aliasing behavior between `lhs` and `rhs`. The conservative "I don't
  ///       know" result of this method should be MayAlias.
  ///   * ModRefResult getModRef(Operation *op, Value location)
  ///     - This method returns a `ModRefResult` that corresponds to the
  ///       modify-reference behavior of `op` on the given `location`. The
  ///       conservative "I don't know" result of this method should be ModRef.
  template <typename AnalysisT>
  void addAnalysisImplementation(AnalysisT &&analysis) {
    aliasImpls.push_back(
        std::make_unique<Model<AnalysisT>>(std::forward<AnalysisT>(analysis)));
  }

  //===--------------------------------------------------------------------===//
  // Alias Queries
  //===--------------------------------------------------------------------===//

  /// Given two values, return their aliasing behavior.
  AliasResult alias(Value lhs, Value rhs);

  //===--------------------------------------------------------------------===//
  // ModRef Queries
  //===--------------------------------------------------------------------===//

  /// Return the modify-reference behavior of `op` on `location`.
  ModRefResult getModRef(Operation *op, Value location);

private:
  /// A set of internal alias analysis implementations.
  SmallVector<std::unique_ptr<Concept>, 4> aliasImpls;
};

} // namespace mlir

#endif // MLIR_ANALYSIS_ALIASANALYSIS_H_
