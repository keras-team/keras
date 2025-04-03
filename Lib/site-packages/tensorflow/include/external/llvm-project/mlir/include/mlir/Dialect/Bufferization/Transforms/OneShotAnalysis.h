//===- OneShotAnalysis.h - One-Shot (Single Pass) Analysis ------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_BUFFERIZATION_TRANSFORMS_ONESHOTANALYSIS_H
#define MLIR_DIALECT_BUFFERIZATION_TRANSFORMS_ONESHOTANALYSIS_H

#include "mlir/Dialect/Bufferization/IR/BufferizableOpInterface.h"
#include "llvm/ADT/EquivalenceClasses.h"
#include <string>

namespace mlir {
class DominanceInfo;

namespace bufferization {

struct OneShotBufferizationOptions;
struct BufferizationStatistics;
class OneShotAnalysisState;

/// Options for analysis-enabled bufferization.
struct OneShotBufferizationOptions : public BufferizationOptions {
  enum class AnalysisHeuristic {
    BottomUp,
    TopDown,
    BottomUpFromTerminators,
    Fuzzer
  };

  OneShotBufferizationOptions() = default;

  /// Specifies whether returning newly allocated memrefs from loops should be
  /// allowed.  Otherwise, a pass failure is triggered.
  bool allowReturnAllocsFromLoops = false;

  /// Specifies whether the tensor IR should be annotated with alias sets.
  bool dumpAliasSets = false;

  /// The heuristic controls the order in which ops are traversed during the
  /// analysis.
  AnalysisHeuristic analysisHeuristic = AnalysisHeuristic::BottomUp;

  /// Specify the functions that should not be analyzed. copyBeforeWrite will be
  /// set to true when bufferizing them.
  llvm::ArrayRef<std::string> noAnalysisFuncFilter;

  /// Seed for the analysis fuzzer. Used only if the heuristic is set to
  /// `AnalysisHeuristic::Fuzzer`. The fuzzer should be used only with
  /// `testAnalysisOnly = true`.
  unsigned analysisFuzzerSeed = 0;
};

/// State for analysis-enabled bufferization. This class keeps track of alias
/// sets, equivalence sets, in-place OpOperands and other things.
///
/// Note: Modifying the IR generally invalidates the result of the analysis.
/// Adding new operations is safe if they are analyzed subsequently.
class OneShotAnalysisState : public AnalysisState {
public:
  OneShotAnalysisState(Operation *op,
                       const OneShotBufferizationOptions &options);

  OneShotAnalysisState(const OneShotAnalysisState &) = delete;

  ~OneShotAnalysisState() override = default;

  static bool classof(const AnalysisState *base) {
    return base->getType() == TypeID::get<OneShotAnalysisState>();
  }

  /// Return a reference to the BufferizationOptions.
  const OneShotBufferizationOptions &getOptions() const {
    return static_cast<const OneShotBufferizationOptions &>(
        AnalysisState::getOptions());
  }

  /// Analyze the given op and its nested ops.
  LogicalResult analyzeOp(Operation *op, const DominanceInfo &domInfo);

  /// Analyze a single op (without nested ops).
  LogicalResult analyzeSingleOp(Operation *op, const DominanceInfo &domInfo);

  /// Apply `fun` to all the members of the equivalence class of `v`.
  void applyOnEquivalenceClass(Value v, function_ref<void(Value)> fun) const;

  /// Apply `fun` to all aliases of `v`.
  void applyOnAliases(Value v, function_ref<void(Value)> fun) const;

  /// Return true if `v1` and `v2` bufferize to equivalent buffers.
  bool areEquivalentBufferizedValues(Value v1, Value v2) const override;

  /// Return true if `v1` and `v2` may bufferize to aliasing buffers.
  bool areAliasingBufferizedValues(Value v1, Value v2) const override;

  /// Mark the given OpOperand as in-place and merge the results' and operand's
  /// aliasing sets.
  void bufferizeInPlace(OpOperand &operand);

  /// Mark the given OpOperand as out-of-place.
  void bufferizeOutOfPlace(OpOperand &operand);

  /// Add a new entry for `v` in the `aliasInfo` and `equivalentInfo`. In the
  /// beginning the alias and equivalence sets only contain `v` itself.
  void createAliasInfoEntry(Value v);

  /// Find all tensor values in the given operation that have undefined contents
  /// and store them in `undefinedTensorUses`.
  void gatherUndefinedTensorUses(Operation *op);

  int64_t getStatNumTensorOutOfPlace() const { return statNumTensorOutOfPlace; }
  int64_t getStatNumTensorInPlace() const { return statNumTensorInPlace; }

  /// Return `true` if the given tensor has undefined contents.
  bool hasUndefinedContents(OpOperand *opOperand) const override;

  /// Return `true` if the given OpResult has been decided to bufferize inplace.
  bool isInPlace(OpOperand &opOperand) const override;

  /// Return true if the buffer of the given tensor value is written to. Must
  /// not be called for values inside not yet analyzed functions.
  bool isValueWritten(Value value) const;

  /// Return true if the buffer of the given tensor value is writable.
  bool isWritable(Value value) const;

  /// Find the definitions of the given tensor value or retrieve them from the
  /// cache.
  const SetVector<Value> &findDefinitionsCached(Value value);

  /// Reset cached data structures.
  void resetCache() override;

  /// Union the alias sets of `v1` and `v2`.
  void unionAliasSets(Value v1, Value v2);

  /// Union the equivalence classes of `v1` and `v2`.
  void unionEquivalenceClasses(Value v1, Value v2);

  /// Base class for OneShotAnalysisState extensions that allow
  /// OneShotAnalysisState to contain user-specified information in the state
  /// object. Clients are expected to derive this class, add the desired fields,
  /// and make the derived class compatible with the MLIR TypeID mechanism.
  ///
  /// ```mlir
  /// class MyExtension final : public OneShotAnalysisState::Extension {
  /// public:
  ///   MyExtension(OneShotAnalysisState &state, int myData)
  ///       : Extension(state) {...}
  /// private:
  ///   int mySupplementaryData;
  /// };
  /// ```
  ///
  /// Instances of this and derived classes are not expected to be created by
  /// the user, instead they are directly constructed within a
  /// OneShotAnalysisState. A OneShotAnalysisState can only contain one
  /// extension with the given TypeID. Extensions can be obtained from a
  /// OneShotAnalysisState instance.
  ///
  /// ```mlir
  /// state.addExtension<MyExtension>(/*myData=*/42);
  /// MyExtension *ext = state.getExtension<MyExtension>();
  /// ext->doSomething();
  /// ```
  class Extension {
    // Allow OneShotAnalysisState to allocate Extensions.
    friend class OneShotAnalysisState;

  public:
    /// Base virtual destructor.
    // Out-of-line definition ensures symbols are emitted in a single object
    // file.
    virtual ~Extension();

  protected:
    /// Constructs an extension of the given state object.
    Extension(OneShotAnalysisState &state) : state(state) {}

    /// Provides read-only access to the parent OneShotAnalysisState object.
    const OneShotAnalysisState &getAnalysisState() const { return state; }

  private:
    /// Back-reference to the state that is being extended.
    OneShotAnalysisState &state;
  };

  /// Adds a new Extension of the type specified as template parameter,
  /// constructing it with the arguments provided. The extension is owned by the
  /// OneShotAnalysisState. It is expected that the state does not already have
  /// an extension of the same type. Extension constructors are expected to take
  /// a reference to OneShotAnalysisState as first argument, automatically
  /// supplied by this call.
  template <typename Ty, typename... Args>
  Ty &addExtension(Args &&...args) {
    static_assert(
        std::is_base_of<Extension, Ty>::value,
        "only a class derived from OneShotAnalysisState::Extension is allowed");
    auto ptr = std::make_unique<Ty>(*this, std::forward<Args>(args)...);
    auto result = extensions.try_emplace(TypeID::get<Ty>(), std::move(ptr));
    assert(result.second && "extension already added");
    return *static_cast<Ty *>(result.first->second.get());
  }

  /// Returns the extension of the specified type.
  template <typename Ty>
  Ty *getExtension() {
    static_assert(
        std::is_base_of<Extension, Ty>::value,
        "only a class derived from OneShotAnalysisState::Extension is allowed");
    auto iter = extensions.find(TypeID::get<Ty>());
    if (iter == extensions.end())
      return nullptr;
    return static_cast<Ty *>(iter->second.get());
  }

  /// Returns the extension of the specified type.
  template <typename Ty>
  const Ty *getExtension() const {
    return const_cast<OneShotAnalysisState *>(this)->getExtension<Ty>();
  }

private:
  /// llvm::EquivalenceClasses wants comparable elements. This comparator uses
  /// pointer comparison on the defining op. This is a poor man's comparison
  /// but it's not like UnionFind needs ordering anyway.
  struct ValueComparator {
    bool operator()(const Value &lhs, const Value &rhs) const {
      return lhs.getImpl() < rhs.getImpl();
    }
  };

  using EquivalenceClassRangeType = llvm::iterator_range<
      llvm::EquivalenceClasses<Value, ValueComparator>::member_iterator>;
  /// Check that aliasInfo for `v` exists and return a reference to it.
  EquivalenceClassRangeType getAliases(Value v) const;

  /// Cache definitions of tensor values.
  DenseMap<Value, SetVector<Value>> cachedDefinitions;

  /// Set of all OpResults that were decided to bufferize in-place.
  llvm::DenseSet<OpOperand *> inplaceBufferized;

  /// Auxiliary structure to store all the values a given value may alias with.
  /// Alias information is "may be" conservative: In the presence of branches, a
  /// value may alias with one of multiple other values. The concrete aliasing
  /// value may not even be known at compile time. All such values are
  /// considered to be aliases.
  llvm::EquivalenceClasses<Value, ValueComparator> aliasInfo;

  /// Auxiliary structure to store all the equivalent buffer classes. Equivalent
  /// buffer information is "must be" conservative: Only if two values are
  /// guaranteed to be equivalent at runtime, they said to be equivalent. It is
  /// possible that, in the presence of branches, it cannot be determined
  /// statically if two values are equivalent. In that case, the values are
  /// considered to be not equivalent.
  llvm::EquivalenceClasses<Value, ValueComparator> equivalentInfo;

  // Bufferization statistics.
  int64_t statNumTensorOutOfPlace = 0;
  int64_t statNumTensorInPlace = 0;

  /// A set of uses of tensors that have undefined contents.
  DenseSet<OpOperand *> undefinedTensorUses;

  /// Extensions attached to the state, identified by the TypeID of their type.
  /// Only one extension of any given type is allowed.
  DenseMap<TypeID, std::unique_ptr<Extension>> extensions;
};

/// Analyze `op` and its nested ops. Bufferization decisions are stored in
/// `state`.
LogicalResult analyzeOp(Operation *op, OneShotAnalysisState &state,
                        BufferizationStatistics *statistics = nullptr);

/// Run One-Shot Bufferize on the given op: Analysis + Bufferization
LogicalResult
runOneShotBufferize(Operation *op, const OneShotBufferizationOptions &options,
                    BufferizationStatistics *statistics = nullptr);

} // namespace bufferization
} // namespace mlir

MLIR_DECLARE_EXPLICIT_TYPE_ID(mlir::bufferization::OneShotAnalysisState)

#endif // MLIR_DIALECT_BUFFERIZATION_TRANSFORMS_ONESHOTANALYSIS_H
