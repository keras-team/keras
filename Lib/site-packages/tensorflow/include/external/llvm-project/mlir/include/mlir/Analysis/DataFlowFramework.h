//===- DataFlowFramework.h - A generic framework for data-flow analysis ---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines a generic framework for writing data-flow analysis in MLIR.
// The framework consists of a solver, which runs the fixed-point iteration and
// manages analysis dependencies, and a data-flow analysis class used to
// implement specific analyses.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_ANALYSIS_DATAFLOWFRAMEWORK_H
#define MLIR_ANALYSIS_DATAFLOWFRAMEWORK_H

#include "mlir/IR/Operation.h"
#include "mlir/Support/StorageUniquer.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/TypeName.h"
#include <queue>

namespace mlir {

//===----------------------------------------------------------------------===//
// ChangeResult
//===----------------------------------------------------------------------===//

/// A result type used to indicate if a change happened. Boolean operations on
/// ChangeResult behave as though `Change` is truth.
enum class [[nodiscard]] ChangeResult {
  NoChange,
  Change,
};
inline ChangeResult operator|(ChangeResult lhs, ChangeResult rhs) {
  return lhs == ChangeResult::Change ? lhs : rhs;
}
inline ChangeResult &operator|=(ChangeResult &lhs, ChangeResult rhs) {
  lhs = lhs | rhs;
  return lhs;
}
inline ChangeResult operator&(ChangeResult lhs, ChangeResult rhs) {
  return lhs == ChangeResult::NoChange ? lhs : rhs;
}

/// Forward declare the analysis state class.
class AnalysisState;

/// Program point represents a specific location in the execution of a program.
/// A sequence of program points can be combined into a control flow graph.
struct ProgramPoint : public PointerUnion<Operation *, Block *> {
  using ParentTy = PointerUnion<Operation *, Block *>;
  /// Inherit constructors.
  using ParentTy::PointerUnion;
  /// Allow implicit conversion from the parent type.
  ProgramPoint(ParentTy point = nullptr) : ParentTy(point) {}
  /// Allow implicit conversions from operation wrappers.
  /// TODO: For Windows only. Find a better solution.
  template <typename OpT, typename = std::enable_if_t<
                              std::is_convertible<OpT, Operation *>::value &&
                              !std::is_same<OpT, Operation *>::value>>
  ProgramPoint(OpT op) : ParentTy(op) {}

  /// Print the program point.
  void print(raw_ostream &os) const;
};

//===----------------------------------------------------------------------===//
// GenericLatticeAnchor
//===----------------------------------------------------------------------===//

/// Abstract class for generic lattice anchor. In classical data-flow analysis,
/// lattice anchor represent positions in a program to which lattice elements
/// are attached. In sparse data-flow analysis, these can be SSA values, and in
/// dense data-flow analysis, these are the program points before and after
/// every operation.
///
/// Lattice anchor are implemented using MLIR's storage uniquer framework and
/// type ID system to provide RTTI.
class GenericLatticeAnchor : public StorageUniquer::BaseStorage {
public:
  virtual ~GenericLatticeAnchor();

  /// Get the abstract lattice anchor's type identifier.
  TypeID getTypeID() const { return typeID; }

  /// Get a derived source location for the lattice anchor.
  virtual Location getLoc() const = 0;

  /// Print the lattice anchor.
  virtual void print(raw_ostream &os) const = 0;

protected:
  /// Create an abstract lattice anchor with type identifier.
  explicit GenericLatticeAnchor(TypeID typeID) : typeID(typeID) {}

private:
  /// The type identifier of the lattice anchor.
  TypeID typeID;
};

//===----------------------------------------------------------------------===//
// GenericLatticeAnchorBase
//===----------------------------------------------------------------------===//

/// Base class for generic lattice anchor based on a concrete lattice anchor
/// type and a content key. This class defines the common methods required for
/// operability with the storage uniquer framework.
///
/// The provided key type uniquely identifies the concrete lattice anchor
/// instance and are the data members of the class.
template <typename ConcreteT, typename Value>
class GenericLatticeAnchorBase : public GenericLatticeAnchor {
public:
  /// The concrete key type used by the storage uniquer. This class is uniqued
  /// by its contents.
  using KeyTy = Value;
  /// Alias for the base class.
  using Base = GenericLatticeAnchorBase<ConcreteT, Value>;

  /// Construct an instance of the lattice anchor using the provided value and
  /// the type ID of the concrete type.
  template <typename ValueT>
  explicit GenericLatticeAnchorBase(ValueT &&value)
      : GenericLatticeAnchor(TypeID::get<ConcreteT>()),
        value(std::forward<ValueT>(value)) {}

  /// Get a uniqued instance of this lattice anchor class with the given
  /// arguments.
  template <typename... Args>
  static ConcreteT *get(StorageUniquer &uniquer, Args &&...args) {
    return uniquer.get<ConcreteT>(/*initFn=*/{}, std::forward<Args>(args)...);
  }

  /// Allocate space for a lattice anchor and construct it in-place.
  template <typename ValueT>
  static ConcreteT *construct(StorageUniquer::StorageAllocator &alloc,
                              ValueT &&value) {
    return new (alloc.allocate<ConcreteT>())
        ConcreteT(std::forward<ValueT>(value));
  }

  /// Two lattice anchors are equal if their values are equal.
  bool operator==(const Value &value) const { return this->value == value; }

  /// Provide LLVM-style RTTI using type IDs.
  static bool classof(const GenericLatticeAnchor *point) {
    return point->getTypeID() == TypeID::get<ConcreteT>();
  }

  /// Get the contents of the lattice anchor.
  const Value &getValue() const { return value; }

private:
  /// The lattice anchor value.
  Value value;
};

//===----------------------------------------------------------------------===//
// LatticeAnchor
//===----------------------------------------------------------------------===//

/// Fundamental IR components are supported as first-class lattice anchor.
struct LatticeAnchor
    : public PointerUnion<GenericLatticeAnchor *, ProgramPoint, Value> {
  using ParentTy = PointerUnion<GenericLatticeAnchor *, ProgramPoint, Value>;
  /// Inherit constructors.
  using ParentTy::PointerUnion;
  /// Allow implicit conversion from the parent type.
  LatticeAnchor(ParentTy point = nullptr) : ParentTy(point) {}
  /// Allow implicit conversions from operation wrappers.
  /// TODO: For Windows only. Find a better solution.
  template <typename OpT, typename = std::enable_if_t<
                              std::is_convertible<OpT, Operation *>::value &&
                              !std::is_same<OpT, Operation *>::value>>
  LatticeAnchor(OpT op) : ParentTy(ProgramPoint(op)) {}

  LatticeAnchor(Operation *op) : ParentTy(ProgramPoint(op)) {}
  LatticeAnchor(Block *block) : ParentTy(ProgramPoint(block)) {}

  /// Print the lattice anchor.
  void print(raw_ostream &os) const;

  /// Get the source location of the lattice anchor.
  Location getLoc() const;
};

/// Forward declaration of the data-flow analysis class.
class DataFlowAnalysis;

//===----------------------------------------------------------------------===//
// DataFlowConfig
//===----------------------------------------------------------------------===//

/// Configuration class for data flow solver and child analyses. Follows the
/// fluent API pattern.
class DataFlowConfig {
public:
  DataFlowConfig() = default;

  /// Set whether the solver should operate interpocedurally, i.e. enter the
  /// callee body when available. Interprocedural analyses may be more precise,
  /// but also more expensive as more states need to be computed and the
  /// fixpoint convergence takes longer.
  DataFlowConfig &setInterprocedural(bool enable) {
    interprocedural = enable;
    return *this;
  }

  /// Return `true` if the solver operates interprocedurally, `false` otherwise.
  bool isInterprocedural() const { return interprocedural; }

private:
  bool interprocedural = true;
};

//===----------------------------------------------------------------------===//
// DataFlowSolver
//===----------------------------------------------------------------------===//

/// The general data-flow analysis solver. This class is responsible for
/// orchestrating child data-flow analyses, running the fixed-point iteration
/// algorithm, managing analysis state and lattice anchor memory, and tracking
/// dependencies between analyses, lattice anchor, and analysis states.
///
/// Steps to run a data-flow analysis:
///
/// 1. Load and initialize children analyses. Children analyses are instantiated
///    in the solver and initialized, building their dependency relations.
/// 2. Configure and run the analysis. The solver invokes the children analyses
///    according to their dependency relations until a fixed point is reached.
/// 3. Query analysis state results from the solver.
///
/// TODO: Optimize the internal implementation of the solver.
class DataFlowSolver {
public:
  explicit DataFlowSolver(const DataFlowConfig &config = DataFlowConfig())
      : config(config) {}

  /// Load an analysis into the solver. Return the analysis instance.
  template <typename AnalysisT, typename... Args>
  AnalysisT *load(Args &&...args);

  /// Initialize the children analyses starting from the provided top-level
  /// operation and run the analysis until fixpoint.
  LogicalResult initializeAndRun(Operation *top);

  /// Lookup an analysis state for the given lattice anchor. Returns null if one
  /// does not exist.
  template <typename StateT, typename AnchorT>
  const StateT *lookupState(AnchorT anchor) const {
    auto it =
        analysisStates.find({LatticeAnchor(anchor), TypeID::get<StateT>()});
    if (it == analysisStates.end())
      return nullptr;
    return static_cast<const StateT *>(it->second.get());
  }

  /// Erase any analysis state associated with the given lattice anchor.
  template <typename AnchorT>
  void eraseState(AnchorT anchor) {
    LatticeAnchor la(anchor);

    for (auto it = analysisStates.begin(); it != analysisStates.end(); ++it) {
      if (it->first.first == la)
        analysisStates.erase(it);
    }
  }

  /// Get a uniqued lattice anchor instance. If one is not present, it is
  /// created with the provided arguments.
  template <typename AnchorT, typename... Args>
  AnchorT *getLatticeAnchor(Args &&...args) {
    return AnchorT::get(uniquer, std::forward<Args>(args)...);
  }

  /// A work item on the solver queue is a program point, child analysis pair.
  /// Each item is processed by invoking the child analysis at the program
  /// point.
  using WorkItem = std::pair<ProgramPoint, DataFlowAnalysis *>;
  /// Push a work item onto the worklist.
  void enqueue(WorkItem item) { worklist.push(std::move(item)); }

  /// Get the state associated with the given lattice anchor. If it does not
  /// exist, create an uninitialized state.
  template <typename StateT, typename AnchorT>
  StateT *getOrCreateState(AnchorT anchor);

  /// Propagate an update to an analysis state if it changed by pushing
  /// dependent work items to the back of the queue.
  void propagateIfChanged(AnalysisState *state, ChangeResult changed);

  /// Get the configuration of the solver.
  const DataFlowConfig &getConfig() const { return config; }

private:
  /// Configuration of the dataflow solver.
  DataFlowConfig config;

  /// The solver's work queue. Work items can be inserted to the front of the
  /// queue to be processed greedily, speeding up computations that otherwise
  /// quickly degenerate to quadratic due to propagation of state updates.
  std::queue<WorkItem> worklist;

  /// Type-erased instances of the children analyses.
  SmallVector<std::unique_ptr<DataFlowAnalysis>> childAnalyses;

  /// The storage uniquer instance that owns the memory of the allocated lattice
  /// anchors
  StorageUniquer uniquer;

  /// A type-erased map of lattice anchors to associated analysis states for
  /// first-class lattice anchors.
  DenseMap<std::pair<LatticeAnchor, TypeID>, std::unique_ptr<AnalysisState>>
      analysisStates;

  /// Allow the base child analysis class to access the internals of the solver.
  friend class DataFlowAnalysis;
};

//===----------------------------------------------------------------------===//
// AnalysisState
//===----------------------------------------------------------------------===//

/// Base class for generic analysis states. Analysis states contain data-flow
/// information that are attached to lattice anchors and which evolve as the
/// analysis iterates.
///
/// This class places no restrictions on the semantics of analysis states beyond
/// these requirements.
///
/// 1. Querying the state of a lattice anchor prior to visiting that anchor
///    results in uninitialized state. Analyses must be aware of unintialized
///    states.
/// 2. Analysis states can reach fixpoints, where subsequent updates will never
///    trigger a change in the state.
/// 3. Analysis states that are uninitialized can be forcefully initialized to a
///    default value.
class AnalysisState {
public:
  virtual ~AnalysisState();

  /// Create the analysis state at the given lattice anchor.
  AnalysisState(LatticeAnchor anchor) : anchor(anchor) {}

  /// Returns the lattice anchor this state is located at.
  LatticeAnchor getAnchor() const { return anchor; }

  /// Print the contents of the analysis state.
  virtual void print(raw_ostream &os) const = 0;
  LLVM_DUMP_METHOD void dump() const;

  /// Add a dependency to this analysis state on a lattice anchor and an
  /// analysis. If this state is updated, the analysis will be invoked on the
  /// given lattice anchor again (in onUpdate()).
  void addDependency(ProgramPoint point, DataFlowAnalysis *analysis);

protected:
  /// This function is called by the solver when the analysis state is updated
  /// to enqueue more work items. For example, if a state tracks dependents
  /// through the IR (e.g. use-def chains), this function can be implemented to
  /// push those dependents on the worklist.
  virtual void onUpdate(DataFlowSolver *solver) const {
    for (const DataFlowSolver::WorkItem &item : dependents)
      solver->enqueue(item);
  }

  /// The lattice anchor to which the state belongs.
  LatticeAnchor anchor;

#if LLVM_ENABLE_ABI_BREAKING_CHECKS
  /// When compiling with debugging, keep a name for the analysis state.
  StringRef debugName;
#endif // LLVM_ENABLE_ABI_BREAKING_CHECKS

private:
  /// The dependency relations originating from this analysis state. An entry
  /// `state -> (analysis, anchor)` is created when `analysis` queries `state`
  /// when updating `anchor`.
  ///
  /// When this state is updated, all dependent child analysis invocations are
  /// pushed to the back of the queue. Use a `SetVector` to keep the analysis
  /// deterministic.
  ///
  /// Store the dependents on the analysis state for efficiency.
  SetVector<DataFlowSolver::WorkItem> dependents;

  /// Allow the framework to access the dependents.
  friend class DataFlowSolver;
};

//===----------------------------------------------------------------------===//
// DataFlowAnalysis
//===----------------------------------------------------------------------===//

/// Base class for all data-flow analyses. A child analysis is expected to build
/// an initial dependency graph (and optionally provide an initial state) when
/// initialized and define transfer functions when visiting program points.
///
/// In classical data-flow analysis, the dependency graph is fixed and analyses
/// define explicit transfer functions between input states and output states.
/// In this framework, however, the dependency graph can change during the
/// analysis, and transfer functions are opaque such that the solver doesn't
/// know what states calling `visit` on an analysis will be updated. This allows
/// multiple analyses to plug in and provide values for the same state.
///
/// Generally, when an analysis queries an uninitialized state, it is expected
/// to "bail out", i.e., not provide any updates. When the value is initialized,
/// the solver will re-invoke the analysis. If the solver exhausts its worklist,
/// however, and there are still uninitialized states, the solver "nudges" the
/// analyses by default-initializing those states.
class DataFlowAnalysis {
public:
  virtual ~DataFlowAnalysis();

  /// Create an analysis with a reference to the parent solver.
  explicit DataFlowAnalysis(DataFlowSolver &solver);

  /// Initialize the analysis from the provided top-level operation by building
  /// an initial dependency graph between all lattice anchors of interest. This
  /// can be implemented by calling `visit` on all program points of interest
  /// below the top-level operation.
  ///
  /// An analysis can optionally provide initial values to certain analysis
  /// states to influence the evolution of the analysis.
  virtual LogicalResult initialize(Operation *top) = 0;

  /// Visit the given program point. This function is invoked by the solver on
  /// this analysis with a given program point when a dependent analysis state
  /// is updated. The function is similar to a transfer function; it queries
  /// certain analysis states and sets other states.
  ///
  /// The function is expected to create dependencies on queried states and
  /// propagate updates on changed states. A dependency can be created by
  /// calling `addDependency` between the input state and a program point,
  /// indicating that, if the state is updated, the solver should invoke `solve`
  /// on the program point. The dependent point does not have to be the same as
  /// the provided point. An update to a state is propagated by calling
  /// `propagateIfChange` on the state. If the state has changed, then all its
  /// dependents are placed on the worklist.
  ///
  /// The dependency graph does not need to be static. Each invocation of
  /// `visit` can add new dependencies, but these dependencies will not be
  /// dynamically added to the worklist because the solver doesn't know what
  /// will provide a value for then.
  virtual LogicalResult visit(ProgramPoint point) = 0;

protected:
  /// Create a dependency between the given analysis state and lattice anchor
  /// on this analysis.
  void addDependency(AnalysisState *state, ProgramPoint point);

  /// Propagate an update to a state if it changed.
  void propagateIfChanged(AnalysisState *state, ChangeResult changed);

  /// Register a custom lattice anchor class.
  template <typename AnchorT>
  void registerAnchorKind() {
    solver.uniquer.registerParametricStorageType<AnchorT>();
  }

  /// Get or create a custom lattice anchor.
  template <typename AnchorT, typename... Args>
  AnchorT *getLatticeAnchor(Args &&...args) {
    return solver.getLatticeAnchor<AnchorT>(std::forward<Args>(args)...);
  }

  /// Get the analysis state associated with the lattice anchor. The returned
  /// state is expected to be "write-only", and any updates need to be
  /// propagated by `propagateIfChanged`.
  template <typename StateT, typename AnchorT>
  StateT *getOrCreate(AnchorT anchor) {
    return solver.getOrCreateState<StateT>(anchor);
  }

  /// Get a read-only analysis state for the given point and create a dependency
  /// on `dependent`. If the return state is updated elsewhere, this analysis is
  /// re-invoked on the dependent.
  template <typename StateT, typename AnchorT>
  const StateT *getOrCreateFor(ProgramPoint dependent, AnchorT anchor) {
    StateT *state = getOrCreate<StateT>(anchor);
    addDependency(state, dependent);
    return state;
  }

  /// Return the configuration of the solver used for this analysis.
  const DataFlowConfig &getSolverConfig() const { return solver.getConfig(); }

#if LLVM_ENABLE_ABI_BREAKING_CHECKS
  /// When compiling with debugging, keep a name for the analyis.
  StringRef debugName;
#endif // LLVM_ENABLE_ABI_BREAKING_CHECKS

private:
  /// The parent data-flow solver.
  DataFlowSolver &solver;

  /// Allow the data-flow solver to access the internals of this class.
  friend class DataFlowSolver;
};

template <typename AnalysisT, typename... Args>
AnalysisT *DataFlowSolver::load(Args &&...args) {
  childAnalyses.emplace_back(new AnalysisT(*this, std::forward<Args>(args)...));
#if LLVM_ENABLE_ABI_BREAKING_CHECKS
  childAnalyses.back().get()->debugName = llvm::getTypeName<AnalysisT>();
#endif // LLVM_ENABLE_ABI_BREAKING_CHECKS
  return static_cast<AnalysisT *>(childAnalyses.back().get());
}

template <typename StateT, typename AnchorT>
StateT *DataFlowSolver::getOrCreateState(AnchorT anchor) {
  std::unique_ptr<AnalysisState> &state =
      analysisStates[{LatticeAnchor(anchor), TypeID::get<StateT>()}];
  if (!state) {
    state = std::unique_ptr<StateT>(new StateT(anchor));
#if LLVM_ENABLE_ABI_BREAKING_CHECKS
    state->debugName = llvm::getTypeName<StateT>();
#endif // LLVM_ENABLE_ABI_BREAKING_CHECKS
  }
  return static_cast<StateT *>(state.get());
}

inline raw_ostream &operator<<(raw_ostream &os, const AnalysisState &state) {
  state.print(os);
  return os;
}

inline raw_ostream &operator<<(raw_ostream &os, LatticeAnchor anchor) {
  anchor.print(os);
  return os;
}

} // end namespace mlir

namespace llvm {
/// Allow hashing of lattice anchors and program points.
template <>
struct DenseMapInfo<mlir::LatticeAnchor>
    : public DenseMapInfo<mlir::LatticeAnchor::ParentTy> {};

template <>
struct DenseMapInfo<mlir::ProgramPoint>
    : public DenseMapInfo<mlir::ProgramPoint::ParentTy> {};

// Allow llvm::cast style functions.
template <typename To>
struct CastInfo<To, mlir::LatticeAnchor>
    : public CastInfo<To, mlir::LatticeAnchor::PointerUnion> {};

template <typename To>
struct CastInfo<To, const mlir::LatticeAnchor>
    : public CastInfo<To, const mlir::LatticeAnchor::PointerUnion> {};

template <typename To>
struct CastInfo<To, mlir::ProgramPoint>
    : public CastInfo<To, mlir::ProgramPoint::PointerUnion> {};

template <typename To>
struct CastInfo<To, const mlir::ProgramPoint>
    : public CastInfo<To, const mlir::ProgramPoint::PointerUnion> {};

/// Allow stealing the low bits of a ProgramPoint.
template <>
struct PointerLikeTypeTraits<mlir::ProgramPoint>
    : public PointerLikeTypeTraits<mlir::ProgramPoint::ParentTy> {};

} // end namespace llvm

#endif // MLIR_ANALYSIS_DATAFLOWFRAMEWORK_H
