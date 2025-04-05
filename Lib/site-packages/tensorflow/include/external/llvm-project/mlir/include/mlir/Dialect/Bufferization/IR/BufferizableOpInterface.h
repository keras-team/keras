//===- BufferizableOpInterface.h - Bufferizable Ops -------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_BUFFERIZATION_IR_BUFFERIZABLEOPINTERFACE_H_
#define MLIR_DIALECT_BUFFERIZATION_IR_BUFFERIZABLEOPINTERFACE_H_

#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/DenseMapInfoVariant.h"
#include "llvm/ADT/SetVector.h"
#include <optional>

#include "mlir/Dialect/Bufferization/IR/BufferizationEnums.h.inc"

namespace mlir {
class OpBuilder;
namespace func {
class FuncOp;
}

namespace bufferization {

class AnalysisState;
class BufferizableOpInterface;

/// Specifies a fine-grain relationship between buffers to enable more analysis.
enum class BufferRelation {
  Unknown,
  // TODO: ResultContainsOperand,
  // TODO: OperandContainsResult,
  Equivalent
};

/// A maybe aliasing OpOperand. If `isDefinite` is `true`, the OpOperand is
/// guaranteed to alias at runtime.
struct AliasingOpOperand {
  AliasingOpOperand(OpOperand *opOperand, BufferRelation relation,
                    bool isDefinite = true)
      : opOperand(opOperand), relation(relation), isDefinite(isDefinite) {}

  OpOperand *opOperand;
  BufferRelation relation;
  bool isDefinite;
};

/// A maybe aliasing Value. If `isDefinite` is `true`, the Value is guaranteed
/// to alias at runtime.
struct AliasingValue {
  AliasingValue(Value value, BufferRelation relation, bool isDefinite = true)
      : value(value), relation(relation), isDefinite(isDefinite) {}

  Value value;
  BufferRelation relation;
  bool isDefinite;
};

template <typename T> class AliasList {
public:
  /// Create an empty list of aliases.
  AliasList() = default;

  /// Create a list of aliases.
  AliasList(std::initializer_list<T> elems) {
    for (T alias : elems)
      addAlias(alias);
  }

  /// Create a list of aliases.
  AliasList(SmallVector<T> &&aliases) : aliases(std::move(aliases)) {}

  ArrayRef<T> getAliases() const { return aliases; }

  size_t getNumAliases() const { return aliases.size(); }

  void addAlias(T alias) { aliases.push_back(alias); }

  auto begin() const { return aliases.begin(); }
  auto end() const { return aliases.end(); }

private:
  /// The list of aliases.
  SmallVector<T> aliases;
};

/// A list of possible aliasing OpOperands. This list models the runtime
/// aliasing relationship for a Value.
using AliasingOpOperandList = AliasList<AliasingOpOperand>;

/// A list of possible aliasing Values. This list models the runtime aliasing
/// relationship for an OpOperand.
using AliasingValueList = AliasList<AliasingValue>;

class OpFilter {
public:
  /// An op filter entry. Filters can be used to specify which ops should be
  /// processed by the bufferization.
  struct Entry {
    /// If the filter function evaluates to `true`, the filter matches.
    using FilterFn = std::function<bool(Operation *)>;

    /// Filter type: A filter can either be a DENY filter or an ALLOW filter.
    enum FilterType : int8_t { DENY = 0, ALLOW = 1 };

    FilterFn fn;
    FilterType type;
  };

  /// Return whether the op is allowed or not.
  ///
  /// If the filter does not have an ALLOW rule, ops are allowed by default,
  /// unless they are explicitly marked as DENY. If the filter has at least one
  /// ALLOW rule, ops are denied by default and only allowed if they match
  /// an ALLOW rule and no DENY rule.
  bool isOpAllowed(Operation *op) const;

  /// Allow the given dialects.
  ///
  /// This function adds one or multiple ALLOW entries.
  template <typename... DialectTs>
  void allowDialect() {
    // The following expands a call to allowDialectImpl for each dialect
    // in 'DialectTs'.
    (allowDialectImpl<DialectTs>(), ...);
  }

  /// Deny the given dialects.
  ///
  /// This function adds one or multiple DENY entries.
  template <typename... DialectTs>
  void denyDialect() {
    (denyDialectImpl<DialectTs>(), ...);
  }

  /// Allow the given dialect.
  ///
  /// This function adds an ALLOW entry.
  void allowDialect(StringRef dialectNamespace) {
    Entry::FilterFn filterFn = [=](Operation *op) {
      return op->getName().getDialectNamespace() == dialectNamespace;
    };
    entries.push_back(Entry{filterFn, Entry::FilterType::ALLOW});
  }

  /// Deny the given dialect.
  ///
  /// This function adds a DENY entry.
  void denyDialect(StringRef dialectNamespace) {
    Entry::FilterFn filterFn = [=](Operation *op) {
      return op->getName().getDialectNamespace() == dialectNamespace;
    };
    entries.push_back(Entry{filterFn, Entry::FilterType::DENY});
  }

  /// Allow the given ops.
  ///
  /// This function adds one or multiple ALLOW entries.
  template <typename... OpTys>
  void allowOperation() {
    (allowOperationImpl<OpTys>(), ...);
  }

  /// Deny the given ops.
  ///
  /// This function adds one or multiple DENY entries.
  template <typename... OpTys>
  void denyOperation() {
    (denyOperationImpl<OpTys>(), ...);
  }

  /// Allow the given op.
  ///
  /// This function adds an ALLOW entry.
  void allowOperation(StringRef opName) {
    Entry::FilterFn filterFn = [=](Operation *op) {
      return op->getName().getStringRef() == opName;
    };
    allowOperation(filterFn);
  }

  /// Deny the given op.
  ///
  /// This function adds a DENY entry.
  void denyOperation(StringRef opName) {
    Entry::FilterFn filterFn = [=](Operation *op) {
      return op->getName().getStringRef() == opName;
    };
    denyOperation(filterFn);
  }

  /// Allow ops that are matched by `fn`.
  ///
  /// This function adds an ALLOW entry.
  void allowOperation(Entry::FilterFn fn) {
    entries.push_back(Entry{fn, Entry::FilterType::ALLOW});
  }

  /// Deny ops that are matched by `fn`.
  ///
  /// This function adds a DENY entry.
  void denyOperation(Entry::FilterFn fn) {
    entries.push_back(Entry{fn, Entry::FilterType::DENY});
  }

private:
  /// Return `true` if the filter has at least one ALLOW rule.
  bool hasAllowRule() const {
    for (const Entry &e : entries)
      if (e.type == Entry::FilterType::ALLOW)
        return true;
    return false;
  }

  /// Allow a dialect.
  template <typename DialectT>
  void allowDialectImpl() {
    allowDialect(DialectT::getDialectNamespace());
  }

  /// Deny a dialect.
  template <typename DialectT>
  void denyDialectImpl() {
    denyDialect(DialectT::getDialectNamespace());
  }

  /// Allow an op.
  template <typename OpTy>
  void allowOperationImpl() {
    allowOperation(OpTy::getOperationName());
  }

  /// Deny an op.
  template <typename OpTy>
  void denyOperationImpl() {
    denyOperation(OpTy::getOperationName());
  }

  /// A list of filter entries that determine whether an op should be allowed or
  /// denied. If the filter has an ALLOW rule, only ops that are allowed and not
  /// denied are allowed. If the filter does not have an ALLOW rule, only ops
  /// that are not denied are allowed.
  SmallVector<Entry> entries;
};

/// Options for BufferizableOpInterface-based bufferization.
struct BufferizationOptions {
  /// Allocator function: Generate a memref allocation with the given type,
  /// dynamic extents and alignment.
  using AllocationFn = std::function<FailureOr<Value>(
      OpBuilder &, Location, MemRefType, ValueRange, unsigned int)>;
  /// Memcpy function: Generate a memcpy between two buffers.
  using MemCpyFn =
      std::function<LogicalResult(OpBuilder &, Location, Value, Value)>;
  /// Initializer function for analysis state.
  using AnalysisStateInitFn = std::function<void(AnalysisState &)>;
  /// Tensor -> MemRef type converter.
  /// Parameters: Value, memory space, func op, bufferization options
  using FunctionArgTypeConverterFn =
      std::function<BaseMemRefType(TensorType, Attribute memorySpace,
                                   func::FuncOp, const BufferizationOptions &)>;
  /// Tensor -> MemRef type converter.
  /// Parameters: Value, memory space, bufferization options
  using UnknownTypeConverterFn = std::function<BaseMemRefType(
      Value, Attribute memorySpace, const BufferizationOptions &)>;
  // Produce a MemorySpace attribute from a tensor type
  using DefaultMemorySpaceFn =
      std::function<std::optional<Attribute>(TensorType t)>;

  BufferizationOptions();

  /// Try to cast the given op to BufferizableOpInterface if the op is allow
  /// listed.
  BufferizableOpInterface dynCastBufferizableOp(Operation *op) const;

  /// Try to cast the given value to BufferizableOpInterface if the op is allow
  /// listed.
  BufferizableOpInterface dynCastBufferizableOp(Value value) const;

  /// A filter that specifies which ops should be bufferized and which ops
  /// should be ignored.
  OpFilter opFilter;

  /// Return `true` if the given op should be bufferized.
  bool isOpAllowed(Operation *op) const;

  /// Helper functions for allocation and memory copying.
  std::optional<AllocationFn> allocationFn;
  std::optional<MemCpyFn> memCpyFn;

  /// Create a memref allocation with the given type and dynamic extents.
  FailureOr<Value> createAlloc(OpBuilder &b, Location loc, MemRefType type,
                               ValueRange dynShape) const;

  /// Creates a memcpy between two given buffers.
  LogicalResult createMemCpy(OpBuilder &b, Location loc, Value from,
                             Value to) const;

  /// Specifies whether not bufferizable ops are allowed in the input. If so,
  /// bufferization.to_memref and bufferization.to_tensor ops are inserted at
  /// the boundaries.
  bool allowUnknownOps = false;

  /// Specifies whether function boundaries (ops in the func dialect) should be
  /// bufferized or not.
  bool bufferizeFunctionBoundaries = false;

  // Specifies whether to account for parallel regions in RaW analysis. If true,
  // then writes inside of parallel regions that write to buffers defined
  // outside of the parallel region will be given a new buffer.
  bool checkParallelRegions = true;

  /// Certain ops have aliasing OpOperand/OpResult invariants (e.g., scf.for).
  /// If this flag is set to `false`, those invariants are no longer enforced
  /// with buffer copies.
  ///
  /// Note: Deactivating this flag can lead to incorrect bufferization results
  /// when used incorrectly. This flag is useful with
  /// `AlwaysCopyAnalysisState` which bufferizes all writing tensor
  /// OpOperands out-of-place.
  bool enforceAliasingInvariants = true;

  /// This function controls buffer types on function signatures. Sets
  /// `functionArgTypeConverterFn` and `inferFunctionResultLayout` accordingly.
  ///
  /// * InferLayoutMap: All function parameter types have a fully dynamic layout
  ///   map, but function result types are inferred from the body of the
  ///   function.
  /// * FullyDynamicLayoutMap: All function parameter types and result types
  ///   have a fully dynamic layout map. This option is most efficient because
  ///   any layout map can be casted to a fully dynamic one.
  /// * IdentityLayoutMap: All function parameter types and result types have a
  ///   static identity layout (i.e., no layout map). This option may introduce
  ///   additional buffer allocs and copies because layout maps cannot be casted
  ///   away.
  ///
  /// Note: Inferred layout maps may not be desireable when interacting with
  /// external functions, because the generated function signatures will be less
  /// predictable.
  void setFunctionBoundaryTypeConversion(LayoutMapOption layoutMapOption);

  /// Type converter from tensors to memrefs. This type converter is used to
  /// determine bufferized function argument types. By default, a type
  /// converter that returns a memref type with a fully dynamic layout map is
  /// used.
  ///
  /// If `bufferizeFunctionBoundaries` is not set, this function isn't used.
  FunctionArgTypeConverterFn functionArgTypeConverterFn = nullptr;

  /// If true, function result types are inferred from the body of the function.
  /// Otherwise, function result type is determined by
  /// `functionArgTypeConverterFn`.
  ///
  /// If `bufferizeFunctionBoundaries` is not set, this flag has no effect.
  bool inferFunctionResultLayout = true;

  /// Type converter from tensors to memrefs. This type converter is used if no
  /// memref type could be inferred during bufferization. By default, a type
  /// converter that returns a memref type with a fully dynamic layout map is
  /// used.
  UnknownTypeConverterFn unknownTypeConverterFn = nullptr;

  // Use during type conversion to determine the memory space for memref based
  // on the original tensor type if the memory space cannot be inferred.
  // Returning std::nullopt will cause bufferization to fail (useful to indicate
  // failure to determine memory space for a tensor type).
  DefaultMemorySpaceFn defaultMemorySpaceFn =
      [](TensorType t) -> std::optional<Attribute> { return Attribute(); };

  /// If set to `true`, the analysis is skipped. A buffer is copied before every
  /// write. This flag cannot be used together with `testAnalysisOnly = true`.
  bool copyBeforeWrite = false;

  /// If set to `true`, does not modify the IR apart from adding attributes (for
  /// checking the results of the analysis) and post analysis steps.
  bool testAnalysisOnly = false;

  /// If set to `true`, the IR is annotated with details about RaW conflicts.
  /// For debugging only. Should be used together with `testAnalysisOnly`.
  bool printConflicts = false;

  /// Buffer alignment for new memory allocations.
  unsigned int bufferAlignment = 64;

  /// Initializer functions for analysis state. These can be used to
  /// initialize dialect-specific analysis state.
  SmallVector<AnalysisStateInitFn> stateInitializers;
};

/// Traversal parameters for `findValueInReverseUseDefChain`.
struct TraversalConfig {
  /// Specifies if leaves (that do not have further OpOperands to follow)
  /// should be returned even if they do not match the specified filter.
  bool alwaysIncludeLeaves = true;

  /// Specifies whether out-of-place/undecided OpOperands should be followed.
  bool followInPlaceOnly = false;

  /// Specifies whether non-equivalent OpOperands should be followed.
  bool followEquivalentOnly = false;

  /// Specifies whether unknown/non-bufferizable/ops not included in the
  /// OpFilter of BufferizationOptions should be followed.
  bool followUnknownOps = false;

  /// Specifies whether OpOperands with a different type that are not the result
  /// of a CastOpInterface op should be followed.
  bool followSameTypeOrCastsOnly = false;

  /// Specifies whether already visited values should be visited again.
  /// (Note: This can result in infinite looping.)
  bool revisitAlreadyVisitedValues = false;
};

/// AnalysisState provides a variety of helper functions for dealing with
/// tensor values.
class AnalysisState {
public:
  /// Determine which OpOperand* will alias with `value` if the op is
  /// bufferized in place. Return all tensor OpOperand* if the op is not
  /// bufferizable.
  AliasingOpOperandList getAliasingOpOperands(Value value) const;

  /// Determine which Value will alias with `opOperand` if the op is bufferized
  /// in place. Return all tensor Values if the op is not bufferizable.
  AliasingValueList getAliasingValues(OpOperand &opOperand) const;

  /// Return true if `opOperand` bufferizes to a memory read. Return `true` if
  /// the op is not bufferizable.
  bool bufferizesToMemoryRead(OpOperand &opOperand) const;

  /// Return true if `opOperand` bufferizes to a memory write. Return true` if
  /// the op is not bufferizable.
  bool bufferizesToMemoryWrite(OpOperand &opOperand) const;

  /// Return true if the given `value` bufferizes to a memory write. Return
  /// true if the value is a block argument. Return `true` if the defining op is
  /// not bufferizable. Otherwise, consult the BufferizableOpInterface.
  bool bufferizesToMemoryWrite(Value value) const;

  /// Return true if `opOperand` does neither read nor write but bufferizes to
  /// an alias. Return false if the op is not bufferizable.
  bool bufferizesToAliasOnly(OpOperand &opOperand) const;

  /// Return true if a copy can always be avoided when allocating a new tensor
  /// for the given OpOperand.
  bool canOmitTensorCopy(OpOperand &opOperand) const;

  /// Return true if the given value is read by an op that bufferizes to a
  /// memory read. Also takes into account ops that create an alias but do not
  /// read by themselves (e.g., ExtractSliceOp).
  bool isValueRead(Value value) const;

  /// Starting from `value`, follow the use-def chain in reverse, always
  /// selecting the aliasing OpOperands. Find and return Values for which
  /// `condition` evaluates to true. OpOperands of such matching Values are not
  /// traversed any further.
  ///
  /// When reaching the end of a chain, also return the last Value of that
  /// chain if `config.alwaysIncludeLeaves` is set.
  ///
  /// Example:
  ///
  ///                               8
  ///                               |
  ///   6*         7*         +-----+----+
  ///   |          |          |          |
  ///   2*         3          4*         5
  ///   |          |          |          |
  ///   +----------+----------+----------+
  ///              |
  ///              1
  ///
  /// In the above example, Values with a star satisfy the condition. When
  /// starting the traversal from Value 1, the resulting SetVector is:
  /// { 2, 7, 8, 5 }
  ///
  /// Additional stopping conditions for the traversal can be specified in
  /// `config`.
  SetVector<Value> findValueInReverseUseDefChain(
      Value value, llvm::function_ref<bool(Value)> condition,
      TraversalConfig config = TraversalConfig()) const;

  /// Find the values that may define the contents of the given value at
  /// runtime. A block argument is always a definition. An OpResult is a
  /// definition if it bufferizes to memory write. If it does not bufferize to
  /// a memory write but has aliasing operands, we continue the lookup on these
  /// values.
  ///
  /// Example: %r = tensor.insert %f into %t[%c0] : tensor<?xf32>
  /// findDefinitions(%r) = {%r} because %r bufferizes to memory write.
  ///
  /// Example: %r = tensor.empty() : tensor<10xf32>
  /// findDefinitions(%r) = {} because tensor.empty does not the define the
  /// contents of its result (i.e., it does not bufferize to a memory write)
  /// and it has no aliasing OpOperands.
  ///
  /// Example:
  /// %a = arith.constant ... : tensor<10xf32>
  /// %b1 = tensor.insert %f into %t : tensor<50xf32>
  /// %b2 = tensor.extract_slice %b1[0][10][1] : tensor<50xf32> tensor<10xf32>
  /// %r = arith.select %cond, %a, %b : tensor<10xf32>
  /// findDefinitions(%r) = {%a, %b1}. %r and %b2 are skipped (lookup continues
  /// in the operands) because their defining ops do not define the contents of
  /// the tensor.
  ///
  /// Example:
  /// %a = tensor.empty() : tensor<10xf32>
  /// %b = arith.constant ... : tensor<10xf32>
  /// %r = arith.select %cond, %a, %b : tensor<10xf32>
  /// findDefinitions(%r) = {%b}. %a is excluded because it does not define the
  /// contents of the tensor.
  ///
  /// Note: OpResults of unknown ops are handled conservatively and assumed to
  /// be definitions.
  SetVector<Value> findDefinitions(Value value) const;

  /// Return `true` if the given OpResult has been decided to bufferize inplace.
  virtual bool isInPlace(OpOperand &opOperand) const;

  /// Return true if `v1` and `v2` bufferize to equivalent buffers.
  virtual bool areEquivalentBufferizedValues(Value v1, Value v2) const;

  /// Return true if `v1` and `v2` may bufferize to aliasing buffers.
  virtual bool areAliasingBufferizedValues(Value v1, Value v2) const;

  /// Return `true` if the given tensor has undefined contents.
  virtual bool hasUndefinedContents(OpOperand *opOperand) const;

  /// Return a reference to the BufferizationOptions.
  const BufferizationOptions &getOptions() const { return options; }

  AnalysisState(const BufferizationOptions &options);

  // AnalysisState should be passed as a reference.
  AnalysisState(const AnalysisState &) = delete;

  virtual ~AnalysisState() = default;

  static bool classof(const AnalysisState *base) { return true; }

  TypeID getType() const { return type; }

  /// Return the closest enclosing repetitive region around the given op.
  Region *getEnclosingRepetitiveRegion(Operation *op,
                                       const BufferizationOptions &options);

  /// Return the closest enclosing repetitive region around the place where the
  /// given value is defined.
  Region *getEnclosingRepetitiveRegion(Value value,
                                       const BufferizationOptions &options);

  /// Return the closest enclosing repetitive region around the given block.
  Region *getEnclosingRepetitiveRegion(Block *block,
                                       const BufferizationOptions &options);

  virtual void resetCache();

protected:
  AnalysisState(const BufferizationOptions &options, TypeID type);

private:
  /// A reference to current bufferization options.
  const BufferizationOptions &options;

  /// The type of analysis.
  TypeID type;

  /// Cache containing closest ancestor repetitive Region.
  DenseMap<std::variant<Operation *, Block *, Region *, Value>, Region *>
      enclosingRepetitiveRegionCache;
};

/// Create an AllocTensorOp for the given shaped value (memref or tensor).
/// If `copy` is set, the shaped value is copied. Otherwise, a tensor with
/// undefined contents is allocated.
FailureOr<Value>
allocateTensorForShapedValue(OpBuilder &b, Location loc, Value shapedValue,
                             const BufferizationOptions &options,
                             bool copy = true);

/// Lookup the buffer for the given value. If the value was not bufferized
/// yet, wrap it in a ToMemrefOp. Otherwise, it is the result of a ToTensorOp,
/// from which the memref operand is returned.
FailureOr<Value> getBuffer(RewriterBase &rewriter, Value value,
                           const BufferizationOptions &options);

/// Return the buffer type for a given Value (tensor) after bufferization
/// without bufferizing any IR.
///
/// Note: It should be sufficient to call `getBuffer()->getType()` in most
/// cases. However, when a buffer type should be predicted without modifying any
/// IR, this function can be used.
///
/// This function is a wrapper around BufferizableOpInterface::getBufferType.
FailureOr<BaseMemRefType> getBufferType(Value value,
                                        const BufferizationOptions &options);

/// Return the buffer type for a given Value (tensor) after bufferization
/// without bufferizing any IR. This function (and not the other overload
/// without `invocationStack`) can be used from `getBufferType` implementations
/// of the `BufferizableOpInterface`.
///
/// Note: It should be sufficient to call `getBuffer()->getType()` in most
/// cases. However, when a buffer type should be predicted without modifying any
/// IR, this function can be used.
///
/// This function is a wrapper around `BufferizableOpInterface::getBufferType`.
FailureOr<BaseMemRefType> getBufferType(Value value,
                                        const BufferizationOptions &options,
                                        SmallVector<Value> &invocationStack);

/// Return "true" if the given op has tensor semantics and should be bufferized.
/// If the op is bufferizable, the BufferizableOpInterface is queried.
/// Otherwise, an op has tensor semantics if it has tensor operands, tensor
/// op results and/or tensor block arguments.
bool hasTensorSemantics(Operation *op);

/// Replace an op with replacement values. The op is deleted. Tensor OpResults
/// must be replaced with memref values.
void replaceOpWithBufferizedValues(RewriterBase &rewriter, Operation *op,
                                   ValueRange values);

/// Replace an op with a new op. The new op must have the same number of
/// results as the replaced op. The new op may not return any tensor values.
template <typename OpTy, typename... Args>
OpTy replaceOpWithNewBufferizedOp(RewriterBase &rewriter, Operation *op,
                                  Args &&...args) {
  auto newOp = rewriter.create<OpTy>(op->getLoc(), std::forward<Args>(args)...);
  replaceOpWithBufferizedValues(rewriter, op, newOp->getResults());
  return newOp;
}

/// Return a MemRefType to which the type of the given value can be bufferized.
///
/// If possible, op bufferization implementations should not use this function
/// and instead infer precise memref types for tensor results by themselves.
///
/// Unless a layout map was specified, `options.unknownTypeConverterFn`
/// determines what kind of layout map will be used. For best composability
/// (without copies), the fully dynamic layout map is used by default.
///
/// Note: Canonicalization patterns could clean up layout maps and infer more
/// precise layout maps after bufferization. However, many possible
/// canonicalizations are currently not implemented.
BaseMemRefType getMemRefType(Value value, const BufferizationOptions &options,
                             MemRefLayoutAttrInterface layout = {},
                             Attribute memorySpace = nullptr);

/// Return a MemRef type with fully dynamic layout. If the given tensor type
/// is unranked, return an unranked MemRef type.
BaseMemRefType
getMemRefTypeWithFullyDynamicLayout(TensorType tensorType,
                                    Attribute memorySpace = nullptr);

/// Return a MemRef type with a static identity layout (i.e., no layout map). If
/// the given tensor type is unranked, return an unranked MemRef type.
BaseMemRefType
getMemRefTypeWithStaticIdentityLayout(TensorType tensorType,
                                      Attribute memorySpace = nullptr);

/// Return the owner of the given value. In case of a BlockArgument that is the
/// owner of the block. In case of an OpResult that is the defining op.
Operation *getOwnerOfValue(Value value);

/// Assuming that the given region is repetitive, find the next enclosing
/// repetitive region.
Region *getNextEnclosingRepetitiveRegion(Region *region,
                                         const BufferizationOptions &options);

/// If `region` is a parallel region, return `region`. Otherwise, find the first
/// enclosing parallel region of `region`. If there is no such region, return
/// "nullptr".
///
/// Note: Whether a region is parallel or sequential is queried from the
/// `BufferizableOpInterface`.
Region *getParallelRegion(Region *region, const BufferizationOptions &options);

namespace detail {
/// This is the default implementation of
/// BufferizableOpInterface::getAliasingOpOperands. Should not be called from
/// other places.
AliasingOpOperandList defaultGetAliasingOpOperands(Value value,
                                                   const AnalysisState &state);

/// This is the default implementation of
/// BufferizableOpInterface::getBufferType. Should not be called from other
/// places.
FailureOr<BaseMemRefType>
defaultGetBufferType(Value value, const BufferizationOptions &options,
                     SmallVector<Value> &invocationStack);

/// This is the default implementation of
/// BufferizableOpInterface::resultBufferizesToMemoryWrite. Should not be called
/// from other places.
bool defaultResultBufferizesToMemoryWrite(OpResult opResult,
                                          const AnalysisState &state);

/// This is the default implementation of
/// BufferizableOpInterface::isRepetitiveRegion. Should not be called from other
/// places.
bool defaultIsRepetitiveRegion(BufferizableOpInterface bufferizableOp,
                               unsigned index);

/// This is the default implementation of getAliasingOpOperands in case the
/// defining op does not implement the BufferizableOpInterface.
AliasingOpOperandList unknownGetAliasingOpOperands(Value value);

/// This is the default implementation of getAliasingValues in case the owner
/// op does not implement the BufferizableOpInterface.
AliasingValueList unknownGetAliasingValues(OpOperand &opOperand);

/// This is the default implementation of
/// BufferizableOpInterface::hasTensorSemantics
bool defaultHasTensorSemantics(Operation *op);
} // namespace detail

} // namespace bufferization
} // namespace mlir

MLIR_DECLARE_EXPLICIT_TYPE_ID(mlir::bufferization::AnalysisState)

//===----------------------------------------------------------------------===//
// Bufferization Interfaces
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Bufferization/IR/BufferizableOpInterface.h.inc"

#endif // MLIR_DIALECT_BUFFERIZATION_IR_BUFFERIZABLEOPINTERFACE_H_
