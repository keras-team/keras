//===- TransformInterfaces.h - Transform Dialect Interfaces -----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_TRANSFORM_INTERFACES_TRANSFORMINTERFACES_H
#define MLIR_DIALECT_TRANSFORM_INTERFACES_TRANSFORMINTERFACES_H

#include "mlir/Dialect/Transform/Utils/DiagnosedSilenceableFailure.h"
#include "mlir/Dialect/Transform/Utils/RaggedArray.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Transforms/DialectConversion.h"

#include "mlir/Dialect/Transform/Interfaces/TransformTypeInterfaces.h.inc"

namespace mlir {
namespace transform {

class TransformOpInterface;
class TransformResults;
class TransformRewriter;
class TransformState;

using Param = Attribute;
using MappedValue = llvm::PointerUnion<Operation *, Param, Value>;

namespace detail {
/// Maps the only block argument of the op with PossibleTopLevelTransformOpTrait
/// to either the list of operations associated with its operand or the root of
/// the payload IR, depending on what is available in the context.
LogicalResult
mapPossibleTopLevelTransformOpBlockArguments(TransformState &state,
                                             Operation *op, Region &region);

/// Verification hook for PossibleTopLevelTransformOpTrait.
LogicalResult verifyPossibleTopLevelTransformOpTrait(Operation *op);

/// Populates `effects` with side effects implied by
/// PossibleTopLevelTransformOpTrait for the given operation. The operation may
/// have an optional `root` operand, indicating it is not in fact top-level. It
/// is also expected to have a single-block body.
void getPotentialTopLevelEffects(
    Operation *operation, Value root, Block &body,
    SmallVectorImpl<MemoryEffects::EffectInstance> &effects);

/// Verification hook for TransformOpInterface.
LogicalResult verifyTransformOpInterface(Operation *op);

/// Appends the entities associated with the given transform values in `state`
/// to the pre-existing list of mappings. The array of mappings must have as
/// many elements as values. If `flatten` is set, multiple values may be
/// associated with each transform value, and this always succeeds. Otherwise,
/// checks that each value has exactly one mapping associated and return failure
/// otherwise.
LogicalResult appendValueMappings(
    MutableArrayRef<SmallVector<transform::MappedValue>> mappings,
    ValueRange values, const transform::TransformState &state,
    bool flatten = true);

/// Populates `mappings` with mapped values associated with the given transform
/// IR values in the given `state`.
void prepareValueMappings(
    SmallVectorImpl<SmallVector<transform::MappedValue>> &mappings,
    ValueRange values, const transform::TransformState &state);

/// Populates `results` with payload associations that match exactly those of
/// the operands to `block`'s terminator.
void forwardTerminatorOperands(Block *block, transform::TransformState &state,
                               transform::TransformResults &results);

/// Make a dummy transform state for testing purposes. This MUST NOT be used
/// outside of test cases.
TransformState makeTransformStateForTesting(Region *region,
                                            Operation *payloadRoot);

/// Returns all operands that are handles and being consumed by the given op.
SmallVector<OpOperand *>
getConsumedHandleOpOperands(transform::TransformOpInterface transformOp);
} // namespace detail
} // namespace transform
} // namespace mlir

#include "mlir/Dialect/Transform/Interfaces/TransformInterfaces.h.inc"

namespace mlir {
namespace transform {

/// Options controlling the application of transform operations by the
/// TransformState.
class TransformOptions {
public:
  TransformOptions() = default;
  TransformOptions(const TransformOptions &) = default;
  TransformOptions &operator=(const TransformOptions &) = default;

  /// Requests computationally expensive checks of the transform and payload IR
  /// well-formedness to be performed before each transformation. In particular,
  /// these ensure that the handles still point to valid operations when used.
  TransformOptions &enableExpensiveChecks(bool enable = true) {
    expensiveChecksEnabled = enable;
    return *this;
  }

  // Ensures that only a single top-level transform op is present in the IR.
  TransformOptions &enableEnforceSingleToplevelTransformOp(bool enable = true) {
    enforceSingleToplevelTransformOp = enable;
    return *this;
  }

  /// Returns true if the expensive checks are requested.
  bool getExpensiveChecksEnabled() const { return expensiveChecksEnabled; }

  // Returns true if enforcing a single top-level transform op is requested.
  bool getEnforceSingleToplevelTransformOp() const {
    return enforceSingleToplevelTransformOp;
  }

private:
  bool expensiveChecksEnabled = true;
  bool enforceSingleToplevelTransformOp = true;
};

/// Entry point to the Transform dialect infrastructure. Applies the
/// transformation specified by `transform` to payload IR contained in
/// `payloadRoot`. The `transform` operation may contain other operations that
/// will be executed following the internal logic of the operation. It must
/// have the `PossibleTopLevelTransformOp` trait and not have any operands.
/// This function internally keeps track of the transformation state.
LogicalResult applyTransforms(
    Operation *payloadRoot, TransformOpInterface transform,
    const RaggedArray<MappedValue> &extraMapping = {},
    const TransformOptions &options = TransformOptions(),
    bool enforceToplevelTransformOp = true,
    function_ref<void(TransformState &)> stateInitializer = nullptr,
    function_ref<LogicalResult(TransformState &)> stateExporter = nullptr);

/// The state maintained across applications of various ops implementing the
/// TransformOpInterface. The operations implementing this interface and the
/// surrounding structure are referred to as transform IR. The operations to
/// which transformations apply are referred to as payload IR. Transform IR
/// operates on values that can be associated either with a list of payload IR
/// operations (such values are referred to as handles) or with a list of
/// parameters represented as attributes. The state thus contains the mapping
/// between values defined in the transform IR ops and either payload IR ops or
/// parameters. For payload ops, the mapping is many-to-many and the reverse
/// mapping is also stored. The "expensive-checks" option can be passed to the
/// constructor at transformation execution time that transform IR values used
/// as operands by a transform IR operation are not associated with dangling
/// pointers to payload IR operations that are known to have been erased by
/// previous transformation through the same or a different transform IR value.
///
/// A reference to this class is passed as an argument to "apply" methods of the
/// transform op interface. Thus the "apply" method can call either
/// `state.getPayloadOps( getSomeOperand() )` to obtain the list of operations
/// or `state.getParams( getSomeOperand() )` to obtain the list of parameters
/// associated with its operand. The method is expected to populate the
/// `TransformResults` class instance in order to update the mapping. The
/// `applyTransform` method takes care of propagating the state of
/// `TransformResults` into the instance of this class.
///
/// When applying transform IR operations with regions, the client is expected
/// to create a `RegionScope` RAII object to create a new "stack frame" for
/// values defined inside the region. The mappings from and to these values will
/// be automatically dropped when the object goes out of scope, typically at the
/// end of the `apply` function of the parent operation. If a region contains
/// blocks with arguments, the client can map those arguments to payload IR ops
/// using `mapBlockArguments`.
class TransformState {
public:
  using Param = transform::Param;

private:
  /// Mapping between a Value in the transform IR and the corresponding set of
  /// operations in the payload IR.
  using TransformOpMapping = DenseMap<Value, SmallVector<Operation *, 2>>;

  /// Mapping between a payload IR operation and the transform IR values it is
  /// associated with.
  using TransformOpReverseMapping =
      DenseMap<Operation *, SmallVector<Value, 2>>;

  /// Mapping between a Value in the transform IR and the corresponding list of
  /// parameters.
  using ParamMapping = DenseMap<Value, SmallVector<Param>>;

  /// Mapping between a Value in the transform IR and the corrsponding list of
  /// values in the payload IR. Also works for reverse mappings.
  using ValueMapping = DenseMap<Value, SmallVector<Value>>;

  /// Mapping between a Value in the transform IR and an error message that
  /// should be emitted when the value is used.
  using InvalidatedHandleMap = DenseMap<Value, std::function<void(Location)>>;

#ifdef LLVM_ENABLE_ABI_BREAKING_CHECKS
  /// Debug only: A timestamp is associated with each transform IR value, so
  /// that invalid iterator usage can be detected more reliably.
  using TransformIRTimestampMapping = DenseMap<Value, int64_t>;
#endif // LLVM_ENABLE_ABI_BREAKING_CHECKS

  /// The bidirectional mappings between transform IR values and payload IR
  /// operations, and the mapping between transform IR values and parameters.
  struct Mappings {
    TransformOpMapping direct;
    TransformOpReverseMapping reverse;
    ParamMapping params;
    ValueMapping values;
    ValueMapping reverseValues;

#ifdef LLVM_ENABLE_ABI_BREAKING_CHECKS
    TransformIRTimestampMapping timestamps;
    void incrementTimestamp(Value value) { ++timestamps[value]; }
#endif // LLVM_ENABLE_ABI_BREAKING_CHECKS
  };

  friend LogicalResult
  applyTransforms(Operation *, TransformOpInterface,
                  const RaggedArray<MappedValue> &, const TransformOptions &,
                  bool, function_ref<void(TransformState &)>,
                  function_ref<LogicalResult(TransformState &)>);

  friend TransformState
  detail::makeTransformStateForTesting(Region *region, Operation *payloadRoot);

public:
  const TransformOptions &getOptions() const { return options; }

  /// Returns the op at which the transformation state is rooted. This is
  /// typically helpful for transformations that apply globally.
  Operation *getTopLevel() const;

  /// Returns the number of extra mappings for the top-level operation.
  size_t getNumTopLevelMappings() const { return topLevelMappedValues.size(); }

  /// Returns the position-th extra mapping for the top-level operation.
  ArrayRef<MappedValue> getTopLevelMapping(size_t position) const {
    return topLevelMappedValues[position];
  }

  /// Returns an iterator that enumerates all ops that the given transform IR
  /// value corresponds to. Ops may be erased while iterating; erased ops are
  /// not enumerated. This function is helpful for transformations that apply to
  /// a particular handle.
  auto getPayloadOps(Value value) const {
    ArrayRef<Operation *> view = getPayloadOpsView(value);

#ifdef LLVM_ENABLE_ABI_BREAKING_CHECKS
    // Memorize the current timestamp and make sure that it has not changed
    // when incrementing or dereferencing the iterator returned by this
    // function. The timestamp is incremented when the "direct" mapping is
    // resized; this would invalidate the iterator returned by this function.
    int64_t currentTimestamp = getMapping(value).timestamps.lookup(value);
#endif // LLVM_ENABLE_ABI_BREAKING_CHECKS

    // When ops are replaced/erased, they are replaced with nullptr (until
    // the data structure is compacted). Do not enumerate these ops.
    return llvm::make_filter_range(view, [=](Operation *op) {
#ifdef LLVM_ENABLE_ABI_BREAKING_CHECKS
      [[maybe_unused]] bool sameTimestamp =
          currentTimestamp == this->getMapping(value).timestamps.lookup(value);
      assert(sameTimestamp && "iterator was invalidated during iteration");
#endif // LLVM_ENABLE_ABI_BREAKING_CHECKS
      return op != nullptr;
    });
  }

  /// Returns the list of parameters that the given transform IR value
  /// corresponds to.
  ArrayRef<Attribute> getParams(Value value) const;

  /// Returns an iterator that enumerates all payload IR values that the given
  /// transform IR value corresponds to.
  auto getPayloadValues(Value handleValue) const {
    ArrayRef<Value> view = getPayloadValuesView(handleValue);

#ifdef LLVM_ENABLE_ABI_BREAKING_CHECKS
    // Memorize the current timestamp and make sure that it has not changed
    // when incrementing or dereferencing the iterator returned by this
    // function. The timestamp is incremented when the "values" mapping is
    // resized; this would invalidate the iterator returned by this function.
    int64_t currentTimestamp =
        getMapping(handleValue).timestamps.lookup(handleValue);
    return llvm::make_filter_range(view, [=](Value v) {
      [[maybe_unused]] bool sameTimestamp =
          currentTimestamp ==
          this->getMapping(handleValue).timestamps.lookup(handleValue);
      assert(sameTimestamp && "iterator was invalidated during iteration");
      return true;
    });
#else
    return llvm::make_range(view.begin(), view.end());
#endif // LLVM_ENABLE_ABI_BREAKING_CHECKS
  }

  /// Populates `handles` with all handles pointing to the given Payload IR op.
  /// Returns success if such handles exist, failure otherwise.
  /// If `includeOutOfScope` is set to "true", handles that are defined in
  /// regions beyond the most recent isolated from above region are included.
  LogicalResult getHandlesForPayloadOp(Operation *op,
                                       SmallVectorImpl<Value> &handles,
                                       bool includeOutOfScope = false) const;

  /// Populates `handles` with all handles pointing to the given payload IR
  /// value. Returns success if such handles exist, failure otherwise.
  /// If `includeOutOfScope` is set to "true", handles that are defined in
  /// regions beyond the most recent isolated from above region are included.
  LogicalResult getHandlesForPayloadValue(Value payloadValue,
                                          SmallVectorImpl<Value> &handles,
                                          bool includeOutOfScope = false) const;

  /// Applies the transformation specified by the given transform op and updates
  /// the state accordingly.
  DiagnosedSilenceableFailure applyTransform(TransformOpInterface transform);

  /// Records the mapping between a block argument in the transform IR and a
  /// list of operations in the payload IR. The arguments must be defined in
  /// blocks of the currently processed transform IR region, typically after a
  /// region scope is defined.
  ///
  /// Returns failure if the payload does not satisfy the conditions associated
  /// with the type of the handle value.
  LogicalResult mapBlockArguments(BlockArgument argument,
                                  ArrayRef<Operation *> operations) {
    assert(argument.getParentRegion() == regionStack.back()->region &&
           "mapping block arguments from a region other than the active one");
    return setPayloadOps(argument, operations);
  }
  LogicalResult mapBlockArgument(BlockArgument argument,
                                 ArrayRef<MappedValue> values);
  LogicalResult mapBlockArguments(Block::BlockArgListType arguments,
                                  ArrayRef<SmallVector<MappedValue>> mapping);

  // Forward declarations to support limited visibility.
  class RegionScope;

  /// Creates a new region scope for the given region. The region is expected to
  /// be nested in the currently processed region.
  // Implementation note: this method is inline but implemented outside of the
  // class body to comply with visibility and full-declaration requirements.
  inline RegionScope make_region_scope(Region &region);

  /// A RAII object maintaining a "stack frame" for a transform IR region. When
  /// applying a transform IR operation that contains a region, the caller is
  /// expected to create a RegionScope before applying the ops contained in the
  /// region. This ensures that the mappings between values defined in the
  /// transform IR region and payload IR operations are cleared when the region
  /// processing ends; such values cannot be accessed outside the region.
  class RegionScope {
  public:
    /// Forgets the mapping from or to values defined in the associated
    /// transform IR region, and restores the mapping that existed before
    /// entering this scope.
    ~RegionScope();

  private:
    /// Creates a new scope for mappings between values defined in the given
    /// transform IR region and payload IR objects.
    RegionScope(TransformState &state, Region &region)
        : state(state), region(&region) {
      auto res = state.mappings.insert(
          std::make_pair(&region, std::make_unique<Mappings>()));
      assert(res.second && "the region scope is already present");
      (void)res;
      state.regionStack.push_back(this);
    }

    /// Back-reference to the transform state.
    TransformState &state;

    /// The region this scope is associated with.
    Region *region;

    /// The transform op within this region that is currently being applied.
    TransformOpInterface currentTransform;

    friend class transform::TransformState;
  };
  friend class RegionScope;

  /// Base class for TransformState extensions that allow TransformState to
  /// contain user-specified information in the state object. Clients are
  /// expected to derive this class, add the desired fields, and make the
  /// derived class compatible with the MLIR TypeID mechanism:
  ///
  /// ```mlir
  /// class MyExtension final : public TransformState::Extension {
  /// public:
  ///   MyExtension(TranfsormState &state, int myData)
  ///     : Extension(state) {...}
  /// private:
  ///   int mySupplementaryData;
  /// };
  /// ```
  ///
  /// Instances of this and derived classes are not expected to be created by
  /// the user, instead they are directly constructed within a TransformState. A
  /// TransformState can only contain one extension with the given TypeID.
  /// Extensions can be obtained from a TransformState instance, and can be
  /// removed when they are no longer required.
  ///
  /// ```mlir
  /// transformState.addExtension<MyExtension>(/*myData=*/42);
  /// MyExtension *ext = transformState.getExtension<MyExtension>();
  /// ext->doSomething();
  /// ```
  class Extension {
    // Allow TransformState to allocate Extensions.
    friend class TransformState;

  public:
    /// Base virtual destructor.
    // Out-of-line definition ensures symbols are emitted in a single object
    // file.
    virtual ~Extension();

  protected:
    /// Constructs an extension of the given TransformState object.
    Extension(TransformState &state) : state(state) {}

    /// Provides read-only access to the parent TransformState object.
    const TransformState &getTransformState() const { return state; }

    /// Replaces the given payload op with another op. If the replacement op is
    /// null, removes the association of the payload op with its handle. Returns
    /// failure if the op is not associated with any handle.
    ///
    /// Note: This function does not update value handles. None of the original
    /// op's results are allowed to be mapped to any value handle.
    LogicalResult replacePayloadOp(Operation *op, Operation *replacement);

    /// Replaces the given payload value with another value. If the replacement
    /// value is null, removes the association of the payload value with its
    /// handle. Returns failure if the value is not associated with any handle.
    LogicalResult replacePayloadValue(Value value, Value replacement);

  private:
    /// Back-reference to the state that is being extended.
    TransformState &state;
  };

  /// Adds a new Extension of the type specified as template parameter,
  /// constructing it with the arguments provided. The extension is owned by the
  /// TransformState. It is expected that the state does not already have an
  /// extension of the same type. Extension constructors are expected to take
  /// a reference to TransformState as first argument, automatically supplied
  /// by this call.
  template <typename Ty, typename... Args>
  Ty &addExtension(Args &&...args) {
    static_assert(
        std::is_base_of<Extension, Ty>::value,
        "only an class derived from TransformState::Extension is allowed here");
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
        "only an class derived from TransformState::Extension is allowed here");
    auto iter = extensions.find(TypeID::get<Ty>());
    if (iter == extensions.end())
      return nullptr;
    return static_cast<Ty *>(iter->second.get());
  }

  /// Removes the extension of the specified type.
  template <typename Ty>
  void removeExtension() {
    static_assert(
        std::is_base_of<Extension, Ty>::value,
        "only an class derived from TransformState::Extension is allowed here");
    extensions.erase(TypeID::get<Ty>());
  }

private:
  /// Identifier for storing top-level value in the `operations` mapping.
  static constexpr Value kTopLevelValue = Value();

  /// Creates a state for transform ops living in the given region. The second
  /// argument points to the root operation in the payload IR being transformed,
  /// which may or may not contain the region with transform ops. Additional
  /// options can be provided through the trailing configuration object.
  TransformState(Region *region, Operation *payloadRoot,
                 const RaggedArray<MappedValue> &extraMappings = {},
                 const TransformOptions &options = TransformOptions());

  /// Returns the mappings frame for the region in which the value is defined.
  /// If `allowOutOfScope` is set to "false", asserts that the value is in
  /// scope, based on the current stack of frames.
  const Mappings &getMapping(Value value, bool allowOutOfScope = false) const {
    return const_cast<TransformState *>(this)->getMapping(value,
                                                          allowOutOfScope);
  }
  Mappings &getMapping(Value value, bool allowOutOfScope = false) {
    Region *region = value.getParentRegion();
    auto it = mappings.find(region);
    assert(it != mappings.end() &&
           "trying to find a mapping for a value from an unmapped region");
#ifndef NDEBUG
    if (!allowOutOfScope) {
      for (Region *r : llvm::reverse(llvm::make_first_range(mappings))) {
        if (r == region)
          break;
        if (r->getParentOp()->hasTrait<OpTrait::IsIsolatedFromAbove>())
          llvm_unreachable("trying to get mapping beyond region that is "
                           "isolated from above");
      }
    }
#endif // NDEBUG
    return *it->second;
  }

  /// Returns the mappings frame for the region in which the operation resides.
  /// If `allowOutOfScope` is set to "false", asserts that the operation is in
  /// scope, based on the current stack of frames.
  const Mappings &getMapping(Operation *operation,
                             bool allowOutOfScope = false) const {
    return const_cast<TransformState *>(this)->getMapping(operation,
                                                          allowOutOfScope);
  }
  Mappings &getMapping(Operation *operation, bool allowOutOfScope = false) {
    Region *region = operation->getParentRegion();
    auto it = mappings.find(region);
    assert(it != mappings.end() &&
           "trying to find a mapping for an operation from an unmapped region");
#ifndef NDEBUG
    if (!allowOutOfScope) {
      for (Region *r : llvm::reverse(llvm::make_first_range(mappings))) {
        if (r == region)
          break;
        if (r->getParentOp()->hasTrait<OpTrait::IsIsolatedFromAbove>())
          llvm_unreachable("trying to get mapping beyond region that is "
                           "isolated from above");
      }
    }
#endif // NDEBUG
    return *it->second;
  }

  /// Updates the state to include the associations between op results and the
  /// provided result of applying a transform op.
  LogicalResult updateStateFromResults(const TransformResults &results,
                                       ResultRange opResults);

  /// Returns a list of all ops that the given transform IR value corresponds
  /// to. In case an op was erased, the returned list contains nullptr. This
  /// function is helpful for transformations that apply to a particular handle.
  ArrayRef<Operation *> getPayloadOpsView(Value value) const;

  /// Returns a list of payload IR values that the given transform IR value
  /// corresponds to.
  ArrayRef<Value> getPayloadValuesView(Value handleValue) const;

  /// Sets the payload IR ops associated with the given transform IR value
  /// (handle). A payload op may be associated multiple handles as long as
  /// at most one of them gets consumed by further transformations.
  /// For example, a hypothetical "find function by name" may be called twice in
  /// a row to produce two handles pointing to the same function:
  ///
  ///   %0 = transform.find_func_by_name { name = "myfunc" }
  ///   %1 = transform.find_func_by_name { name = "myfunc" }
  ///
  /// which is valid by itself. However, calling a hypothetical "rewrite and
  /// rename function" transform on both handles:
  ///
  ///   transform.rewrite_and_rename %0 { new_name = "func" }
  ///   transform.rewrite_and_rename %1 { new_name = "func" }
  ///
  /// is invalid given the transformation "consumes" the handle as expressed
  /// by side effects. Practically, a transformation consuming a handle means
  /// that the associated payload operation may no longer exist.
  ///
  /// Similarly, operation handles may be invalidate and should not be used
  /// after a transform that consumed a value handle pointing to a payload value
  /// defined by the operation as either block argument or op result. For
  /// example, in the following sequence, the last transform operation rewrites
  /// the callee to not return a specified result:
  ///
  ///   %0 = transform.find_call "myfunc"
  ///   %1 = transform.find_results_of_calling "myfunc"
  ///   transform.drop_call_result_from_signature %1[0]
  ///
  /// which requires the call operations to be recreated. Therefore, the handle
  /// %0 becomes associated with a dangling pointer and should not be used.
  ///
  /// Returns failure if the payload does not satisfy the conditions associated
  /// with the type of the handle value. The value is expected to have a type
  /// implementing TransformHandleTypeInterface.
  LogicalResult setPayloadOps(Value value, ArrayRef<Operation *> targets);

  /// Sets the payload IR values association with the given transform IR value
  /// (handle). A payload value may be associated with multiple handles as long
  /// as at most one of them is consumed by further transformations. For
  /// example, a hypothetical "get results of calls to function with the given
  /// name" transform may be performed twice in a row producing handles pointing
  /// to the same values:
  ///
  ///   %0 = transform.find_results_of_calling "myfunc"
  ///   %1 = transform.find_results_of_calling "myfunc"
  ///
  /// which is valid by itself. However, calling a hypothetical "erase value
  /// producer" transform on both handles:
  ///
  ///   transform.erase_value_produce %0
  ///   transform.erase_value_produce %1
  ///
  /// is invalid provided the transformation "consumes" the handle as expressed
  /// by side effects (which themselves reflect the semantics of the transform
  /// erasing the producer and making the handle dangling). Practically, a
  /// transformation consuming a handle means the associated payload value may
  /// no longer exist.
  ///
  /// Similarly, value handles are invalidated and should not be used after a
  /// transform that consumed an operation handle pointing to the payload IR
  /// operation defining the values associated the value handle, as either block
  /// arguments or op results, or any ancestor operation. For example,
  ///
  ///   %0 = transform.find_call "myfunc"
  ///   %1 = transform.find_results_of_calling "myfunc"
  ///   transform.rewrite_and_rename %0 { new_name = "func" }
  ///
  /// makes %1 unusable after the last transformation if it consumes %0. When an
  /// operation handle is consumed, it usually indicates that the operation was
  /// destroyed or heavily modified, meaning that the values it defines may no
  /// longer exist.
  ///
  /// Returns failure if the payload values do not satisfy the conditions
  /// associated with the type of the handle value. The value is expected to
  /// have a type implementing TransformValueHandleTypeInterface.
  LogicalResult setPayloadValues(Value handle, ValueRange payloadValues);

  /// Sets the parameters associated with the given transform IR value. Returns
  /// failure if the parameters do not satisfy the conditions associated with
  /// the type of the value. The value is expected to have a type implementing
  /// TransformParamTypeInterface.
  LogicalResult setParams(Value value, ArrayRef<Param> params);

  /// Forgets the payload IR ops associated with the given transform IR value,
  /// as well as any association between value handles and the results of said
  /// payload IR op.
  ///
  /// If `allowOutOfScope` is set to "false", asserts that the handle is in
  /// scope, based on the current stack of frames.
  void forgetMapping(Value opHandle, ValueRange origOpFlatResults,
                     bool allowOutOfScope = false);

  void forgetValueMapping(Value valueHandle,
                          ArrayRef<Operation *> payloadOperations);

  /// Replaces the given payload op with another op. If the replacement op is
  /// null, removes the association of the payload op with its handle. Returns
  /// failure if the op is not associated with any handle.
  ///
  /// Note: This function does not update value handles. None of the original
  /// op's results are allowed to be mapped to any value handle.
  LogicalResult replacePayloadOp(Operation *op, Operation *replacement);

  /// Replaces the given payload value with another value. If the replacement
  /// value is null, removes the association of the payload value with its
  /// handle. Returns failure if the value is not associated with any handle.
  LogicalResult replacePayloadValue(Value value, Value replacement);

  /// Records handle invalidation reporters into `newlyInvalidated`.
  /// Specifically,
  ///  - `handle` is the op operand that consumes the handle,
  ///  - `potentialAncestors` is a list of ancestors of the payload operation
  ///     that the consumed handle is associated with, including itself,
  ///  - `throughValue` is the payload value the handle to which is consumed,
  ///     when it is the case, null when the operation handle is consumed
  ///     directly.
  /// Iterates over all known operation and value handles and records reporters
  /// for any potential future use of `handle` or any other handle that is
  /// invalidated by its consumption, i.e., any handle pointing to any payload
  /// IR entity (operation or value) associated with the same payload IR entity
  /// as the consumed handle, or any nested payload IR entity. If
  /// `potentialAncestors` is empty, records the reporter anyway. Does not
  /// override existing reporters. This must remain a const method so it doesn't
  /// inadvertently mutate `invalidatedHandles` too early.
  void recordOpHandleInvalidation(OpOperand &consumingHandle,
                                  ArrayRef<Operation *> potentialAncestors,
                                  Value throughValue,
                                  InvalidatedHandleMap &newlyInvalidated) const;

  /// Records handle invalidation reporters into `newlyInvalidated`.
  /// Specifically,
  ///  - `consumingHandle` is the op operand that consumes the handle,
  ///  - `potentialAncestors` is a list of ancestors of the payload operation
  ///     that the consumed handle is associated with, including itself,
  ///  - `payloadOp` is the operation itself,
  ///  - `otherHandle` is another that may be associated with the affected
  ///     payload operations
  ///  - `throughValue` is the payload value the handle to which is consumed,
  ///     when it is the case, null when the operation handle is consumed
  ///     directly.
  /// Looks at the payload opreations associated with `otherHandle` and if any
  /// of these operations has an ancestor (or is itself) listed in
  /// `potentialAncestors`, records the error message describing the use of the
  /// invalidated handle. Does nothing if `otherHandle` already has a reporter
  /// associated with it. This must remain a const method so it doesn't
  /// inadvertently mutate `invalidatedHandles` too early.
  void recordOpHandleInvalidationOne(
      OpOperand &consumingHandle, ArrayRef<Operation *> potentialAncestors,
      Operation *payloadOp, Value otherHandle, Value throughValue,
      InvalidatedHandleMap &newlyInvalidated) const;

  /// Records handle invalidation reporters into `newlyInvalidated`.
  /// Specifically,
  ///  - `opHandle` is the op operand that consumes the handle;
  ///  - `potentialAncestors` is a list of ancestors of the payload operation
  ///     that the consumed handle is associated with, including itself;
  ///  - `payloadValue` is the value defined by the operation associated with
  ///     the consuming handle as either op result or block argument;
  ///  - `valueHandle` is another that may be associated with the payload value.
  /// Looks at the payload values associated with `valueHandle` and if any of
  /// these values is defined, as op result or block argument, by an operation
  /// whose ancestor (or the operation itself) is listed in
  /// `potentialAncestors`, records the error message describing the use of the
  /// invalidated handle. Does nothing if `valueHandle` already has a reporter
  /// associated with it. This must remain a const method so it doesn't
  /// inadvertently mutate `invalidatedHandles` too early.
  void recordValueHandleInvalidationByOpHandleOne(
      OpOperand &opHandle, ArrayRef<Operation *> potentialAncestors,
      Value payloadValue, Value valueHandle,
      InvalidatedHandleMap &newlyInvalidated) const;

  /// Records handle invalidation reporters into `newlyInvalidated`.
  /// Specifically,
  ///  - `valueHandle` is the op operand that consumes the handle,
  ///  - `throughValue` is the payload value the handle to which is consumed,
  ///     when it is the case, null when the operation handle is consumed
  ///     directly.
  /// Iterates over all known operation and value handles and records reporters
  /// for any potential future use of `handle` or any other handle that is
  /// invalidated by its consumption, i.e., any handle pointing to any payload
  /// IR entity (operation or value) associated with the same payload IR entity
  /// as the consumed handle, or any nested payload IR entity. Does not override
  /// existing reporters. This must remain a const method so it doesn't
  /// inadvertently mutate `invalidatedHandles` too early.
  void
  recordValueHandleInvalidation(OpOperand &valueHandle,
                                InvalidatedHandleMap &newlyInvalidated) const;

  /// Checks that the operation does not use invalidated handles as operands.
  /// Reports errors and returns failure if it does. Otherwise, invalidates the
  /// handles consumed by the operation as well as any handles pointing to
  /// payload IR operations nested in the operations associated with the
  /// consumed handles.
  LogicalResult
  checkAndRecordHandleInvalidation(TransformOpInterface transform);

  /// Implementation of the checkAndRecordHandleInvalidation. This must remain a
  /// const method so it doesn't inadvertently mutate `invalidatedHandles` too
  /// early.
  LogicalResult checkAndRecordHandleInvalidationImpl(
      transform::TransformOpInterface transform,
      transform::TransformState::InvalidatedHandleMap &newlyInvalidated) const;

  /// Remove all nullptrs from op handles that were added by `replacePayloadOp`.
  void compactOpHandles();

  /// A stack of mappings between transform IR values and payload IR ops,
  /// aggregated by the region in which the transform IR values are defined.
  /// We use a pointer to the Mappings struct so that reallocations inside
  /// MapVector don't invalidate iterators when we apply nested transform ops
  /// while also iterating over the mappings.
  llvm::MapVector<Region *, std::unique_ptr<Mappings>> mappings;

  /// Op handles may be temporarily mapped to nullptr to avoid invalidating
  /// payload op iterators. This set contains all op handles with nullptrs.
  /// These handles are "compacted" (i.e., nullptrs removed) at the end of each
  /// transform.
  DenseSet<Value> opHandlesToCompact;

  /// Extensions attached to the TransformState, identified by the TypeID of
  /// their type. Only one extension of any given type is allowed.
  DenseMap<TypeID, std::unique_ptr<Extension>> extensions;

  /// The top-level operation that contains all payload IR, typically a module.
  Operation *topLevel;

  /// Extra mapped values (payload operations, values or parameters) to be
  /// associated with additional entry block arguments of the top-level
  /// transform operation.
  RaggedArray<MappedValue> topLevelMappedValues;

  /// Additional options controlling the transformation state behavior.
  TransformOptions options;

  /// The mapping from invalidated handles to the error-reporting functions that
  /// describe when the handles were invalidated. Calling such a function emits
  /// a user-visible diagnostic with an additional note pointing to the given
  /// location.
  InvalidatedHandleMap invalidatedHandles;

  /// A stack of nested regions that are being processed in the transform IR.
  /// Each region must be an ancestor of the following regions in this list.
  /// These are also the keys for "mappings".
  SmallVector<RegionScope *> regionStack;

  /// The top-level region scope. The first (bottom) element of `regionStack`
  /// is the top-level region scope object.
  std::unique_ptr<RegionScope> topLevelRegionScope;
};

/// Local mapping between values defined by a specific op implementing the
/// TransformOpInterface and the payload IR ops they correspond to.
class TransformResults {
  friend class TransformState;

public:
  /// Indicates that the result of the transform IR op at the given position
  /// corresponds to the given list of payload IR ops. Each result must be set
  /// by the transformation exactly once in case of transformation succeeding.
  /// The value must have a type implementing TransformHandleTypeInterface.
  template <typename Range>
  void set(OpResult value, Range &&ops) {
    int64_t position = value.getResultNumber();
    assert(position < static_cast<int64_t>(operations.size()) &&
           "setting results for a non-existent handle");
    assert(operations[position].data() == nullptr && "results already set");
    assert(params[position].data() == nullptr &&
           "another kind of results already set");
    assert(values[position].data() == nullptr &&
           "another kind of results already set");
    operations.replace(position, std::forward<Range>(ops));
  }

  /// Indicates that the result of the transform IR op at the given position
  /// corresponds to the given list of payload IR ops. Each result must be set
  /// by the transformation exactly once in case of transformation succeeding.
  /// The value must have a type implementing TransformHandleTypeInterface.
  void set(OpResult value, std::initializer_list<Operation *> ops) {
    set(value, ArrayRef<Operation *>(ops));
  }

  /// Indicates that the result of the transform IR op at the given position
  /// corresponds to the given list of parameters. Each result must be set by
  /// the transformation exactly once in case of transformation succeeding. The
  /// value must have a type implementing TransformParamTypeInterface.
  void setParams(OpResult value, ArrayRef<TransformState::Param> params);

  /// Indicates that the result of the transform IR op at the given position
  /// corresponds to the given range of payload IR values. Each result must be
  /// set by the transformation exactly once in case of transformation
  /// succeeding. The value must have a type implementing
  /// TransformValueHandleTypeInterface.
  template <typename Range>
  void setValues(OpResult handle, Range &&values) {
    int64_t position = handle.getResultNumber();
    assert(position < static_cast<int64_t>(this->values.size()) &&
           "setting values for a non-existent handle");
    assert(this->values[position].data() == nullptr && "values already set");
    assert(operations[position].data() == nullptr &&
           "another kind of results already set");
    assert(params[position].data() == nullptr &&
           "another kind of results already set");
    this->values.replace(position, std::forward<Range>(values));
  }

  /// Indicates that the result of the transform IR op at the given position
  /// corresponds to the given range of payload IR values. Each result must be
  /// set by the transformation exactly once in case of transformation
  /// succeeding. The value must have a type implementing
  /// TransformValueHandleTypeInterface.
  void setValues(OpResult handle, std::initializer_list<Value> values) {
    setValues(handle, ArrayRef<Value>(values));
  }

  /// Indicates that the result of the transform IR op at the given position
  /// corresponds to the given range of mapped values. All mapped values are
  /// expected to be compatible with the type of the result, e.g., if the result
  /// is an operation handle, all mapped values are expected to be payload
  /// operations.
  void setMappedValues(OpResult handle, ArrayRef<MappedValue> values);

  /// Sets the currently unset results to empty lists of the kind expected by
  /// the corresponding results of the given `transform` op.
  void setRemainingToEmpty(TransformOpInterface transform);

private:
  /// Creates an instance of TransformResults that expects mappings for
  /// `numSegments` values, which may be associated with payload operations or
  /// parameters.
  explicit TransformResults(unsigned numSegments);

  /// Gets the list of operations associated with the result identified by its
  /// number in the list of operation results. The result must have been set to
  /// be associated with payload IR operations.
  ArrayRef<Operation *> get(unsigned resultNumber) const;

  /// Gets the list of parameters associated with the result identified by its
  /// number in the list of operation results. The result must have been set to
  /// be associated with parameters.
  ArrayRef<TransformState::Param> getParams(unsigned resultNumber) const;

  /// Gets the list of payload IR values associated with the result identified
  /// by its number in the list of operation results. The result must have been
  /// set to be associated with payload IR values.
  ArrayRef<Value> getValues(unsigned resultNumber) const;

  /// Returns `true` if the result identified by its number in the list of
  /// operation results is associated with a list of parameters, `false`
  /// otherwise.
  bool isParam(unsigned resultNumber) const;

  /// Returns `true` if the result identified by its number in the list of
  /// operation results is associated with a list of payload IR value, `false`
  /// otherwise.
  bool isValue(unsigned resultNumber) const;

  /// Returns `true` if the result identified by its number in the list of
  /// operation results is associated with something.
  bool isSet(unsigned resultNumber) const;

  /// Pointers to payload IR ops that are associated with results of a transform
  /// IR op.
  RaggedArray<Operation *> operations;

  /// Parameters that are associated with results of the transform IR op.
  RaggedArray<Param> params;

  /// Payload IR values that are associated with results of a transform IR op.
  RaggedArray<Value> values;
};

/// Creates a RAII object the lifetime of which corresponds to the new mapping
/// for transform IR values defined in the given region. Values defined in
/// surrounding regions remain accessible.
TransformState::RegionScope TransformState::make_region_scope(Region &region) {
  return RegionScope(*this, region);
}

/// A configuration object for customizing a `TrackingListener`.
struct TrackingListenerConfig {
  using SkipHandleFn = std::function<bool(Value)>;

  /// An optional function that returns "true" for handles that do not have to
  /// be updated. These are typically dead or consumed handles.
  SkipHandleFn skipHandleFn = nullptr;

  /// If set to "true", the name of a replacement op must match the name of the
  /// original op. If set to "false", the names of the payload ops tracked in a
  /// handle may change as the tracking listener updates the transform state.
  bool requireMatchingReplacementOpName = true;

  /// If set to "true", cast ops (that implement the CastOpInterface) are
  /// skipped and the replacement op search continues with the operands of the
  /// cast op.
  bool skipCastOps = true;
};

/// A listener that updates a TransformState based on IR modifications. This
/// listener can be used during a greedy pattern rewrite to keep the transform
/// state up-to-date.
class TrackingListener : public RewriterBase::Listener,
                         public TransformState::Extension {
public:
  /// Create a new TrackingListener for usage in the specified transform op.
  /// Optionally, a function can be specified to identify handles that should
  /// do not have to be updated.
  TrackingListener(TransformState &state, TransformOpInterface op,
                   TrackingListenerConfig config = TrackingListenerConfig());

protected:
  /// Return a replacement payload op for the given op, which is going to be
  /// replaced with the given values. By default, if all values are defined by
  /// the same op, which also has the same type as the given op, that defining
  /// op is used as a replacement.
  ///
  /// A "failure" return value indicates that no replacement operation could be
  /// found. A "nullptr" return value indicates that no replacement op is needed
  /// (e.g., handle is dead or was consumed) and that the payload op should
  /// be dropped from the mapping.
  ///
  /// Example: A tracked "linalg.generic" with two results is replaced with two
  /// values defined by (another) "linalg.generic". It is reasonable to assume
  /// that the replacement "linalg.generic" represents the same "computation".
  /// Therefore, the payload op mapping is updated to the defining op of the
  /// replacement values.
  ///
  /// Counter Example: A "linalg.generic" is replaced with values defined by an
  /// "scf.for". Without further investigation, the relationship between the
  /// "linalg.generic" and the "scf.for" is unclear. They may not represent the
  /// same computation; e.g., there may be tiled "linalg.generic" inside the
  /// loop body that represents the original computation. Therefore, the
  /// TrackingListener is conservative by default: it drops the mapping and
  /// triggers the "payload replacement not found" notification. This default
  /// behavior can be customized in `TrackingListenerConfig`.
  ///
  /// If no replacement op could be found according to the rules mentioned
  /// above, this function tries to skip over cast-like ops that implement
  /// `CastOpInterface`.
  ///
  /// Example: A tracked "linalg.generic" is replaced with "linalg.generic",
  /// wrapped in a "tensor.cast". A cast is a metadata-only operation and it is
  /// reasonable to assume that the wrapped "linalg.generic" represents the same
  /// computation as the original "linalg.generic". The mapping is updated
  /// accordingly.
  ///
  /// Certain ops (typically also metadata-only ops) are not considered casts,
  /// but should be skipped nonetheless. Such ops should implement
  /// `FindPayloadReplacementOpInterface` to specify with which operands the
  /// lookup should continue.
  ///
  /// Example: A tracked "linalg.generic" is replaced with "linalg.generic",
  /// wrapped in a "tensor.reshape". A reshape is a metadata-only operation but
  /// not cast. (Implementing `CastOpInterface` would be incorrect and cause
  /// invalid foldings.) However, due to its `FindPayloadReplacementOpInterface`
  /// implementation, the replacement op lookup continues with the wrapped
  /// "linalg.generic" and the mapping is updated accordingly.
  ///
  /// Derived classes may override `findReplacementOp` to specify custom
  /// replacement rules.
  virtual DiagnosedSilenceableFailure
  findReplacementOp(Operation *&result, Operation *op,
                    ValueRange newValues) const;

  /// Notify the listener that the pattern failed to match the given operation,
  /// and provide a callback to populate a diagnostic with the reason why the
  /// failure occurred.
  void
  notifyMatchFailure(Location loc,
                     function_ref<void(Diagnostic &)> reasonCallback) override;

  /// This function is called when a tracked payload op is dropped because no
  /// replacement op was found. Derived classes can implement this function for
  /// custom error handling.
  virtual void
  notifyPayloadReplacementNotFound(Operation *op, ValueRange values,
                                   DiagnosedSilenceableFailure &&diag) {}

  /// Return the single op that defines all given values (if any).
  static Operation *getCommonDefiningOp(ValueRange values);

  /// Return the transform op in which this TrackingListener is used.
  TransformOpInterface getTransformOp() const { return transformOp; }

private:
  friend class TransformRewriter;

  void notifyOperationErased(Operation *op) override;

  void notifyOperationReplaced(Operation *op, ValueRange newValues) override;
  using Listener::notifyOperationReplaced;

  /// The transform op in which this TrackingListener is used.
  TransformOpInterface transformOp;

  /// The handles that are consumed by the transform op.
  DenseSet<Value> consumedHandles;

  /// Tracking listener configuration.
  TrackingListenerConfig config;
};

/// A specialized listener that keeps track of cases in which no replacement
/// payload could be found. The error state of this listener must be checked
/// before the end of its lifetime.
class ErrorCheckingTrackingListener : public TrackingListener {
public:
  using transform::TrackingListener::TrackingListener;

  ~ErrorCheckingTrackingListener() override;

  /// Check and return the current error state of this listener. Afterwards,
  /// resets the error state to "success".
  DiagnosedSilenceableFailure checkAndResetError();

  /// Return "true" if this tracking listener had a failure.
  bool failed() const;

protected:
  void
  notifyPayloadReplacementNotFound(Operation *op, ValueRange values,
                                   DiagnosedSilenceableFailure &&diag) override;

private:
  /// The error state of this listener. "Success" indicates that no error
  /// happened so far.
  DiagnosedSilenceableFailure status = DiagnosedSilenceableFailure::success();

  /// The number of errors that have been encountered.
  int64_t errorCounter = 0;
};

/// This is a special rewriter to be used in transform op implementations,
/// providing additional helper functions to update the transform state, etc.
// TODO: Helper functions will be added in a subsequent change.
class TransformRewriter : public RewriterBase {
protected:
  friend class TransformState;

  /// Create a new TransformRewriter.
  explicit TransformRewriter(MLIRContext *ctx,
                             ErrorCheckingTrackingListener *listener);

public:
  /// Return "true" if the tracking listener had failures.
  bool hasTrackingFailures() const;

  /// Silence all tracking failures that have been encountered so far.
  void silenceTrackingFailure();

  /// Notify the transform dialect interpreter that the given op has been
  /// replaced with another op and that the mapping between handles and payload
  /// ops/values should be updated. This function should be called before the
  /// original op is erased. It fails if the operation could not be replaced,
  /// e.g., because the original operation is not tracked.
  ///
  /// Note: As long as IR modifications are performed through this rewriter,
  /// the transform state is usually updated automatically. This function should
  /// be used when unsupported rewriter API is used; e.g., updating all uses of
  /// a tracked operation one-by-one instead of using `RewriterBase::replaceOp`.
  LogicalResult notifyPayloadOperationReplaced(Operation *op,
                                               Operation *replacement);

private:
  ErrorCheckingTrackingListener *const listener;
};

/// This trait is supposed to be attached to Transform dialect operations that
/// can be standalone top-level transforms. Such operations typically contain
/// other Transform dialect operations that can be executed following some
/// control flow logic specific to the current operation. The operations with
/// this trait are expected to have at least one single-block region with at
/// least one argument of type implementing TransformHandleTypeInterface. The
/// operations are also expected to be valid without operands, in which case
/// they are considered top-level, and with one or more arguments, in which case
/// they are considered nested. Top-level operations have the block argument of
/// the entry block in the Transform IR correspond to the root operation of
/// Payload IR. Nested operations have the block argument of the entry block in
/// the Transform IR correspond to a list of Payload IR operations mapped to the
/// first operand of the Transform IR operation. The operation must implement
/// TransformOpInterface.
template <typename OpTy>
class PossibleTopLevelTransformOpTrait
    : public OpTrait::TraitBase<OpTy, PossibleTopLevelTransformOpTrait> {
public:
  /// Verifies that `op` satisfies the invariants of this trait. Not expected to
  /// be called directly.
  static LogicalResult verifyTrait(Operation *op) {
    return detail::verifyPossibleTopLevelTransformOpTrait(op);
  }

  /// Returns the single block of the given region.
  Block *getBodyBlock(unsigned region = 0) {
    return &this->getOperation()->getRegion(region).front();
  }

  /// Populates `effects` with side effects implied by this trait.
  void getPotentialTopLevelEffects(
      SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
    detail::getPotentialTopLevelEffects(
        this->getOperation(), cast<OpTy>(this->getOperation()).getRoot(),
        *getBodyBlock(), effects);
  }

  /// Sets up the mapping between the entry block of the given region of this op
  /// and the relevant list of Payload IR operations in the given state. The
  /// state is expected to be already scoped at the region of this operation.
  LogicalResult mapBlockArguments(TransformState &state, Region &region) {
    assert(region.getParentOp() == this->getOperation() &&
           "op comes from the wrong region");
    return detail::mapPossibleTopLevelTransformOpBlockArguments(
        state, this->getOperation(), region);
  }
  LogicalResult mapBlockArguments(TransformState &state) {
    assert(
        this->getOperation()->getNumRegions() == 1 &&
        "must indicate the region to map if the operation has more than one");
    return mapBlockArguments(state, this->getOperation()->getRegion(0));
  }
};

class ApplyToEachResultList;

/// Trait implementing the TransformOpInterface for operations applying a
/// transformation to a single operation handle and producing an arbitrary
/// number of handles and parameter values.
/// The op must implement a method with the following signature:
///   - DiagnosedSilenceableFailure applyToOne(OpTy,
///       ApplyToEachResultList &results, TransformState &state)
/// to perform a transformation that is applied in turn to all payload IR
/// operations that correspond to the handle of the transform IR operation.
/// In `applyToOne`, OpTy is either Operation* or a concrete payload IR Op class
/// that the transformation is applied to (and NOT the class of the transform IR
/// op).
/// The `applyToOne` method takes an empty `results` vector that it fills with
/// zero, one or multiple operations depending on the number of results expected
/// by the transform op.
/// The number of results must match the number of results of the transform op.
/// `applyToOne` is allowed to fill the `results` with all null elements to
/// signify that the transformation did not apply to the payload IR operations.
/// Such null elements are filtered out from results before return.
///
/// The transform op having this trait is expected to have a single operand.
template <typename OpTy>
class TransformEachOpTrait
    : public OpTrait::TraitBase<OpTy, TransformEachOpTrait> {
public:
  /// Calls `applyToOne` for every payload operation associated with the operand
  /// of this transform IR op, the following case disjunction happens:
  ///   1. If not target payload ops are associated to the operand then fill the
  ///      results vector with the expected number of null elements and return
  ///      success. This is the corner case handling that allows propagating
  ///      the "no-op" case gracefully to improve usability.
  ///   2. If any `applyToOne` returns definiteFailure, the transformation is
  ///      immediately considered definitely failed and we return.
  ///   3. All applications of `applyToOne` are checked to return a number of
  ///      results expected by the transform IR op. If not, this is a definite
  ///      failure and we return early.
  ///   4. If `applyToOne` produces ops, associate them with the result of this
  ///      transform op.
  ///   5. If any `applyToOne` return silenceableFailure, the transformation is
  ///      considered silenceable.
  ///   6. Otherwise the transformation is considered successful.
  DiagnosedSilenceableFailure apply(transform::TransformRewriter &rewriter,
                                    TransformResults &transformResults,
                                    TransformState &state);

  /// Checks that the op matches the expectations of this trait.
  static LogicalResult verifyTrait(Operation *op);
};

/// Side effect resource corresponding to the mapping between Transform IR
/// values and Payload IR operations. An Allocate effect from this resource
/// means creating a new mapping entry, it is always accompanied by a Write
/// effect. A Read effect from this resource means accessing the mapping. A Free
/// effect on this resource indicates the removal of the mapping entry,
/// typically after a transformation that modifies the Payload IR operations
/// associated with one of the Transform IR operation's operands. It is always
/// accompanied by a Read effect. Read-after-Free and double-Free are not
/// allowed (they would be problematic with "regular" memory effects too) as
/// they indicate an attempt to access Payload IR operations that have been
/// modified, potentially erased, by the previous transformations.
// TODO: consider custom effects if these are not enabling generic passes such
// as CSE/DCE to work.
struct TransformMappingResource
    : public SideEffects::Resource::Base<TransformMappingResource> {
  StringRef getName() override { return "transform.mapping"; }
};

/// Side effect resource corresponding to the Payload IR itself. Only Read and
/// Write effects are expected on this resource, with Write always accompanied
/// by a Read (short of fully replacing the top-level Payload IR operation, one
/// cannot modify the Payload IR without reading it first). This is intended
/// to disallow reordering of Transform IR operations that mutate the Payload IR
/// while still allowing the reordering of those that only access it.
struct PayloadIRResource
    : public SideEffects::Resource::Base<PayloadIRResource> {
  StringRef getName() override { return "transform.payload_ir"; }
};

/// Populates `effects` with the memory effects indicating the operation on the
/// given handle value:
///   - consumes = Read + Free,
///   - produces = Allocate + Write,
///   - onlyReads = Read.
void consumesHandle(MutableArrayRef<OpOperand> handles,
                    SmallVectorImpl<MemoryEffects::EffectInstance> &effects);
void producesHandle(ResultRange handles,
                    SmallVectorImpl<MemoryEffects::EffectInstance> &effects);
void producesHandle(MutableArrayRef<BlockArgument> handles,
                    SmallVectorImpl<MemoryEffects::EffectInstance> &effects);
void onlyReadsHandle(MutableArrayRef<OpOperand> handles,
                     SmallVectorImpl<MemoryEffects::EffectInstance> &effects);

/// Checks whether the transform op consumes the given handle.
bool isHandleConsumed(Value handle, transform::TransformOpInterface transform);

/// Populates `effects` with the memory effects indicating the access to payload
/// IR resource.
void modifiesPayload(SmallVectorImpl<MemoryEffects::EffectInstance> &effects);
void onlyReadsPayload(SmallVectorImpl<MemoryEffects::EffectInstance> &effects);

/// Checks whether the transform op modifies the payload.
bool doesModifyPayload(transform::TransformOpInterface transform);
/// Checks whether the transform op reads the payload.
bool doesReadPayload(transform::TransformOpInterface transform);

/// Populates `consumedArguments` with positions of `block` arguments that are
/// consumed by the operations in the `block`.
void getConsumedBlockArguments(
    Block &block, llvm::SmallDenseSet<unsigned> &consumedArguments);

/// Trait implementing the MemoryEffectOpInterface for operations that "consume"
/// their operands and produce new results.
template <typename OpTy>
class FunctionalStyleTransformOpTrait
    : public OpTrait::TraitBase<OpTy, FunctionalStyleTransformOpTrait> {
public:
  /// This op "consumes" the operands by reading and freeing then, "produces"
  /// the results by allocating and writing it and reads/writes the payload IR
  /// in the process.
  void getEffects(SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
    consumesHandle(this->getOperation()->getOpOperands(), effects);
    producesHandle(this->getOperation()->getOpResults(), effects);
    modifiesPayload(effects);
  }

  /// Checks that the op matches the expectations of this trait.
  static LogicalResult verifyTrait(Operation *op) {
    if (!op->getName().getInterface<MemoryEffectOpInterface>()) {
      op->emitError()
          << "FunctionalStyleTransformOpTrait should only be attached to ops "
             "that implement MemoryEffectOpInterface";
    }
    return success();
  }
};

/// Trait implementing the MemoryEffectOpInterface for operations that use their
/// operands without consuming and without modifying the Payload IR to
/// potentially produce new handles.
template <typename OpTy>
class NavigationTransformOpTrait
    : public OpTrait::TraitBase<OpTy, NavigationTransformOpTrait> {
public:
  /// This op produces handles to the Payload IR without consuming the original
  /// handles and without modifying the IR itself.
  void getEffects(SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
    onlyReadsHandle(this->getOperation()->getOpOperands(), effects);
    producesHandle(this->getOperation()->getOpResults(), effects);
    if (llvm::any_of(this->getOperation()->getOperandTypes(), [](Type t) {
          return isa<TransformHandleTypeInterface,
                     TransformValueHandleTypeInterface>(t);
        })) {
      onlyReadsPayload(effects);
    }
  }

  /// Checks that the op matches the expectation of this trait.
  static LogicalResult verifyTrait(Operation *op) {
    if (!op->getName().getInterface<MemoryEffectOpInterface>()) {
      op->emitError() << "NavigationTransformOpTrait should only be attached "
                         "to ops that implement MemoryEffectOpInterface";
    }
    return success();
  }
};

namespace detail {
/// Non-template implementation of ParamProducerTransformOpTrait::getEffects().
void getParamProducerTransformOpTraitEffects(
    Operation *op, SmallVectorImpl<MemoryEffects::EffectInstance> &effects);
/// Non-template implementation of ParamProducerTransformOpTrait::verify().
LogicalResult verifyParamProducerTransformOpTrait(Operation *op);
} // namespace detail

/// Trait implementing the MemoryEffectsOpInterface for operations that produce
/// transform dialect parameters. It marks all op results of
/// TransformHandleTypeInterface as produced by the op, all operands as only
/// read by the op and, if at least one of the operand is a handle to payload
/// ops, the entire payload as potentially read. The op must only produce
/// parameter-typed results.
template <typename OpTy>
class ParamProducerTransformOpTrait
    : public OpTrait::TraitBase<OpTy, ParamProducerTransformOpTrait> {
public:
  /// Populates `effects` with effect instances described in the trait
  /// documentation.
  void getEffects(SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
    detail::getParamProducerTransformOpTraitEffects(this->getOperation(),
                                                    effects);
  }

  /// Checks that the op matches the expectation of this trait, i.e., that it
  /// implements the MemoryEffectsOpInterface and only produces parameter-typed
  /// results.
  static LogicalResult verifyTrait(Operation *op) {
    return detail::verifyParamProducerTransformOpTrait(op);
  }
};

/// `TrackingListener` failures are reported only for ops that have this trait.
/// The purpose of this trait is to give users more time to update their custom
/// transform ops to use the provided `TransformRewriter` for all IR
/// modifications. This trait will eventually be removed, and failures will be
/// reported for all transform ops.
template <typename OpTy>
class ReportTrackingListenerFailuresOpTrait
    : public OpTrait::TraitBase<OpTy, ReportTrackingListenerFailuresOpTrait> {};

/// A single result of applying a transform op with `ApplyEachOpTrait` to a
/// single payload operation.
using ApplyToEachResult = MappedValue;

/// A list of results of applying a transform op with `ApplyEachOpTrait` to a
/// single payload operation, co-indexed with the results of the transform op.
class ApplyToEachResultList {
public:
  ApplyToEachResultList() = default;
  explicit ApplyToEachResultList(unsigned size) : results(size) {}

  /// Sets the list of results to `size` null pointers.
  void assign(unsigned size, std::nullptr_t) { results.assign(size, nullptr); }

  /// Sets the list of results to the given range of values.
  template <typename Range>
  void assign(Range &&range) {
    // This is roughly the implementation of SmallVectorImpl::assign.
    // Dispatching to it with map_range and template type inference would result
    // in more complex code here.
    results.clear();
    results.reserve(llvm::size(range));
    for (auto element : range) {
      if constexpr (std::is_convertible_v<decltype(*std::begin(range)),
                                          Operation *>) {
        results.push_back(static_cast<Operation *>(element));
      } else if constexpr (std::is_convertible_v<decltype(*std::begin(range)),
                                                 Value>) {
        results.push_back(element.template get<Value>());
      } else {
        results.push_back(static_cast<Attribute>(element));
      }
    }
  }

  /// Appends an element to the list.
  // Using ApplyToEachResult that can be implicitly constructed from a Value but
  // not from a concrete Op that is implicitly convertible to a Value to avoid
  // ambiguity.
  void push_back(Operation *op) { results.push_back(op); }
  void push_back(Attribute attr) { results.push_back(attr); }
  void push_back(ApplyToEachResult r) { results.push_back(r); }

  /// Reserves space for `size` elements in the list.
  void reserve(unsigned size) { results.reserve(size); }

  /// Iterators over the list.
  auto begin() { return results.begin(); }
  auto end() { return results.end(); }
  auto begin() const { return results.begin(); }
  auto end() const { return results.end(); }

  /// Returns the number of elements in the list.
  size_t size() const { return results.size(); }

  /// Element access. Expects the index to be in bounds.
  ApplyToEachResult &operator[](size_t index) { return results[index]; }
  const ApplyToEachResult &operator[](size_t index) const {
    return results[index];
  }

private:
  /// Underlying storage.
  SmallVector<ApplyToEachResult> results;
};

namespace detail {

/// Check that the contents of `partialResult` matches the number, kind (payload
/// op or parameter) and nullity (either all or none) requirements of
/// `transformOp`. Report errors and return failure otherwise.
LogicalResult checkApplyToOne(Operation *transformOp, Location payloadOpLoc,
                              const ApplyToEachResultList &partialResult);

/// "Transpose" the results produced by individual applications, arranging them
/// per result value of the transform op, and populate `transformResults` with
/// that. The number, kind and nullity of per-application results are assumed to
/// have been verified.
void setApplyToOneResults(Operation *transformOp,
                          TransformResults &transformResults,
                          ArrayRef<ApplyToEachResultList> results);

/// Applies a one-to-one or a one-to-many transform to each of the given
/// targets. Puts the results of transforms, if any, in `results` in the same
/// order. Fails if any of the application fails. Individual transforms must be
/// callable with the following signature:
///   - DiagnosedSilenceableFailure(OpTy,
///       SmallVector<Operation*> &results, state)
/// where OpTy is either
///   - Operation *, in which case the transform is always applied;
///   - a concrete Op class, in which case a check is performed whether
///   `targets` contains operations of the same class and a silenceable failure
///   is reported if it does not.
template <typename TransformOpTy, typename Range>
DiagnosedSilenceableFailure applyTransformToEach(
    TransformOpTy transformOp, TransformRewriter &rewriter, Range &&targets,
    SmallVectorImpl<ApplyToEachResultList> &results, TransformState &state) {
  using OpTy = typename llvm::function_traits<
      decltype(&TransformOpTy::applyToOne)>::template arg_t<1>;
  static_assert(std::is_convertible<OpTy, Operation *>::value,
                "expected transform function to take an operation");
  OpBuilder::InsertionGuard g(rewriter);

  SmallVector<Diagnostic> silenceableStack;
  unsigned expectedNumResults = transformOp->getNumResults();
  for (Operation *target : targets) {
    auto specificOp = dyn_cast<OpTy>(target);
    if (!specificOp) {
      Diagnostic diag(transformOp->getLoc(), DiagnosticSeverity::Error);
      diag << "transform applied to the wrong op kind";
      diag.attachNote(target->getLoc()) << "when applied to this op";
      silenceableStack.push_back(std::move(diag));
      continue;
    }

    ApplyToEachResultList partialResults;
    partialResults.reserve(expectedNumResults);
    Location specificOpLoc = specificOp->getLoc();
    rewriter.setInsertionPoint(specificOp);
    DiagnosedSilenceableFailure res =
        transformOp.applyToOne(rewriter, specificOp, partialResults, state);
    if (res.isDefiniteFailure())
      return DiagnosedSilenceableFailure::definiteFailure();

    if (res.isSilenceableFailure()) {
      res.takeDiagnostics(silenceableStack);
      continue;
    }

    if (failed(detail::checkApplyToOne(transformOp, specificOpLoc,
                                       partialResults))) {
      return DiagnosedSilenceableFailure::definiteFailure();
    }
    results.push_back(std::move(partialResults));
  }
  if (!silenceableStack.empty()) {
    return DiagnosedSilenceableFailure::silenceableFailure(
        std::move(silenceableStack));
  }
  return DiagnosedSilenceableFailure::success();
}

/// Reports an error and returns failure if `targets` contains an ancestor
/// operation before its descendant (or a copy of itself). Implementation detail
/// for expensive checks during `TransformEachOpTrait::apply`.
LogicalResult checkNestedConsumption(Location loc,
                                     ArrayRef<Operation *> targets);

} // namespace detail
} // namespace transform
} // namespace mlir

template <typename OpTy>
mlir::DiagnosedSilenceableFailure
mlir::transform::TransformEachOpTrait<OpTy>::apply(
    TransformRewriter &rewriter, TransformResults &transformResults,
    TransformState &state) {
  Value handle = this->getOperation()->getOperand(0);
  auto targets = state.getPayloadOps(handle);

  // If the operand is consumed, check if it is associated with operations that
  // may be erased before their nested operations are.
  if (state.getOptions().getExpensiveChecksEnabled() &&
      isHandleConsumed(handle, cast<transform::TransformOpInterface>(
                                   this->getOperation())) &&
      failed(detail::checkNestedConsumption(this->getOperation()->getLoc(),
                                            llvm::to_vector(targets)))) {
    return DiagnosedSilenceableFailure::definiteFailure();
  }

  // Step 1. Handle the corner case where no target is specified.
  // This is typically the case when the matcher fails to apply and we need to
  // propagate gracefully.
  // In this case, we fill all results with an empty vector.
  if (std::empty(targets)) {
    SmallVector<Operation *> emptyPayload;
    SmallVector<Attribute> emptyParams;
    for (OpResult r : this->getOperation()->getResults()) {
      if (isa<TransformParamTypeInterface>(r.getType()))
        transformResults.setParams(r, emptyParams);
      else if (isa<TransformValueHandleTypeInterface>(r.getType()))
        transformResults.setValues(r, ValueRange());
      else
        transformResults.set(r, emptyPayload);
    }
    return DiagnosedSilenceableFailure::success();
  }

  // Step 2. Call applyToOne on each target and record newly produced ops in its
  // corresponding results entry.
  SmallVector<ApplyToEachResultList, 1> results;
  DiagnosedSilenceableFailure result = detail::applyTransformToEach(
      cast<OpTy>(this->getOperation()), rewriter, targets, results, state);

  // Step 3. Propagate the definite failure if any and bail out.
  if (result.isDefiniteFailure())
    return result;

  // Step 4. "Transpose" the results produced by individual applications,
  // arranging them per result value of the transform op. The number, kind and
  // nullity of per-application results have been verified by the callback
  // above.
  detail::setApplyToOneResults(this->getOperation(), transformResults, results);

  // Step 5. ApplyToOne may have returned silenceableFailure, propagate it.
  return result;
}

template <typename OpTy>
llvm::LogicalResult
mlir::transform::TransformEachOpTrait<OpTy>::verifyTrait(Operation *op) {
  static_assert(OpTy::template hasTrait<OpTrait::OneOperand>(),
                "expected single-operand op");
  if (!op->getName().getInterface<TransformOpInterface>()) {
    return op->emitError() << "TransformEachOpTrait should only be attached to "
                              "ops that implement TransformOpInterface";
  }

  return success();
}

#endif // DIALECT_TRANSFORM_INTERFACES_TRANSFORMINTERFACES_H
