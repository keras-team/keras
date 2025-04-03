//===- PDLPatternMatch.h - PDLPatternMatcher classes -------==---*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_IR_PDLPATTERNMATCH_H
#define MLIR_IR_PDLPATTERNMATCH_H

#include "mlir/Config/mlir-config.h"

#if MLIR_ENABLE_PDL_IN_PATTERNMATCH
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"

namespace mlir {
//===----------------------------------------------------------------------===//
// PDL Patterns
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// PDLValue

/// Storage type of byte-code interpreter values. These are passed to constraint
/// functions as arguments.
class PDLValue {
public:
  /// The underlying kind of a PDL value.
  enum class Kind { Attribute, Operation, Type, TypeRange, Value, ValueRange };

  /// Construct a new PDL value.
  PDLValue(const PDLValue &other) = default;
  PDLValue(std::nullptr_t = nullptr) {}
  PDLValue(Attribute value)
      : value(value.getAsOpaquePointer()), kind(Kind::Attribute) {}
  PDLValue(Operation *value) : value(value), kind(Kind::Operation) {}
  PDLValue(Type value) : value(value.getAsOpaquePointer()), kind(Kind::Type) {}
  PDLValue(TypeRange *value) : value(value), kind(Kind::TypeRange) {}
  PDLValue(Value value)
      : value(value.getAsOpaquePointer()), kind(Kind::Value) {}
  PDLValue(ValueRange *value) : value(value), kind(Kind::ValueRange) {}

  /// Returns true if the type of the held value is `T`.
  template <typename T>
  bool isa() const {
    assert(value && "isa<> used on a null value");
    return kind == getKindOf<T>();
  }

  /// Attempt to dynamically cast this value to type `T`, returns null if this
  /// value is not an instance of `T`.
  template <typename T,
            typename ResultT = std::conditional_t<
                std::is_convertible<T, bool>::value, T, std::optional<T>>>
  ResultT dyn_cast() const {
    return isa<T>() ? castImpl<T>() : ResultT();
  }

  /// Cast this value to type `T`, asserts if this value is not an instance of
  /// `T`.
  template <typename T>
  T cast() const {
    assert(isa<T>() && "expected value to be of type `T`");
    return castImpl<T>();
  }

  /// Get an opaque pointer to the value.
  const void *getAsOpaquePointer() const { return value; }

  /// Return if this value is null or not.
  explicit operator bool() const { return value; }

  /// Return the kind of this value.
  Kind getKind() const { return kind; }

  /// Print this value to the provided output stream.
  void print(raw_ostream &os) const;

  /// Print the specified value kind to an output stream.
  static void print(raw_ostream &os, Kind kind);

private:
  /// Find the index of a given type in a range of other types.
  template <typename...>
  struct index_of_t;
  template <typename T, typename... R>
  struct index_of_t<T, T, R...> : std::integral_constant<size_t, 0> {};
  template <typename T, typename F, typename... R>
  struct index_of_t<T, F, R...>
      : std::integral_constant<size_t, 1 + index_of_t<T, R...>::value> {};

  /// Return the kind used for the given T.
  template <typename T>
  static Kind getKindOf() {
    return static_cast<Kind>(index_of_t<T, Attribute, Operation *, Type,
                                        TypeRange, Value, ValueRange>::value);
  }

  /// The internal implementation of `cast`, that returns the underlying value
  /// as the given type `T`.
  template <typename T>
  std::enable_if_t<llvm::is_one_of<T, Attribute, Type, Value>::value, T>
  castImpl() const {
    return T::getFromOpaquePointer(value);
  }
  template <typename T>
  std::enable_if_t<llvm::is_one_of<T, TypeRange, ValueRange>::value, T>
  castImpl() const {
    return *reinterpret_cast<T *>(const_cast<void *>(value));
  }
  template <typename T>
  std::enable_if_t<std::is_pointer<T>::value, T> castImpl() const {
    return reinterpret_cast<T>(const_cast<void *>(value));
  }

  /// The internal opaque representation of a PDLValue.
  const void *value{nullptr};
  /// The kind of the opaque value.
  Kind kind{Kind::Attribute};
};

inline raw_ostream &operator<<(raw_ostream &os, PDLValue value) {
  value.print(os);
  return os;
}

inline raw_ostream &operator<<(raw_ostream &os, PDLValue::Kind kind) {
  PDLValue::print(os, kind);
  return os;
}

//===----------------------------------------------------------------------===//
// PDLResultList

/// The class represents a list of PDL results, returned by a native rewrite
/// method. It provides the mechanism with which to pass PDLValues back to the
/// PDL bytecode.
class PDLResultList {
public:
  /// Push a new Attribute value onto the result list.
  void push_back(Attribute value) { results.push_back(value); }

  /// Push a new Operation onto the result list.
  void push_back(Operation *value) { results.push_back(value); }

  /// Push a new Type onto the result list.
  void push_back(Type value) { results.push_back(value); }

  /// Push a new TypeRange onto the result list.
  void push_back(TypeRange value) {
    // The lifetime of a TypeRange can't be guaranteed, so we'll need to
    // allocate a storage for it.
    llvm::OwningArrayRef<Type> storage(value.size());
    llvm::copy(value, storage.begin());
    allocatedTypeRanges.emplace_back(std::move(storage));
    typeRanges.push_back(allocatedTypeRanges.back());
    results.push_back(&typeRanges.back());
  }
  void push_back(ValueTypeRange<OperandRange> value) {
    typeRanges.push_back(value);
    results.push_back(&typeRanges.back());
  }
  void push_back(ValueTypeRange<ResultRange> value) {
    typeRanges.push_back(value);
    results.push_back(&typeRanges.back());
  }

  /// Push a new Value onto the result list.
  void push_back(Value value) { results.push_back(value); }

  /// Push a new ValueRange onto the result list.
  void push_back(ValueRange value) {
    // The lifetime of a ValueRange can't be guaranteed, so we'll need to
    // allocate a storage for it.
    llvm::OwningArrayRef<Value> storage(value.size());
    llvm::copy(value, storage.begin());
    allocatedValueRanges.emplace_back(std::move(storage));
    valueRanges.push_back(allocatedValueRanges.back());
    results.push_back(&valueRanges.back());
  }
  void push_back(OperandRange value) {
    valueRanges.push_back(value);
    results.push_back(&valueRanges.back());
  }
  void push_back(ResultRange value) {
    valueRanges.push_back(value);
    results.push_back(&valueRanges.back());
  }

protected:
  /// Create a new result list with the expected number of results.
  PDLResultList(unsigned maxNumResults) {
    // For now just reserve enough space for all of the results. We could do
    // separate counts per range type, but it isn't really worth it unless there
    // are a "large" number of results.
    typeRanges.reserve(maxNumResults);
    valueRanges.reserve(maxNumResults);
  }

  /// The PDL results held by this list.
  SmallVector<PDLValue> results;
  /// Memory used to store ranges held by the list.
  SmallVector<TypeRange> typeRanges;
  SmallVector<ValueRange> valueRanges;
  /// Memory allocated to store ranges in the result list whose lifetime was
  /// generated in the native function.
  SmallVector<llvm::OwningArrayRef<Type>> allocatedTypeRanges;
  SmallVector<llvm::OwningArrayRef<Value>> allocatedValueRanges;
};

//===----------------------------------------------------------------------===//
// PDLPatternConfig

/// An individual configuration for a pattern, which can be accessed by native
/// functions via the PDLPatternConfigSet. This allows for injecting additional
/// configuration into PDL patterns that is specific to certain compilation
/// flows.
class PDLPatternConfig {
public:
  virtual ~PDLPatternConfig() = default;

  /// Hooks that are invoked at the beginning and end of a rewrite of a matched
  /// pattern. These can be used to setup any specific state necessary for the
  /// rewrite.
  virtual void notifyRewriteBegin(PatternRewriter &rewriter) {}
  virtual void notifyRewriteEnd(PatternRewriter &rewriter) {}

  /// Return the TypeID that represents this configuration.
  TypeID getTypeID() const { return id; }

protected:
  PDLPatternConfig(TypeID id) : id(id) {}

private:
  TypeID id;
};

/// This class provides a base class for users implementing a type of pattern
/// configuration.
template <typename T>
class PDLPatternConfigBase : public PDLPatternConfig {
public:
  /// Support LLVM style casting.
  static bool classof(const PDLPatternConfig *config) {
    return config->getTypeID() == getConfigID();
  }

  /// Return the type id used for this configuration.
  static TypeID getConfigID() { return TypeID::get<T>(); }

protected:
  PDLPatternConfigBase() : PDLPatternConfig(getConfigID()) {}
};

/// This class contains a set of configurations for a specific pattern.
/// Configurations are uniqued by TypeID, meaning that only one configuration of
/// each type is allowed.
class PDLPatternConfigSet {
public:
  PDLPatternConfigSet() = default;

  /// Construct a set with the given configurations.
  template <typename... ConfigsT>
  PDLPatternConfigSet(ConfigsT &&...configs) {
    (addConfig(std::forward<ConfigsT>(configs)), ...);
  }

  /// Get the configuration defined by the given type. Asserts that the
  /// configuration of the provided type exists.
  template <typename T>
  const T &get() const {
    const T *config = tryGet<T>();
    assert(config && "configuration not found");
    return *config;
  }

  /// Get the configuration defined by the given type, returns nullptr if the
  /// configuration does not exist.
  template <typename T>
  const T *tryGet() const {
    for (const auto &configIt : configs)
      if (const T *config = dyn_cast<T>(configIt.get()))
        return config;
    return nullptr;
  }

  /// Notify the configurations within this set at the beginning or end of a
  /// rewrite of a matched pattern.
  void notifyRewriteBegin(PatternRewriter &rewriter) {
    for (const auto &config : configs)
      config->notifyRewriteBegin(rewriter);
  }
  void notifyRewriteEnd(PatternRewriter &rewriter) {
    for (const auto &config : configs)
      config->notifyRewriteEnd(rewriter);
  }

protected:
  /// Add a configuration to the set.
  template <typename T>
  void addConfig(T &&config) {
    assert(!tryGet<std::decay_t<T>>() && "configuration already exists");
    configs.emplace_back(
        std::make_unique<std::decay_t<T>>(std::forward<T>(config)));
  }

  /// The set of configurations for this pattern. This uses a vector instead of
  /// a map with the expectation that the number of configurations per set is
  /// small (<= 1).
  SmallVector<std::unique_ptr<PDLPatternConfig>> configs;
};

//===----------------------------------------------------------------------===//
// PDLPatternModule

/// A generic PDL pattern constraint function. This function applies a
/// constraint to a given set of opaque PDLValue entities. Returns success if
/// the constraint successfully held, failure otherwise.
using PDLConstraintFunction = std::function<LogicalResult(
    PatternRewriter &, PDLResultList &, ArrayRef<PDLValue>)>;

/// A native PDL rewrite function. This function performs a rewrite on the
/// given set of values. Any results from this rewrite that should be passed
/// back to PDL should be added to the provided result list. This method is only
/// invoked when the corresponding match was successful. Returns failure if an
/// invariant of the rewrite was broken (certain rewriters may recover from
/// partial pattern application).
using PDLRewriteFunction = std::function<LogicalResult(
    PatternRewriter &, PDLResultList &, ArrayRef<PDLValue>)>;

namespace detail {
namespace pdl_function_builder {
/// A utility variable that always resolves to false. This is useful for static
/// asserts that are always false, but only should fire in certain templated
/// constructs. For example, if a templated function should never be called, the
/// function could be defined as:
///
/// template <typename T>
/// void foo() {
///  static_assert(always_false<T>, "This function should never be called");
/// }
///
template <class... T>
constexpr bool always_false = false;

//===----------------------------------------------------------------------===//
// PDL Function Builder: Type Processing
//===----------------------------------------------------------------------===//

/// This struct provides a convenient way to determine how to process a given
/// type as either a PDL parameter, or a result value. This allows for
/// supporting complex types in constraint and rewrite functions, without
/// requiring the user to hand-write the necessary glue code themselves.
/// Specializations of this class should implement the following methods to
/// enable support as a PDL argument or result type:
///
///   static LogicalResult verifyAsArg(
///     function_ref<LogicalResult(const Twine &)> errorFn, PDLValue pdlValue,
///     size_t argIdx);
///
///     * This method verifies that the given PDLValue is valid for use as a
///       value of `T`.
///
///   static T processAsArg(PDLValue pdlValue);
///
///     *  This method processes the given PDLValue as a value of `T`.
///
///   static void processAsResult(PatternRewriter &, PDLResultList &results,
///                               const T &value);
///
///     *  This method processes the given value of `T` as the result of a
///        function invocation. The method should package the value into an
///        appropriate form and append it to the given result list.
///
/// If the type `T` is based on a higher order value, consider using
/// `ProcessPDLValueBasedOn` as a base class of the specialization to simplify
/// the implementation.
///
template <typename T, typename Enable = void>
struct ProcessPDLValue;

/// This struct provides a simplified model for processing types that are based
/// on another type, e.g. APInt is based on the handling for IntegerAttr. This
/// allows for building the necessary processing functions on top of the base
/// value instead of a PDLValue. Derived users should implement the following
/// (which subsume the ProcessPDLValue variants):
///
///   static LogicalResult verifyAsArg(
///     function_ref<LogicalResult(const Twine &)> errorFn,
///     const BaseT &baseValue, size_t argIdx);
///
///     * This method verifies that the given PDLValue is valid for use as a
///       value of `T`.
///
///   static T processAsArg(BaseT baseValue);
///
///     *  This method processes the given base value as a value of `T`.
///
template <typename T, typename BaseT>
struct ProcessPDLValueBasedOn {
  static LogicalResult
  verifyAsArg(function_ref<LogicalResult(const Twine &)> errorFn,
              PDLValue pdlValue, size_t argIdx) {
    // Verify the base class before continuing.
    if (failed(ProcessPDLValue<BaseT>::verifyAsArg(errorFn, pdlValue, argIdx)))
      return failure();
    return ProcessPDLValue<T>::verifyAsArg(
        errorFn, ProcessPDLValue<BaseT>::processAsArg(pdlValue), argIdx);
  }
  static T processAsArg(PDLValue pdlValue) {
    return ProcessPDLValue<T>::processAsArg(
        ProcessPDLValue<BaseT>::processAsArg(pdlValue));
  }

  /// Explicitly add the expected parent API to ensure the parent class
  /// implements the necessary API (and doesn't implicitly inherit it from
  /// somewhere else).
  static LogicalResult
  verifyAsArg(function_ref<LogicalResult(const Twine &)> errorFn, BaseT value,
              size_t argIdx) {
    return success();
  }
  static T processAsArg(BaseT baseValue);
};

/// This struct provides a simplified model for processing types that have
/// "builtin" PDLValue support:
///   * Attribute, Operation *, Type, TypeRange, ValueRange
template <typename T>
struct ProcessBuiltinPDLValue {
  static LogicalResult
  verifyAsArg(function_ref<LogicalResult(const Twine &)> errorFn,
              PDLValue pdlValue, size_t argIdx) {
    if (pdlValue)
      return success();
    return errorFn("expected a non-null value for argument " + Twine(argIdx) +
                   " of type: " + llvm::getTypeName<T>());
  }

  static T processAsArg(PDLValue pdlValue) { return pdlValue.cast<T>(); }
  static void processAsResult(PatternRewriter &, PDLResultList &results,
                              T value) {
    results.push_back(value);
  }
};

/// This struct provides a simplified model for processing types that inherit
/// from builtin PDLValue types. For example, derived attributes like
/// IntegerAttr, derived types like IntegerType, derived operations like
/// ModuleOp, Interfaces, etc.
template <typename T, typename BaseT>
struct ProcessDerivedPDLValue : public ProcessPDLValueBasedOn<T, BaseT> {
  static LogicalResult
  verifyAsArg(function_ref<LogicalResult(const Twine &)> errorFn,
              BaseT baseValue, size_t argIdx) {
    return TypeSwitch<BaseT, LogicalResult>(baseValue)
        .Case([&](T) { return success(); })
        .Default([&](BaseT) {
          return errorFn("expected argument " + Twine(argIdx) +
                         " to be of type: " + llvm::getTypeName<T>());
        });
  }
  using ProcessPDLValueBasedOn<T, BaseT>::verifyAsArg;

  static T processAsArg(BaseT baseValue) {
    return baseValue.template cast<T>();
  }
  using ProcessPDLValueBasedOn<T, BaseT>::processAsArg;

  static void processAsResult(PatternRewriter &, PDLResultList &results,
                              T value) {
    results.push_back(value);
  }
};

//===----------------------------------------------------------------------===//
// Attribute

template <>
struct ProcessPDLValue<Attribute> : public ProcessBuiltinPDLValue<Attribute> {};
template <typename T>
struct ProcessPDLValue<T,
                       std::enable_if_t<std::is_base_of<Attribute, T>::value>>
    : public ProcessDerivedPDLValue<T, Attribute> {};

/// Handling for various Attribute value types.
template <>
struct ProcessPDLValue<StringRef>
    : public ProcessPDLValueBasedOn<StringRef, StringAttr> {
  static StringRef processAsArg(StringAttr value) { return value.getValue(); }
  using ProcessPDLValueBasedOn<StringRef, StringAttr>::processAsArg;

  static void processAsResult(PatternRewriter &rewriter, PDLResultList &results,
                              StringRef value) {
    results.push_back(rewriter.getStringAttr(value));
  }
};
template <>
struct ProcessPDLValue<std::string>
    : public ProcessPDLValueBasedOn<std::string, StringAttr> {
  template <typename T>
  static std::string processAsArg(T value) {
    static_assert(always_false<T>,
                  "`std::string` arguments require a string copy, use "
                  "`StringRef` for string-like arguments instead");
    return {};
  }
  static void processAsResult(PatternRewriter &rewriter, PDLResultList &results,
                              StringRef value) {
    results.push_back(rewriter.getStringAttr(value));
  }
};

//===----------------------------------------------------------------------===//
// Operation

template <>
struct ProcessPDLValue<Operation *>
    : public ProcessBuiltinPDLValue<Operation *> {};
template <typename T>
struct ProcessPDLValue<T, std::enable_if_t<std::is_base_of<OpState, T>::value>>
    : public ProcessDerivedPDLValue<T, Operation *> {
  static T processAsArg(Operation *value) { return cast<T>(value); }
};

//===----------------------------------------------------------------------===//
// Type

template <>
struct ProcessPDLValue<Type> : public ProcessBuiltinPDLValue<Type> {};
template <typename T>
struct ProcessPDLValue<T, std::enable_if_t<std::is_base_of<Type, T>::value>>
    : public ProcessDerivedPDLValue<T, Type> {};

//===----------------------------------------------------------------------===//
// TypeRange

template <>
struct ProcessPDLValue<TypeRange> : public ProcessBuiltinPDLValue<TypeRange> {};
template <>
struct ProcessPDLValue<ValueTypeRange<OperandRange>> {
  static void processAsResult(PatternRewriter &, PDLResultList &results,
                              ValueTypeRange<OperandRange> types) {
    results.push_back(types);
  }
};
template <>
struct ProcessPDLValue<ValueTypeRange<ResultRange>> {
  static void processAsResult(PatternRewriter &, PDLResultList &results,
                              ValueTypeRange<ResultRange> types) {
    results.push_back(types);
  }
};
template <unsigned N>
struct ProcessPDLValue<SmallVector<Type, N>> {
  static void processAsResult(PatternRewriter &, PDLResultList &results,
                              SmallVector<Type, N> values) {
    results.push_back(TypeRange(values));
  }
};

//===----------------------------------------------------------------------===//
// Value

template <>
struct ProcessPDLValue<Value> : public ProcessBuiltinPDLValue<Value> {};

//===----------------------------------------------------------------------===//
// ValueRange

template <>
struct ProcessPDLValue<ValueRange> : public ProcessBuiltinPDLValue<ValueRange> {
};
template <>
struct ProcessPDLValue<OperandRange> {
  static void processAsResult(PatternRewriter &, PDLResultList &results,
                              OperandRange values) {
    results.push_back(values);
  }
};
template <>
struct ProcessPDLValue<ResultRange> {
  static void processAsResult(PatternRewriter &, PDLResultList &results,
                              ResultRange values) {
    results.push_back(values);
  }
};
template <unsigned N>
struct ProcessPDLValue<SmallVector<Value, N>> {
  static void processAsResult(PatternRewriter &, PDLResultList &results,
                              SmallVector<Value, N> values) {
    results.push_back(ValueRange(values));
  }
};

//===----------------------------------------------------------------------===//
// PDL Function Builder: Argument Handling
//===----------------------------------------------------------------------===//

/// Validate the given PDLValues match the constraints defined by the argument
/// types of the given function. In the case of failure, a match failure
/// diagnostic is emitted.
/// FIXME: This should be completely removed in favor of `assertArgs`, but PDL
/// does not currently preserve Constraint application ordering.
template <typename PDLFnT, std::size_t... I>
LogicalResult verifyAsArgs(PatternRewriter &rewriter, ArrayRef<PDLValue> values,
                           std::index_sequence<I...>) {
  using FnTraitsT = llvm::function_traits<PDLFnT>;

  auto errorFn = [&](const Twine &msg) {
    return rewriter.notifyMatchFailure(rewriter.getUnknownLoc(), msg);
  };
  return success(
      (succeeded(ProcessPDLValue<typename FnTraitsT::template arg_t<I + 1>>::
                     verifyAsArg(errorFn, values[I], I)) &&
       ...));
}

/// Assert that the given PDLValues match the constraints defined by the
/// arguments of the given function. In the case of failure, a fatal error
/// is emitted.
template <typename PDLFnT, std::size_t... I>
void assertArgs(PatternRewriter &rewriter, ArrayRef<PDLValue> values,
                std::index_sequence<I...>) {
  // We only want to do verification in debug builds, same as with `assert`.
#if LLVM_ENABLE_ABI_BREAKING_CHECKS
  using FnTraitsT = llvm::function_traits<PDLFnT>;
  auto errorFn = [&](const Twine &msg) -> LogicalResult {
    llvm::report_fatal_error(msg);
  };
  (void)errorFn;
  assert((succeeded(ProcessPDLValue<typename FnTraitsT::template arg_t<I + 1>>::
                        verifyAsArg(errorFn, values[I], I)) &&
          ...));
#endif
  (void)values;
}

//===----------------------------------------------------------------------===//
// PDL Function Builder: Results Handling
//===----------------------------------------------------------------------===//

/// Store a single result within the result list.
template <typename T>
static LogicalResult processResults(PatternRewriter &rewriter,
                                    PDLResultList &results, T &&value) {
  ProcessPDLValue<T>::processAsResult(rewriter, results,
                                      std::forward<T>(value));
  return success();
}

/// Store a std::pair<> as individual results within the result list.
template <typename T1, typename T2>
static LogicalResult processResults(PatternRewriter &rewriter,
                                    PDLResultList &results,
                                    std::pair<T1, T2> &&pair) {
  if (failed(processResults(rewriter, results, std::move(pair.first))) ||
      failed(processResults(rewriter, results, std::move(pair.second))))
    return failure();
  return success();
}

/// Store a std::tuple<> as individual results within the result list.
template <typename... Ts>
static LogicalResult processResults(PatternRewriter &rewriter,
                                    PDLResultList &results,
                                    std::tuple<Ts...> &&tuple) {
  auto applyFn = [&](auto &&...args) {
    return (succeeded(processResults(rewriter, results, std::move(args))) &&
            ...);
  };
  return success(std::apply(applyFn, std::move(tuple)));
}

/// Handle LogicalResult propagation.
inline LogicalResult processResults(PatternRewriter &rewriter,
                                    PDLResultList &results,
                                    LogicalResult &&result) {
  return result;
}
template <typename T>
static LogicalResult processResults(PatternRewriter &rewriter,
                                    PDLResultList &results,
                                    FailureOr<T> &&result) {
  if (failed(result))
    return failure();
  return processResults(rewriter, results, std::move(*result));
}

//===----------------------------------------------------------------------===//
// PDL Constraint Builder
//===----------------------------------------------------------------------===//

/// Process the arguments of a native constraint and invoke it.
template <typename PDLFnT, std::size_t... I,
          typename FnTraitsT = llvm::function_traits<PDLFnT>>
typename FnTraitsT::result_t
processArgsAndInvokeConstraint(PDLFnT &fn, PatternRewriter &rewriter,
                               ArrayRef<PDLValue> values,
                               std::index_sequence<I...>) {
  return fn(
      rewriter,
      (ProcessPDLValue<typename FnTraitsT::template arg_t<I + 1>>::processAsArg(
          values[I]))...);
}

/// Build a constraint function from the given function `ConstraintFnT`. This
/// allows for enabling the user to define simpler, more direct constraint
/// functions without needing to handle the low-level PDL goop.
///
/// If the constraint function is already in the correct form, we just forward
/// it directly.
template <typename ConstraintFnT>
std::enable_if_t<
    std::is_convertible<ConstraintFnT, PDLConstraintFunction>::value,
    PDLConstraintFunction>
buildConstraintFn(ConstraintFnT &&constraintFn) {
  return std::forward<ConstraintFnT>(constraintFn);
}
/// Otherwise, we generate a wrapper that will unpack the PDLValues in the form
/// we desire.
template <typename ConstraintFnT>
std::enable_if_t<
    !std::is_convertible<ConstraintFnT, PDLConstraintFunction>::value,
    PDLConstraintFunction>
buildConstraintFn(ConstraintFnT &&constraintFn) {
  return [constraintFn = std::forward<ConstraintFnT>(constraintFn)](
             PatternRewriter &rewriter, PDLResultList &,
             ArrayRef<PDLValue> values) -> LogicalResult {
    auto argIndices = std::make_index_sequence<
        llvm::function_traits<ConstraintFnT>::num_args - 1>();
    if (failed(verifyAsArgs<ConstraintFnT>(rewriter, values, argIndices)))
      return failure();
    return processArgsAndInvokeConstraint(constraintFn, rewriter, values,
                                          argIndices);
  };
}

//===----------------------------------------------------------------------===//
// PDL Rewrite Builder
//===----------------------------------------------------------------------===//

/// Process the arguments of a native rewrite and invoke it.
/// This overload handles the case of no return values.
template <typename PDLFnT, std::size_t... I,
          typename FnTraitsT = llvm::function_traits<PDLFnT>>
std::enable_if_t<std::is_same<typename FnTraitsT::result_t, void>::value,
                 LogicalResult>
processArgsAndInvokeRewrite(PDLFnT &fn, PatternRewriter &rewriter,
                            PDLResultList &, ArrayRef<PDLValue> values,
                            std::index_sequence<I...>) {
  fn(rewriter,
     (ProcessPDLValue<typename FnTraitsT::template arg_t<I + 1>>::processAsArg(
         values[I]))...);
  return success();
}
/// This overload handles the case of return values, which need to be packaged
/// into the result list.
template <typename PDLFnT, std::size_t... I,
          typename FnTraitsT = llvm::function_traits<PDLFnT>>
std::enable_if_t<!std::is_same<typename FnTraitsT::result_t, void>::value,
                 LogicalResult>
processArgsAndInvokeRewrite(PDLFnT &fn, PatternRewriter &rewriter,
                            PDLResultList &results, ArrayRef<PDLValue> values,
                            std::index_sequence<I...>) {
  return processResults(
      rewriter, results,
      fn(rewriter, (ProcessPDLValue<typename FnTraitsT::template arg_t<I + 1>>::
                        processAsArg(values[I]))...));
  (void)values;
}

/// Build a rewrite function from the given function `RewriteFnT`. This
/// allows for enabling the user to define simpler, more direct rewrite
/// functions without needing to handle the low-level PDL goop.
///
/// If the rewrite function is already in the correct form, we just forward
/// it directly.
template <typename RewriteFnT>
std::enable_if_t<std::is_convertible<RewriteFnT, PDLRewriteFunction>::value,
                 PDLRewriteFunction>
buildRewriteFn(RewriteFnT &&rewriteFn) {
  return std::forward<RewriteFnT>(rewriteFn);
}
/// Otherwise, we generate a wrapper that will unpack the PDLValues in the form
/// we desire.
template <typename RewriteFnT>
std::enable_if_t<!std::is_convertible<RewriteFnT, PDLRewriteFunction>::value,
                 PDLRewriteFunction>
buildRewriteFn(RewriteFnT &&rewriteFn) {
  return [rewriteFn = std::forward<RewriteFnT>(rewriteFn)](
             PatternRewriter &rewriter, PDLResultList &results,
             ArrayRef<PDLValue> values) {
    auto argIndices =
        std::make_index_sequence<llvm::function_traits<RewriteFnT>::num_args -
                                 1>();
    assertArgs<RewriteFnT>(rewriter, values, argIndices);
    return processArgsAndInvokeRewrite(rewriteFn, rewriter, results, values,
                                       argIndices);
  };
}

} // namespace pdl_function_builder
} // namespace detail

//===----------------------------------------------------------------------===//
// PDLPatternModule

/// This class contains all of the necessary data for a set of PDL patterns, or
/// pattern rewrites specified in the form of the PDL dialect. This PDL module
/// contained by this pattern may contain any number of `pdl.pattern`
/// operations.
class PDLPatternModule {
public:
  PDLPatternModule() = default;

  /// Construct a PDL pattern with the given module and configurations.
  PDLPatternModule(OwningOpRef<ModuleOp> module)
      : pdlModule(std::move(module)) {}
  template <typename... ConfigsT>
  PDLPatternModule(OwningOpRef<ModuleOp> module, ConfigsT &&...patternConfigs)
      : PDLPatternModule(std::move(module)) {
    auto configSet = std::make_unique<PDLPatternConfigSet>(
        std::forward<ConfigsT>(patternConfigs)...);
    attachConfigToPatterns(*pdlModule, *configSet);
    configs.emplace_back(std::move(configSet));
  }

  /// Merge the state in `other` into this pattern module.
  void mergeIn(PDLPatternModule &&other);

  /// Return the internal PDL module of this pattern.
  ModuleOp getModule() { return pdlModule.get(); }

  /// Return the MLIR context of this pattern.
  MLIRContext *getContext() { return getModule()->getContext(); }

  //===--------------------------------------------------------------------===//
  // Function Registry

  /// Register a constraint function with PDL. A constraint function may be
  /// specified in one of two ways:
  ///
  ///   * `LogicalResult (PatternRewriter &,
  ///                     PDLResultList &,
  ///                     ArrayRef<PDLValue>)`
  ///
  ///   In this overload the arguments of the constraint function are passed via
  ///   the low-level PDLValue form, and the results are manually appended to
  ///   the given result list.
  ///
  ///   * `LogicalResult (PatternRewriter &, ValueTs... values)`
  ///
  ///   In this form the arguments of the constraint function are passed via the
  ///   expected high level C++ type. In this form, the framework will
  ///   automatically unwrap PDLValues and convert them to the expected ValueTs.
  ///   For example, if the constraint function accepts a `Operation *`, the
  ///   framework will automatically cast the input PDLValue. In the case of a
  ///   `StringRef`, the framework will automatically unwrap the argument as a
  ///   StringAttr and pass the underlying string value. To see the full list of
  ///   supported types, or to see how to add handling for custom types, view
  ///   the definition of `ProcessPDLValue` above.
  void registerConstraintFunction(StringRef name,
                                  PDLConstraintFunction constraintFn);
  template <typename ConstraintFnT>
  void registerConstraintFunction(StringRef name,
                                  ConstraintFnT &&constraintFn) {
    registerConstraintFunction(name,
                               detail::pdl_function_builder::buildConstraintFn(
                                   std::forward<ConstraintFnT>(constraintFn)));
  }

  /// Register a rewrite function with PDL. A rewrite function may be specified
  /// in one of two ways:
  ///
  ///   * `void (PatternRewriter &, PDLResultList &, ArrayRef<PDLValue>)`
  ///
  ///   In this overload the arguments of the constraint function are passed via
  ///   the low-level PDLValue form, and the results are manually appended to
  ///   the given result list.
  ///
  ///   * `ResultT (PatternRewriter &, ValueTs... values)`
  ///
  ///   In this form the arguments and result of the rewrite function are passed
  ///   via the expected high level C++ type. In this form, the framework will
  ///   automatically unwrap the PDLValues arguments and convert them to the
  ///   expected ValueTs. It will also automatically handle the processing and
  ///   packaging of the result value to the result list. For example, if the
  ///   rewrite function takes a `Operation *`, the framework will automatically
  ///   cast the input PDLValue. In the case of a `StringRef`, the framework
  ///   will automatically unwrap the argument as a StringAttr and pass the
  ///   underlying string value. In the reverse case, if the rewrite returns a
  ///   StringRef or std::string, it will automatically package this as a
  ///   StringAttr and append it to the result list. To see the full list of
  ///   supported types, or to see how to add handling for custom types, view
  ///   the definition of `ProcessPDLValue` above.
  void registerRewriteFunction(StringRef name, PDLRewriteFunction rewriteFn);
  template <typename RewriteFnT>
  void registerRewriteFunction(StringRef name, RewriteFnT &&rewriteFn) {
    registerRewriteFunction(name, detail::pdl_function_builder::buildRewriteFn(
                                      std::forward<RewriteFnT>(rewriteFn)));
  }

  /// Return the set of the registered constraint functions.
  const llvm::StringMap<PDLConstraintFunction> &getConstraintFunctions() const {
    return constraintFunctions;
  }
  llvm::StringMap<PDLConstraintFunction> takeConstraintFunctions() {
    return constraintFunctions;
  }
  /// Return the set of the registered rewrite functions.
  const llvm::StringMap<PDLRewriteFunction> &getRewriteFunctions() const {
    return rewriteFunctions;
  }
  llvm::StringMap<PDLRewriteFunction> takeRewriteFunctions() {
    return rewriteFunctions;
  }

  /// Return the set of the registered pattern configs.
  SmallVector<std::unique_ptr<PDLPatternConfigSet>> takeConfigs() {
    return std::move(configs);
  }
  DenseMap<Operation *, PDLPatternConfigSet *> takeConfigMap() {
    return std::move(configMap);
  }

  /// Clear out the patterns and functions within this module.
  void clear() {
    pdlModule = nullptr;
    constraintFunctions.clear();
    rewriteFunctions.clear();
  }

private:
  /// Attach the given pattern config set to the patterns defined within the
  /// given module.
  void attachConfigToPatterns(ModuleOp module, PDLPatternConfigSet &configSet);

  /// The module containing the `pdl.pattern` operations.
  OwningOpRef<ModuleOp> pdlModule;

  /// The set of configuration sets referenced by patterns within `pdlModule`.
  SmallVector<std::unique_ptr<PDLPatternConfigSet>> configs;
  DenseMap<Operation *, PDLPatternConfigSet *> configMap;

  /// The external functions referenced from within the PDL module.
  llvm::StringMap<PDLConstraintFunction> constraintFunctions;
  llvm::StringMap<PDLRewriteFunction> rewriteFunctions;
};
} // namespace mlir

#else

namespace mlir {
// Stubs for when PDL in pattern rewrites is not enabled.

class PDLValue {
public:
  template <typename T>
  T dyn_cast() const {
    return nullptr;
  }
};
class PDLResultList {};
using PDLConstraintFunction = std::function<LogicalResult(
    PatternRewriter &, PDLResultList &, ArrayRef<PDLValue>)>;
using PDLRewriteFunction = std::function<LogicalResult(
    PatternRewriter &, PDLResultList &, ArrayRef<PDLValue>)>;

class PDLPatternModule {
public:
  PDLPatternModule() = default;

  PDLPatternModule(OwningOpRef<ModuleOp> /*module*/) {}
  MLIRContext *getContext() {
    llvm_unreachable("Error: PDL for rewrites when PDL is not enabled");
  }
  void mergeIn(PDLPatternModule &&other) {}
  void clear() {}
  template <typename ConstraintFnT>
  void registerConstraintFunction(StringRef name,
                                  ConstraintFnT &&constraintFn) {}
  void registerRewriteFunction(StringRef name, PDLRewriteFunction rewriteFn) {}
  template <typename RewriteFnT>
  void registerRewriteFunction(StringRef name, RewriteFnT &&rewriteFn) {}
  const llvm::StringMap<PDLConstraintFunction> &getConstraintFunctions() const {
    return constraintFunctions;
  }

private:
  llvm::StringMap<PDLConstraintFunction> constraintFunctions;
};

} // namespace mlir
#endif

#endif // MLIR_IR_PDLPATTERNMATCH_H
