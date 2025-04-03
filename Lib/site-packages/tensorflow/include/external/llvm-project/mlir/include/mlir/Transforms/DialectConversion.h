//===- DialectConversion.h - MLIR dialect conversion pass -------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares a generic pass for converting between MLIR dialects.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_TRANSFORMS_DIALECTCONVERSION_H_
#define MLIR_TRANSFORMS_DIALECTCONVERSION_H_

#include "mlir/Config/mlir-config.h"
#include "mlir/Rewrite/FrozenRewritePatternSet.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/StringMap.h"
#include <type_traits>

namespace mlir {

// Forward declarations.
class Attribute;
class Block;
struct ConversionConfig;
class ConversionPatternRewriter;
class MLIRContext;
class Operation;
struct OperationConverter;
class Type;
class Value;

//===----------------------------------------------------------------------===//
// Type Conversion
//===----------------------------------------------------------------------===//

/// Type conversion class. Specific conversions and materializations can be
/// registered using addConversion and addMaterialization, respectively.
class TypeConverter {
public:
  virtual ~TypeConverter() = default;
  TypeConverter() = default;
  // Copy the registered conversions, but not the caches
  TypeConverter(const TypeConverter &other)
      : conversions(other.conversions),
        argumentMaterializations(other.argumentMaterializations),
        sourceMaterializations(other.sourceMaterializations),
        targetMaterializations(other.targetMaterializations),
        typeAttributeConversions(other.typeAttributeConversions) {}
  TypeConverter &operator=(const TypeConverter &other) {
    conversions = other.conversions;
    argumentMaterializations = other.argumentMaterializations;
    sourceMaterializations = other.sourceMaterializations;
    targetMaterializations = other.targetMaterializations;
    typeAttributeConversions = other.typeAttributeConversions;
    return *this;
  }

  /// This class provides all of the information necessary to convert a type
  /// signature.
  class SignatureConversion {
  public:
    SignatureConversion(unsigned numOrigInputs)
        : remappedInputs(numOrigInputs) {}

    /// This struct represents a range of new types or a single value that
    /// remaps an existing signature input.
    struct InputMapping {
      size_t inputNo, size;
      Value replacementValue;
    };

    /// Return the argument types for the new signature.
    ArrayRef<Type> getConvertedTypes() const { return argTypes; }

    /// Get the input mapping for the given argument.
    std::optional<InputMapping> getInputMapping(unsigned input) const {
      return remappedInputs[input];
    }

    //===------------------------------------------------------------------===//
    // Conversion Hooks
    //===------------------------------------------------------------------===//

    /// Remap an input of the original signature with a new set of types. The
    /// new types are appended to the new signature conversion.
    void addInputs(unsigned origInputNo, ArrayRef<Type> types);

    /// Append new input types to the signature conversion, this should only be
    /// used if the new types are not intended to remap an existing input.
    void addInputs(ArrayRef<Type> types);

    /// Remap an input of the original signature to another `replacement`
    /// value. This drops the original argument.
    void remapInput(unsigned origInputNo, Value replacement);

  private:
    /// Remap an input of the original signature with a range of types in the
    /// new signature.
    void remapInput(unsigned origInputNo, unsigned newInputNo,
                    unsigned newInputCount = 1);

    /// The remapping information for each of the original arguments.
    SmallVector<std::optional<InputMapping>, 4> remappedInputs;

    /// The set of new argument types.
    SmallVector<Type, 4> argTypes;
  };

  /// The general result of a type attribute conversion callback, allowing
  /// for early termination. The default constructor creates the na case.
  class AttributeConversionResult {
  public:
    constexpr AttributeConversionResult() : impl() {}
    AttributeConversionResult(Attribute attr) : impl(attr, resultTag) {}

    static AttributeConversionResult result(Attribute attr);
    static AttributeConversionResult na();
    static AttributeConversionResult abort();

    bool hasResult() const;
    bool isNa() const;
    bool isAbort() const;

    Attribute getResult() const;

  private:
    AttributeConversionResult(Attribute attr, unsigned tag) : impl(attr, tag) {}

    llvm::PointerIntPair<Attribute, 2> impl;
    // Note that na is 0 so that we can use PointerIntPair's default
    // constructor.
    static constexpr unsigned naTag = 0;
    static constexpr unsigned resultTag = 1;
    static constexpr unsigned abortTag = 2;
  };

  /// Register a conversion function. A conversion function must be convertible
  /// to any of the following forms(where `T` is a class derived from `Type`:
  ///   * std::optional<Type>(T)
  ///     - This form represents a 1-1 type conversion. It should return nullptr
  ///       or `std::nullopt` to signify failure. If `std::nullopt` is returned,
  ///       the converter is allowed to try another conversion function to
  ///       perform the conversion.
  ///   * std::optional<LogicalResult>(T, SmallVectorImpl<Type> &)
  ///     - This form represents a 1-N type conversion. It should return
  ///       `failure` or `std::nullopt` to signify a failed conversion. If the
  ///       new set of types is empty, the type is removed and any usages of the
  ///       existing value are expected to be removed during conversion. If
  ///       `std::nullopt` is returned, the converter is allowed to try another
  ///       conversion function to perform the conversion.
  ///   * std::optional<LogicalResult>(T, SmallVectorImpl<Type> &,
  ///                                  ArrayRef<Type>)
  ///     - This form represents a 1-N type conversion supporting recursive
  ///       types. The first two arguments and the return value are the same as
  ///       for the regular 1-N form. The third argument is contains is the
  ///       "call stack" of the recursive conversion: it contains the list of
  ///       types currently being converted, with the current type being the
  ///       last one. If it is present more than once in the list, the
  ///       conversion concerns a recursive type.
  /// Note: When attempting to convert a type, e.g. via 'convertType', the
  ///       mostly recently added conversions will be invoked first.
  template <typename FnT, typename T = typename llvm::function_traits<
                              std::decay_t<FnT>>::template arg_t<0>>
  void addConversion(FnT &&callback) {
    registerConversion(wrapCallback<T>(std::forward<FnT>(callback)));
  }

  /// All of the following materializations require function objects that are
  /// convertible to the following form:
  ///   `std::optional<Value>(OpBuilder &, T, ValueRange, Location)`,
  /// where `T` is any subclass of `Type`. This function is responsible for
  /// creating an operation, using the OpBuilder and Location provided, that
  /// "casts" a range of values into a single value of the given type `T`. It
  /// must return a Value of the type `T` on success, an `std::nullopt` if
  /// it failed but other materialization can be attempted, and `nullptr` on
  /// unrecoverable failure. Materialization functions must be provided when a
  /// type conversion may persist after the conversion has finished.

  /// This method registers a materialization that will be called when
  /// converting (potentially multiple) block arguments that were the result of
  /// a signature conversion of a single block argument, to a single SSA value
  /// with the old block argument type.
  template <typename FnT, typename T = typename llvm::function_traits<
                              std::decay_t<FnT>>::template arg_t<1>>
  void addArgumentMaterialization(FnT &&callback) {
    argumentMaterializations.emplace_back(
        wrapMaterialization<T>(std::forward<FnT>(callback)));
  }

  /// This method registers a materialization that will be called when
  /// converting a legal replacement value back to an illegal source type.
  /// This is used when some uses of the original, illegal value must persist
  /// beyond the main conversion.
  template <typename FnT, typename T = typename llvm::function_traits<
                              std::decay_t<FnT>>::template arg_t<1>>
  void addSourceMaterialization(FnT &&callback) {
    sourceMaterializations.emplace_back(
        wrapMaterialization<T>(std::forward<FnT>(callback)));
  }

  /// This method registers a materialization that will be called when
  /// converting an illegal (source) value to a legal (target) type.
  template <typename FnT, typename T = typename llvm::function_traits<
                              std::decay_t<FnT>>::template arg_t<1>>
  void addTargetMaterialization(FnT &&callback) {
    targetMaterializations.emplace_back(
        wrapMaterialization<T>(std::forward<FnT>(callback)));
  }

  /// Register a conversion function for attributes within types. Type
  /// converters may call this function in order to allow hoking into the
  /// translation of attributes that exist within types. For example, a type
  /// converter for the `memref` type could use these conversions to convert
  /// memory spaces or layouts in an extensible way.
  ///
  /// The conversion functions take a non-null Type or subclass of Type and a
  /// non-null Attribute (or subclass of Attribute), and returns a
  /// `AttributeConversionResult`. This result can either contan an `Attribute`,
  /// which may be `nullptr`, representing the conversion's success,
  /// `AttributeConversionResult::na()` (the default empty value), indicating
  /// that the conversion function did not apply and that further conversion
  /// functions should be checked, or `AttributeConversionResult::abort()`
  /// indicating that the conversion process should be aborted.
  ///
  /// Registered conversion functions are callled in the reverse of the order in
  /// which they were registered.
  template <
      typename FnT,
      typename T =
          typename llvm::function_traits<std::decay_t<FnT>>::template arg_t<0>,
      typename A =
          typename llvm::function_traits<std::decay_t<FnT>>::template arg_t<1>>
  void addTypeAttributeConversion(FnT &&callback) {
    registerTypeAttributeConversion(
        wrapTypeAttributeConversion<T, A>(std::forward<FnT>(callback)));
  }

  /// Convert the given type. This function should return failure if no valid
  /// conversion exists, success otherwise. If the new set of types is empty,
  /// the type is removed and any usages of the existing value are expected to
  /// be removed during conversion.
  LogicalResult convertType(Type t, SmallVectorImpl<Type> &results) const;

  /// This hook simplifies defining 1-1 type conversions. This function returns
  /// the type to convert to on success, and a null type on failure.
  Type convertType(Type t) const;

  /// Attempts a 1-1 type conversion, expecting the result type to be
  /// `TargetType`. Returns the converted type cast to `TargetType` on success,
  /// and a null type on conversion or cast failure.
  template <typename TargetType>
  TargetType convertType(Type t) const {
    return dyn_cast_or_null<TargetType>(convertType(t));
  }

  /// Convert the given set of types, filling 'results' as necessary. This
  /// returns failure if the conversion of any of the types fails, success
  /// otherwise.
  LogicalResult convertTypes(TypeRange types,
                             SmallVectorImpl<Type> &results) const;

  /// Return true if the given type is legal for this type converter, i.e. the
  /// type converts to itself.
  bool isLegal(Type type) const;

  /// Return true if all of the given types are legal for this type converter.
  template <typename RangeT>
  std::enable_if_t<!std::is_convertible<RangeT, Type>::value &&
                       !std::is_convertible<RangeT, Operation *>::value,
                   bool>
  isLegal(RangeT &&range) const {
    return llvm::all_of(range, [this](Type type) { return isLegal(type); });
  }
  /// Return true if the given operation has legal operand and result types.
  bool isLegal(Operation *op) const;

  /// Return true if the types of block arguments within the region are legal.
  bool isLegal(Region *region) const;

  /// Return true if the inputs and outputs of the given function type are
  /// legal.
  bool isSignatureLegal(FunctionType ty) const;

  /// This method allows for converting a specific argument of a signature. It
  /// takes as inputs the original argument input number, type.
  /// On success, it populates 'result' with any new mappings.
  LogicalResult convertSignatureArg(unsigned inputNo, Type type,
                                    SignatureConversion &result) const;
  LogicalResult convertSignatureArgs(TypeRange types,
                                     SignatureConversion &result,
                                     unsigned origInputOffset = 0) const;

  /// This function converts the type signature of the given block, by invoking
  /// 'convertSignatureArg' for each argument. This function should return a
  /// valid conversion for the signature on success, std::nullopt otherwise.
  std::optional<SignatureConversion> convertBlockSignature(Block *block) const;

  /// Materialize a conversion from a set of types into one result type by
  /// generating a cast sequence of some kind. See the respective
  /// `add*Materialization` for more information on the context for these
  /// methods.
  Value materializeArgumentConversion(OpBuilder &builder, Location loc,
                                      Type resultType,
                                      ValueRange inputs) const {
    return materializeConversion(argumentMaterializations, builder, loc,
                                 resultType, inputs);
  }
  Value materializeSourceConversion(OpBuilder &builder, Location loc,
                                    Type resultType, ValueRange inputs) const {
    return materializeConversion(sourceMaterializations, builder, loc,
                                 resultType, inputs);
  }
  Value materializeTargetConversion(OpBuilder &builder, Location loc,
                                    Type resultType, ValueRange inputs) const {
    return materializeConversion(targetMaterializations, builder, loc,
                                 resultType, inputs);
  }

  /// Convert an attribute present `attr` from within the type `type` using
  /// the registered conversion functions. If no applicable conversion has been
  /// registered, return std::nullopt. Note that the empty attribute/`nullptr`
  /// is a valid return value for this function.
  std::optional<Attribute> convertTypeAttribute(Type type,
                                                Attribute attr) const;

private:
  /// The signature of the callback used to convert a type. If the new set of
  /// types is empty, the type is removed and any usages of the existing value
  /// are expected to be removed during conversion.
  using ConversionCallbackFn = std::function<std::optional<LogicalResult>(
      Type, SmallVectorImpl<Type> &)>;

  /// The signature of the callback used to materialize a conversion.
  using MaterializationCallbackFn = std::function<std::optional<Value>(
      OpBuilder &, Type, ValueRange, Location)>;

  /// The signature of the callback used to convert a type attribute.
  using TypeAttributeConversionCallbackFn =
      std::function<AttributeConversionResult(Type, Attribute)>;

  /// Attempt to materialize a conversion using one of the provided
  /// materialization functions.
  Value
  materializeConversion(ArrayRef<MaterializationCallbackFn> materializations,
                        OpBuilder &builder, Location loc, Type resultType,
                        ValueRange inputs) const;

  /// Generate a wrapper for the given callback. This allows for accepting
  /// different callback forms, that all compose into a single version.
  /// With callback of form: `std::optional<Type>(T)`
  template <typename T, typename FnT>
  std::enable_if_t<std::is_invocable_v<FnT, T>, ConversionCallbackFn>
  wrapCallback(FnT &&callback) const {
    return wrapCallback<T>([callback = std::forward<FnT>(callback)](
                               T type, SmallVectorImpl<Type> &results) {
      if (std::optional<Type> resultOpt = callback(type)) {
        bool wasSuccess = static_cast<bool>(*resultOpt);
        if (wasSuccess)
          results.push_back(*resultOpt);
        return std::optional<LogicalResult>(success(wasSuccess));
      }
      return std::optional<LogicalResult>();
    });
  }
  /// With callback of form: `std::optional<LogicalResult>(
  ///     T, SmallVectorImpl<Type> &, ArrayRef<Type>)`.
  template <typename T, typename FnT>
  std::enable_if_t<std::is_invocable_v<FnT, T, SmallVectorImpl<Type> &>,
                   ConversionCallbackFn>
  wrapCallback(FnT &&callback) const {
    return [callback = std::forward<FnT>(callback)](
               Type type,
               SmallVectorImpl<Type> &results) -> std::optional<LogicalResult> {
      T derivedType = dyn_cast<T>(type);
      if (!derivedType)
        return std::nullopt;
      return callback(derivedType, results);
    };
  }

  /// Register a type conversion.
  void registerConversion(ConversionCallbackFn callback) {
    conversions.emplace_back(std::move(callback));
    cachedDirectConversions.clear();
    cachedMultiConversions.clear();
  }

  /// Generate a wrapper for the given materialization callback. The callback
  /// may take any subclass of `Type` and the wrapper will check for the target
  /// type to be of the expected class before calling the callback.
  template <typename T, typename FnT>
  MaterializationCallbackFn wrapMaterialization(FnT &&callback) const {
    return [callback = std::forward<FnT>(callback)](
               OpBuilder &builder, Type resultType, ValueRange inputs,
               Location loc) -> std::optional<Value> {
      if (T derivedType = dyn_cast<T>(resultType))
        return callback(builder, derivedType, inputs, loc);
      return std::nullopt;
    };
  }

  /// Generate a wrapper for the given memory space conversion callback. The
  /// callback may take any subclass of `Attribute` and the wrapper will check
  /// for the target attribute to be of the expected class before calling the
  /// callback.
  template <typename T, typename A, typename FnT>
  TypeAttributeConversionCallbackFn
  wrapTypeAttributeConversion(FnT &&callback) const {
    return [callback = std::forward<FnT>(callback)](
               Type type, Attribute attr) -> AttributeConversionResult {
      if (T derivedType = dyn_cast<T>(type)) {
        if (A derivedAttr = dyn_cast_or_null<A>(attr))
          return callback(derivedType, derivedAttr);
      }
      return AttributeConversionResult::na();
    };
  }

  /// Register a memory space conversion, clearing caches.
  void
  registerTypeAttributeConversion(TypeAttributeConversionCallbackFn callback) {
    typeAttributeConversions.emplace_back(std::move(callback));
    // Clear type conversions in case a memory space is lingering inside.
    cachedDirectConversions.clear();
    cachedMultiConversions.clear();
  }

  /// The set of registered conversion functions.
  SmallVector<ConversionCallbackFn, 4> conversions;

  /// The list of registered materialization functions.
  SmallVector<MaterializationCallbackFn, 2> argumentMaterializations;
  SmallVector<MaterializationCallbackFn, 2> sourceMaterializations;
  SmallVector<MaterializationCallbackFn, 2> targetMaterializations;

  /// The list of registered type attribute conversion functions.
  SmallVector<TypeAttributeConversionCallbackFn, 2> typeAttributeConversions;

  /// A set of cached conversions to avoid recomputing in the common case.
  /// Direct 1-1 conversions are the most common, so this cache stores the
  /// successful 1-1 conversions as well as all failed conversions.
  mutable DenseMap<Type, Type> cachedDirectConversions;
  /// This cache stores the successful 1->N conversions, where N != 1.
  mutable DenseMap<Type, SmallVector<Type, 2>> cachedMultiConversions;
  /// A mutex used for cache access
  mutable llvm::sys::SmartRWMutex<true> cacheMutex;
};

//===----------------------------------------------------------------------===//
// Conversion Patterns
//===----------------------------------------------------------------------===//

/// Base class for the conversion patterns. This pattern class enables type
/// conversions, and other uses specific to the conversion framework. As such,
/// patterns of this type can only be used with the 'apply*' methods below.
class ConversionPattern : public RewritePattern {
public:
  /// Hook for derived classes to implement rewriting. `op` is the (first)
  /// operation matched by the pattern, `operands` is a list of the rewritten
  /// operand values that are passed to `op`, `rewriter` can be used to emit the
  /// new operations. This function should not fail. If some specific cases of
  /// the operation are not supported, these cases should not be matched.
  virtual void rewrite(Operation *op, ArrayRef<Value> operands,
                       ConversionPatternRewriter &rewriter) const {
    llvm_unreachable("unimplemented rewrite");
  }

  /// Hook for derived classes to implement combined matching and rewriting.
  virtual LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const {
    if (failed(match(op)))
      return failure();
    rewrite(op, operands, rewriter);
    return success();
  }

  /// Attempt to match and rewrite the IR root at the specified operation.
  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const final;

  /// Return the type converter held by this pattern, or nullptr if the pattern
  /// does not require type conversion.
  const TypeConverter *getTypeConverter() const { return typeConverter; }

  template <typename ConverterTy>
  std::enable_if_t<std::is_base_of<TypeConverter, ConverterTy>::value,
                   const ConverterTy *>
  getTypeConverter() const {
    return static_cast<const ConverterTy *>(typeConverter);
  }

protected:
  /// See `RewritePattern::RewritePattern` for information on the other
  /// available constructors.
  using RewritePattern::RewritePattern;
  /// Construct a conversion pattern with the given converter, and forward the
  /// remaining arguments to RewritePattern.
  template <typename... Args>
  ConversionPattern(const TypeConverter &typeConverter, Args &&...args)
      : RewritePattern(std::forward<Args>(args)...),
        typeConverter(&typeConverter) {}

protected:
  /// An optional type converter for use by this pattern.
  const TypeConverter *typeConverter = nullptr;

private:
  using RewritePattern::rewrite;
};

/// OpConversionPattern is a wrapper around ConversionPattern that allows for
/// matching and rewriting against an instance of a derived operation class as
/// opposed to a raw Operation.
template <typename SourceOp>
class OpConversionPattern : public ConversionPattern {
public:
  using OpAdaptor = typename SourceOp::Adaptor;

  OpConversionPattern(MLIRContext *context, PatternBenefit benefit = 1)
      : ConversionPattern(SourceOp::getOperationName(), benefit, context) {}
  OpConversionPattern(const TypeConverter &typeConverter, MLIRContext *context,
                      PatternBenefit benefit = 1)
      : ConversionPattern(typeConverter, SourceOp::getOperationName(), benefit,
                          context) {}

  /// Wrappers around the ConversionPattern methods that pass the derived op
  /// type.
  LogicalResult match(Operation *op) const final {
    return match(cast<SourceOp>(op));
  }
  void rewrite(Operation *op, ArrayRef<Value> operands,
               ConversionPatternRewriter &rewriter) const final {
    auto sourceOp = cast<SourceOp>(op);
    rewrite(sourceOp, OpAdaptor(operands, sourceOp), rewriter);
  }
  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const final {
    auto sourceOp = cast<SourceOp>(op);
    return matchAndRewrite(sourceOp, OpAdaptor(operands, sourceOp), rewriter);
  }

  /// Rewrite and Match methods that operate on the SourceOp type. These must be
  /// overridden by the derived pattern class.
  virtual LogicalResult match(SourceOp op) const {
    llvm_unreachable("must override match or matchAndRewrite");
  }
  virtual void rewrite(SourceOp op, OpAdaptor adaptor,
                       ConversionPatternRewriter &rewriter) const {
    llvm_unreachable("must override matchAndRewrite or a rewrite method");
  }
  virtual LogicalResult
  matchAndRewrite(SourceOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const {
    if (failed(match(op)))
      return failure();
    rewrite(op, adaptor, rewriter);
    return success();
  }

private:
  using ConversionPattern::matchAndRewrite;
};

/// OpInterfaceConversionPattern is a wrapper around ConversionPattern that
/// allows for matching and rewriting against an instance of an OpInterface
/// class as opposed to a raw Operation.
template <typename SourceOp>
class OpInterfaceConversionPattern : public ConversionPattern {
public:
  OpInterfaceConversionPattern(MLIRContext *context, PatternBenefit benefit = 1)
      : ConversionPattern(Pattern::MatchInterfaceOpTypeTag(),
                          SourceOp::getInterfaceID(), benefit, context) {}
  OpInterfaceConversionPattern(const TypeConverter &typeConverter,
                               MLIRContext *context, PatternBenefit benefit = 1)
      : ConversionPattern(typeConverter, Pattern::MatchInterfaceOpTypeTag(),
                          SourceOp::getInterfaceID(), benefit, context) {}

  /// Wrappers around the ConversionPattern methods that pass the derived op
  /// type.
  void rewrite(Operation *op, ArrayRef<Value> operands,
               ConversionPatternRewriter &rewriter) const final {
    rewrite(cast<SourceOp>(op), operands, rewriter);
  }
  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const final {
    return matchAndRewrite(cast<SourceOp>(op), operands, rewriter);
  }

  /// Rewrite and Match methods that operate on the SourceOp type. These must be
  /// overridden by the derived pattern class.
  virtual void rewrite(SourceOp op, ArrayRef<Value> operands,
                       ConversionPatternRewriter &rewriter) const {
    llvm_unreachable("must override matchAndRewrite or a rewrite method");
  }
  virtual LogicalResult
  matchAndRewrite(SourceOp op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const {
    if (failed(match(op)))
      return failure();
    rewrite(op, operands, rewriter);
    return success();
  }

private:
  using ConversionPattern::matchAndRewrite;
};

/// OpTraitConversionPattern is a wrapper around ConversionPattern that allows
/// for matching and rewriting against instances of an operation that possess a
/// given trait.
template <template <typename> class TraitType>
class OpTraitConversionPattern : public ConversionPattern {
public:
  OpTraitConversionPattern(MLIRContext *context, PatternBenefit benefit = 1)
      : ConversionPattern(Pattern::MatchTraitOpTypeTag(),
                          TypeID::get<TraitType>(), benefit, context) {}
  OpTraitConversionPattern(const TypeConverter &typeConverter,
                           MLIRContext *context, PatternBenefit benefit = 1)
      : ConversionPattern(typeConverter, Pattern::MatchTraitOpTypeTag(),
                          TypeID::get<TraitType>(), benefit, context) {}
};

/// Generic utility to convert op result types according to type converter
/// without knowing exact op type.
/// Clones existing op with new result types and returns it.
FailureOr<Operation *>
convertOpResultTypes(Operation *op, ValueRange operands,
                     const TypeConverter &converter,
                     ConversionPatternRewriter &rewriter);

/// Add a pattern to the given pattern list to convert the signature of a
/// FunctionOpInterface op with the given type converter. This only supports
/// ops which use FunctionType to represent their type.
void populateFunctionOpInterfaceTypeConversionPattern(
    StringRef functionLikeOpName, RewritePatternSet &patterns,
    const TypeConverter &converter);

template <typename FuncOpT>
void populateFunctionOpInterfaceTypeConversionPattern(
    RewritePatternSet &patterns, const TypeConverter &converter) {
  populateFunctionOpInterfaceTypeConversionPattern(FuncOpT::getOperationName(),
                                                   patterns, converter);
}

void populateAnyFunctionOpInterfaceTypeConversionPattern(
    RewritePatternSet &patterns, const TypeConverter &converter);

//===----------------------------------------------------------------------===//
// Conversion PatternRewriter
//===----------------------------------------------------------------------===//

namespace detail {
struct ConversionPatternRewriterImpl;
} // namespace detail

/// This class implements a pattern rewriter for use with ConversionPatterns. It
/// extends the base PatternRewriter and provides special conversion specific
/// hooks.
class ConversionPatternRewriter final : public PatternRewriter {
public:
  ~ConversionPatternRewriter() override;

  /// Apply a signature conversion to given block. This replaces the block with
  /// a new block containing the updated signature. The operations of the given
  /// block are inlined into the newly-created block, which is returned.
  ///
  /// If no block argument types are changing, the original block will be
  /// left in place and returned.
  ///
  /// A signature converison must be provided. (Type converters can construct
  /// a signature conversion with `convertBlockSignature`.)
  ///
  /// Optionally, a type converter can be provided to build materializations.
  /// Note: If no type converter was provided or the type converter does not
  /// specify any suitable argument/target materialization rules, the dialect
  /// conversion may fail to legalize unresolved materializations.
  Block *
  applySignatureConversion(Block *block,
                           TypeConverter::SignatureConversion &conversion,
                           const TypeConverter *converter = nullptr);

  /// Apply a signature conversion to each block in the given region. This
  /// replaces each block with a new block containing the updated signature. If
  /// an updated signature would match the current signature, the respective
  /// block is left in place as is. (See `applySignatureConversion` for
  /// details.) The new entry block of the region is returned.
  ///
  /// SignatureConversions are computed with the specified type converter.
  /// This function returns "failure" if the type converter failed to compute
  /// a SignatureConversion for at least one block.
  ///
  /// Optionally, a special SignatureConversion can be specified for the entry
  /// block. This is because the types of the entry block arguments are often
  /// tied semantically to the operation.
  FailureOr<Block *> convertRegionTypes(
      Region *region, const TypeConverter &converter,
      TypeConverter::SignatureConversion *entryConversion = nullptr);

  /// Replace all the uses of the block argument `from` with value `to`.
  void replaceUsesOfBlockArgument(BlockArgument from, Value to);

  /// Return the converted value of 'key' with a type defined by the type
  /// converter of the currently executing pattern. Return nullptr in the case
  /// of failure, the remapped value otherwise.
  Value getRemappedValue(Value key);

  /// Return the converted values that replace 'keys' with types defined by the
  /// type converter of the currently executing pattern. Returns failure if the
  /// remap failed, success otherwise.
  LogicalResult getRemappedValues(ValueRange keys,
                                  SmallVectorImpl<Value> &results);

  //===--------------------------------------------------------------------===//
  // PatternRewriter Hooks
  //===--------------------------------------------------------------------===//

  /// Indicate that the conversion rewriter can recover from rewrite failure.
  /// Recovery is supported via rollback, allowing for continued processing of
  /// patterns even if a failure is encountered during the rewrite step.
  bool canRecoverFromRewriteFailure() const override { return true; }

  /// PatternRewriter hook for replacing an operation.
  void replaceOp(Operation *op, ValueRange newValues) override;

  /// PatternRewriter hook for replacing an operation.
  void replaceOp(Operation *op, Operation *newOp) override;

  /// PatternRewriter hook for erasing a dead operation. The uses of this
  /// operation *must* be made dead by the end of the conversion process,
  /// otherwise an assert will be issued.
  void eraseOp(Operation *op) override;

  /// PatternRewriter hook for erase all operations in a block. This is not yet
  /// implemented for dialect conversion.
  void eraseBlock(Block *block) override;

  /// PatternRewriter hook for inlining the ops of a block into another block.
  void inlineBlockBefore(Block *source, Block *dest, Block::iterator before,
                         ValueRange argValues = std::nullopt) override;
  using PatternRewriter::inlineBlockBefore;

  /// PatternRewriter hook for updating the given operation in-place.
  /// Note: These methods only track updates to the given operation itself,
  /// and not nested regions. Updates to regions will still require notification
  /// through other more specific hooks above.
  void startOpModification(Operation *op) override;

  /// PatternRewriter hook for updating the given operation in-place.
  void finalizeOpModification(Operation *op) override;

  /// PatternRewriter hook for updating the given operation in-place.
  void cancelOpModification(Operation *op) override;

  /// Return a reference to the internal implementation.
  detail::ConversionPatternRewriterImpl &getImpl();

private:
  // Allow OperationConverter to construct new rewriters.
  friend struct OperationConverter;

  /// Conversion pattern rewriters must not be used outside of dialect
  /// conversions. They apply some IR rewrites in a delayed fashion and could
  /// bring the IR into an inconsistent state when used standalone.
  explicit ConversionPatternRewriter(MLIRContext *ctx,
                                     const ConversionConfig &config);

  // Hide unsupported pattern rewriter API.
  using OpBuilder::setListener;

  std::unique_ptr<detail::ConversionPatternRewriterImpl> impl;
};

//===----------------------------------------------------------------------===//
// ConversionTarget
//===----------------------------------------------------------------------===//

/// This class describes a specific conversion target.
class ConversionTarget {
public:
  /// This enumeration corresponds to the specific action to take when
  /// considering an operation legal for this conversion target.
  enum class LegalizationAction {
    /// The target supports this operation.
    Legal,

    /// This operation has dynamic legalization constraints that must be checked
    /// by the target.
    Dynamic,

    /// The target explicitly does not support this operation.
    Illegal,
  };

  /// A structure containing additional information describing a specific legal
  /// operation instance.
  struct LegalOpDetails {
    /// A flag that indicates if this operation is 'recursively' legal. This
    /// means that if an operation is legal, either statically or dynamically,
    /// all of the operations nested within are also considered legal.
    bool isRecursivelyLegal = false;
  };

  /// The signature of the callback used to determine if an operation is
  /// dynamically legal on the target.
  using DynamicLegalityCallbackFn =
      std::function<std::optional<bool>(Operation *)>;

  ConversionTarget(MLIRContext &ctx) : ctx(ctx) {}
  virtual ~ConversionTarget() = default;

  //===--------------------------------------------------------------------===//
  // Legality Registration
  //===--------------------------------------------------------------------===//

  /// Register a legality action for the given operation.
  void setOpAction(OperationName op, LegalizationAction action);
  template <typename OpT>
  void setOpAction(LegalizationAction action) {
    setOpAction(OperationName(OpT::getOperationName(), &ctx), action);
  }

  /// Register the given operations as legal.
  void addLegalOp(OperationName op) {
    setOpAction(op, LegalizationAction::Legal);
  }
  template <typename OpT>
  void addLegalOp() {
    addLegalOp(OperationName(OpT::getOperationName(), &ctx));
  }
  template <typename OpT, typename OpT2, typename... OpTs>
  void addLegalOp() {
    addLegalOp<OpT>();
    addLegalOp<OpT2, OpTs...>();
  }

  /// Register the given operation as dynamically legal and set the dynamic
  /// legalization callback to the one provided.
  void addDynamicallyLegalOp(OperationName op,
                             const DynamicLegalityCallbackFn &callback) {
    setOpAction(op, LegalizationAction::Dynamic);
    setLegalityCallback(op, callback);
  }
  template <typename OpT>
  void addDynamicallyLegalOp(const DynamicLegalityCallbackFn &callback) {
    addDynamicallyLegalOp(OperationName(OpT::getOperationName(), &ctx),
                          callback);
  }
  template <typename OpT, typename OpT2, typename... OpTs>
  void addDynamicallyLegalOp(const DynamicLegalityCallbackFn &callback) {
    addDynamicallyLegalOp<OpT>(callback);
    addDynamicallyLegalOp<OpT2, OpTs...>(callback);
  }
  template <typename OpT, class Callable>
  std::enable_if_t<!std::is_invocable_v<Callable, Operation *>>
  addDynamicallyLegalOp(Callable &&callback) {
    addDynamicallyLegalOp<OpT>(
        [=](Operation *op) { return callback(cast<OpT>(op)); });
  }

  /// Register the given operation as illegal, i.e. this operation is known to
  /// not be supported by this target.
  void addIllegalOp(OperationName op) {
    setOpAction(op, LegalizationAction::Illegal);
  }
  template <typename OpT>
  void addIllegalOp() {
    addIllegalOp(OperationName(OpT::getOperationName(), &ctx));
  }
  template <typename OpT, typename OpT2, typename... OpTs>
  void addIllegalOp() {
    addIllegalOp<OpT>();
    addIllegalOp<OpT2, OpTs...>();
  }

  /// Mark an operation, that *must* have either been set as `Legal` or
  /// `DynamicallyLegal`, as being recursively legal. This means that in
  /// addition to the operation itself, all of the operations nested within are
  /// also considered legal. An optional dynamic legality callback may be
  /// provided to mark subsets of legal instances as recursively legal.
  void markOpRecursivelyLegal(OperationName name,
                              const DynamicLegalityCallbackFn &callback);
  template <typename OpT>
  void markOpRecursivelyLegal(const DynamicLegalityCallbackFn &callback = {}) {
    OperationName opName(OpT::getOperationName(), &ctx);
    markOpRecursivelyLegal(opName, callback);
  }
  template <typename OpT, typename OpT2, typename... OpTs>
  void markOpRecursivelyLegal(const DynamicLegalityCallbackFn &callback = {}) {
    markOpRecursivelyLegal<OpT>(callback);
    markOpRecursivelyLegal<OpT2, OpTs...>(callback);
  }
  template <typename OpT, class Callable>
  std::enable_if_t<!std::is_invocable_v<Callable, Operation *>>
  markOpRecursivelyLegal(Callable &&callback) {
    markOpRecursivelyLegal<OpT>(
        [=](Operation *op) { return callback(cast<OpT>(op)); });
  }

  /// Register a legality action for the given dialects.
  void setDialectAction(ArrayRef<StringRef> dialectNames,
                        LegalizationAction action);

  /// Register the operations of the given dialects as legal.
  template <typename... Names>
  void addLegalDialect(StringRef name, Names... names) {
    SmallVector<StringRef, 2> dialectNames({name, names...});
    setDialectAction(dialectNames, LegalizationAction::Legal);
  }
  template <typename... Args>
  void addLegalDialect() {
    SmallVector<StringRef, 2> dialectNames({Args::getDialectNamespace()...});
    setDialectAction(dialectNames, LegalizationAction::Legal);
  }

  /// Register the operations of the given dialects as dynamically legal, i.e.
  /// requiring custom handling by the callback.
  template <typename... Names>
  void addDynamicallyLegalDialect(const DynamicLegalityCallbackFn &callback,
                                  StringRef name, Names... names) {
    SmallVector<StringRef, 2> dialectNames({name, names...});
    setDialectAction(dialectNames, LegalizationAction::Dynamic);
    setLegalityCallback(dialectNames, callback);
  }
  template <typename... Args>
  void addDynamicallyLegalDialect(DynamicLegalityCallbackFn callback) {
    addDynamicallyLegalDialect(std::move(callback),
                               Args::getDialectNamespace()...);
  }

  /// Register unknown operations as dynamically legal. For operations(and
  /// dialects) that do not have a set legalization action, treat them as
  /// dynamically legal and invoke the given callback.
  void markUnknownOpDynamicallyLegal(const DynamicLegalityCallbackFn &fn) {
    setLegalityCallback(fn);
  }

  /// Register the operations of the given dialects as illegal, i.e.
  /// operations of this dialect are not supported by the target.
  template <typename... Names>
  void addIllegalDialect(StringRef name, Names... names) {
    SmallVector<StringRef, 2> dialectNames({name, names...});
    setDialectAction(dialectNames, LegalizationAction::Illegal);
  }
  template <typename... Args>
  void addIllegalDialect() {
    SmallVector<StringRef, 2> dialectNames({Args::getDialectNamespace()...});
    setDialectAction(dialectNames, LegalizationAction::Illegal);
  }

  //===--------------------------------------------------------------------===//
  // Legality Querying
  //===--------------------------------------------------------------------===//

  /// Get the legality action for the given operation.
  std::optional<LegalizationAction> getOpAction(OperationName op) const;

  /// If the given operation instance is legal on this target, a structure
  /// containing legality information is returned. If the operation is not
  /// legal, std::nullopt is returned. Also returns std::nullopt if operation
  /// legality wasn't registered by user or dynamic legality callbacks returned
  /// None.
  ///
  /// Note: Legality is actually a 4-state: Legal(recursive=true),
  /// Legal(recursive=false), Illegal or Unknown, where Unknown is treated
  /// either as Legal or Illegal depending on context.
  std::optional<LegalOpDetails> isLegal(Operation *op) const;

  /// Returns true is operation instance is illegal on this target. Returns
  /// false if operation is legal, operation legality wasn't registered by user
  /// or dynamic legality callbacks returned None.
  bool isIllegal(Operation *op) const;

private:
  /// Set the dynamic legality callback for the given operation.
  void setLegalityCallback(OperationName name,
                           const DynamicLegalityCallbackFn &callback);

  /// Set the dynamic legality callback for the given dialects.
  void setLegalityCallback(ArrayRef<StringRef> dialects,
                           const DynamicLegalityCallbackFn &callback);

  /// Set the dynamic legality callback for the unknown ops.
  void setLegalityCallback(const DynamicLegalityCallbackFn &callback);

  /// The set of information that configures the legalization of an operation.
  struct LegalizationInfo {
    /// The legality action this operation was given.
    LegalizationAction action = LegalizationAction::Illegal;

    /// If some legal instances of this operation may also be recursively legal.
    bool isRecursivelyLegal = false;

    /// The legality callback if this operation is dynamically legal.
    DynamicLegalityCallbackFn legalityFn;
  };

  /// Get the legalization information for the given operation.
  std::optional<LegalizationInfo> getOpInfo(OperationName op) const;

  /// A deterministic mapping of operation name and its respective legality
  /// information.
  llvm::MapVector<OperationName, LegalizationInfo> legalOperations;

  /// A set of legality callbacks for given operation names that are used to
  /// check if an operation instance is recursively legal.
  DenseMap<OperationName, DynamicLegalityCallbackFn> opRecursiveLegalityFns;

  /// A deterministic mapping of dialect name to the specific legality action to
  /// take.
  llvm::StringMap<LegalizationAction> legalDialects;

  /// A set of dynamic legality callbacks for given dialect names.
  llvm::StringMap<DynamicLegalityCallbackFn> dialectLegalityFns;

  /// An optional legality callback for unknown operations.
  DynamicLegalityCallbackFn unknownLegalityFn;

  /// The current context this target applies to.
  MLIRContext &ctx;
};

#if MLIR_ENABLE_PDL_IN_PATTERNMATCH
//===----------------------------------------------------------------------===//
// PDL Configuration
//===----------------------------------------------------------------------===//

/// A PDL configuration that is used to supported dialect conversion
/// functionality.
class PDLConversionConfig final
    : public PDLPatternConfigBase<PDLConversionConfig> {
public:
  PDLConversionConfig(const TypeConverter *converter) : converter(converter) {}
  ~PDLConversionConfig() final = default;

  /// Return the type converter used by this configuration, which may be nullptr
  /// if no type conversions are expected.
  const TypeConverter *getTypeConverter() const { return converter; }

  /// Hooks that are invoked at the beginning and end of a rewrite of a matched
  /// pattern.
  void notifyRewriteBegin(PatternRewriter &rewriter) final;
  void notifyRewriteEnd(PatternRewriter &rewriter) final;

private:
  /// An optional type converter to use for the pattern.
  const TypeConverter *converter;
};

/// Register the dialect conversion PDL functions with the given pattern set.
void registerConversionPDLFunctions(RewritePatternSet &patterns);

#else

// Stubs for when PDL in rewriting is not enabled.

inline void registerConversionPDLFunctions(RewritePatternSet &patterns) {}

class PDLConversionConfig final {
public:
  PDLConversionConfig(const TypeConverter * /*converter*/) {}
};

#endif // MLIR_ENABLE_PDL_IN_PATTERNMATCH

//===----------------------------------------------------------------------===//
// ConversionConfig
//===----------------------------------------------------------------------===//

/// Dialect conversion configuration.
struct ConversionConfig {
  /// An optional callback used to notify about match failure diagnostics during
  /// the conversion. Diagnostics reported to this callback may only be
  /// available in debug mode.
  function_ref<void(Diagnostic &)> notifyCallback = nullptr;

  /// Partial conversion only. All operations that are found not to be
  /// legalizable are placed in this set. (Note that if there is an op
  /// explicitly marked as illegal, the conversion terminates and the set will
  /// not necessarily be complete.)
  DenseSet<Operation *> *unlegalizedOps = nullptr;

  /// Analysis conversion only. All operations that are found to be legalizable
  /// are placed in this set. Note that no actual rewrites are applied to the
  /// IR during an analysis conversion and only pre-existing operations are
  /// added to the set.
  DenseSet<Operation *> *legalizableOps = nullptr;

  /// An optional listener that is notified about all IR modifications in case
  /// dialect conversion succeeds. If the dialect conversion fails and no IR
  /// modifications are visible (i.e., they were all rolled back), or if the
  /// dialect conversion is an "analysis conversion", no notifications are
  /// sent (apart from `notifyPatternBegin`/notifyPatternEnd`).
  ///
  /// Note: Notifications are sent in a delayed fashion, when the dialect
  /// conversion is guaranteed to succeed. At that point, some IR modifications
  /// may already have been materialized. Consequently, operations/blocks that
  /// are passed to listener callbacks should not be accessed. (Ops/blocks are
  /// guaranteed to be valid pointers and accessing op names is allowed. But
  /// there are no guarantees about the state of ops/blocks at the time that a
  /// callback is triggered.)
  ///
  /// Example: Consider a dialect conversion a new op ("test.foo") is created
  /// and inserted, and later moved to another block. (Moving ops also triggers
  /// "notifyOperationInserted".)
  ///
  /// (1) notifyOperationInserted: "test.foo" (into block "b1")
  /// (2) notifyOperationInserted: "test.foo" (moved to another block "b2")
  ///
  /// When querying "op->getBlock()" during the first "notifyOperationInserted",
  /// "b2" would be returned because "moving an op" is a kind of rewrite that is
  /// immediately performed by the dialect conversion (and rolled back upon
  /// failure).
  //
  // Note: When receiving a "notifyBlockInserted"/"notifyOperationInserted"
  // callback, the previous region/block is provided to the callback, but not
  // the iterator pointing to the exact location within the region/block. That
  // is because these notifications are sent with a delay (after the IR has
  // already been modified) and iterators into past IR state cannot be
  // represented at the moment.
  RewriterBase::Listener *listener = nullptr;

  /// If set to "true", the dialect conversion attempts to build source/target/
  /// argument materializations through the type converter API in lieu of
  /// "builtin.unrealized_conversion_cast ops". The conversion process fails if
  /// at least one materialization could not be built.
  ///
  /// If set to "false", the dialect conversion does not build any custom
  /// materializations and instead inserts "builtin.unrealized_conversion_cast"
  /// ops to ensure that the resulting IR is valid.
  bool buildMaterializations = true;
};

//===----------------------------------------------------------------------===//
// Reconcile Unrealized Casts
//===----------------------------------------------------------------------===//

/// Try to reconcile all given UnrealizedConversionCastOps and store the
/// left-over ops in `remainingCastOps` (if provided).
///
/// This function processes cast ops in a worklist-driven fashion. For each
/// cast op, if the chain of input casts eventually reaches a cast op where the
/// input types match the output types of the matched op, replace the matched
/// op with the inputs.
///
/// Example:
/// %1 = unrealized_conversion_cast %0 : !A to !B
/// %2 = unrealized_conversion_cast %1 : !B to !C
/// %3 = unrealized_conversion_cast %2 : !C to !A
///
/// In the above example, %0 can be used instead of %3 and all cast ops are
/// folded away.
void reconcileUnrealizedCasts(
    ArrayRef<UnrealizedConversionCastOp> castOps,
    SmallVectorImpl<UnrealizedConversionCastOp> *remainingCastOps = nullptr);

//===----------------------------------------------------------------------===//
// Op Conversion Entry Points
//===----------------------------------------------------------------------===//

/// Below we define several entry points for operation conversion. It is
/// important to note that the patterns provided to the conversion framework may
/// have additional constraints. See the `PatternRewriter Hooks` section of the
/// ConversionPatternRewriter, to see what additional constraints are imposed on
/// the use of the PatternRewriter.

/// Apply a partial conversion on the given operations and all nested
/// operations. This method converts as many operations to the target as
/// possible, ignoring operations that failed to legalize. This method only
/// returns failure if there ops explicitly marked as illegal.
LogicalResult
applyPartialConversion(ArrayRef<Operation *> ops,
                       const ConversionTarget &target,
                       const FrozenRewritePatternSet &patterns,
                       ConversionConfig config = ConversionConfig());
LogicalResult
applyPartialConversion(Operation *op, const ConversionTarget &target,
                       const FrozenRewritePatternSet &patterns,
                       ConversionConfig config = ConversionConfig());

/// Apply a complete conversion on the given operations, and all nested
/// operations. This method returns failure if the conversion of any operation
/// fails, or if there are unreachable blocks in any of the regions nested
/// within 'ops'.
LogicalResult applyFullConversion(ArrayRef<Operation *> ops,
                                  const ConversionTarget &target,
                                  const FrozenRewritePatternSet &patterns,
                                  ConversionConfig config = ConversionConfig());
LogicalResult applyFullConversion(Operation *op, const ConversionTarget &target,
                                  const FrozenRewritePatternSet &patterns,
                                  ConversionConfig config = ConversionConfig());

/// Apply an analysis conversion on the given operations, and all nested
/// operations. This method analyzes which operations would be successfully
/// converted to the target if a conversion was applied. All operations that
/// were found to be legalizable to the given 'target' are placed within the
/// provided 'config.legalizableOps' set; note that no actual rewrites are
/// applied to the operations on success. This method only returns failure if
/// there are unreachable blocks in any of the regions nested within 'ops'.
LogicalResult
applyAnalysisConversion(ArrayRef<Operation *> ops, ConversionTarget &target,
                        const FrozenRewritePatternSet &patterns,
                        ConversionConfig config = ConversionConfig());
LogicalResult
applyAnalysisConversion(Operation *op, ConversionTarget &target,
                        const FrozenRewritePatternSet &patterns,
                        ConversionConfig config = ConversionConfig());
} // namespace mlir

#endif // MLIR_TRANSFORMS_DIALECTCONVERSION_H_
