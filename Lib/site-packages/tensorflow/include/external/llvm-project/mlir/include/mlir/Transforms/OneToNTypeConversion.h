//===-- OneToNTypeConversion.h - Utils for 1:N type conversion --*- C++ -*-===//
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file provides utils for implementing (poor-man's) dialect conversion
// passes with 1:N type conversions.
//
// The main function, `applyPartialOneToNConversion`, first applies a set of
// `RewritePattern`s, which produce unrealized casts to convert the operands and
// results from and to the source types, and then replaces all newly added
// unrealized casts by user-provided materializations. For this to work, the
// main function requires a special `TypeConverter`, a special
// `PatternRewriter`, and special RewritePattern`s, which extend their
// respective base classes for 1:N type converions.
//
// Note that this is much more simple-minded than the "real" dialect conversion,
// which checks for legality before applying patterns and does probably many
// other additional things. Ideally, some of the extensions here could be
// integrated there.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_TRANSFORMS_ONETONTYPECONVERSION_H
#define MLIR_TRANSFORMS_ONETONTYPECONVERSION_H

#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir {

/// Extends `TypeConverter` with 1:N target materializations. Such
/// materializations have to provide the "reverse" of 1:N type conversions,
/// i.e., they need to materialize N values with target types into one value
/// with a source type (which isn't possible in the base class currently).
class OneToNTypeConverter : public TypeConverter {
public:
  /// Callback that expresses user-provided materialization logic from the given
  /// value to N values of the given types. This is useful for expressing target
  /// materializations for 1:N type conversions, which materialize one value in
  /// a source type as N values in target types.
  using OneToNMaterializationCallbackFn =
      std::function<std::optional<SmallVector<Value>>(OpBuilder &, TypeRange,
                                                      Value, Location)>;

  /// Creates the mapping of the given range of original types to target types
  /// of the conversion and stores that mapping in the given (signature)
  /// conversion. This function simply calls
  /// `TypeConverter::convertSignatureArgs` and exists here with a different
  /// name to reflect the broader semantic.
  LogicalResult computeTypeMapping(TypeRange types,
                                   SignatureConversion &result) {
    return convertSignatureArgs(types, result);
  }

  /// Applies one of the user-provided 1:N target materializations. If several
  /// exists, they are tried out in the reverse order in which they have been
  /// added until the first one succeeds. If none succeeds, the functions
  /// returns `std::nullopt`.
  std::optional<SmallVector<Value>>
  materializeTargetConversion(OpBuilder &builder, Location loc,
                              TypeRange resultTypes, Value input) const;

  /// Adds a 1:N target materialization to the converter. Such materializations
  /// build IR that converts N values with target types into 1 value of the
  /// source type.
  void addTargetMaterialization(OneToNMaterializationCallbackFn &&callback) {
    oneToNTargetMaterializations.emplace_back(std::move(callback));
  }

private:
  SmallVector<OneToNMaterializationCallbackFn> oneToNTargetMaterializations;
};

/// Stores a 1:N mapping of types and provides several useful accessors. This
/// class extends `SignatureConversion`, which already supports 1:N type
/// mappings but lacks some accessors into the mapping as well as access to the
/// original types.
class OneToNTypeMapping : public TypeConverter::SignatureConversion {
public:
  OneToNTypeMapping(TypeRange originalTypes)
      : TypeConverter::SignatureConversion(originalTypes.size()),
        originalTypes(originalTypes) {}

  using TypeConverter::SignatureConversion::getConvertedTypes;

  /// Returns the list of types that corresponds to the original type at the
  /// given index.
  TypeRange getConvertedTypes(unsigned originalTypeNo) const;

  /// Returns the list of original types.
  TypeRange getOriginalTypes() const { return originalTypes; }

  /// Returns the slice of converted values that corresponds the original value
  /// at the given index.
  ValueRange getConvertedValues(ValueRange convertedValues,
                                unsigned originalValueNo) const;

  /// Fills the given result vector with as many copies of the location of the
  /// original value as the number of values it is converted to.
  void convertLocation(Value originalValue, unsigned originalValueNo,
                       llvm::SmallVectorImpl<Location> &result) const;

  /// Fills the given result vector with as many copies of the lociation of each
  /// original value as the number of values they are respectively converted to.
  void convertLocations(ValueRange originalValues,
                        llvm::SmallVectorImpl<Location> &result) const;

  /// Returns true iff at least one type conversion maps an input type to a type
  /// that is different from itself.
  bool hasNonIdentityConversion() const;

private:
  llvm::SmallVector<Type> originalTypes;
};

/// Extends the basic `RewritePattern` class with a type converter member and
/// some accessors to it. This is useful for patterns that are not
/// `ConversionPattern`s but still require access to a type converter.
class RewritePatternWithConverter : public mlir::RewritePattern {
public:
  /// Construct a conversion pattern with the given converter, and forward the
  /// remaining arguments to RewritePattern.
  template <typename... Args>
  RewritePatternWithConverter(TypeConverter &typeConverter, Args &&...args)
      : RewritePattern(std::forward<Args>(args)...),
        typeConverter(&typeConverter) {}

  /// Return the type converter held by this pattern, or nullptr if the pattern
  /// does not require type conversion.
  TypeConverter *getTypeConverter() const { return typeConverter; }

  template <typename ConverterTy>
  std::enable_if_t<std::is_base_of<TypeConverter, ConverterTy>::value,
                   ConverterTy *>
  getTypeConverter() const {
    return static_cast<ConverterTy *>(typeConverter);
  }

protected:
  /// A type converter for use by this pattern.
  TypeConverter *const typeConverter;
};

/// Specialization of `PatternRewriter` that `OneToNConversionPattern`s use. The
/// class provides additional rewrite methods that are specific to 1:N type
/// conversions.
class OneToNPatternRewriter : public PatternRewriter {
public:
  OneToNPatternRewriter(MLIRContext *context,
                        OpBuilder::Listener *listener = nullptr)
      : PatternRewriter(context, listener) {}

  /// Replaces the results of the operation with the specified list of values
  /// mapped back to the original types as specified in the provided type
  /// mapping. That type mapping must match the replaced op (i.e., the original
  /// types must be the same as the result types of the op) and the new values
  /// (i.e., the converted types must be the same as the types of the new
  /// values).
  void replaceOp(Operation *op, ValueRange newValues,
                 const OneToNTypeMapping &resultMapping);
  using PatternRewriter::replaceOp;

  /// Applies the given argument conversion to the given block. This consists of
  /// replacing each original argument with N arguments as specified in the
  /// argument conversion and inserting unrealized casts from the converted
  /// values to the original types, which are then used in lieu of the original
  /// ones. (Eventually, `applyPartialOneToNConversion` replaces these casts
  /// with a user-provided argument materialization if necessary.) This is
  /// similar to `ArgConverter::applySignatureConversion` but (1) handles 1:N
  /// type conversion properly and probably (2) doesn't handle many other edge
  /// cases.
  Block *applySignatureConversion(Block *block,
                                  OneToNTypeMapping &argumentConversion);
};

/// Base class for patterns with 1:N type conversions. Derived classes have to
/// overwrite the `matchAndRewrite` overlaod that provides additional
/// information for 1:N type conversions.
class OneToNConversionPattern : public RewritePatternWithConverter {
public:
  using RewritePatternWithConverter::RewritePatternWithConverter;

  /// This function has to be implemented by derived classes and is called from
  /// the usual overloads. Like in "normal" `DialectConversion`, the function is
  /// provided with the converted operands (which thus have target types). Since
  /// 1:N conversions are supported, there is usually no 1:1 relationship
  /// between the original and the converted operands. Instead, the provided
  /// `operandMapping` can be used to access the converted operands that
  /// correspond to a particular original operand. Similarly, `resultMapping`
  /// is provided to help with assembling the result values, which may have 1:N
  /// correspondences as well. In that case, the original op should be replaced
  /// with the overload of `replaceOp` that takes the provided `resultMapping`
  /// in order to deal with the mapping of converted result values to their
  /// usages in the original types correctly.
  virtual LogicalResult matchAndRewrite(Operation *op,
                                        OneToNPatternRewriter &rewriter,
                                        const OneToNTypeMapping &operandMapping,
                                        const OneToNTypeMapping &resultMapping,
                                        ValueRange convertedOperands) const = 0;

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const final;
};

/// This class is a wrapper around `OneToNConversionPattern` for matching
/// against instances of a particular op class.
template <typename SourceOp>
class OneToNOpConversionPattern : public OneToNConversionPattern {
public:
  OneToNOpConversionPattern(TypeConverter &typeConverter, MLIRContext *context,
                            PatternBenefit benefit = 1,
                            ArrayRef<StringRef> generatedNames = {})
      : OneToNConversionPattern(typeConverter, SourceOp::getOperationName(),
                                benefit, context, generatedNames) {}
  /// Generic adaptor around the root op of this pattern using the converted
  /// operands. Importantly, each operand is represented as a *range* of values,
  /// namely the N values each original operand gets converted to. Concretely,
  /// this makes the result type of the accessor functions of the adaptor class
  /// be a `ValueRange`.
  class OpAdaptor
      : public SourceOp::template GenericAdaptor<ArrayRef<ValueRange>> {
  public:
    using RangeT = ArrayRef<ValueRange>;
    using BaseT = typename SourceOp::template GenericAdaptor<RangeT>;
    using Properties = typename SourceOp::template InferredProperties<SourceOp>;

    OpAdaptor(const OneToNTypeMapping *operandMapping,
              const OneToNTypeMapping *resultMapping,
              const ValueRange *convertedOperands, RangeT values, SourceOp op)
        : BaseT(values, op), operandMapping(operandMapping),
          resultMapping(resultMapping), convertedOperands(convertedOperands) {}

    /// Get the type mapping of the original operands to the converted operands.
    const OneToNTypeMapping &getOperandMapping() const {
      return *operandMapping;
    }

    /// Get the type mapping of the original results to the converted results.
    const OneToNTypeMapping &getResultMapping() const { return *resultMapping; }

    /// Get a flat range of all converted operands. Unlike `getOperands`, which
    /// returns an `ArrayRef` with one `ValueRange` for each original operand,
    /// this function returns a `ValueRange` that contains all converted
    /// operands irrespectively of which operand they originated from.
    ValueRange getFlatOperands() const { return *convertedOperands; }

  private:
    const OneToNTypeMapping *operandMapping;
    const OneToNTypeMapping *resultMapping;
    const ValueRange *convertedOperands;
  };

  using OneToNConversionPattern::matchAndRewrite;

  /// Overload that derived classes have to override for their op type.
  virtual LogicalResult
  matchAndRewrite(SourceOp op, OpAdaptor adaptor,
                  OneToNPatternRewriter &rewriter) const = 0;

  LogicalResult matchAndRewrite(Operation *op, OneToNPatternRewriter &rewriter,
                                const OneToNTypeMapping &operandMapping,
                                const OneToNTypeMapping &resultMapping,
                                ValueRange convertedOperands) const final {
    // Wrap converted operands and type mappings into an adaptor.
    SmallVector<ValueRange> valueRanges;
    for (int64_t i = 0; i < op->getNumOperands(); i++) {
      auto values = operandMapping.getConvertedValues(convertedOperands, i);
      valueRanges.push_back(values);
    }
    OpAdaptor adaptor(&operandMapping, &resultMapping, &convertedOperands,
                      valueRanges, cast<SourceOp>(op));

    // Call overload implemented by the derived class.
    return matchAndRewrite(cast<SourceOp>(op), adaptor, rewriter);
  }
};

/// Applies the given set of patterns recursively on the given op and adds user
/// materializations where necessary. The patterns are expected to be
/// `OneToNConversionPattern`, which help converting the types of the operands
/// and results of the matched ops. The provided type converter is used to
/// convert the operands of matched ops from their original types to operands
/// with different types. Unlike in `DialectConversion`, this supports 1:N type
/// conversions. Those conversions at the "boundary" of the pattern application,
/// where converted results are not consumed by replaced ops that expect the
/// converted operands or vice versa, the function inserts user materializations
/// from the type converter. Also unlike `DialectConversion`, there are no legal
/// or illegal types; the function simply applies the given patterns and does
/// not fail if some ops or types remain unconverted (i.e., the conversion is
/// only "partial").
LogicalResult
applyPartialOneToNConversion(Operation *op, OneToNTypeConverter &typeConverter,
                             const FrozenRewritePatternSet &patterns);

/// Add a pattern to the given pattern list to convert the signature of a
/// FunctionOpInterface op with the given type converter. This only supports
/// ops which use FunctionType to represent their type. This is intended to be
/// used with the 1:N dialect conversion.
void populateOneToNFunctionOpInterfaceTypeConversionPattern(
    StringRef functionLikeOpName, TypeConverter &converter,
    RewritePatternSet &patterns);
template <typename FuncOpT>
void populateOneToNFunctionOpInterfaceTypeConversionPattern(
    TypeConverter &converter, RewritePatternSet &patterns) {
  populateOneToNFunctionOpInterfaceTypeConversionPattern(
      FuncOpT::getOperationName(), converter, patterns);
}

} // namespace mlir

#endif // MLIR_TRANSFORMS_ONETONTYPECONVERSION_H
