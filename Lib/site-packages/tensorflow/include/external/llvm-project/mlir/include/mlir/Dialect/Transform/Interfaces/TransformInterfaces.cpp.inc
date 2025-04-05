/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Interface Definitions                                                      *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

/// Populate conversion patterns into the given pattern set with the
/// given type converter.
void mlir::transform::ConversionPatternDescriptorOpInterface::populatePatterns(::mlir::TypeConverter & typeConverter, ::mlir::RewritePatternSet & patterns) {
      return getImpl()->populatePatterns(getImpl(), getOperation(), typeConverter, patterns);
  }
/// Populate the ConversionTarget using the final TypeConverter. The default
/// implementation is to do nothing. Overriding this method can be useful
/// in order to setup the ConversionTarget for structural type conversions.
/// In such a situation, an op's legality depends on using the TypeConverter
/// to determine whether the op's operand and result types are legal
/// (defined as converting to themselves).
/// 
void mlir::transform::ConversionPatternDescriptorOpInterface::populateConversionTargetRules(const ::mlir::TypeConverter & typeConverter, ::mlir::ConversionTarget & conversionTarget) {
      return getImpl()->populateConversionTargetRules(getImpl(), getOperation(), typeConverter, conversionTarget);
  }
/// Return the type converter to be used with this pattern set. If no
/// type converter is specified, the default type converter of the enclosing
/// "apply_conversion_patterns" op is used.
std::unique_ptr<::mlir::TypeConverter> mlir::transform::ConversionPatternDescriptorOpInterface::getTypeConverter() {
      return getImpl()->getTypeConverter(getImpl(), getOperation());
  }
/// Verify the default type converter that is provided by the enclosing
/// "apply_conversion_patterns" op.
::llvm::LogicalResult mlir::transform::ConversionPatternDescriptorOpInterface::verifyTypeConverter(TypeConverterBuilderOpInterface builder) {
      return getImpl()->verifyTypeConverter(getImpl(), getOperation(), builder);
  }
/// Return the operands at which the lookup for replacement payload ops
/// should continue.
::llvm::SmallVector<::mlir::Value> mlir::transform::FindPayloadReplacementOpInterface::getNextOperands() {
      return getImpl()->getNextOperands(getImpl(), getOperation());
  }
/// Populate rewrite patterns into the given pattern set.
void mlir::transform::PatternDescriptorOpInterface::populatePatterns(::mlir::RewritePatternSet & patterns) {
      return getImpl()->populatePatterns(getImpl(), getOperation(), patterns);
  }
/// Populate rewrite patterns into the given pattern set taking into account
/// the transform state.
void mlir::transform::PatternDescriptorOpInterface::populatePatternsWithState(::mlir::RewritePatternSet & patterns, ::mlir::transform::TransformState & state) {
      return getImpl()->populatePatternsWithState(getImpl(), getOperation(), patterns, state);
  }
/// Applies the transformation represented by the current operation. This
/// accepts as arguments the object that must be populated with results of
/// the current transformation and a transformation state object that can be
/// used for queries, e.g., to obtain the list of operations on which the
/// transformation represented by the current op is targeted. Returns a
/// special status object indicating whether the transformation succeeded
/// or failed, and, if it failed, whether the failure is recoverable or not.
/// 
/// IR must be created, modified and deleted with the provided rewriter.
/// implementations are responsible for setting the insertion point of the
/// rewriter to the desired location.
::mlir::DiagnosedSilenceableFailure mlir::transform::TransformOpInterface::apply(::mlir::transform::TransformRewriter & rewriter, ::mlir::transform::TransformResults & transformResults, ::mlir::transform::TransformState & state) {
      return getImpl()->apply(getImpl(), getOperation(), rewriter, transformResults, state);
  }
/// Indicates whether the op instance allows its handle operands to be
/// associated with the same payload operations.
bool mlir::transform::TransformOpInterface::allowsRepeatedHandleOperands() {
      return getImpl()->allowsRepeatedHandleOperands(getImpl(), getOperation());
  }
/// Return the type converter to be used with a dialect conversion.
std::unique_ptr<::mlir::TypeConverter> mlir::transform::TypeConverterBuilderOpInterface::getTypeConverter() {
      return getImpl()->getTypeConverter(getImpl(), getOperation());
  }
/// Return the type of type converter that this `getTypeConverter` returns.
/// This function is used for op verification.
StringRef mlir::transform::TypeConverterBuilderOpInterface::getTypeConverterType() {
      return getImpl()->getTypeConverterType();
  }
/// Populate the given type converter with source/target materialization
/// functions.
void mlir::transform::TypeConverterBuilderOpInterface::populateTypeMaterializations(::mlir::TypeConverter & converter) {
      return getImpl()->populateTypeMaterializations(getImpl(), getOperation(), converter);
  }
