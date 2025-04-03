/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Interface Declarations                                                     *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

namespace mlir {
namespace transform {
class TransformOpInterface;
namespace detail {
struct TransformOpInterfaceInterfaceTraits {
  struct Concept {
    /// The methods defined by the interface.
    ::mlir::DiagnosedSilenceableFailure (*apply)(const Concept *impl, ::mlir::Operation *, ::mlir::transform::TransformRewriter &, ::mlir::transform::TransformResults &, ::mlir::transform::TransformState &);
    bool (*allowsRepeatedHandleOperands)(const Concept *impl, ::mlir::Operation *);
  };
  template<typename ConcreteOp>
  class Model : public Concept {
  public:
    using Interface = ::mlir::transform::TransformOpInterface;
    Model() : Concept{apply, allowsRepeatedHandleOperands} {}

    static inline ::mlir::DiagnosedSilenceableFailure apply(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, ::mlir::transform::TransformRewriter & rewriter, ::mlir::transform::TransformResults & transformResults, ::mlir::transform::TransformState & state);
    static inline bool allowsRepeatedHandleOperands(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
  };
  template<typename ConcreteOp>
  class FallbackModel : public Concept {
  public:
    using Interface = ::mlir::transform::TransformOpInterface;
    FallbackModel() : Concept{apply, allowsRepeatedHandleOperands} {}

    static inline ::mlir::DiagnosedSilenceableFailure apply(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, ::mlir::transform::TransformRewriter & rewriter, ::mlir::transform::TransformResults & transformResults, ::mlir::transform::TransformState & state);
    static inline bool allowsRepeatedHandleOperands(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
  };
  template<typename ConcreteModel, typename ConcreteOp>
  class ExternalModel : public FallbackModel<ConcreteModel> {
  public:
    using ConcreteEntity = ConcreteOp;
    bool allowsRepeatedHandleOperands(::mlir::Operation *tablegen_opaque_val) const;
  };
};
template <typename ConcreteOp>
struct TransformOpInterfaceTrait;

} // namespace detail
class TransformOpInterface : public ::mlir::OpInterface<TransformOpInterface, detail::TransformOpInterfaceInterfaceTraits> {
public:
  using ::mlir::OpInterface<TransformOpInterface, detail::TransformOpInterfaceInterfaceTraits>::OpInterface;
  template <typename ConcreteOp>
  struct Trait : public detail::TransformOpInterfaceTrait<ConcreteOp> {};
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
  ::mlir::DiagnosedSilenceableFailure apply(::mlir::transform::TransformRewriter & rewriter, ::mlir::transform::TransformResults & transformResults, ::mlir::transform::TransformState & state);
  /// Indicates whether the op instance allows its handle operands to be
  /// associated with the same payload operations.
  bool allowsRepeatedHandleOperands();

    /// Creates the silenceable failure object with a diagnostic located at the
    /// current operation. Silenceable failure must be suppressed or reported
    /// explicitly at some later time.
    DiagnosedSilenceableFailure
    emitSilenceableError(const ::llvm::Twine &message = {}) {
      return ::mlir::emitSilenceableFailure((*this), message);
    }

    /// Creates the definite failure object with a diagnostic located at the
    /// current operation. Definite failure will be reported when the object
    /// is destroyed or converted.
    DiagnosedDefiniteFailure
    emitDefiniteFailure(const ::llvm::Twine &message = {}) {
      return ::mlir::emitDefiniteFailure((*this), message);
    }

    /// Emits a generic definite failure for the current transform operation
    /// targeting the given Payload IR operation and returns failure. Should
    /// be only used as a last resort when the transformation itself provides
    /// no further indication as to the reason of the failure.
    DiagnosedDefiniteFailure emitDefaultDefiniteFailure(
        ::mlir::Operation *target) {
      auto diag = ::mlir::emitDefiniteFailure((*this), "failed to apply");
      diag.attachNote(target->getLoc()) << "attempted to apply to this op";
      return diag;
    }

    /// Creates the default silenceable failure for a transform op that failed
    /// to properly apply to a target.
    DiagnosedSilenceableFailure emitDefaultSilenceableFailure(
        Operation *target) {
      DiagnosedSilenceableFailure diag = emitSilenceableFailure((*this)->getLoc());
      diag << (*this)->getName() << " failed to apply";
      diag.attachNote(target->getLoc()) << "when applied to this op";
      return diag;
    }

    /// Returns all operands that are handles and being consumed by this op.
    ::llvm::SmallVector<OpOperand *> getConsumedHandleOpOperands() {
      return ::mlir::transform::detail::getConsumedHandleOpOperands((*this));
    }
};
namespace detail {
  template <typename ConcreteOp>
  struct TransformOpInterfaceTrait : public ::mlir::OpInterface<TransformOpInterface, detail::TransformOpInterfaceInterfaceTraits>::Trait<ConcreteOp> {
    /// Indicates whether the op instance allows its handle operands to be
    /// associated with the same payload operations.
    bool allowsRepeatedHandleOperands() {
      return false;
    }
    static ::llvm::LogicalResult verifyTrait(::mlir::Operation *op) {
      return ::mlir::transform::detail::verifyTransformOpInterface(op);
    }

    /// Creates the silenceable failure object with a diagnostic located at the
    /// current operation. Silenceable failure must be suppressed or reported
    /// explicitly at some later time.
    DiagnosedSilenceableFailure
    emitSilenceableError(const ::llvm::Twine &message = {}) {
      return ::mlir::emitSilenceableFailure((*static_cast<ConcreteOp *>(this)), message);
    }

    /// Creates the definite failure object with a diagnostic located at the
    /// current operation. Definite failure will be reported when the object
    /// is destroyed or converted.
    DiagnosedDefiniteFailure
    emitDefiniteFailure(const ::llvm::Twine &message = {}) {
      return ::mlir::emitDefiniteFailure((*static_cast<ConcreteOp *>(this)), message);
    }

    /// Emits a generic definite failure for the current transform operation
    /// targeting the given Payload IR operation and returns failure. Should
    /// be only used as a last resort when the transformation itself provides
    /// no further indication as to the reason of the failure.
    DiagnosedDefiniteFailure emitDefaultDefiniteFailure(
        ::mlir::Operation *target) {
      auto diag = ::mlir::emitDefiniteFailure((*static_cast<ConcreteOp *>(this)), "failed to apply");
      diag.attachNote(target->getLoc()) << "attempted to apply to this op";
      return diag;
    }

    /// Creates the default silenceable failure for a transform op that failed
    /// to properly apply to a target.
    DiagnosedSilenceableFailure emitDefaultSilenceableFailure(
        Operation *target) {
      DiagnosedSilenceableFailure diag = emitSilenceableFailure((*static_cast<ConcreteOp *>(this))->getLoc());
      diag << (*static_cast<ConcreteOp *>(this))->getName() << " failed to apply";
      diag.attachNote(target->getLoc()) << "when applied to this op";
      return diag;
    }

    /// Returns all operands that are handles and being consumed by this op.
    ::llvm::SmallVector<OpOperand *> getConsumedHandleOpOperands() {
      return ::mlir::transform::detail::getConsumedHandleOpOperands((*static_cast<ConcreteOp *>(this)));
    }
  
  };
}// namespace detail
} // namespace transform
} // namespace mlir
namespace mlir {
namespace transform {
class FindPayloadReplacementOpInterface;
namespace detail {
struct FindPayloadReplacementOpInterfaceInterfaceTraits {
  struct Concept {
    /// The methods defined by the interface.
    ::llvm::SmallVector<::mlir::Value> (*getNextOperands)(const Concept *impl, ::mlir::Operation *);
  };
  template<typename ConcreteOp>
  class Model : public Concept {
  public:
    using Interface = ::mlir::transform::FindPayloadReplacementOpInterface;
    Model() : Concept{getNextOperands} {}

    static inline ::llvm::SmallVector<::mlir::Value> getNextOperands(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
  };
  template<typename ConcreteOp>
  class FallbackModel : public Concept {
  public:
    using Interface = ::mlir::transform::FindPayloadReplacementOpInterface;
    FallbackModel() : Concept{getNextOperands} {}

    static inline ::llvm::SmallVector<::mlir::Value> getNextOperands(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
  };
  template<typename ConcreteModel, typename ConcreteOp>
  class ExternalModel : public FallbackModel<ConcreteModel> {
  public:
    using ConcreteEntity = ConcreteOp;
  };
};
template <typename ConcreteOp>
struct FindPayloadReplacementOpInterfaceTrait;

} // namespace detail
class FindPayloadReplacementOpInterface : public ::mlir::OpInterface<FindPayloadReplacementOpInterface, detail::FindPayloadReplacementOpInterfaceInterfaceTraits> {
public:
  using ::mlir::OpInterface<FindPayloadReplacementOpInterface, detail::FindPayloadReplacementOpInterfaceInterfaceTraits>::OpInterface;
  template <typename ConcreteOp>
  struct Trait : public detail::FindPayloadReplacementOpInterfaceTrait<ConcreteOp> {};
  /// Return the operands at which the lookup for replacement payload ops
  /// should continue.
  ::llvm::SmallVector<::mlir::Value> getNextOperands();

    /// Name of the attribute attachable to an operation, indicating that
    /// TrackingListener failures should be silenced.
    constexpr const static ::llvm::StringLiteral
        kSilenceTrackingFailuresAttrName =
            "transform.silence_tracking_failures";
};
namespace detail {
  template <typename ConcreteOp>
  struct FindPayloadReplacementOpInterfaceTrait : public ::mlir::OpInterface<FindPayloadReplacementOpInterface, detail::FindPayloadReplacementOpInterfaceInterfaceTraits>::Trait<ConcreteOp> {

    /// Name of the attribute attachable to an operation, indicating that
    /// TrackingListener failures should be silenced.
    constexpr const static ::llvm::StringLiteral
        kSilenceTrackingFailuresAttrName =
            "transform.silence_tracking_failures";
  
  };
}// namespace detail
} // namespace transform
} // namespace mlir
namespace mlir {
namespace transform {
class PatternDescriptorOpInterface;
namespace detail {
struct PatternDescriptorOpInterfaceInterfaceTraits {
  struct Concept {
    /// The methods defined by the interface.
    void (*populatePatterns)(const Concept *impl, ::mlir::Operation *, ::mlir::RewritePatternSet &);
    void (*populatePatternsWithState)(const Concept *impl, ::mlir::Operation *, ::mlir::RewritePatternSet &, ::mlir::transform::TransformState &);
  };
  template<typename ConcreteOp>
  class Model : public Concept {
  public:
    using Interface = ::mlir::transform::PatternDescriptorOpInterface;
    Model() : Concept{populatePatterns, populatePatternsWithState} {}

    static inline void populatePatterns(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, ::mlir::RewritePatternSet & patterns);
    static inline void populatePatternsWithState(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, ::mlir::RewritePatternSet & patterns, ::mlir::transform::TransformState & state);
  };
  template<typename ConcreteOp>
  class FallbackModel : public Concept {
  public:
    using Interface = ::mlir::transform::PatternDescriptorOpInterface;
    FallbackModel() : Concept{populatePatterns, populatePatternsWithState} {}

    static inline void populatePatterns(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, ::mlir::RewritePatternSet & patterns);
    static inline void populatePatternsWithState(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, ::mlir::RewritePatternSet & patterns, ::mlir::transform::TransformState & state);
  };
  template<typename ConcreteModel, typename ConcreteOp>
  class ExternalModel : public FallbackModel<ConcreteModel> {
  public:
    using ConcreteEntity = ConcreteOp;
    void populatePatternsWithState(::mlir::Operation *tablegen_opaque_val, ::mlir::RewritePatternSet &patterns, ::mlir::transform::TransformState &state) const;
  };
};
template <typename ConcreteOp>
struct PatternDescriptorOpInterfaceTrait;

} // namespace detail
class PatternDescriptorOpInterface : public ::mlir::OpInterface<PatternDescriptorOpInterface, detail::PatternDescriptorOpInterfaceInterfaceTraits> {
public:
  using ::mlir::OpInterface<PatternDescriptorOpInterface, detail::PatternDescriptorOpInterfaceInterfaceTraits>::OpInterface;
  template <typename ConcreteOp>
  struct Trait : public detail::PatternDescriptorOpInterfaceTrait<ConcreteOp> {};
  /// Populate rewrite patterns into the given pattern set.
  void populatePatterns(::mlir::RewritePatternSet & patterns);
  /// Populate rewrite patterns into the given pattern set taking into account
  /// the transform state.
  void populatePatternsWithState(::mlir::RewritePatternSet & patterns, ::mlir::transform::TransformState & state);
};
namespace detail {
  template <typename ConcreteOp>
  struct PatternDescriptorOpInterfaceTrait : public ::mlir::OpInterface<PatternDescriptorOpInterface, detail::PatternDescriptorOpInterfaceInterfaceTraits>::Trait<ConcreteOp> {
    /// Populate rewrite patterns into the given pattern set taking into account
    /// the transform state.
    void populatePatternsWithState(::mlir::RewritePatternSet & patterns, ::mlir::transform::TransformState & state) {
      (*static_cast<ConcreteOp *>(this)).populatePatterns(patterns);
    }
  };
}// namespace detail
} // namespace transform
} // namespace mlir
namespace mlir {
namespace transform {
class TypeConverterBuilderOpInterface;
namespace detail {
struct TypeConverterBuilderOpInterfaceInterfaceTraits {
  struct Concept {
    /// The methods defined by the interface.
    std::unique_ptr<::mlir::TypeConverter> (*getTypeConverter)(const Concept *impl, ::mlir::Operation *);
    StringRef (*getTypeConverterType)();
    void (*populateTypeMaterializations)(const Concept *impl, ::mlir::Operation *, ::mlir::TypeConverter &);
  };
  template<typename ConcreteOp>
  class Model : public Concept {
  public:
    using Interface = ::mlir::transform::TypeConverterBuilderOpInterface;
    Model() : Concept{getTypeConverter, getTypeConverterType, populateTypeMaterializations} {}

    static inline std::unique_ptr<::mlir::TypeConverter> getTypeConverter(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline StringRef getTypeConverterType();
    static inline void populateTypeMaterializations(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, ::mlir::TypeConverter & converter);
  };
  template<typename ConcreteOp>
  class FallbackModel : public Concept {
  public:
    using Interface = ::mlir::transform::TypeConverterBuilderOpInterface;
    FallbackModel() : Concept{getTypeConverter, getTypeConverterType, populateTypeMaterializations} {}

    static inline std::unique_ptr<::mlir::TypeConverter> getTypeConverter(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline StringRef getTypeConverterType();
    static inline void populateTypeMaterializations(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, ::mlir::TypeConverter & converter);
  };
  template<typename ConcreteModel, typename ConcreteOp>
  class ExternalModel : public FallbackModel<ConcreteModel> {
  public:
    using ConcreteEntity = ConcreteOp;
    std::unique_ptr<::mlir::TypeConverter> getTypeConverter(::mlir::Operation *tablegen_opaque_val) const;
    static StringRef getTypeConverterType();
    void populateTypeMaterializations(::mlir::Operation *tablegen_opaque_val, ::mlir::TypeConverter &converter) const;
  };
};
template <typename ConcreteOp>
struct TypeConverterBuilderOpInterfaceTrait;

} // namespace detail
class TypeConverterBuilderOpInterface : public ::mlir::OpInterface<TypeConverterBuilderOpInterface, detail::TypeConverterBuilderOpInterfaceInterfaceTraits> {
public:
  using ::mlir::OpInterface<TypeConverterBuilderOpInterface, detail::TypeConverterBuilderOpInterfaceInterfaceTraits>::OpInterface;
  template <typename ConcreteOp>
  struct Trait : public detail::TypeConverterBuilderOpInterfaceTrait<ConcreteOp> {};
  /// Return the type converter to be used with a dialect conversion.
  std::unique_ptr<::mlir::TypeConverter> getTypeConverter();
  /// Return the type of type converter that this `getTypeConverter` returns.
  /// This function is used for op verification.
  StringRef getTypeConverterType();
  /// Populate the given type converter with source/target materialization
  /// functions.
  void populateTypeMaterializations(::mlir::TypeConverter & converter);
};
namespace detail {
  template <typename ConcreteOp>
  struct TypeConverterBuilderOpInterfaceTrait : public ::mlir::OpInterface<TypeConverterBuilderOpInterface, detail::TypeConverterBuilderOpInterfaceInterfaceTraits>::Trait<ConcreteOp> {
    /// Return the type converter to be used with a dialect conversion.
    std::unique_ptr<::mlir::TypeConverter> getTypeConverter() {
      return std::make_unique<::mlir::TypeConverter>();
    }
    /// Return the type of type converter that this `getTypeConverter` returns.
    /// This function is used for op verification.
    static StringRef getTypeConverterType() {
      return "TypeConverter";
    }
    /// Populate the given type converter with source/target materialization
    /// functions.
    void populateTypeMaterializations(::mlir::TypeConverter & converter) {
      return;
    }
  };
}// namespace detail
} // namespace transform
} // namespace mlir
namespace mlir {
namespace transform {
class ConversionPatternDescriptorOpInterface;
namespace detail {
struct ConversionPatternDescriptorOpInterfaceInterfaceTraits {
  struct Concept {
    /// The methods defined by the interface.
    void (*populatePatterns)(const Concept *impl, ::mlir::Operation *, ::mlir::TypeConverter &, ::mlir::RewritePatternSet &);
    void (*populateConversionTargetRules)(const Concept *impl, ::mlir::Operation *, const ::mlir::TypeConverter &, ::mlir::ConversionTarget &);
    std::unique_ptr<::mlir::TypeConverter> (*getTypeConverter)(const Concept *impl, ::mlir::Operation *);
    ::llvm::LogicalResult (*verifyTypeConverter)(const Concept *impl, ::mlir::Operation *, TypeConverterBuilderOpInterface);
  };
  template<typename ConcreteOp>
  class Model : public Concept {
  public:
    using Interface = ::mlir::transform::ConversionPatternDescriptorOpInterface;
    Model() : Concept{populatePatterns, populateConversionTargetRules, getTypeConverter, verifyTypeConverter} {}

    static inline void populatePatterns(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, ::mlir::TypeConverter & typeConverter, ::mlir::RewritePatternSet & patterns);
    static inline void populateConversionTargetRules(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, const ::mlir::TypeConverter & typeConverter, ::mlir::ConversionTarget & conversionTarget);
    static inline std::unique_ptr<::mlir::TypeConverter> getTypeConverter(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline ::llvm::LogicalResult verifyTypeConverter(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, TypeConverterBuilderOpInterface builder);
  };
  template<typename ConcreteOp>
  class FallbackModel : public Concept {
  public:
    using Interface = ::mlir::transform::ConversionPatternDescriptorOpInterface;
    FallbackModel() : Concept{populatePatterns, populateConversionTargetRules, getTypeConverter, verifyTypeConverter} {}

    static inline void populatePatterns(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, ::mlir::TypeConverter & typeConverter, ::mlir::RewritePatternSet & patterns);
    static inline void populateConversionTargetRules(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, const ::mlir::TypeConverter & typeConverter, ::mlir::ConversionTarget & conversionTarget);
    static inline std::unique_ptr<::mlir::TypeConverter> getTypeConverter(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline ::llvm::LogicalResult verifyTypeConverter(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, TypeConverterBuilderOpInterface builder);
  };
  template<typename ConcreteModel, typename ConcreteOp>
  class ExternalModel : public FallbackModel<ConcreteModel> {
  public:
    using ConcreteEntity = ConcreteOp;
    void populateConversionTargetRules(::mlir::Operation *tablegen_opaque_val, const ::mlir::TypeConverter &typeConverter, ::mlir::ConversionTarget &conversionTarget) const;
    std::unique_ptr<::mlir::TypeConverter> getTypeConverter(::mlir::Operation *tablegen_opaque_val) const;
    ::llvm::LogicalResult verifyTypeConverter(::mlir::Operation *tablegen_opaque_val, TypeConverterBuilderOpInterface builder) const;
  };
};
template <typename ConcreteOp>
struct ConversionPatternDescriptorOpInterfaceTrait;

} // namespace detail
class ConversionPatternDescriptorOpInterface : public ::mlir::OpInterface<ConversionPatternDescriptorOpInterface, detail::ConversionPatternDescriptorOpInterfaceInterfaceTraits> {
public:
  using ::mlir::OpInterface<ConversionPatternDescriptorOpInterface, detail::ConversionPatternDescriptorOpInterfaceInterfaceTraits>::OpInterface;
  template <typename ConcreteOp>
  struct Trait : public detail::ConversionPatternDescriptorOpInterfaceTrait<ConcreteOp> {};
  /// Populate conversion patterns into the given pattern set with the
  /// given type converter.
  void populatePatterns(::mlir::TypeConverter & typeConverter, ::mlir::RewritePatternSet & patterns);
  /// Populate the ConversionTarget using the final TypeConverter. The default
  /// implementation is to do nothing. Overriding this method can be useful
  /// in order to setup the ConversionTarget for structural type conversions.
  /// In such a situation, an op's legality depends on using the TypeConverter
  /// to determine whether the op's operand and result types are legal
  /// (defined as converting to themselves).
  /// 
  void populateConversionTargetRules(const ::mlir::TypeConverter & typeConverter, ::mlir::ConversionTarget & conversionTarget);
  /// Return the type converter to be used with this pattern set. If no
  /// type converter is specified, the default type converter of the enclosing
  /// "apply_conversion_patterns" op is used.
  std::unique_ptr<::mlir::TypeConverter> getTypeConverter();
  /// Verify the default type converter that is provided by the enclosing
  /// "apply_conversion_patterns" op.
  ::llvm::LogicalResult verifyTypeConverter(TypeConverterBuilderOpInterface builder);
};
namespace detail {
  template <typename ConcreteOp>
  struct ConversionPatternDescriptorOpInterfaceTrait : public ::mlir::OpInterface<ConversionPatternDescriptorOpInterface, detail::ConversionPatternDescriptorOpInterfaceInterfaceTraits>::Trait<ConcreteOp> {
    /// Populate the ConversionTarget using the final TypeConverter. The default
    /// implementation is to do nothing. Overriding this method can be useful
    /// in order to setup the ConversionTarget for structural type conversions.
    /// In such a situation, an op's legality depends on using the TypeConverter
    /// to determine whether the op's operand and result types are legal
    /// (defined as converting to themselves).
    /// 
    void populateConversionTargetRules(const ::mlir::TypeConverter & typeConverter, ::mlir::ConversionTarget & conversionTarget) {
      return;
    }
    /// Return the type converter to be used with this pattern set. If no
    /// type converter is specified, the default type converter of the enclosing
    /// "apply_conversion_patterns" op is used.
    std::unique_ptr<::mlir::TypeConverter> getTypeConverter() {
      return nullptr;
    }
    /// Verify the default type converter that is provided by the enclosing
    /// "apply_conversion_patterns" op.
    ::llvm::LogicalResult verifyTypeConverter(TypeConverterBuilderOpInterface builder) {
      return success();
    }
  };
}// namespace detail
} // namespace transform
} // namespace mlir
namespace mlir {
namespace transform {
template<typename ConcreteOp>
::mlir::DiagnosedSilenceableFailure detail::TransformOpInterfaceInterfaceTraits::Model<ConcreteOp>::apply(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, ::mlir::transform::TransformRewriter & rewriter, ::mlir::transform::TransformResults & transformResults, ::mlir::transform::TransformState & state) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).apply(rewriter, transformResults, state);
}
template<typename ConcreteOp>
bool detail::TransformOpInterfaceInterfaceTraits::Model<ConcreteOp>::allowsRepeatedHandleOperands(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).allowsRepeatedHandleOperands();
}
template<typename ConcreteOp>
::mlir::DiagnosedSilenceableFailure detail::TransformOpInterfaceInterfaceTraits::FallbackModel<ConcreteOp>::apply(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, ::mlir::transform::TransformRewriter & rewriter, ::mlir::transform::TransformResults & transformResults, ::mlir::transform::TransformState & state) {
  return static_cast<const ConcreteOp *>(impl)->apply(tablegen_opaque_val, rewriter, transformResults, state);
}
template<typename ConcreteOp>
bool detail::TransformOpInterfaceInterfaceTraits::FallbackModel<ConcreteOp>::allowsRepeatedHandleOperands(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return static_cast<const ConcreteOp *>(impl)->allowsRepeatedHandleOperands(tablegen_opaque_val);
}
template<typename ConcreteModel, typename ConcreteOp>
bool detail::TransformOpInterfaceInterfaceTraits::ExternalModel<ConcreteModel, ConcreteOp>::allowsRepeatedHandleOperands(::mlir::Operation *tablegen_opaque_val) const {
return false;
}
} // namespace transform
} // namespace mlir
namespace mlir {
namespace transform {
template<typename ConcreteOp>
::llvm::SmallVector<::mlir::Value> detail::FindPayloadReplacementOpInterfaceInterfaceTraits::Model<ConcreteOp>::getNextOperands(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getNextOperands();
}
template<typename ConcreteOp>
::llvm::SmallVector<::mlir::Value> detail::FindPayloadReplacementOpInterfaceInterfaceTraits::FallbackModel<ConcreteOp>::getNextOperands(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return static_cast<const ConcreteOp *>(impl)->getNextOperands(tablegen_opaque_val);
}
} // namespace transform
} // namespace mlir
namespace mlir {
namespace transform {
template<typename ConcreteOp>
void detail::PatternDescriptorOpInterfaceInterfaceTraits::Model<ConcreteOp>::populatePatterns(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, ::mlir::RewritePatternSet & patterns) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).populatePatterns(patterns);
}
template<typename ConcreteOp>
void detail::PatternDescriptorOpInterfaceInterfaceTraits::Model<ConcreteOp>::populatePatternsWithState(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, ::mlir::RewritePatternSet & patterns, ::mlir::transform::TransformState & state) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).populatePatternsWithState(patterns, state);
}
template<typename ConcreteOp>
void detail::PatternDescriptorOpInterfaceInterfaceTraits::FallbackModel<ConcreteOp>::populatePatterns(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, ::mlir::RewritePatternSet & patterns) {
  return static_cast<const ConcreteOp *>(impl)->populatePatterns(tablegen_opaque_val, patterns);
}
template<typename ConcreteOp>
void detail::PatternDescriptorOpInterfaceInterfaceTraits::FallbackModel<ConcreteOp>::populatePatternsWithState(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, ::mlir::RewritePatternSet & patterns, ::mlir::transform::TransformState & state) {
  return static_cast<const ConcreteOp *>(impl)->populatePatternsWithState(tablegen_opaque_val, patterns, state);
}
template<typename ConcreteModel, typename ConcreteOp>
void detail::PatternDescriptorOpInterfaceInterfaceTraits::ExternalModel<ConcreteModel, ConcreteOp>::populatePatternsWithState(::mlir::Operation *tablegen_opaque_val, ::mlir::RewritePatternSet &patterns, ::mlir::transform::TransformState &state) const {
(llvm::cast<ConcreteOp>(tablegen_opaque_val)).populatePatterns(patterns);
}
} // namespace transform
} // namespace mlir
namespace mlir {
namespace transform {
template<typename ConcreteOp>
std::unique_ptr<::mlir::TypeConverter> detail::TypeConverterBuilderOpInterfaceInterfaceTraits::Model<ConcreteOp>::getTypeConverter(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getTypeConverter();
}
template<typename ConcreteOp>
StringRef detail::TypeConverterBuilderOpInterfaceInterfaceTraits::Model<ConcreteOp>::getTypeConverterType() {
  return ConcreteOp::getTypeConverterType();
}
template<typename ConcreteOp>
void detail::TypeConverterBuilderOpInterfaceInterfaceTraits::Model<ConcreteOp>::populateTypeMaterializations(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, ::mlir::TypeConverter & converter) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).populateTypeMaterializations(converter);
}
template<typename ConcreteOp>
std::unique_ptr<::mlir::TypeConverter> detail::TypeConverterBuilderOpInterfaceInterfaceTraits::FallbackModel<ConcreteOp>::getTypeConverter(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return static_cast<const ConcreteOp *>(impl)->getTypeConverter(tablegen_opaque_val);
}
template<typename ConcreteOp>
StringRef detail::TypeConverterBuilderOpInterfaceInterfaceTraits::FallbackModel<ConcreteOp>::getTypeConverterType() {
  return ConcreteOp::getTypeConverterType();
}
template<typename ConcreteOp>
void detail::TypeConverterBuilderOpInterfaceInterfaceTraits::FallbackModel<ConcreteOp>::populateTypeMaterializations(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, ::mlir::TypeConverter & converter) {
  return static_cast<const ConcreteOp *>(impl)->populateTypeMaterializations(tablegen_opaque_val, converter);
}
template<typename ConcreteModel, typename ConcreteOp>
std::unique_ptr<::mlir::TypeConverter> detail::TypeConverterBuilderOpInterfaceInterfaceTraits::ExternalModel<ConcreteModel, ConcreteOp>::getTypeConverter(::mlir::Operation *tablegen_opaque_val) const {
return std::make_unique<::mlir::TypeConverter>();
}
template<typename ConcreteModel, typename ConcreteOp>
StringRef detail::TypeConverterBuilderOpInterfaceInterfaceTraits::ExternalModel<ConcreteModel, ConcreteOp>::getTypeConverterType() {
return "TypeConverter";
}
template<typename ConcreteModel, typename ConcreteOp>
void detail::TypeConverterBuilderOpInterfaceInterfaceTraits::ExternalModel<ConcreteModel, ConcreteOp>::populateTypeMaterializations(::mlir::Operation *tablegen_opaque_val, ::mlir::TypeConverter &converter) const {
return;
}
} // namespace transform
} // namespace mlir
namespace mlir {
namespace transform {
template<typename ConcreteOp>
void detail::ConversionPatternDescriptorOpInterfaceInterfaceTraits::Model<ConcreteOp>::populatePatterns(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, ::mlir::TypeConverter & typeConverter, ::mlir::RewritePatternSet & patterns) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).populatePatterns(typeConverter, patterns);
}
template<typename ConcreteOp>
void detail::ConversionPatternDescriptorOpInterfaceInterfaceTraits::Model<ConcreteOp>::populateConversionTargetRules(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, const ::mlir::TypeConverter & typeConverter, ::mlir::ConversionTarget & conversionTarget) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).populateConversionTargetRules(typeConverter, conversionTarget);
}
template<typename ConcreteOp>
std::unique_ptr<::mlir::TypeConverter> detail::ConversionPatternDescriptorOpInterfaceInterfaceTraits::Model<ConcreteOp>::getTypeConverter(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getTypeConverter();
}
template<typename ConcreteOp>
::llvm::LogicalResult detail::ConversionPatternDescriptorOpInterfaceInterfaceTraits::Model<ConcreteOp>::verifyTypeConverter(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, TypeConverterBuilderOpInterface builder) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).verifyTypeConverter(builder);
}
template<typename ConcreteOp>
void detail::ConversionPatternDescriptorOpInterfaceInterfaceTraits::FallbackModel<ConcreteOp>::populatePatterns(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, ::mlir::TypeConverter & typeConverter, ::mlir::RewritePatternSet & patterns) {
  return static_cast<const ConcreteOp *>(impl)->populatePatterns(tablegen_opaque_val, typeConverter, patterns);
}
template<typename ConcreteOp>
void detail::ConversionPatternDescriptorOpInterfaceInterfaceTraits::FallbackModel<ConcreteOp>::populateConversionTargetRules(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, const ::mlir::TypeConverter & typeConverter, ::mlir::ConversionTarget & conversionTarget) {
  return static_cast<const ConcreteOp *>(impl)->populateConversionTargetRules(tablegen_opaque_val, typeConverter, conversionTarget);
}
template<typename ConcreteOp>
std::unique_ptr<::mlir::TypeConverter> detail::ConversionPatternDescriptorOpInterfaceInterfaceTraits::FallbackModel<ConcreteOp>::getTypeConverter(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return static_cast<const ConcreteOp *>(impl)->getTypeConverter(tablegen_opaque_val);
}
template<typename ConcreteOp>
::llvm::LogicalResult detail::ConversionPatternDescriptorOpInterfaceInterfaceTraits::FallbackModel<ConcreteOp>::verifyTypeConverter(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, TypeConverterBuilderOpInterface builder) {
  return static_cast<const ConcreteOp *>(impl)->verifyTypeConverter(tablegen_opaque_val, builder);
}
template<typename ConcreteModel, typename ConcreteOp>
void detail::ConversionPatternDescriptorOpInterfaceInterfaceTraits::ExternalModel<ConcreteModel, ConcreteOp>::populateConversionTargetRules(::mlir::Operation *tablegen_opaque_val, const ::mlir::TypeConverter &typeConverter, ::mlir::ConversionTarget &conversionTarget) const {
return;
}
template<typename ConcreteModel, typename ConcreteOp>
std::unique_ptr<::mlir::TypeConverter> detail::ConversionPatternDescriptorOpInterfaceInterfaceTraits::ExternalModel<ConcreteModel, ConcreteOp>::getTypeConverter(::mlir::Operation *tablegen_opaque_val) const {
return nullptr;
}
template<typename ConcreteModel, typename ConcreteOp>
::llvm::LogicalResult detail::ConversionPatternDescriptorOpInterfaceInterfaceTraits::ExternalModel<ConcreteModel, ConcreteOp>::verifyTypeConverter(::mlir::Operation *tablegen_opaque_val, TypeConverterBuilderOpInterface builder) const {
return success();
}
} // namespace transform
} // namespace mlir
