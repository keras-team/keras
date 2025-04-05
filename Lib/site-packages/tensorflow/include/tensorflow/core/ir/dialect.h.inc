/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Dialect Declarations                                                       *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|* From: ops.td                                                               *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

namespace mlir {
namespace tfg {

class TFGraphDialect : public ::mlir::Dialect {
  explicit TFGraphDialect(::mlir::MLIRContext *context);

  void initialize();
  friend class ::mlir::MLIRContext;
public:
  ~TFGraphDialect() override;
  static constexpr ::llvm::StringLiteral getDialectNamespace() {
    return ::llvm::StringLiteral("tfg");
  }

  /// Parse an attribute registered to this dialect.
  ::mlir::Attribute parseAttribute(::mlir::DialectAsmParser &parser,
                                   ::mlir::Type type) const override;

  /// Print an attribute registered to this dialect.
  void printAttribute(::mlir::Attribute attr,
                      ::mlir::DialectAsmPrinter &os) const override;

    /// Provides a hook for op interface.
    void *getRegisteredInterfaceForOp(mlir::TypeID interfaceID,
                                      mlir::OperationName opName) override;

    StringAttr getNameAttrIdentifier() const { return name_key_; }
    static constexpr StringLiteral getNameAttrKey() { return {"_mlir_name"}; }

    StringAttr getDeviceAttrIdentifier() const { return device_key_; }
    static constexpr StringLiteral getDeviceAttrKey() {
      return {"_mlir_device"};
    }

    StringAttr getAssignedDeviceAttrIdentifier() const {
      return assigned_device_key_;
    }
    static constexpr StringLiteral getAssignedDeviceAttrKey() {
      return {"_mlir_assigned_device"};
    }

    StringAttr getFullTypeAttrIdentifier() const { return fulltype_key_; }
    static constexpr StringLiteral getFullTypeAttrKey() {
      return {"_mlir_fulltype"};
    }

    StringAttr getTfgNameAttrIdentifier() const { return tfg_name_key_; }
    static constexpr StringRef getTfgNameAttrKey() { return "tfg.name"; }

    StringAttr getTfgDescriptionAttrIdentifier() const {
      return tfg_description_key_;
    }
    static constexpr StringRef getTfgDescriptionAttrKey() {
      return {"tfg.description"};
    }

    StringAttr getTfgIsRefAttrIdentifier() const { return tfg_is_ref_key_; }
    static constexpr StringRef getTfgIsRefAttrKey() { return {"tfg.is_ref"}; }

    StringAttr getTfgHandleDataAttrIdentifier() const {
      return tfg_handle_data_key_;
    }
    static constexpr StringRef getTfgHandleDataAttrKey() {
      return {"tfg.handle_data"};
    }

    StringAttr getTfgFullTypeAttrIdentifier() const {
      return tfg_full_type_key_;
    }
    static constexpr StringRef getTfgFullTypeAttrKey() {
      return {"tfg.experimental_full_type"};
    }

    StringAttr getLiftedGraphFuncNameAttrIdentifier() const {
      return lifted_graph_func_name_;
    }
    static constexpr StringRef getLiftedGraphFuncNameKey() {
      return {"_mlir_lifted_graph"};
    }

    // Cached accessor for the control type.
    ControlType getControlType() const { return control_ty_; }

    // Print an operation that belongs to this dialect if unregistered.
    void printCustomTfOp(Operation *op, OpAsmPrinter &printer) const;

    // Returns the hook to parse an operation belonging to this dialect, even
    // if unregistered.
    std::optional<ParseOpHook> getParseOperationHook(StringRef opName) const
      override;

    // Returns the took to print an operation belonging to this dialect, even
    // if unregistered.
    llvm::unique_function<void(Operation *, OpAsmPrinter &)>
    getOperationPrinter(Operation *op) const override;

    // Functions for checking operation categories.
    #define GET_OP_CATEGORIES
    #include "tensorflow/core/ir/tf_op_names.inc"

  private:
    // Fallback implementation of OpAsmOpInterface.
    TFGraphOpAsmInterface *fallbackOpAsmInterface_ = nullptr;

    // Cached TensorFlow operation names.
    #define GET_OP_NAME_DECLS
    #include "tensorflow/core/ir/tf_op_names.inc"

    // Cached identifier for efficiency purpose.
    StringAttr assigned_device_key_;
    StringAttr device_key_;
    StringAttr fulltype_key_;
    StringAttr lifted_graph_func_name_;
    StringAttr name_key_;
    StringAttr tfg_description_key_;
    StringAttr tfg_full_type_key_;
    StringAttr tfg_handle_data_key_;
    StringAttr tfg_is_ref_key_;
    StringAttr tfg_name_key_;

    // Cached control type.
    ControlType control_ty_;
  };
} // namespace tfg
} // namespace mlir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::tfg::TFGraphDialect)
