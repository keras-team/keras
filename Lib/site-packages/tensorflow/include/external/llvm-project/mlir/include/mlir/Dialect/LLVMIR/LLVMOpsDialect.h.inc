/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Dialect Declarations                                                       *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|* From: LLVMOps.td                                                           *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

namespace mlir {
namespace LLVM {

class LLVMDialect : public ::mlir::Dialect {
  explicit LLVMDialect(::mlir::MLIRContext *context);

  void initialize();
  friend class ::mlir::MLIRContext;
public:
  ~LLVMDialect() override;
  static constexpr ::llvm::StringLiteral getDialectNamespace() {
    return ::llvm::StringLiteral("llvm");
  }

  /// Parse an attribute registered to this dialect.
  ::mlir::Attribute parseAttribute(::mlir::DialectAsmParser &parser,
                                   ::mlir::Type type) const override;

  /// Print an attribute registered to this dialect.
  void printAttribute(::mlir::Attribute attr,
                      ::mlir::DialectAsmPrinter &os) const override;

  /// Materialize a single constant operation from a given attribute value with
  /// the desired resultant type.
  ::mlir::Operation *materializeConstant(::mlir::OpBuilder &builder,
                                         ::mlir::Attribute value,
                                         ::mlir::Type type,
                                         ::mlir::Location loc) override;

    /// Provides a hook for verifying dialect attributes attached to the given
    /// op.
    ::llvm::LogicalResult verifyOperationAttribute(
        ::mlir::Operation *op, ::mlir::NamedAttribute attribute) override;

    /// Provides a hook for verifying dialect attributes attached to the given
    /// op's region argument.
    ::llvm::LogicalResult verifyRegionArgAttribute(
        ::mlir::Operation *op, unsigned regionIndex, unsigned argIndex,
        ::mlir::NamedAttribute attribute) override;

    /// Provides a hook for verifying dialect attributes attached to the given
    /// op's region result.
    ::llvm::LogicalResult verifyRegionResultAttribute(
        ::mlir::Operation *op, unsigned regionIndex, unsigned resultIndex,
        ::mlir::NamedAttribute attribute) override;

    /// Helper to manage the discardable attribute `workgroup_attribution`.
    class WorkgroupAttributionAttrHelper {
      ::mlir::StringAttr name;
    public:
      static constexpr ::llvm::StringLiteral getNameStr() {
        return "llvm.workgroup_attribution";
      }
      constexpr ::mlir::StringAttr getName() {
        return name;
      }

      WorkgroupAttributionAttrHelper(::mlir::MLIRContext *ctx)
        : name(::mlir::StringAttr::get(ctx, getNameStr())) {}

     WorkgroupAttributionAttr getAttr(::mlir::Operation *op) {
       return op->getAttrOfType<WorkgroupAttributionAttr>(name);
     }
     void setAttr(::mlir::Operation *op, WorkgroupAttributionAttr val) {
       op->setAttr(name, val);
     }
     bool isAttrPresent(::mlir::Operation *op) {
       return op->hasAttrOfType<WorkgroupAttributionAttr>(name);
     }
     void removeAttr(::mlir::Operation *op) {
       assert(op->hasAttrOfType<WorkgroupAttributionAttr>(name));
       op->removeAttr(name);
     }
   };
   WorkgroupAttributionAttrHelper getWorkgroupAttributionAttrHelper() {
     return workgroupAttributionAttrName;
   }
 private:
   WorkgroupAttributionAttrHelper workgroupAttributionAttrName;
 public:

    /// Name of the data layout attributes.
    static StringRef getDataLayoutAttrName() { return "llvm.data_layout"; }
    static StringRef getNoAliasScopesAttrName() { return "noalias_scopes"; }
    static StringRef getAliasScopesAttrName() { return "alias_scopes"; }
    static StringRef getAccessGroupsAttrName() { return "access_groups"; }
    static StringRef getIdentAttrName() { return "llvm.ident"; }

    /// Names of llvm parameter attributes.
    static StringRef getAlignAttrName() { return "llvm.align"; }
    static StringRef getAllocAlignAttrName() { return "llvm.allocalign"; }
    static StringRef getAllocatedPointerAttrName() { return "llvm.allocptr"; }
    static StringRef getByValAttrName() { return "llvm.byval"; }
    static StringRef getByRefAttrName() { return "llvm.byref"; }
    static StringRef getNoUndefAttrName() { return "llvm.noundef"; }
    static StringRef getDereferenceableAttrName() { return "llvm.dereferenceable"; }
    static StringRef getDereferenceableOrNullAttrName() { return "llvm.dereferenceable_or_null"; }
    static StringRef getInAllocaAttrName() { return "llvm.inalloca"; }
    static StringRef getInRegAttrName() { return "llvm.inreg"; }
    static StringRef getNestAttrName() { return "llvm.nest"; }
    static StringRef getNoAliasAttrName() { return "llvm.noalias"; }
    static StringRef getNoCaptureAttrName() { return "llvm.nocapture"; }
    static StringRef getNoFreeAttrName() { return "llvm.nofree"; }
    static StringRef getNonNullAttrName() { return "llvm.nonnull"; }
    static StringRef getPreallocatedAttrName() { return "llvm.preallocated"; }
    static StringRef getReadonlyAttrName() { return "llvm.readonly"; }
    static StringRef getReturnedAttrName() { return "llvm.returned"; }
    static StringRef getSExtAttrName() { return "llvm.signext"; }
    static StringRef getStackAlignmentAttrName() { return "llvm.alignstack"; }
    static StringRef getStructRetAttrName() { return "llvm.sret"; }
    static StringRef getWriteOnlyAttrName() { return "llvm.writeonly"; }
    static StringRef getZExtAttrName() { return "llvm.zeroext"; }
    // TODO Restrict the usage of this to parameter attributes once there is an
    // alternative way of modeling memory effects on FunctionOpInterface.
    /// Name of the attribute that will cause the creation of a readnone memory
    /// effect when lowering to the LLVMDialect.
    static StringRef getReadnoneAttrName() { return "llvm.readnone"; }

    /// Verifies if the given string is a well-formed data layout descriptor.
    /// Uses `reportError` to report errors.
    static LogicalResult verifyDataLayoutString(
        StringRef descr, llvm::function_ref<void (const Twine &)> reportError);

    /// Name of the target triple attribute.
    static StringRef getTargetTripleAttrName() { return "llvm.target_triple"; }

    /// Name of the C wrapper emission attribute.
    static StringRef getEmitCWrapperAttrName() {
      return "llvm.emit_c_interface";
    }

    /// Returns `true` if the given type is compatible with the LLVM dialect.
    static bool isCompatibleType(Type);


    Type parseType(DialectAsmParser &p) const override;
    void printType(Type, DialectAsmPrinter &p) const override;

  private:
    /// Verifies a parameter attribute attached to a parameter of type
    /// paramType.
    LogicalResult verifyParameterAttribute(Operation *op,
                                           Type paramType,
                                           NamedAttribute paramAttr);

    /// Register all types.
    void registerTypes();

    /// A cache storing compatible LLVM types that have been verified. This
    /// can save us lots of verification time if there are many occurrences
    /// of some deeply-nested aggregate types in the program.
    ThreadLocalCache<DenseSet<Type>> compatibleTypes;

    /// Register the attributes of this dialect.
    void registerAttributes();
  };
} // namespace LLVM
} // namespace mlir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::LLVM::LLVMDialect)
