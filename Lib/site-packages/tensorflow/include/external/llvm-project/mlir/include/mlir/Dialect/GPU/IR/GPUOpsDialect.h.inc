/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Dialect Declarations                                                       *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|* From: GPUOps.td                                                            *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

namespace mlir {
namespace gpu {

class GPUDialect : public ::mlir::Dialect {
  explicit GPUDialect(::mlir::MLIRContext *context);

  void initialize();
  friend class ::mlir::MLIRContext;
public:
  ~GPUDialect() override;
  static constexpr ::llvm::StringLiteral getDialectNamespace() {
    return ::llvm::StringLiteral("gpu");
  }

  /// Parse an attribute registered to this dialect.
  ::mlir::Attribute parseAttribute(::mlir::DialectAsmParser &parser,
                                   ::mlir::Type type) const override;

  /// Print an attribute registered to this dialect.
  void printAttribute(::mlir::Attribute attr,
                      ::mlir::DialectAsmPrinter &os) const override;

  /// Parse a type registered to this dialect.
  ::mlir::Type parseType(::mlir::DialectAsmParser &parser) const override;

  /// Print a type registered to this dialect.
  void printType(::mlir::Type type,
                 ::mlir::DialectAsmPrinter &os) const override;

    /// Provides a hook for verifying dialect attributes attached to the given
    /// op.
    ::llvm::LogicalResult verifyOperationAttribute(
        ::mlir::Operation *op, ::mlir::NamedAttribute attribute) override;

    /// Helper to manage the discardable attribute `known_block_size`.
    class KnownBlockSizeAttrHelper {
      ::mlir::StringAttr name;
    public:
      static constexpr ::llvm::StringLiteral getNameStr() {
        return "gpu.known_block_size";
      }
      constexpr ::mlir::StringAttr getName() {
        return name;
      }

      KnownBlockSizeAttrHelper(::mlir::MLIRContext *ctx)
        : name(::mlir::StringAttr::get(ctx, getNameStr())) {}

     ::mlir::DenseI32ArrayAttr getAttr(::mlir::Operation *op) {
       return op->getAttrOfType<::mlir::DenseI32ArrayAttr>(name);
     }
     void setAttr(::mlir::Operation *op, ::mlir::DenseI32ArrayAttr val) {
       op->setAttr(name, val);
     }
     bool isAttrPresent(::mlir::Operation *op) {
       return op->hasAttrOfType<::mlir::DenseI32ArrayAttr>(name);
     }
     void removeAttr(::mlir::Operation *op) {
       assert(op->hasAttrOfType<::mlir::DenseI32ArrayAttr>(name));
       op->removeAttr(name);
     }
   };
   KnownBlockSizeAttrHelper getKnownBlockSizeAttrHelper() {
     return knownBlockSizeAttrName;
   }
 private:
   KnownBlockSizeAttrHelper knownBlockSizeAttrName;
 public:

    /// Helper to manage the discardable attribute `known_grid_size`.
    class KnownGridSizeAttrHelper {
      ::mlir::StringAttr name;
    public:
      static constexpr ::llvm::StringLiteral getNameStr() {
        return "gpu.known_grid_size";
      }
      constexpr ::mlir::StringAttr getName() {
        return name;
      }

      KnownGridSizeAttrHelper(::mlir::MLIRContext *ctx)
        : name(::mlir::StringAttr::get(ctx, getNameStr())) {}

     ::mlir::DenseI32ArrayAttr getAttr(::mlir::Operation *op) {
       return op->getAttrOfType<::mlir::DenseI32ArrayAttr>(name);
     }
     void setAttr(::mlir::Operation *op, ::mlir::DenseI32ArrayAttr val) {
       op->setAttr(name, val);
     }
     bool isAttrPresent(::mlir::Operation *op) {
       return op->hasAttrOfType<::mlir::DenseI32ArrayAttr>(name);
     }
     void removeAttr(::mlir::Operation *op) {
       assert(op->hasAttrOfType<::mlir::DenseI32ArrayAttr>(name));
       op->removeAttr(name);
     }
   };
   KnownGridSizeAttrHelper getKnownGridSizeAttrHelper() {
     return knownGridSizeAttrName;
   }
 private:
   KnownGridSizeAttrHelper knownGridSizeAttrName;
 public:

    /// Get the name of the attribute used to annotate the modules that contain
    /// kernel modules.
    static StringRef getContainerModuleAttrName() {
      return "gpu.container_module";
    }
    /// Get the name of the attribute used to annotate external kernel
    /// functions.
    static StringRef getKernelFuncAttrName() { return "gpu.kernel"; }

    /// Returns whether the given function is a kernel function, i.e., has the
    /// 'gpu.kernel' attribute.
    static bool isKernel(Operation *op);

    /// Returns the number of workgroup (thread, block) dimensions supported in
    /// the GPU dialect.
    // TODO: consider generalizing this.
    static unsigned getNumWorkgroupDimensions() { return 3; }

    /// Returns the numeric value used to identify the workgroup memory address
    /// space.
    static AddressSpace getWorkgroupAddressSpace() { return AddressSpace::Workgroup; }

    /// Returns the numeric value used to identify the private memory address
    /// space.
    static AddressSpace getPrivateAddressSpace() { return AddressSpace::Private; }

    /// Return true if the given MemRefType has an address space that matches
    /// with the gpu::AddressSpaceAttr attribute with value 'workgroup`.
    static bool hasWorkgroupMemoryAddressSpace(MemRefType type);

    /// Return true if the given Attribute is an gpu::AddressSpaceAttr
    /// attribute with value 'workgroup`.
    static bool isWorkgroupMemoryAddressSpace(Attribute memorySpace);
  };
} // namespace gpu
} // namespace mlir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::gpu::GPUDialect)
