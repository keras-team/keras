/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Dialect Declarations                                                       *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|* From: BufferizationBase.td                                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

namespace mlir {
namespace bufferization {

class BufferizationDialect : public ::mlir::Dialect {
  explicit BufferizationDialect(::mlir::MLIRContext *context);

  void initialize();
  friend class ::mlir::MLIRContext;
public:
  ~BufferizationDialect() override;
  static constexpr ::llvm::StringLiteral getDialectNamespace() {
    return ::llvm::StringLiteral("bufferization");
  }

    /// Provides a hook for verifying dialect attributes attached to the given
    /// op.
    ::llvm::LogicalResult verifyOperationAttribute(
        ::mlir::Operation *op, ::mlir::NamedAttribute attribute) override;

    /// Verify an attribute from this dialect on the argument at 'argIndex' for
    /// the region at 'regionIndex' on the given operation. Returns failure if
    /// the verification failed, success otherwise. This hook may optionally be
    /// invoked from any operation containing a region.
    LogicalResult verifyRegionArgAttribute(Operation *,
                                           unsigned regionIndex,
                                           unsigned argIndex,
                                           NamedAttribute) override;

    /// An attribute that can override writability of buffers of tensor function
    /// arguments during One-Shot Module Bufferize.
    constexpr const static ::llvm::StringLiteral
        kWritableAttrName = "bufferization.writable";

    /// An attribute for function arguments that describes how the function
    /// accesses the buffer. Can be one "none", "read", "write" or "read-write".
    ///
    /// When no attribute is specified, the analysis tries to infer the access
    /// behavior from its body. In case of external functions, for which no
    /// function body is available, "read-write" is assumed by default.
    constexpr const static ::llvm::StringLiteral
        kBufferAccessAttrName = "bufferization.access";

    /// Attribute name used to mark the bufferization layout for region
    /// arguments during One-Shot Module Bufferize.
    constexpr const static ::llvm::StringLiteral
        kBufferLayoutAttrName = "bufferization.buffer_layout";

    /// An attribute that can be attached to ops with an allocation and/or
    /// deallocation side effect. It indicates that the op is under a "manual
    /// deallocation" scheme. In the case of an allocation op, the returned
    /// value is *not* an automatically managed allocation and assigned an
    /// ownership of "false". Furthermore, only deallocation ops that are
    /// guaranteed to deallocate a buffer under "manual deallocation" are
    /// allowed to have this attribute. (Deallocation ops without this
    /// attribute are rejected by the ownership-based buffer deallocation pass.)
    constexpr const static ::llvm::StringLiteral
        kManualDeallocation = "bufferization.manual_deallocation";
  };
} // namespace bufferization
} // namespace mlir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::bufferization::BufferizationDialect)
