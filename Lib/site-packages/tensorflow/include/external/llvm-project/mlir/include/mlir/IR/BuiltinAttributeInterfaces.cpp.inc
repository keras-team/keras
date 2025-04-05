/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Interface Definitions                                                      *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

/// This method returns an opaque range indexer for the given elementID, which
/// corresponds to a desired C++ element data type. Returns the indexer if the
/// attribute supports the given data type, failure otherwise.
::mlir::FailureOr<::mlir::detail::ElementsAttrIndexer> mlir::ElementsAttr::getValuesImpl(::mlir::TypeID elementID) const {
      return getImpl()->getValuesImpl(getImpl(), *this, elementID);
  }
/// Returns true if the attribute elements correspond to a splat, i.e. that
/// all elements of the attribute are the same value.
bool mlir::ElementsAttr::isSplat() const {
      return getImpl()->isSplat(getImpl(), *this);
  }
/// Returns the shaped type of the elements attribute.
::mlir::ShapedType mlir::ElementsAttr::getShapedType() const {
      return getImpl()->getShapedType(getImpl(), *this);
  }
/// Get the attribute's type
::mlir::Type mlir::ElementsAttr::getType() const {
      return getImpl()->implTypedAttr->getType(getImpl()->implTypedAttr, *this);
  }
/// Get the MemRef layout as an AffineMap, the method must not return NULL
::mlir::AffineMap mlir::MemRefLayoutAttrInterface::getAffineMap() const {
      return getImpl()->getAffineMap(getImpl(), *this);
  }
/// Return true if this attribute represents the identity layout
bool mlir::MemRefLayoutAttrInterface::isIdentity() const {
      return getImpl()->isIdentity(getImpl(), *this);
  }
/// Check if the current layout is applicable to the provided shape
::llvm::LogicalResult mlir::MemRefLayoutAttrInterface::verifyLayout(::llvm::ArrayRef<int64_t> shape, ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError) const {
      return getImpl()->verifyLayout(getImpl(), *this, shape, emitError);
  }
/// Get the attribute's type
::mlir::Type mlir::TypedAttr::getType() const {
      return getImpl()->getType(getImpl(), *this);
  }
