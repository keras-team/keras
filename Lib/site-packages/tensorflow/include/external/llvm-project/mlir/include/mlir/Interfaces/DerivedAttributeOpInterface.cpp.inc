/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Interface Definitions                                                      *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

/// Returns whether name corresponds to a derived attribute.
bool mlir::DerivedAttributeOpInterface::isDerivedAttribute(::mlir::StringRef name) {
      return getImpl()->isDerivedAttribute(name);
  }
/// Materializes the derived attributes. Returns null attribute where
/// unable to materialize a derived attribute as attribute.
::mlir::DictionaryAttr mlir::DerivedAttributeOpInterface::materializeDerivedAttributes() {
      return getImpl()->materializeDerivedAttributes(getImpl(), getOperation());
  }
