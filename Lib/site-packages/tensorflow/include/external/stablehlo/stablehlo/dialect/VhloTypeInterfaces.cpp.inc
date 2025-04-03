/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Interface Definitions                                                      *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

/// Returns the minimum version of the VHLO dialect an attribute is supported in.
mlir::vhlo::Version mlir::vhlo::VersionedTypeInterface::getMinVersion() const {
      return getImpl()->getMinVersion(getImpl(), *this);
  }
/// Returns the maximum version (inclusive) of the VHLO dialect an attribute is supported in.
mlir::vhlo::Version mlir::vhlo::VersionedTypeInterface::getMaxVersion() const {
      return getImpl()->getMaxVersion(getImpl(), *this);
  }
