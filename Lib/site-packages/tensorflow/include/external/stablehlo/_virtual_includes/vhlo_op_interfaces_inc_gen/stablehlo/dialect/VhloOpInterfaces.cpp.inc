/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Interface Definitions                                                      *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

/// Validate versioned constraints on a versioned op.
///        Used if the spec'ed constraints of an op change over time.
mlir::LogicalResult mlir::vhlo::VersionedOpConstraintInterface::validateConstraint(mlir::Operation* op, mlir::vhlo::Version targetVersion) {
      return getImpl()->validateConstraint(getImpl(), getOperation(), op, targetVersion);
  }
/// Returns the minimum version of the VHLO dialect an op is supported in.
mlir::vhlo::Version mlir::vhlo::VersionedOpInterface::getMinVersion() {
      return getImpl()->getMinVersion(getImpl(), getOperation());
  }
/// Returns the maximum version (inclusive) of the VHLO dialect an op is supported in.
mlir::vhlo::Version mlir::vhlo::VersionedOpInterface::getMaxVersion() {
      return getImpl()->getMaxVersion(getImpl(), getOperation());
  }
