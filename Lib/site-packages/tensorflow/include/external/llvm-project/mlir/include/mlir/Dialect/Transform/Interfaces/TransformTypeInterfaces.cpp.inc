/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Interface Definitions                                                      *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

/// Checks if the given associated objects (Payload IR operations or attributes)
/// satisfy the conditions defined by this type. If not, produces a silenceable
/// error at the specified location.
::mlir::DiagnosedSilenceableFailure mlir::transform::TransformHandleTypeInterface::checkPayload(::mlir::Location loc, ::mlir::ArrayRef<::mlir::Operation *> payload) const {
      return getImpl()->checkPayload(getImpl(), *this, loc, payload);
  }
/// Checks if the given associated objects (Payload IR operations or attributes)
/// satisfy the conditions defined by this type. If not, produces a silenceable
/// error at the specified location.
::mlir::DiagnosedSilenceableFailure mlir::transform::TransformParamTypeInterface::checkPayload(::mlir::Location loc, ::mlir::ArrayRef<::mlir::Attribute> payload) const {
      return getImpl()->checkPayload(getImpl(), *this, loc, payload);
  }
/// Checks if the given associated objects (Payload IR operations or attributes)
/// satisfy the conditions defined by this type. If not, produces a silenceable
/// error at the specified location.
::mlir::DiagnosedSilenceableFailure mlir::transform::TransformValueHandleTypeInterface::checkPayload(::mlir::Location loc, ::mlir::ArrayRef<::mlir::Value> payload) const {
      return getImpl()->checkPayload(getImpl(), *this, loc, payload);
  }
