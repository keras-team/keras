/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Interface Definitions                                                      *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

/// Returns the callee of this call-like operation. A `callee` is either a
/// reference to a symbol, via SymbolRefAttr, or a reference to a defined
/// SSA value. If the reference is an SSA value, the SSA value corresponds
/// to a region of a lambda-like operation.
::mlir::CallInterfaceCallable mlir::CallOpInterface::getCallableForCallee() {
      return getImpl()->getCallableForCallee(getImpl(), getOperation());
  }
/// Sets the callee of this call-like operation. A `callee` is either a
/// reference to a symbol, via SymbolRefAttr, or a reference to a defined
/// SSA value. The type of the `callee` is expected to be the same as the
/// return type of `getCallableForCallee`, e.g., `callee` should be
/// SymbolRefAttr for `func.call`.
void mlir::CallOpInterface::setCalleeFromCallable(::mlir::CallInterfaceCallable callee) {
      return getImpl()->setCalleeFromCallable(getImpl(), getOperation(), callee);
  }
/// Returns the operands within this call that are used as arguments to the
/// callee.
::mlir::Operation::operand_range mlir::CallOpInterface::getArgOperands() {
      return getImpl()->getArgOperands(getImpl(), getOperation());
  }
/// Returns the operands within this call that are used as arguments to the
/// callee as a mutable range.
::mlir::MutableOperandRange mlir::CallOpInterface::getArgOperandsMutable() {
      return getImpl()->getArgOperandsMutable(getImpl(), getOperation());
  }
/// Resolve the callable operation for given callee to a
/// CallableOpInterface, or nullptr if a valid callable was not resolved.
/// `symbolTable` parameter allow for using a cached symbol table for symbol
/// lookups instead of performing an O(N) scan.
::mlir::Operation *mlir::CallOpInterface::resolveCallableInTable(::mlir::SymbolTableCollection * symbolTable) {
      return getImpl()->resolveCallableInTable(getImpl(), getOperation(), symbolTable);
  }
/// Resolve the callable operation for given callee to a
/// CallableOpInterface, or nullptr if a valid callable was not resolved.
::mlir::Operation *mlir::CallOpInterface::resolveCallable() {
      return getImpl()->resolveCallable(getImpl(), getOperation());
  }
/// Returns the region on the current operation that is callable. This may
/// return null in the case of an external callable object, e.g. an external
/// function.
::mlir::Region *mlir::CallableOpInterface::getCallableRegion() {
      return getImpl()->getCallableRegion(getImpl(), getOperation());
  }
/// Returns the callable's argument types based exclusively on the type (to
/// allow for this method may be called on function declarations).
::llvm::ArrayRef<::mlir::Type> mlir::CallableOpInterface::getArgumentTypes() {
      return getImpl()->getArgumentTypes(getImpl(), getOperation());
  }
/// Returns the callable's result types based exclusively on the type (to
/// allow for this method may be called on function declarations).
::llvm::ArrayRef<::mlir::Type> mlir::CallableOpInterface::getResultTypes() {
      return getImpl()->getResultTypes(getImpl(), getOperation());
  }
/// Get the array of argument attribute dictionaries. The method should
/// return an array attribute containing only dictionary attributes equal in
/// number to the number of region arguments. Alternatively, the method can
/// return null to indicate that the region has no argument attributes.
::mlir::ArrayAttr mlir::CallableOpInterface::getArgAttrsAttr() {
      return getImpl()->getArgAttrsAttr(getImpl(), getOperation());
  }
/// Get the array of result attribute dictionaries. The method should return
/// an array attribute containing only dictionary attributes equal in number
/// to the number of region results. Alternatively, the method can return
/// null to indicate that the region has no result attributes.
::mlir::ArrayAttr mlir::CallableOpInterface::getResAttrsAttr() {
      return getImpl()->getResAttrsAttr(getImpl(), getOperation());
  }
/// Set the array of argument attribute dictionaries.
void mlir::CallableOpInterface::setArgAttrsAttr(::mlir::ArrayAttr attrs) {
      return getImpl()->setArgAttrsAttr(getImpl(), getOperation(), attrs);
  }
/// Set the array of result attribute dictionaries.
void mlir::CallableOpInterface::setResAttrsAttr(::mlir::ArrayAttr attrs) {
      return getImpl()->setResAttrsAttr(getImpl(), getOperation(), attrs);
  }
/// Remove the array of argument attribute dictionaries. This is the same as
/// setting all argument attributes to an empty dictionary. The method should
/// return the removed attribute.
::mlir::Attribute mlir::CallableOpInterface::removeArgAttrsAttr() {
      return getImpl()->removeArgAttrsAttr(getImpl(), getOperation());
  }
/// Remove the array of result attribute dictionaries. This is the same as
/// setting all result attributes to an empty dictionary. The method should
/// return the removed attribute.
::mlir::Attribute mlir::CallableOpInterface::removeResAttrsAttr() {
      return getImpl()->removeResAttrsAttr(getImpl(), getOperation());
  }
