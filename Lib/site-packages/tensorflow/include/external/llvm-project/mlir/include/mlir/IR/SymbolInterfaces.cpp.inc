/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Interface Definitions                                                      *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

/// Returns the name of this symbol.
::mlir::StringAttr mlir::SymbolOpInterface::getNameAttr() {
      return getImpl()->getNameAttr(getImpl(), getOperation());
  }
/// Sets the name of this symbol.
void mlir::SymbolOpInterface::setName(::mlir::StringAttr name) {
      return getImpl()->setName(getImpl(), getOperation(), name);
  }
/// Gets the visibility of this symbol.
mlir::SymbolTable::Visibility mlir::SymbolOpInterface::getVisibility() {
      return getImpl()->getVisibility(getImpl(), getOperation());
  }
/// Returns true if this symbol has nested visibility.
bool mlir::SymbolOpInterface::isNested() {
      return getImpl()->isNested(getImpl(), getOperation());
  }
/// Returns true if this symbol has private visibility.
bool mlir::SymbolOpInterface::isPrivate() {
      return getImpl()->isPrivate(getImpl(), getOperation());
  }
/// Returns true if this symbol has public visibility.
bool mlir::SymbolOpInterface::isPublic() {
      return getImpl()->isPublic(getImpl(), getOperation());
  }
/// Sets the visibility of this symbol.
void mlir::SymbolOpInterface::setVisibility(mlir::SymbolTable::Visibility vis) {
      return getImpl()->setVisibility(getImpl(), getOperation(), vis);
  }
/// Sets the visibility of this symbol to be nested.
void mlir::SymbolOpInterface::setNested() {
      return getImpl()->setNested(getImpl(), getOperation());
  }
/// Sets the visibility of this symbol to be private.
void mlir::SymbolOpInterface::setPrivate() {
      return getImpl()->setPrivate(getImpl(), getOperation());
  }
/// Sets the visibility of this symbol to be public.
void mlir::SymbolOpInterface::setPublic() {
      return getImpl()->setPublic(getImpl(), getOperation());
  }
/// Get all of the uses of the current symbol that are nested within the
/// given operation 'from'.
/// Note: See mlir::SymbolTable::getSymbolUses for more details.
::std::optional<::mlir::SymbolTable::UseRange> mlir::SymbolOpInterface::getSymbolUses(::mlir::Operation * from) {
      return getImpl()->getSymbolUses(getImpl(), getOperation(), from);
  }
/// Return if the current symbol is known to have no uses that are nested
/// within the given operation 'from'.
/// Note: See mlir::SymbolTable::symbolKnownUseEmpty for more details.
bool mlir::SymbolOpInterface::symbolKnownUseEmpty(::mlir::Operation * from) {
      return getImpl()->symbolKnownUseEmpty(getImpl(), getOperation(), from);
  }
/// Attempt to replace all uses of the current symbol with the provided
/// symbol 'newSymbol' that are nested within the given operation 'from'.
/// Note: See mlir::SymbolTable::replaceAllSymbolUses for more details.
::llvm::LogicalResult mlir::SymbolOpInterface::replaceAllSymbolUses(::mlir::StringAttr newSymbol, ::mlir::Operation * from) {
      return getImpl()->replaceAllSymbolUses(getImpl(), getOperation(), newSymbol, from);
  }
/// Returns true if this operation optionally defines a symbol based on the
/// presence of the symbol name.
bool mlir::SymbolOpInterface::isOptionalSymbol() {
      return getImpl()->isOptionalSymbol(getImpl(), getOperation());
  }
/// Returns true if this operation can be discarded if it has no remaining
/// symbol uses.
bool mlir::SymbolOpInterface::canDiscardOnUseEmpty() {
      return getImpl()->canDiscardOnUseEmpty(getImpl(), getOperation());
  }
/// Returns true if this operation is a declaration of a symbol (as opposed
/// to a definition).
bool mlir::SymbolOpInterface::isDeclaration() {
      return getImpl()->isDeclaration(getImpl(), getOperation());
  }
/// Verify the symbol uses held by this operation.
::llvm::LogicalResult mlir::SymbolUserOpInterface::verifySymbolUses(::mlir::SymbolTableCollection & symbolTable) {
      return getImpl()->verifySymbolUses(getImpl(), getOperation(), symbolTable);
  }
