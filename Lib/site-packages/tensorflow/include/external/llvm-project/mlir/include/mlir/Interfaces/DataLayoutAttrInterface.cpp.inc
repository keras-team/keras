/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Interface Definitions                                                      *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

/// Returns the attribute associated with the key.
::mlir::FailureOr<::mlir::Attribute> mlir::DLTIQueryInterface::query(::mlir::DataLayoutEntryKey key) const {
      return getImpl()->query(getImpl(), *this, key);
  }
/// Returns the key of the this layout entry.
::mlir::DataLayoutEntryKey mlir::DataLayoutEntryInterface::getKey() const {
      return getImpl()->getKey(getImpl(), *this);
  }
/// Returns the value of this layout entry.
::mlir::Attribute mlir::DataLayoutEntryInterface::getValue() const {
      return getImpl()->getValue(getImpl(), *this);
  }
/// Checks that the entry is well-formed, reports errors at the provided location.
::llvm::LogicalResult mlir::DataLayoutEntryInterface::verifyEntry(::mlir::Location loc) const {
      return getImpl()->verifyEntry(getImpl(), *this, loc);
  }
/// Combines the current layout with the given list of layouts, provided from the outermost (oldest) to the innermost (newest). Returns null on failure.
::mlir::DataLayoutSpecInterface mlir::DataLayoutSpecInterface::combineWith(::llvm::ArrayRef<::mlir::DataLayoutSpecInterface> specs) const {
      return getImpl()->combineWith(getImpl(), *this, specs);
  }
/// Returns the list of layout entries.
::mlir::DataLayoutEntryListRef mlir::DataLayoutSpecInterface::getEntries() const {
      return getImpl()->getEntries(getImpl(), *this);
  }
/// Returns the endianness identifier.
::mlir::StringAttr mlir::DataLayoutSpecInterface::getEndiannessIdentifier(::mlir::MLIRContext * context) const {
      return getImpl()->getEndiannessIdentifier(getImpl(), *this, context);
  }
/// Returns the alloca memory space identifier.
::mlir::StringAttr mlir::DataLayoutSpecInterface::getAllocaMemorySpaceIdentifier(::mlir::MLIRContext * context) const {
      return getImpl()->getAllocaMemorySpaceIdentifier(getImpl(), *this, context);
  }
/// Returns the program memory space identifier.
::mlir::StringAttr mlir::DataLayoutSpecInterface::getProgramMemorySpaceIdentifier(::mlir::MLIRContext * context) const {
      return getImpl()->getProgramMemorySpaceIdentifier(getImpl(), *this, context);
  }
/// Returns the global memory space identifier.
::mlir::StringAttr mlir::DataLayoutSpecInterface::getGlobalMemorySpaceIdentifier(::mlir::MLIRContext * context) const {
      return getImpl()->getGlobalMemorySpaceIdentifier(getImpl(), *this, context);
  }
/// Returns the stack alignment identifier.
::mlir::StringAttr mlir::DataLayoutSpecInterface::getStackAlignmentIdentifier(::mlir::MLIRContext * context) const {
      return getImpl()->getStackAlignmentIdentifier(getImpl(), *this, context);
  }
/// Returns a copy of the entries related to a specific type class regardles of type parameters.
::mlir::DataLayoutEntryList mlir::DataLayoutSpecInterface::getSpecForType(::mlir::TypeID type) const {
      return getImpl()->getSpecForType(getImpl(), *this, type);
  }
/// Returns the entry related to the given identifier, if present.
::mlir::DataLayoutEntryInterface mlir::DataLayoutSpecInterface::getSpecForIdentifier(::mlir::StringAttr identifier) const {
      return getImpl()->getSpecForIdentifier(getImpl(), *this, identifier);
  }
/// Verifies the validity of the specification and reports any errors at the given location.
::llvm::LogicalResult mlir::DataLayoutSpecInterface::verifySpec(::mlir::Location loc) const {
      return getImpl()->verifySpec(getImpl(), *this, loc);
  }
/// Returns the attribute associated with the key.
::mlir::FailureOr<::mlir::Attribute> mlir::DataLayoutSpecInterface::query(::mlir::DataLayoutEntryKey key) const {
      return getImpl()->implDLTIQueryInterface->query(getImpl()->implDLTIQueryInterface, *this, key);
  }
/// Returns the list of layout entries.
::mlir::DataLayoutEntryListRef mlir::TargetDeviceSpecInterface::getEntries() const {
      return getImpl()->getEntries(getImpl(), *this);
  }
/// Returns the entry related to the given identifier, if present.
::mlir::DataLayoutEntryInterface mlir::TargetDeviceSpecInterface::getSpecForIdentifier(::mlir::StringAttr identifier) const {
      return getImpl()->getSpecForIdentifier(getImpl(), *this, identifier);
  }
/// Checks that the entry is well-formed, reports errors at the provided location.
::llvm::LogicalResult mlir::TargetDeviceSpecInterface::verifyEntry(::mlir::Location loc) const {
      return getImpl()->verifyEntry(getImpl(), *this, loc);
  }
/// Returns the attribute associated with the key.
::mlir::FailureOr<::mlir::Attribute> mlir::TargetDeviceSpecInterface::query(::mlir::DataLayoutEntryKey key) const {
      return getImpl()->implDLTIQueryInterface->query(getImpl()->implDLTIQueryInterface, *this, key);
  }
/// Returns the list of layout entries.
llvm::ArrayRef<DeviceIDTargetDeviceSpecPair> mlir::TargetSystemSpecInterface::getEntries() const {
      return getImpl()->getEntries(getImpl(), *this);
  }
/// Returns the device description spec for given device ID
std::optional<::mlir::TargetDeviceSpecInterface> mlir::TargetSystemSpecInterface::getDeviceSpecForDeviceID(StringAttr deviceID) const {
      return getImpl()->getDeviceSpecForDeviceID(getImpl(), *this, deviceID);
  }
/// Verifies the validity of the specification and reports any errors at the given location.
::llvm::LogicalResult mlir::TargetSystemSpecInterface::verifySpec(::mlir::Location loc) const {
      return getImpl()->verifySpec(getImpl(), *this, loc);
  }
/// Returns the attribute associated with the key.
::mlir::FailureOr<::mlir::Attribute> mlir::TargetSystemSpecInterface::query(::mlir::DataLayoutEntryKey key) const {
      return getImpl()->implDLTIQueryInterface->query(getImpl()->implDLTIQueryInterface, *this, key);
  }
