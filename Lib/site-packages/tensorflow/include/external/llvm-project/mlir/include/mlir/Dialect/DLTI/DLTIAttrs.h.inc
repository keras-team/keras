/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* AttrDef Declarations                                                       *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

#ifdef GET_ATTRDEF_CLASSES
#undef GET_ATTRDEF_CLASSES


namespace mlir {
class AsmParser;
class AsmPrinter;
} // namespace mlir
namespace mlir {
class DataLayoutEntryAttr;
class DataLayoutSpecAttr;
class MapAttr;
class TargetSystemSpecAttr;
class TargetDeviceSpecAttr;
class DataLayoutEntryAttr : public ::mlir::Attribute::AttrBase<DataLayoutEntryAttr, ::mlir::Attribute, detail::DataLayoutEntryAttrStorage, ::mlir::DataLayoutEntryInterface::Trait> {
public:
  using Base::Base;
  /// Returns the entry with the given key and value.
  static DataLayoutEntryAttr get(StringAttr key, Attribute value);
  static DataLayoutEntryAttr get(MLIRContext *context, Type key, Attribute value);
  static DataLayoutEntryAttr get(Type key, Attribute value);
  static constexpr ::llvm::StringLiteral name = "dlti.dl_entry";
  static constexpr ::llvm::StringLiteral dialectName = "dlti";
  static DataLayoutEntryAttr get(::mlir::MLIRContext *context, DataLayoutEntryKey key, Attribute value);
  static constexpr ::llvm::StringLiteral getMnemonic() {
    return {"dl_entry"};
  }

  static ::mlir::Attribute parse(::mlir::AsmParser &odsParser, ::mlir::Type odsType);
  void print(::mlir::AsmPrinter &odsPrinter) const;
  DataLayoutEntryKey getKey() const;
  Attribute getValue() const;
};
namespace detail {
struct DataLayoutSpecAttrStorage;
} // namespace detail
class DataLayoutSpecAttr : public ::mlir::Attribute::AttrBase<DataLayoutSpecAttr, ::mlir::Attribute, detail::DataLayoutSpecAttrStorage, ::mlir::DLTIQueryInterface::Trait, ::mlir::DataLayoutSpecInterface::Trait> {
public:
  using Base::Base;
  /// Combines this specification with `specs`, enclosing specifications listed
  /// from outermost to innermost. This overwrites the older entries with the
  /// same key as the newer entries if the entries are compatible. Returns null
  /// if the specifications are not compatible.
  DataLayoutSpecAttr combineWith(ArrayRef<DataLayoutSpecInterface> specs) const;

  /// Returns the endiannes identifier.
  StringAttr getEndiannessIdentifier(MLIRContext *context) const;

  /// Returns the alloca memory space identifier.
  StringAttr getAllocaMemorySpaceIdentifier(MLIRContext *context) const;

  /// Returns the program memory space identifier.
  StringAttr getProgramMemorySpaceIdentifier(MLIRContext *context) const;

  /// Returns the global memory space identifier.
  StringAttr getGlobalMemorySpaceIdentifier(MLIRContext *context) const;

  /// Returns the stack alignment identifier.
  StringAttr getStackAlignmentIdentifier(MLIRContext *context) const;

  /// Returns the attribute associated with the key.
  FailureOr<Attribute> query(DataLayoutEntryKey key) {
    return llvm::cast<mlir::DataLayoutSpecInterface>(*this).queryHelper(key);
  }
  static constexpr ::llvm::StringLiteral name = "dlti.dl_spec";
  static constexpr ::llvm::StringLiteral dialectName = "dlti";
  using Base::getChecked;
  static DataLayoutSpecAttr get(::mlir::MLIRContext *context, ::llvm::ArrayRef<DataLayoutEntryInterface> entries);
  static DataLayoutSpecAttr getChecked(::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError, ::mlir::MLIRContext *context, ::llvm::ArrayRef<DataLayoutEntryInterface> entries);
  static ::llvm::LogicalResult verify(::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError, ::llvm::ArrayRef<DataLayoutEntryInterface> entries);
  static ::llvm::LogicalResult verifyInvariants(::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError, ::llvm::ArrayRef<DataLayoutEntryInterface> entries);
  static constexpr ::llvm::StringLiteral getMnemonic() {
    return {"dl_spec"};
  }

  static ::mlir::Attribute parse(::mlir::AsmParser &odsParser, ::mlir::Type odsType);
  void print(::mlir::AsmPrinter &odsPrinter) const;
  ::llvm::ArrayRef<DataLayoutEntryInterface> getEntries() const;
};
namespace detail {
struct MapAttrStorage;
} // namespace detail
class MapAttr : public ::mlir::Attribute::AttrBase<MapAttr, ::mlir::Attribute, detail::MapAttrStorage, ::mlir::DLTIQueryInterface::Trait> {
public:
  using Base::Base;
  /// Returns the attribute associated with the key.
  FailureOr<Attribute> query(DataLayoutEntryKey key) {
    for (DataLayoutEntryInterface entry : getEntries())
      if (entry.getKey() == key)
          return entry.getValue();
    return ::mlir::failure();
  }
  static constexpr ::llvm::StringLiteral name = "dlti.map";
  static constexpr ::llvm::StringLiteral dialectName = "dlti";
  using Base::getChecked;
  static MapAttr get(::mlir::MLIRContext *context, ::llvm::ArrayRef<DataLayoutEntryInterface> entries);
  static MapAttr getChecked(::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError, ::mlir::MLIRContext *context, ::llvm::ArrayRef<DataLayoutEntryInterface> entries);
  static ::llvm::LogicalResult verify(::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError, ::llvm::ArrayRef<DataLayoutEntryInterface> entries);
  static ::llvm::LogicalResult verifyInvariants(::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError, ::llvm::ArrayRef<DataLayoutEntryInterface> entries);
  static constexpr ::llvm::StringLiteral getMnemonic() {
    return {"map"};
  }

  static ::mlir::Attribute parse(::mlir::AsmParser &odsParser, ::mlir::Type odsType);
  void print(::mlir::AsmPrinter &odsPrinter) const;
  ::llvm::ArrayRef<DataLayoutEntryInterface> getEntries() const;
};
namespace detail {
struct TargetSystemSpecAttrStorage;
} // namespace detail
class TargetSystemSpecAttr : public ::mlir::Attribute::AttrBase<TargetSystemSpecAttr, ::mlir::Attribute, detail::TargetSystemSpecAttrStorage, ::mlir::DLTIQueryInterface::Trait, ::mlir::TargetSystemSpecInterface::Trait> {
public:
  using Base::Base;
  /// Return the device specification that matches the given device ID
  std::optional<TargetDeviceSpecInterface>
  getDeviceSpecForDeviceID(
    TargetSystemSpecInterface::DeviceID deviceID);

  /// Returns the attribute associated with the key.
  FailureOr<Attribute> query(DataLayoutEntryKey key) const {
    return llvm::cast<mlir::TargetSystemSpecInterface>(*this).queryHelper(key);
  }
  static constexpr ::llvm::StringLiteral name = "dlti.target_system_spec";
  static constexpr ::llvm::StringLiteral dialectName = "dlti";
  using Base::getChecked;
  static TargetSystemSpecAttr get(::mlir::MLIRContext *context, ::llvm::ArrayRef<DeviceIDTargetDeviceSpecPair> entries);
  static TargetSystemSpecAttr getChecked(::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError, ::mlir::MLIRContext *context, ::llvm::ArrayRef<DeviceIDTargetDeviceSpecPair> entries);
  static ::llvm::LogicalResult verify(::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError, ::llvm::ArrayRef<DeviceIDTargetDeviceSpecPair> entries);
  static ::llvm::LogicalResult verifyInvariants(::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError, ::llvm::ArrayRef<DeviceIDTargetDeviceSpecPair> entries);
  static constexpr ::llvm::StringLiteral getMnemonic() {
    return {"target_system_spec"};
  }

  static ::mlir::Attribute parse(::mlir::AsmParser &odsParser, ::mlir::Type odsType);
  void print(::mlir::AsmPrinter &odsPrinter) const;
  ::llvm::ArrayRef<DeviceIDTargetDeviceSpecPair> getEntries() const;
};
namespace detail {
struct TargetDeviceSpecAttrStorage;
} // namespace detail
class TargetDeviceSpecAttr : public ::mlir::Attribute::AttrBase<TargetDeviceSpecAttr, ::mlir::Attribute, detail::TargetDeviceSpecAttrStorage, ::mlir::DLTIQueryInterface::Trait, ::mlir::TargetDeviceSpecInterface::Trait> {
public:
  using Base::Base;
  /// Returns the attribute associated with the key.
  FailureOr<Attribute> query(DataLayoutEntryKey key) const {
    return llvm::cast<mlir::TargetDeviceSpecInterface>(*this).queryHelper(key);
  }
  static constexpr ::llvm::StringLiteral name = "dlti.target_device_spec";
  static constexpr ::llvm::StringLiteral dialectName = "dlti";
  using Base::getChecked;
  static TargetDeviceSpecAttr get(::mlir::MLIRContext *context, ::llvm::ArrayRef<DataLayoutEntryInterface> entries);
  static TargetDeviceSpecAttr getChecked(::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError, ::mlir::MLIRContext *context, ::llvm::ArrayRef<DataLayoutEntryInterface> entries);
  static ::llvm::LogicalResult verify(::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError, ::llvm::ArrayRef<DataLayoutEntryInterface> entries);
  static ::llvm::LogicalResult verifyInvariants(::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError, ::llvm::ArrayRef<DataLayoutEntryInterface> entries);
  static constexpr ::llvm::StringLiteral getMnemonic() {
    return {"target_device_spec"};
  }

  static ::mlir::Attribute parse(::mlir::AsmParser &odsParser, ::mlir::Type odsType);
  void print(::mlir::AsmPrinter &odsPrinter) const;
  ::llvm::ArrayRef<DataLayoutEntryInterface> getEntries() const;
};
} // namespace mlir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::DataLayoutEntryAttr)
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::DataLayoutSpecAttr)
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::MapAttr)
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::TargetSystemSpecAttr)
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::TargetDeviceSpecAttr)

#endif  // GET_ATTRDEF_CLASSES

