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
class CallSiteLoc;
class FileLineColLoc;
class FusedLoc;
class NameLoc;
class OpaqueLoc;
class UnknownLoc;
namespace detail {
struct CallSiteLocAttrStorage;
} // namespace detail
class CallSiteLoc : public ::mlir::Attribute::AttrBase<CallSiteLoc, ::mlir::LocationAttr, detail::CallSiteLocAttrStorage> {
public:
  using Base::Base;
  static constexpr ::llvm::StringLiteral name = "builtin.call_site_loc";
  static constexpr ::llvm::StringLiteral dialectName = "builtin";
  static CallSiteLoc get(Location callee, Location caller);
  static CallSiteLoc get(Location name, ArrayRef<Location> frames);
  Location getCallee() const;
  Location getCaller() const;
};
namespace detail {
struct FileLineColLocAttrStorage;
} // namespace detail
class FileLineColLoc : public ::mlir::Attribute::AttrBase<FileLineColLoc, ::mlir::LocationAttr, detail::FileLineColLocAttrStorage> {
public:
  using Base::Base;
  static constexpr ::llvm::StringLiteral name = "builtin.file_line_loc";
  static constexpr ::llvm::StringLiteral dialectName = "builtin";
  static FileLineColLoc get(StringAttr filename, unsigned line, unsigned column);
  static FileLineColLoc get(::mlir::MLIRContext *context, StringRef filename, unsigned line, unsigned column);
  StringAttr getFilename() const;
  unsigned getLine() const;
  unsigned getColumn() const;
};
namespace detail {
struct FusedLocAttrStorage;
} // namespace detail
class FusedLoc : public ::mlir::Attribute::AttrBase<FusedLoc, ::mlir::LocationAttr, detail::FusedLocAttrStorage> {
public:
  using Base::Base;
  static Location get(ArrayRef<Location> locs, Attribute metadata,
                      MLIRContext *context);
  static Location get(MLIRContext *context, ArrayRef<Location> locs) {
    return get(locs, Attribute(), context);
  }
  static constexpr ::llvm::StringLiteral name = "builtin.fused_loc";
  static constexpr ::llvm::StringLiteral dialectName = "builtin";
  static FusedLoc get(::mlir::MLIRContext *context, ::llvm::ArrayRef<Location> locations, Attribute metadata);
  ::llvm::ArrayRef<Location> getLocations() const;
  Attribute getMetadata() const;
};
namespace detail {
struct NameLocAttrStorage;
} // namespace detail
class NameLoc : public ::mlir::Attribute::AttrBase<NameLoc, ::mlir::LocationAttr, detail::NameLocAttrStorage> {
public:
  using Base::Base;
  static constexpr ::llvm::StringLiteral name = "builtin.name_loc";
  static constexpr ::llvm::StringLiteral dialectName = "builtin";
  static NameLoc get(StringAttr name, Location childLoc);
  static NameLoc get(StringAttr name);
  StringAttr getName() const;
  Location getChildLoc() const;
};
namespace detail {
struct OpaqueLocAttrStorage;
} // namespace detail
class OpaqueLoc : public ::mlir::Attribute::AttrBase<OpaqueLoc, ::mlir::LocationAttr, detail::OpaqueLocAttrStorage> {
public:
  using Base::Base;
  /// Returns an instance of opaque location which contains a given pointer to
  /// an object. The corresponding MLIR location is set to UnknownLoc.
  template <typename T>
  static OpaqueLoc get(T underlyingLocation, MLIRContext *context);

  /// Returns an instance of opaque location which contains a given pointer to
  /// an object and an additional MLIR location.
  template <typename T>
  static OpaqueLoc get(T underlyingLocation, Location fallbackLocation) {
    return get(reinterpret_cast<uintptr_t>(underlyingLocation),
               TypeID::get<T>(), fallbackLocation);
  }

  /// Returns a pointer to some data structure that opaque location stores.
  template <typename T> static T getUnderlyingLocation(Location location) {
    assert(isa<T>(location));
    return reinterpret_cast<T>(
        mlir::cast<mlir::OpaqueLoc>(static_cast<LocationAttr>(location))
            .getUnderlyingLocation());
  }

  /// Returns a pointer to some data structure that opaque location stores.
  /// Returns nullptr if provided location is not opaque location or if it
  /// contains a pointer of different type.
  template <typename T>
  static T getUnderlyingLocationOrNull(Location location) {
    return isa<T>(location)
              ? reinterpret_cast<T>(mlir::cast<mlir::OpaqueLoc>(
                                        static_cast<LocationAttr>(location))
                                        .getUnderlyingLocation())
              : T(nullptr);
  }

  /// Checks whether provided location is opaque location and contains a
  /// pointer to an object of particular type.
  template <typename T> static bool isa(Location location) {
    auto opaque_loc =
        mlir::dyn_cast<OpaqueLoc>(static_cast<LocationAttr>(location));
    return opaque_loc && opaque_loc.getUnderlyingTypeID() == TypeID::get<T>();
  }
  static constexpr ::llvm::StringLiteral name = "builtin.opaque_loc";
  static constexpr ::llvm::StringLiteral dialectName = "builtin";
  static OpaqueLoc get(uintptr_t underlyingLocation, TypeID underlyingTypeID, Location fallbackLocation);
  uintptr_t getUnderlyingLocation() const;
  TypeID getUnderlyingTypeID() const;
  Location getFallbackLocation() const;
};
class UnknownLoc : public ::mlir::Attribute::AttrBase<UnknownLoc, ::mlir::LocationAttr, ::mlir::AttributeStorage> {
public:
  using Base::Base;
  static UnknownLoc get(MLIRContext *context);
  static constexpr ::llvm::StringLiteral name = "builtin.unknown_loc";
  static constexpr ::llvm::StringLiteral dialectName = "builtin";
};
} // namespace mlir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::CallSiteLoc)
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::FileLineColLoc)
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::FusedLoc)
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::NameLoc)
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::OpaqueLoc)
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::UnknownLoc)

#endif  // GET_ATTRDEF_CLASSES

