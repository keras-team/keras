/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* AttrDef Definitions                                                        *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

#ifdef GET_ATTRDEF_LIST
#undef GET_ATTRDEF_LIST

::mlir::mesh::ReductionKindAttr,
::mlir::mesh::MeshAxesArrayAttr

#endif  // GET_ATTRDEF_LIST

#ifdef GET_ATTRDEF_CLASSES
#undef GET_ATTRDEF_CLASSES

static ::mlir::OptionalParseResult generatedAttributeParser(::mlir::AsmParser &parser, ::llvm::StringRef *mnemonic, ::mlir::Type type, ::mlir::Attribute &value) {
  return ::mlir::AsmParser::KeywordSwitch<::mlir::OptionalParseResult>(parser)
    .Case(::mlir::mesh::ReductionKindAttr::getMnemonic(), [&](llvm::StringRef, llvm::SMLoc) {
      value = ::mlir::mesh::ReductionKindAttr::parse(parser, type);
      return ::mlir::success(!!value);
    })
    .Case(::mlir::mesh::MeshAxesArrayAttr::getMnemonic(), [&](llvm::StringRef, llvm::SMLoc) {
      value = ::mlir::mesh::MeshAxesArrayAttr::parse(parser, type);
      return ::mlir::success(!!value);
    })
    .Default([&](llvm::StringRef keyword, llvm::SMLoc) {
      *mnemonic = keyword;
      return std::nullopt;
    });
}

static ::llvm::LogicalResult generatedAttributePrinter(::mlir::Attribute def, ::mlir::AsmPrinter &printer) {
  return ::llvm::TypeSwitch<::mlir::Attribute, ::llvm::LogicalResult>(def)    .Case<::mlir::mesh::ReductionKindAttr>([&](auto t) {
      printer << ::mlir::mesh::ReductionKindAttr::getMnemonic();
t.print(printer);
      return ::mlir::success();
    })
    .Case<::mlir::mesh::MeshAxesArrayAttr>([&](auto t) {
      printer << ::mlir::mesh::MeshAxesArrayAttr::getMnemonic();
t.print(printer);
      return ::mlir::success();
    })
    .Default([](auto) { return ::mlir::failure(); });
}

namespace mlir {
namespace mesh {
namespace detail {
struct ReductionKindAttrStorage : public ::mlir::AttributeStorage {
  using KeyTy = std::tuple<::mlir::mesh::ReductionKind>;
  ReductionKindAttrStorage(::mlir::mesh::ReductionKind value) : value(std::move(value)) {}

  KeyTy getAsKey() const {
    return KeyTy(value);
  }

  bool operator==(const KeyTy &tblgenKey) const {
    return (value == std::get<0>(tblgenKey));
  }

  static ::llvm::hash_code hashKey(const KeyTy &tblgenKey) {
    return ::llvm::hash_combine(std::get<0>(tblgenKey));
  }

  static ReductionKindAttrStorage *construct(::mlir::AttributeStorageAllocator &allocator, KeyTy &&tblgenKey) {
    auto value = std::move(std::get<0>(tblgenKey));
    return new (allocator.allocate<ReductionKindAttrStorage>()) ReductionKindAttrStorage(std::move(value));
  }

  ::mlir::mesh::ReductionKind value;
};
} // namespace detail
ReductionKindAttr ReductionKindAttr::get(::mlir::MLIRContext *context, ::mlir::mesh::ReductionKind value) {
  return Base::get(context, std::move(value));
}

::mlir::Attribute ReductionKindAttr::parse(::mlir::AsmParser &odsParser, ::mlir::Type odsType) {
  ::mlir::Builder odsBuilder(odsParser.getContext());
  ::llvm::SMLoc odsLoc = odsParser.getCurrentLocation();
  (void) odsLoc;
  ::mlir::FailureOr<::mlir::mesh::ReductionKind> _result_value;

  // Parse variable 'value'
  _result_value = [&]() -> ::mlir::FailureOr<::mlir::mesh::ReductionKind> {
      auto loc = odsParser.getCurrentLocation();
      ::llvm::StringRef enumKeyword;
      if (::mlir::failed(odsParser.parseKeyword(&enumKeyword)))
        return ::mlir::failure();
      auto maybeEnum = ::mlir::mesh::symbolizeReductionKind(enumKeyword);
      if (maybeEnum)
        return *maybeEnum;
      return {(::llvm::LogicalResult)(odsParser.emitError(loc) << "expected " << "::mlir::mesh::ReductionKind" << " to be one of: " << "sum" << ", " << "max" << ", " << "min" << ", " << "product" << ", " << "average" << ", " << "bitwise_and" << ", " << "bitwise_or" << ", " << "bitwise_xor" << ", " << "generic")};
    }();
  if (::mlir::failed(_result_value)) {
    odsParser.emitError(odsParser.getCurrentLocation(), "failed to parse Mesh_ReductionKindAttr parameter 'value' which is to be a `::mlir::mesh::ReductionKind`");
    return {};
  }
  assert(::mlir::succeeded(_result_value));
  return ReductionKindAttr::get(odsParser.getContext(),
      ::mlir::mesh::ReductionKind((*_result_value)));
}

void ReductionKindAttr::print(::mlir::AsmPrinter &odsPrinter) const {
  ::mlir::Builder odsBuilder(getContext());
  odsPrinter << ' ';
  odsPrinter << stringifyReductionKind(getValue());
}

::mlir::mesh::ReductionKind ReductionKindAttr::getValue() const {
  return getImpl()->value;
}

} // namespace mesh
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::mesh::ReductionKindAttr)
namespace mlir {
namespace mesh {
namespace detail {
struct MeshAxesArrayAttrStorage : public ::mlir::AttributeStorage {
  using KeyTy = std::tuple<::llvm::ArrayRef<MeshAxesAttr>>;
  MeshAxesArrayAttrStorage(::llvm::ArrayRef<MeshAxesAttr> axes) : axes(std::move(axes)) {}

  KeyTy getAsKey() const {
    return KeyTy(axes);
  }

  bool operator==(const KeyTy &tblgenKey) const {
    return (axes == std::get<0>(tblgenKey));
  }

  static ::llvm::hash_code hashKey(const KeyTy &tblgenKey) {
    return ::llvm::hash_combine(std::get<0>(tblgenKey));
  }

  static MeshAxesArrayAttrStorage *construct(::mlir::AttributeStorageAllocator &allocator, KeyTy &&tblgenKey) {
    auto axes = std::move(std::get<0>(tblgenKey));
    axes = allocator.copyInto(axes);
    return new (allocator.allocate<MeshAxesArrayAttrStorage>()) MeshAxesArrayAttrStorage(std::move(axes));
  }

  ::llvm::ArrayRef<MeshAxesAttr> axes;
};
} // namespace detail
MeshAxesArrayAttr MeshAxesArrayAttr::get(::mlir::MLIRContext *context, ::llvm::ArrayRef<MeshAxesAttr> axes) {
  return Base::get(context, std::move(axes));
}

::mlir::Attribute MeshAxesArrayAttr::parse(::mlir::AsmParser &odsParser, ::mlir::Type odsType) {
  ::mlir::Builder odsBuilder(odsParser.getContext());
  ::llvm::SMLoc odsLoc = odsParser.getCurrentLocation();
  (void) odsLoc;
  ::mlir::FailureOr<::llvm::SmallVector<MeshAxesAttr>> _result_axes;
  // Parse literal '['
  if (odsParser.parseLSquare()) return {};

  // Parse variable 'axes'
  _result_axes = ::mlir::FieldParser<::llvm::SmallVector<MeshAxesAttr>>::parse(odsParser);
  if (::mlir::failed(_result_axes)) {
    odsParser.emitError(odsParser.getCurrentLocation(), "failed to parse Mesh_MeshAxesArrayAttr parameter 'axes' which is to be a `::llvm::ArrayRef<MeshAxesAttr>`");
    return {};
  }
  // Parse literal ']'
  if (odsParser.parseRSquare()) return {};
  assert(::mlir::succeeded(_result_axes));
  return MeshAxesArrayAttr::get(odsParser.getContext(),
      ::llvm::ArrayRef<MeshAxesAttr>((*_result_axes)));
}

void MeshAxesArrayAttr::print(::mlir::AsmPrinter &odsPrinter) const {
  ::mlir::Builder odsBuilder(getContext());
  odsPrinter << "[";
  odsPrinter.printStrippedAttrOrType(getAxes());
  odsPrinter << "]";
}

::llvm::ArrayRef<MeshAxesAttr> MeshAxesArrayAttr::getAxes() const {
  return getImpl()->axes;
}

} // namespace mesh
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::mesh::MeshAxesArrayAttr)
namespace mlir {
namespace mesh {

/// Parse an attribute registered to this dialect.
::mlir::Attribute MeshDialect::parseAttribute(::mlir::DialectAsmParser &parser,
                                      ::mlir::Type type) const {
  ::llvm::SMLoc typeLoc = parser.getCurrentLocation();
  ::llvm::StringRef attrTag;
  {
    ::mlir::Attribute attr;
    auto parseResult = generatedAttributeParser(parser, &attrTag, type, attr);
    if (parseResult.has_value())
      return attr;
  }
  
  parser.emitError(typeLoc) << "unknown attribute `"
      << attrTag << "` in dialect `" << getNamespace() << "`";
  return {};
}
/// Print an attribute registered to this dialect.
void MeshDialect::printAttribute(::mlir::Attribute attr,
                         ::mlir::DialectAsmPrinter &printer) const {
  if (::mlir::succeeded(generatedAttributePrinter(attr, printer)))
    return;
  
}
} // namespace mesh
} // namespace mlir

#endif  // GET_ATTRDEF_CLASSES

