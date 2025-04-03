static ::mlir::Attribute readArrayAttr(MLIRContext* context, DialectBytecodeReader &reader) {
  SmallVector<Attribute> value;
  if (succeeded(reader.readAttributes(value))) {
    return get<ArrayAttr>(context, value);
  }
  return ArrayAttr();
}

static void write(ArrayAttr attribute, DialectBytecodeWriter &writer) {
  writer.writeVarInt(/* ArrayAttr */ 0);
  writer.writeAttributes(attribute.getValue());
}

static ::mlir::Attribute readCallSiteLoc(MLIRContext* context, DialectBytecodeReader &reader) {
  LocationAttr callee, caller;
  if (succeeded(reader.readAttribute(callee)) &&
      succeeded(reader.readAttribute(caller))) {
    return get<CallSiteLoc>(context, callee, caller);
  }
  return CallSiteLoc();
}

static void write(CallSiteLoc attribute, DialectBytecodeWriter &writer) {
  writer.writeVarInt(/* CallSiteLoc */ 10);
  writer.writeAttribute(attribute.getCallee());
  writer.writeAttribute(attribute.getCaller());
}

static ::mlir::Attribute readDenseArrayAttr(MLIRContext* context, DialectBytecodeReader &reader) {
  Type elementType;
  uint64_t size;
  ArrayRef<char> rawData;
  if (succeeded(reader.readType(elementType)) &&
      succeeded(reader.readVarInt(size)) &&
      succeeded(reader.readBlob(rawData))) {
    return get<DenseArrayAttr>(context, elementType, size, rawData);
  }
  return DenseArrayAttr();
}

static void write(DenseArrayAttr attribute, DialectBytecodeWriter &writer) {
  writer.writeVarInt(/* DenseArrayAttr */ 17);
  writer.writeType(attribute.getElementType());
  writer.writeVarInt(attribute.getSize());
  writer.writeOwnedBlob(attribute.getRawData());
}

static ::mlir::Attribute readDenseIntOrFPElementsAttr(MLIRContext* context, DialectBytecodeReader &reader) {
  ShapedType type;
  ArrayRef<char> rawData;
  if (succeeded(reader.readType(type)) &&
      succeeded(reader.readBlob(rawData))) {
    return cast<DenseIntOrFPElementsAttr>(DenseIntOrFPElementsAttr::getFromRawBuffer(type, rawData));
  }
  return DenseIntOrFPElementsAttr();
}

static void write(DenseIntOrFPElementsAttr attribute, DialectBytecodeWriter &writer) {
  writer.writeVarInt(/* DenseIntOrFPElementsAttr */ 18);
  writer.writeType(attribute.getType());
  writer.writeOwnedBlob(attribute.getRawData());
}

static ::mlir::Attribute readDenseResourceElementsAttr(MLIRContext* context, DialectBytecodeReader &reader) {
  ShapedType type;
  FailureOr<DenseResourceElementsHandle> rawHandle;
  if (succeeded(reader.readType(type)) &&
      succeeded(readResourceHandle<DenseResourceElementsHandle>(reader, rawHandle))) {
    return get<DenseResourceElementsAttr>(context, type, *rawHandle);
  }
  return DenseResourceElementsAttr();
}

static void write(DenseResourceElementsAttr attribute, DialectBytecodeWriter &writer) {
  writer.writeVarInt(/* DenseResourceElementsAttr */ 16);
  writer.writeType(attribute.getType());
  writer.writeResourceHandle(attribute.getRawHandle());
}

static ::mlir::Attribute readDenseStringElementsAttr(MLIRContext* context, DialectBytecodeReader &reader) {
  ShapedType type;
  uint64_t _isSplat;
  SmallVector<StringRef> rawStringData;
  if (succeeded(reader.readType(type)) &&
      succeeded(reader.readVarInt(_isSplat)) &&
      succeeded(readPotentiallySplatString(reader, type, _isSplat, rawStringData))) {
    return get<DenseStringElementsAttr>(context, type, rawStringData);
  }
  return DenseStringElementsAttr();
}

static void write(DenseStringElementsAttr attribute, DialectBytecodeWriter &writer) {
  writer.writeVarInt(/* DenseStringElementsAttr */ 19);
  writer.writeType(attribute.getType());
  writer.writeVarInt(attribute.isSplat());
  writePotentiallySplatString(writer, attribute);
}

static ::mlir::Attribute readDictionaryAttr(MLIRContext* context, DialectBytecodeReader &reader) {
  SmallVector<NamedAttribute> value;
  auto readValue = [&]() -> FailureOr<NamedAttribute> {
    StringAttr name;
    Attribute value;
    if (succeeded(reader.readAttribute<StringAttr>(name)) &&
        succeeded(reader.readAttribute(value))) {
      return NamedAttribute(name, value);
    }
    return failure();
  };
  if (succeeded(reader.readList(value, readValue))) {
    return get<DictionaryAttr>(context, value);
  }
  return DictionaryAttr();
}

static void write(DictionaryAttr attribute, DialectBytecodeWriter &writer) {
  writer.writeVarInt(/* DictionaryAttr */ 1);
  writer.writeList(attribute.getValue(), [&](NamedAttribute attribute) {
    writer.writeAttribute(attribute.getName());
    writer.writeAttribute(attribute.getValue());
  });
}

static ::mlir::Attribute readDistinctAttr(MLIRContext* context, DialectBytecodeReader &reader) {
  Attribute referencedAttr;
  if (succeeded(reader.readAttribute(referencedAttr))) {
    return get<DistinctAttr>(context, referencedAttr);
  }
  return DistinctAttr();
}

static void write(DistinctAttr attribute, DialectBytecodeWriter &writer) {
  writer.writeVarInt(/* DistinctAttr */ 21);
  writer.writeAttribute(attribute.getReferencedAttr());
}

static ::mlir::Attribute readFileLineColLoc(MLIRContext* context, DialectBytecodeReader &reader) {
  StringAttr filename;
  uint64_t line, column;
  if (succeeded(reader.readAttribute<StringAttr>(filename)) &&
      succeeded(reader.readVarInt(line)) &&
      succeeded(reader.readVarInt(column))) {
    return get<FileLineColLoc>(context, filename, line, column);
  }
  return FileLineColLoc();
}

static void write(FileLineColLoc attribute, DialectBytecodeWriter &writer) {
  writer.writeVarInt(/* FileLineColLoc */ 11);
  writer.writeAttribute(attribute.getFilename());
  writer.writeVarInt(attribute.getLine());
  writer.writeVarInt(attribute.getColumn());
}

static ::mlir::Attribute readFlatSymbolRefAttr(MLIRContext* context, DialectBytecodeReader &reader) {
  StringAttr rootReference;
  if (succeeded(reader.readAttribute<StringAttr>(rootReference))) {
    return get<FlatSymbolRefAttr>(context, rootReference);
  }
  return FlatSymbolRefAttr();
}

static void write(FlatSymbolRefAttr attribute, DialectBytecodeWriter &writer) {
  writer.writeVarInt(/* FlatSymbolRefAttr */ 4);
  writer.writeAttribute(attribute.getRootReference());
}

static ::mlir::Attribute readFloatAttr(MLIRContext* context, DialectBytecodeReader &reader) {
  Type type;
  FailureOr<APFloat> value;
  if (succeeded(reader.readType(type)) &&
      succeeded(readAPFloatWithKnownSemantics(reader, type, value))) {
    return get<FloatAttr>(context, type, *value);
  }
  return FloatAttr();
}

static void write(FloatAttr attribute, DialectBytecodeWriter &writer) {
  writer.writeVarInt(/* FloatAttr */ 9);
  writer.writeType(attribute.getType());
  writer.writeAPFloatWithKnownSemantics(attribute.getValue());
}

static ::mlir::Attribute readFusedLoc(MLIRContext* context, DialectBytecodeReader &reader) {
  SmallVector<Location> locations;
  auto readLocations = [&]() -> FailureOr<Location> {
    LocationAttr value;
    if (succeeded(reader.readAttribute(value))) {
      return Location(value);
    }
    return failure();
  };
  if (succeeded(reader.readList(locations, readLocations))) {
    return cast<FusedLoc>(get<FusedLoc>(context, locations));
  }
  return FusedLoc();
}

static ::mlir::Attribute readFusedLocWithMetadata(MLIRContext* context, DialectBytecodeReader &reader) {
  SmallVector<Location> locations;
  Attribute metadata;
  auto readLocations = [&]() -> FailureOr<Location> {
    LocationAttr value;
    if (succeeded(reader.readAttribute(value))) {
      return Location(value);
    }
    return failure();
  };
  if (succeeded(reader.readList(locations, readLocations)) &&
      succeeded(reader.readAttribute(metadata))) {
    return cast<FusedLoc>(get<FusedLoc>(context, locations, metadata));
  }
  return FusedLoc();
}

static void write(FusedLoc attribute, DialectBytecodeWriter &writer) {
  if (!attribute.getMetadata()) {
    writer.writeVarInt(/* FusedLoc */ 12);
    writer.writeList(attribute.getLocations(), [&](Location attribute) {
      writer.writeAttribute((LocationAttr)attribute);
    });
  }
  if (attribute.getMetadata()) {
    writer.writeVarInt(/* FusedLocWithMetadata */ 13);
    writer.writeList(attribute.getLocations(), [&](Location attribute) {
      writer.writeAttribute((LocationAttr)attribute);
    });
    writer.writeAttribute(attribute.getMetadata());
  }
}

static ::mlir::Attribute readIntegerAttr(MLIRContext* context, DialectBytecodeReader &reader) {
  Type type;
  FailureOr<APInt> value;
  if (succeeded(reader.readType(type)) &&
      succeeded(readAPIntWithKnownWidth(reader, type, value))) {
    return get<IntegerAttr>(context, type, *value);
  }
  return IntegerAttr();
}

static void write(IntegerAttr attribute, DialectBytecodeWriter &writer) {
  writer.writeVarInt(/* IntegerAttr */ 8);
  writer.writeType(attribute.getType());
  writer.writeAPIntWithKnownWidth(attribute.getValue());
}

static ::mlir::Attribute readNameLoc(MLIRContext* context, DialectBytecodeReader &reader) {
  StringAttr name;
  LocationAttr childLoc;
  if (succeeded(reader.readAttribute<StringAttr>(name)) &&
      succeeded(reader.readAttribute(childLoc))) {
    return get<NameLoc>(context, name, childLoc);
  }
  return NameLoc();
}

static void write(NameLoc attribute, DialectBytecodeWriter &writer) {
  writer.writeVarInt(/* NameLoc */ 14);
  writer.writeAttribute(attribute.getName());
  writer.writeAttribute(attribute.getChildLoc());
}

static ::mlir::Attribute readSparseElementsAttr(MLIRContext* context, DialectBytecodeReader &reader) {
  ShapedType type;
  DenseIntElementsAttr indices;
  DenseElementsAttr values;
  if (succeeded(reader.readType(type)) &&
      succeeded(reader.readAttribute(indices)) &&
      succeeded(reader.readAttribute(values))) {
    return get<SparseElementsAttr>(context, type, indices, values);
  }
  return SparseElementsAttr();
}

static void write(SparseElementsAttr attribute, DialectBytecodeWriter &writer) {
  writer.writeVarInt(/* SparseElementsAttr */ 20);
  writer.writeType(attribute.getType());
  writer.writeAttribute(attribute.getIndices());
  writer.writeAttribute(attribute.getValues());
}

static ::mlir::Attribute readStringAttr(MLIRContext* context, DialectBytecodeReader &reader) {
  StringRef value;
  if (succeeded(reader.readString(value))) {
    return get<StringAttr>(context, value);
  }
  return StringAttr();
}

static ::mlir::Attribute readStringAttrWithType(MLIRContext* context, DialectBytecodeReader &reader) {
  StringRef value;
  Type type;
  if (succeeded(reader.readString(value)) &&
      succeeded(reader.readType(type))) {
    return get<StringAttr>(context, value, type);
  }
  return StringAttr();
}

static void write(StringAttr attribute, DialectBytecodeWriter &writer) {
  if (::llvm::isa<NoneType>(attribute.getType())) {
    writer.writeVarInt(/* StringAttr */ 2);
    writer.writeOwnedString(attribute);
  }
  if (!::llvm::isa<NoneType>(attribute.getType())) {
    writer.writeVarInt(/* StringAttrWithType */ 3);
    writer.writeOwnedString(attribute);
    writer.writeType(attribute.getType());
  }
}

static ::mlir::Attribute readSymbolRefAttr(MLIRContext* context, DialectBytecodeReader &reader) {
  StringAttr rootReference;
  SmallVector<FlatSymbolRefAttr> nestedReferences;
  if (succeeded(reader.readAttribute<StringAttr>(rootReference)) &&
      succeeded(reader.readAttributes(nestedReferences))) {
    return get<SymbolRefAttr>(context, rootReference, nestedReferences);
  }
  return SymbolRefAttr();
}

static void write(SymbolRefAttr attribute, DialectBytecodeWriter &writer) {
  writer.writeVarInt(/* SymbolRefAttr */ 5);
  writer.writeAttribute(attribute.getRootReference());
  writer.writeAttributes(attribute.getNestedReferences());
}

static ::mlir::Attribute readTypeAttr(MLIRContext* context, DialectBytecodeReader &reader) {
  Type value;
  if (succeeded(reader.readType(value))) {
    return get<TypeAttr>(context, value);
  }
  return TypeAttr();
}

static void write(TypeAttr attribute, DialectBytecodeWriter &writer) {
  writer.writeVarInt(/* TypeAttr */ 6);
  writer.writeType(attribute.getValue());
}

static ::mlir::Attribute readUnitAttr(MLIRContext* context, DialectBytecodeReader &reader) {
  return get<UnitAttr>(context);
}

static void write(UnitAttr attribute, DialectBytecodeWriter &writer) {
  writer.writeVarInt(/* UnitAttr */ 7);
}

static ::mlir::Attribute readUnknownLoc(MLIRContext* context, DialectBytecodeReader &reader) {
  return get<UnknownLoc>(context);
}

static void write(UnknownLoc attribute, DialectBytecodeWriter &writer) {
  writer.writeVarInt(/* UnknownLoc */ 15);
}

static Attribute readAttribute(MLIRContext* context, DialectBytecodeReader &reader) {
  uint64_t kind;
  if (failed(reader.readVarInt(kind)))
    return Attribute();
  switch (kind) {
    case 0:
      return readArrayAttr(context, reader);
    case 1:
      return readDictionaryAttr(context, reader);
    case 2:
      return readStringAttr(context, reader);
    case 3:
      return readStringAttrWithType(context, reader);
    case 4:
      return readFlatSymbolRefAttr(context, reader);
    case 5:
      return readSymbolRefAttr(context, reader);
    case 6:
      return readTypeAttr(context, reader);
    case 7:
      return readUnitAttr(context, reader);
    case 8:
      return readIntegerAttr(context, reader);
    case 9:
      return readFloatAttr(context, reader);
    case 10:
      return readCallSiteLoc(context, reader);
    case 11:
      return readFileLineColLoc(context, reader);
    case 12:
      return readFusedLoc(context, reader);
    case 13:
      return readFusedLocWithMetadata(context, reader);
    case 14:
      return readNameLoc(context, reader);
    case 15:
      return readUnknownLoc(context, reader);
    case 16:
      return readDenseResourceElementsAttr(context, reader);
    case 17:
      return readDenseArrayAttr(context, reader);
    case 18:
      return readDenseIntOrFPElementsAttr(context, reader);
    case 19:
      return readDenseStringElementsAttr(context, reader);
    case 20:
      return readSparseElementsAttr(context, reader);
    case 21:
      return readDistinctAttr(context, reader);
    default:
      reader.emitError() << "unknown attribute code: " << kind;
      return Attribute();
  }
  return Attribute();
}

static LogicalResult writeAttribute(Attribute attribute,
                                DialectBytecodeWriter &writer) {
  return TypeSwitch<Attribute, LogicalResult>(attribute)
    .Case([&](ArrayAttr t) {
      return write(t, writer), success();
    })
    .Case([&](CallSiteLoc t) {
      return write(t, writer), success();
    })
    .Case([&](DenseArrayAttr t) {
      return write(t, writer), success();
    })
    .Case([&](DenseIntOrFPElementsAttr t) {
      return write(t, writer), success();
    })
    .Case([&](DenseResourceElementsAttr t) {
      return write(t, writer), success();
    })
    .Case([&](DenseStringElementsAttr t) {
      return write(t, writer), success();
    })
    .Case([&](DictionaryAttr t) {
      return write(t, writer), success();
    })
    .Case([&](DistinctAttr t) {
      return write(t, writer), success();
    })
    .Case([&](FileLineColLoc t) {
      return write(t, writer), success();
    })
    .Case([&](FlatSymbolRefAttr t) {
      return write(t, writer), success();
    })
    .Case([&](FloatAttr t) {
      return write(t, writer), success();
    })
    .Case([&](FusedLoc t) {
      return write(t, writer), success();
    })
    .Case([&](IntegerAttr t) {
      return write(t, writer), success();
    })
    .Case([&](NameLoc t) {
      return write(t, writer), success();
    })
    .Case([&](SparseElementsAttr t) {
      return write(t, writer), success();
    })
    .Case([&](StringAttr t) {
      return write(t, writer), success();
    })
    .Case([&](SymbolRefAttr t) {
      return write(t, writer), success();
    })
    .Case([&](TypeAttr t) {
      return write(t, writer), success();
    })
    .Case([&](UnitAttr t) {
      return write(t, writer), success();
    })
    .Case([&](UnknownLoc t) {
      return write(t, writer), success();
    })
    .Default([&](Attribute) { return failure(); });
}

static ::mlir::Type readBFloat16Type(MLIRContext* context, DialectBytecodeReader &reader) {
  return get<BFloat16Type>(context);
}

static void write(BFloat16Type type, DialectBytecodeWriter &writer) {
  writer.writeVarInt(/* BFloat16Type */ 3);
}

static ::mlir::Type readComplexType(MLIRContext* context, DialectBytecodeReader &reader) {
  Type elementType;
  if (succeeded(reader.readType(elementType))) {
    return get<ComplexType>(context, elementType);
  }
  return ComplexType();
}

static void write(ComplexType type, DialectBytecodeWriter &writer) {
  writer.writeVarInt(/* ComplexType */ 9);
  writer.writeType(type.getElementType());
}

static ::mlir::Type readFloat128Type(MLIRContext* context, DialectBytecodeReader &reader) {
  return get<Float128Type>(context);
}

static void write(Float128Type type, DialectBytecodeWriter &writer) {
  writer.writeVarInt(/* Float128Type */ 8);
}

static ::mlir::Type readFloat16Type(MLIRContext* context, DialectBytecodeReader &reader) {
  return get<Float16Type>(context);
}

static void write(Float16Type type, DialectBytecodeWriter &writer) {
  writer.writeVarInt(/* Float16Type */ 4);
}

static ::mlir::Type readFloat32Type(MLIRContext* context, DialectBytecodeReader &reader) {
  return get<Float32Type>(context);
}

static void write(Float32Type type, DialectBytecodeWriter &writer) {
  writer.writeVarInt(/* Float32Type */ 5);
}

static ::mlir::Type readFloat64Type(MLIRContext* context, DialectBytecodeReader &reader) {
  return get<Float64Type>(context);
}

static void write(Float64Type type, DialectBytecodeWriter &writer) {
  writer.writeVarInt(/* Float64Type */ 6);
}

static ::mlir::Type readFloat80Type(MLIRContext* context, DialectBytecodeReader &reader) {
  return get<Float80Type>(context);
}

static void write(Float80Type type, DialectBytecodeWriter &writer) {
  writer.writeVarInt(/* Float80Type */ 7);
}

static ::mlir::Type readFunctionType(MLIRContext* context, DialectBytecodeReader &reader) {
  SmallVector<Type> inputs, results;
  if (succeeded(reader.readTypes(inputs)) &&
      succeeded(reader.readTypes(results))) {
    return get<FunctionType>(context, inputs, results);
  }
  return FunctionType();
}

static void write(FunctionType type, DialectBytecodeWriter &writer) {
  writer.writeVarInt(/* FunctionType */ 2);
  writer.writeTypes(type.getInputs());
  writer.writeTypes(type.getResults());
}

static ::mlir::Type readIndexType(MLIRContext* context, DialectBytecodeReader &reader) {
  return get<IndexType>(context);
}

static void write(IndexType type, DialectBytecodeWriter &writer) {
  writer.writeVarInt(/* IndexType */ 1);
}

static ::mlir::Type readIntegerType(MLIRContext* context, DialectBytecodeReader &reader) {
  uint64_t _widthAndSignedness, width;
  IntegerType::SignednessSemantics signedness;
  if (succeeded(reader.readVarInt(_widthAndSignedness)) &&
      ((width = _widthAndSignedness >> 2), true) &&
      ((signedness = static_cast<IntegerType::SignednessSemantics>(_widthAndSignedness & 0x3)), true)) {
    return get<IntegerType>(context, width, signedness);
  }
  return IntegerType();
}

static void write(IntegerType type, DialectBytecodeWriter &writer) {
  writer.writeVarInt(/* IntegerType */ 0);
  writer.writeVarInt(type.getWidth() << 2 | type.getSignedness());
}

static ::mlir::Type readMemRefType(MLIRContext* context, DialectBytecodeReader &reader) {
  SmallVector<int64_t> shape;
  Type elementType;
  MemRefLayoutAttrInterface layout;
  auto readShape = [&]() -> FailureOr<int64_t> {
    int64_t temp;
    if (succeeded(reader.readSignedVarInt(temp))) {
      return temp;
    }
    return failure();
  };
  if (succeeded(reader.readList(shape, readShape)) &&
      succeeded(reader.readType(elementType)) &&
      succeeded(reader.readAttribute(layout))) {
    return get<MemRefType>(context, shape, elementType, layout);
  }
  return MemRefType();
}

static ::mlir::Type readMemRefTypeWithMemSpace(MLIRContext* context, DialectBytecodeReader &reader) {
  Attribute memorySpace;
  SmallVector<int64_t> shape;
  Type elementType;
  MemRefLayoutAttrInterface layout;
  auto readShape = [&]() -> FailureOr<int64_t> {
    int64_t temp;
    if (succeeded(reader.readSignedVarInt(temp))) {
      return temp;
    }
    return failure();
  };
  if (succeeded(reader.readAttribute(memorySpace)) &&
      succeeded(reader.readList(shape, readShape)) &&
      succeeded(reader.readType(elementType)) &&
      succeeded(reader.readAttribute(layout))) {
    return get<MemRefType>(context, shape, elementType, layout, memorySpace);
  }
  return MemRefType();
}

static void write(MemRefType type, DialectBytecodeWriter &writer) {
  if (!type.getMemorySpace()) {
    writer.writeVarInt(/* MemRefType */ 10);
    writer.writeList(type.getShape(), [&](int64_t type) {
      writer.writeSignedVarInt(type);
    });
    writer.writeType(type.getElementType());
    writer.writeAttribute(type.getLayout());
  }
  if (!!type.getMemorySpace()) {
    writer.writeVarInt(/* MemRefTypeWithMemSpace */ 11);
    writer.writeAttribute(type.getMemorySpace());
    writer.writeList(type.getShape(), [&](int64_t type) {
      writer.writeSignedVarInt(type);
    });
    writer.writeType(type.getElementType());
    writer.writeAttribute(type.getLayout());
  }
}

static ::mlir::Type readNoneType(MLIRContext* context, DialectBytecodeReader &reader) {
  return get<NoneType>(context);
}

static void write(NoneType type, DialectBytecodeWriter &writer) {
  writer.writeVarInt(/* NoneType */ 12);
}

static ::mlir::Type readRankedTensorType(MLIRContext* context, DialectBytecodeReader &reader) {
  SmallVector<int64_t> shape;
  Type elementType;
  auto readShape = [&]() -> FailureOr<int64_t> {
    int64_t temp;
    if (succeeded(reader.readSignedVarInt(temp))) {
      return temp;
    }
    return failure();
  };
  if (succeeded(reader.readList(shape, readShape)) &&
      succeeded(reader.readType(elementType))) {
    return get<RankedTensorType>(context, shape, elementType);
  }
  return RankedTensorType();
}

static ::mlir::Type readRankedTensorTypeWithEncoding(MLIRContext* context, DialectBytecodeReader &reader) {
  Attribute encoding;
  SmallVector<int64_t> shape;
  Type elementType;
  auto readShape = [&]() -> FailureOr<int64_t> {
    int64_t temp;
    if (succeeded(reader.readSignedVarInt(temp))) {
      return temp;
    }
    return failure();
  };
  if (succeeded(reader.readAttribute(encoding)) &&
      succeeded(reader.readList(shape, readShape)) &&
      succeeded(reader.readType(elementType))) {
    return get<RankedTensorType>(context, shape, elementType, encoding);
  }
  return RankedTensorType();
}

static void write(RankedTensorType type, DialectBytecodeWriter &writer) {
  if (!type.getEncoding()) {
    writer.writeVarInt(/* RankedTensorType */ 13);
    writer.writeList(type.getShape(), [&](int64_t type) {
      writer.writeSignedVarInt(type);
    });
    writer.writeType(type.getElementType());
  }
  if (type.getEncoding()) {
    writer.writeVarInt(/* RankedTensorTypeWithEncoding */ 14);
    writer.writeAttribute(type.getEncoding());
    writer.writeList(type.getShape(), [&](int64_t type) {
      writer.writeSignedVarInt(type);
    });
    writer.writeType(type.getElementType());
  }
}

static ::mlir::Type readTupleType(MLIRContext* context, DialectBytecodeReader &reader) {
  SmallVector<Type> types;
  if (succeeded(reader.readTypes(types))) {
    return get<TupleType>(context, types);
  }
  return TupleType();
}

static void write(TupleType type, DialectBytecodeWriter &writer) {
  writer.writeVarInt(/* TupleType */ 15);
  writer.writeTypes(type.getTypes());
}

static ::mlir::Type readUnrankedMemRefType(MLIRContext* context, DialectBytecodeReader &reader) {
  Type elementType;
  if (succeeded(reader.readType(elementType))) {
    return get<UnrankedMemRefType>(context, elementType, Attribute());
  }
  return UnrankedMemRefType();
}

static ::mlir::Type readUnrankedMemRefTypeWithMemSpace(MLIRContext* context, DialectBytecodeReader &reader) {
  Attribute memorySpace;
  Type elementType;
  if (succeeded(reader.readAttribute(memorySpace)) &&
      succeeded(reader.readType(elementType))) {
    return get<UnrankedMemRefType>(context, elementType, memorySpace);
  }
  return UnrankedMemRefType();
}

static void write(UnrankedMemRefType type, DialectBytecodeWriter &writer) {
  if (!type.getMemorySpace()) {
    writer.writeVarInt(/* UnrankedMemRefType */ 16);
    writer.writeType(type.getElementType());
  }
  if (type.getMemorySpace()) {
    writer.writeVarInt(/* UnrankedMemRefTypeWithMemSpace */ 17);
    writer.writeAttribute(type.getMemorySpace());
    writer.writeType(type.getElementType());
  }
}

static ::mlir::Type readUnrankedTensorType(MLIRContext* context, DialectBytecodeReader &reader) {
  Type elementType;
  if (succeeded(reader.readType(elementType))) {
    return get<UnrankedTensorType>(context, elementType);
  }
  return UnrankedTensorType();
}

static void write(UnrankedTensorType type, DialectBytecodeWriter &writer) {
  writer.writeVarInt(/* UnrankedTensorType */ 18);
  writer.writeType(type.getElementType());
}

static ::mlir::Type readVectorType(MLIRContext* context, DialectBytecodeReader &reader) {
  SmallVector<int64_t> shape;
  Type elementType;
  auto readShape = [&]() -> FailureOr<int64_t> {
    int64_t temp;
    if (succeeded(reader.readSignedVarInt(temp))) {
      return temp;
    }
    return failure();
  };
  if (succeeded(reader.readList(shape, readShape)) &&
      succeeded(reader.readType(elementType))) {
    return get<VectorType>(context, shape, elementType);
  }
  return VectorType();
}

static ::mlir::Type readVectorTypeWithScalableDims(MLIRContext* context, DialectBytecodeReader &reader) {
  SmallVector<bool> scalableDims;
  SmallVector<int64_t> shape;
  Type elementType;
  auto readScalableDims = [&]() -> FailureOr<bool> {
    bool temp;
    if (succeeded(reader.readBool(temp))) {
      return temp;
    }
    return failure();
  };
  auto readShape = [&]() -> FailureOr<int64_t> {
    int64_t temp;
    if (succeeded(reader.readSignedVarInt(temp))) {
      return temp;
    }
    return failure();
  };
  if (succeeded(reader.readList(scalableDims, readScalableDims)) &&
      succeeded(reader.readList(shape, readShape)) &&
      succeeded(reader.readType(elementType))) {
    return get<VectorType>(context, shape, elementType, scalableDims);
  }
  return VectorType();
}

static void write(VectorType type, DialectBytecodeWriter &writer) {
  if (!type.isScalable()) {
    writer.writeVarInt(/* VectorType */ 19);
    writer.writeList(type.getShape(), [&](int64_t type) {
      writer.writeSignedVarInt(type);
    });
    writer.writeType(type.getElementType());
  }
  if (type.isScalable()) {
    writer.writeVarInt(/* VectorTypeWithScalableDims */ 20);
    writer.writeList(type.getScalableDims(), [&](bool type) {
      writer.writeOwnedBool(type);
    });
    writer.writeList(type.getShape(), [&](int64_t type) {
      writer.writeSignedVarInt(type);
    });
    writer.writeType(type.getElementType());
  }
}

static Type readType(MLIRContext* context, DialectBytecodeReader &reader) {
  uint64_t kind;
  if (failed(reader.readVarInt(kind)))
    return Type();
  switch (kind) {
    case 0:
      return readIntegerType(context, reader);
    case 1:
      return readIndexType(context, reader);
    case 2:
      return readFunctionType(context, reader);
    case 3:
      return readBFloat16Type(context, reader);
    case 4:
      return readFloat16Type(context, reader);
    case 5:
      return readFloat32Type(context, reader);
    case 6:
      return readFloat64Type(context, reader);
    case 7:
      return readFloat80Type(context, reader);
    case 8:
      return readFloat128Type(context, reader);
    case 9:
      return readComplexType(context, reader);
    case 10:
      return readMemRefType(context, reader);
    case 11:
      return readMemRefTypeWithMemSpace(context, reader);
    case 12:
      return readNoneType(context, reader);
    case 13:
      return readRankedTensorType(context, reader);
    case 14:
      return readRankedTensorTypeWithEncoding(context, reader);
    case 15:
      return readTupleType(context, reader);
    case 16:
      return readUnrankedMemRefType(context, reader);
    case 17:
      return readUnrankedMemRefTypeWithMemSpace(context, reader);
    case 18:
      return readUnrankedTensorType(context, reader);
    case 19:
      return readVectorType(context, reader);
    case 20:
      return readVectorTypeWithScalableDims(context, reader);
    default:
      reader.emitError() << "unknown attribute code: " << kind;
      return Type();
  }
  return Type();
}

static LogicalResult writeType(Type type,
                                DialectBytecodeWriter &writer) {
  return TypeSwitch<Type, LogicalResult>(type)
    .Case([&](BFloat16Type t) {
      return write(t, writer), success();
    })
    .Case([&](ComplexType t) {
      return write(t, writer), success();
    })
    .Case([&](Float128Type t) {
      return write(t, writer), success();
    })
    .Case([&](Float16Type t) {
      return write(t, writer), success();
    })
    .Case([&](Float32Type t) {
      return write(t, writer), success();
    })
    .Case([&](Float64Type t) {
      return write(t, writer), success();
    })
    .Case([&](Float80Type t) {
      return write(t, writer), success();
    })
    .Case([&](FunctionType t) {
      return write(t, writer), success();
    })
    .Case([&](IndexType t) {
      return write(t, writer), success();
    })
    .Case([&](IntegerType t) {
      return write(t, writer), success();
    })
    .Case([&](MemRefType t) {
      return write(t, writer), success();
    })
    .Case([&](NoneType t) {
      return write(t, writer), success();
    })
    .Case([&](RankedTensorType t) {
      return write(t, writer), success();
    })
    .Case([&](TupleType t) {
      return write(t, writer), success();
    })
    .Case([&](UnrankedMemRefType t) {
      return write(t, writer), success();
    })
    .Case([&](UnrankedTensorType t) {
      return write(t, writer), success();
    })
    .Case([&](VectorType t) {
      return write(t, writer), success();
    })
    .Default([&](Type) { return failure(); });
}

