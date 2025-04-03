static Attribute readAttribute(MLIRContext* context, DialectBytecodeReader &reader) {
  return reader.emitError() << "unknown attribute", Attribute();
}

static LogicalResult writeAttribute(Attribute attribute,
                                DialectBytecodeWriter &writer) {
  return TypeSwitch<Attribute, LogicalResult>(attribute)
    .Default([&](Attribute) { return failure(); });
}

static ::mlir::Type readAnyQuantizedType(MLIRContext* context, DialectBytecodeReader &reader) {
  uint64_t flags;
  Type storageType;
  int64_t storageTypeMin, storageTypeMax;
  if (succeeded(reader.readVarInt(flags)) &&
      succeeded(reader.readType(storageType)) &&
      succeeded(reader.readSignedVarInt(storageTypeMin)) &&
      succeeded(reader.readSignedVarInt(storageTypeMax))) {
    return get<AnyQuantizedType>(context, flags, storageType, nullptr,
          storageTypeMin, storageTypeMax);
  }
  return AnyQuantizedType();
}

static ::mlir::Type readAnyQuantizedTypeWithExpressedType(MLIRContext* context, DialectBytecodeReader &reader) {
  uint64_t flags;
  Type storageType, expressedType;
  int64_t storageTypeMin, storageTypeMax;
  if (succeeded(reader.readVarInt(flags)) &&
      succeeded(reader.readType(storageType)) &&
      succeeded(reader.readType(expressedType)) &&
      succeeded(reader.readSignedVarInt(storageTypeMin)) &&
      succeeded(reader.readSignedVarInt(storageTypeMax))) {
    return get<AnyQuantizedType>(context, flags, storageType, expressedType, storageTypeMin, storageTypeMax);
  }
  return AnyQuantizedType();
}

static void write(AnyQuantizedType type, DialectBytecodeWriter &writer) {
  if (!type.getExpressedType()) {
    writer.writeVarInt(/* AnyQuantizedType */ 1);
    writer.writeVarInt(type.getFlags());
    writer.writeType(type.getStorageType());
    writer.writeSignedVarInt(type.getStorageTypeMin());
    writer.writeSignedVarInt(type.getStorageTypeMax());
  }
  if (!!type.getExpressedType()) {
    writer.writeVarInt(/* AnyQuantizedTypeWithExpressedType */ 2);
    writer.writeVarInt(type.getFlags());
    writer.writeType(type.getStorageType());
    writer.writeType(type.getExpressedType());
    writer.writeSignedVarInt(type.getStorageTypeMin());
    writer.writeSignedVarInt(type.getStorageTypeMax());
  }
}

static ::mlir::Type readCalibratedQuantizedType(MLIRContext* context, DialectBytecodeReader &reader) {
  Type expressedType;
  double min, max;
  if (succeeded(reader.readType(expressedType)) &&
      succeeded(readDoubleAPFloat(reader, min)) &&
      succeeded(readDoubleAPFloat(reader, max))) {
    return get<CalibratedQuantizedType>(context, expressedType, min, max);
  }
  return CalibratedQuantizedType();
}

static void write(CalibratedQuantizedType type, DialectBytecodeWriter &writer) {
  writer.writeVarInt(/* CalibratedQuantizedType */ 3);
  writer.writeType(type.getExpressedType());
  writer.writeAPFloatWithKnownSemantics(APFloat(type.getMin()));
  writer.writeAPFloatWithKnownSemantics(APFloat(type.getMax()));
}

static ::mlir::Type readUniformQuantizedPerAxisType(MLIRContext* context, DialectBytecodeReader &reader) {
  uint64_t flags;
  Type storageType, expressedType;
  uint64_t quantizedDimension;
  int64_t storageTypeMin, storageTypeMax;
  SmallVector<double> scales;
  SmallVector<int64_t> zeroPoints;
  auto readScales = [&]() -> FailureOr<double> {
    double temp;
    if (succeeded(readDoubleAPFloat(reader, temp))) {
      return temp;
    }
    return failure();
  };
  auto readZeroPoints = [&]() -> FailureOr<int64_t> {
    int64_t temp;
    if (succeeded(reader.readSignedVarInt(temp))) {
      return temp;
    }
    return failure();
  };
  if (succeeded(reader.readVarInt(flags)) &&
      succeeded(reader.readType(storageType)) &&
      succeeded(reader.readType(expressedType)) &&
      succeeded(reader.readVarInt(quantizedDimension)) &&
      succeeded(reader.readSignedVarInt(storageTypeMin)) &&
      succeeded(reader.readSignedVarInt(storageTypeMax)) &&
      succeeded(reader.readList(scales, readScales)) &&
      succeeded(reader.readList(zeroPoints, readZeroPoints))) {
    return get<UniformQuantizedPerAxisType>(context, flags, storageType, expressedType, scales,
            zeroPoints, quantizedDimension, storageTypeMin, storageTypeMax);
  }
  return UniformQuantizedPerAxisType();
}

static void write(UniformQuantizedPerAxisType type, DialectBytecodeWriter &writer) {
  writer.writeVarInt(/* UniformQuantizedPerAxisType */ 5);
  writer.writeVarInt(type.getFlags());
  writer.writeType(type.getStorageType());
  writer.writeType(type.getExpressedType());
  writer.writeVarInt(type.getQuantizedDimension());
  writer.writeSignedVarInt(type.getStorageTypeMin());
  writer.writeSignedVarInt(type.getStorageTypeMax());
  writer.writeList(type.getScales(), [&](double type) {
    writer.writeAPFloatWithKnownSemantics(APFloat(type));
  });
  writer.writeList(type.getZeroPoints(), [&](int64_t type) {
    writer.writeSignedVarInt(type);
  });
}

static ::mlir::Type readUniformQuantizedType(MLIRContext* context, DialectBytecodeReader &reader) {
  uint64_t flags;
  Type storageType, expressedType;
  double scale;
  int64_t zeroPoint, storageTypeMin, storageTypeMax;
  if (succeeded(reader.readVarInt(flags)) &&
      succeeded(reader.readType(storageType)) &&
      succeeded(reader.readType(expressedType)) &&
      succeeded(readDoubleAPFloat(reader, scale)) &&
      succeeded(reader.readSignedVarInt(zeroPoint)) &&
      succeeded(reader.readSignedVarInt(storageTypeMin)) &&
      succeeded(reader.readSignedVarInt(storageTypeMax))) {
    return get<UniformQuantizedType>(context, flags, storageType, expressedType, scale, zeroPoint, storageTypeMin, storageTypeMax);
  }
  return UniformQuantizedType();
}

static void write(UniformQuantizedType type, DialectBytecodeWriter &writer) {
  writer.writeVarInt(/* UniformQuantizedType */ 4);
  writer.writeVarInt(type.getFlags());
  writer.writeType(type.getStorageType());
  writer.writeType(type.getExpressedType());
  writer.writeAPFloatWithKnownSemantics(APFloat(type.getScale()));
  writer.writeSignedVarInt(type.getZeroPoint());
  writer.writeSignedVarInt(type.getStorageTypeMin());
  writer.writeSignedVarInt(type.getStorageTypeMax());
}

static Type readType(MLIRContext* context, DialectBytecodeReader &reader) {
  uint64_t kind;
  if (failed(reader.readVarInt(kind)))
    return Type();
  switch (kind) {
    case 1:
      return readAnyQuantizedType(context, reader);
    case 2:
      return readAnyQuantizedTypeWithExpressedType(context, reader);
    case 3:
      return readCalibratedQuantizedType(context, reader);
    case 4:
      return readUniformQuantizedType(context, reader);
    case 5:
      return readUniformQuantizedPerAxisType(context, reader);
    default:
      reader.emitError() << "unknown attribute code: " << kind;
      return Type();
  }
  return Type();
}

static LogicalResult writeType(Type type,
                                DialectBytecodeWriter &writer) {
  return TypeSwitch<Type, LogicalResult>(type)
    .Case([&](AnyQuantizedType t) {
      return write(t, writer), success();
    })
    .Case([&](CalibratedQuantizedType t) {
      return write(t, writer), success();
    })
    .Case([&](UniformQuantizedPerAxisType t) {
      return write(t, writer), success();
    })
    .Case([&](UniformQuantizedType t) {
      return write(t, writer), success();
    })
    .Default([&](Type) { return failure(); });
}

