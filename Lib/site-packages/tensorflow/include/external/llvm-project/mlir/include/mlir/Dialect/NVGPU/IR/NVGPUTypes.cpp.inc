/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* TypeDef Definitions                                                        *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

#ifdef GET_TYPEDEF_LIST
#undef GET_TYPEDEF_LIST

::mlir::nvgpu::DeviceAsyncTokenType,
::mlir::nvgpu::MBarrierGroupType,
::mlir::nvgpu::MBarrierTokenType,
::mlir::nvgpu::TensorMapDescriptorType,
::mlir::nvgpu::WarpgroupMatrixDescriptorType,
::mlir::nvgpu::WarpgroupAccumulatorType

#endif  // GET_TYPEDEF_LIST

#ifdef GET_TYPEDEF_CLASSES
#undef GET_TYPEDEF_CLASSES

static ::mlir::OptionalParseResult generatedTypeParser(::mlir::AsmParser &parser, ::llvm::StringRef *mnemonic, ::mlir::Type &value) {
  return ::mlir::AsmParser::KeywordSwitch<::mlir::OptionalParseResult>(parser)
    .Case(::mlir::nvgpu::DeviceAsyncTokenType::getMnemonic(), [&](llvm::StringRef, llvm::SMLoc) {
      value = ::mlir::nvgpu::DeviceAsyncTokenType::get(parser.getContext());
      return ::mlir::success(!!value);
    })
    .Case(::mlir::nvgpu::MBarrierGroupType::getMnemonic(), [&](llvm::StringRef, llvm::SMLoc) {
      value = ::mlir::nvgpu::MBarrierGroupType::parse(parser);
      return ::mlir::success(!!value);
    })
    .Case(::mlir::nvgpu::MBarrierTokenType::getMnemonic(), [&](llvm::StringRef, llvm::SMLoc) {
      value = ::mlir::nvgpu::MBarrierTokenType::get(parser.getContext());
      return ::mlir::success(!!value);
    })
    .Case(::mlir::nvgpu::TensorMapDescriptorType::getMnemonic(), [&](llvm::StringRef, llvm::SMLoc) {
      value = ::mlir::nvgpu::TensorMapDescriptorType::parse(parser);
      return ::mlir::success(!!value);
    })
    .Case(::mlir::nvgpu::WarpgroupMatrixDescriptorType::getMnemonic(), [&](llvm::StringRef, llvm::SMLoc) {
      value = ::mlir::nvgpu::WarpgroupMatrixDescriptorType::parse(parser);
      return ::mlir::success(!!value);
    })
    .Case(::mlir::nvgpu::WarpgroupAccumulatorType::getMnemonic(), [&](llvm::StringRef, llvm::SMLoc) {
      value = ::mlir::nvgpu::WarpgroupAccumulatorType::parse(parser);
      return ::mlir::success(!!value);
    })
    .Default([&](llvm::StringRef keyword, llvm::SMLoc) {
      *mnemonic = keyword;
      return std::nullopt;
    });
}

static ::llvm::LogicalResult generatedTypePrinter(::mlir::Type def, ::mlir::AsmPrinter &printer) {
  return ::llvm::TypeSwitch<::mlir::Type, ::llvm::LogicalResult>(def)    .Case<::mlir::nvgpu::DeviceAsyncTokenType>([&](auto t) {
      printer << ::mlir::nvgpu::DeviceAsyncTokenType::getMnemonic();
      return ::mlir::success();
    })
    .Case<::mlir::nvgpu::MBarrierGroupType>([&](auto t) {
      printer << ::mlir::nvgpu::MBarrierGroupType::getMnemonic();
t.print(printer);
      return ::mlir::success();
    })
    .Case<::mlir::nvgpu::MBarrierTokenType>([&](auto t) {
      printer << ::mlir::nvgpu::MBarrierTokenType::getMnemonic();
      return ::mlir::success();
    })
    .Case<::mlir::nvgpu::TensorMapDescriptorType>([&](auto t) {
      printer << ::mlir::nvgpu::TensorMapDescriptorType::getMnemonic();
t.print(printer);
      return ::mlir::success();
    })
    .Case<::mlir::nvgpu::WarpgroupMatrixDescriptorType>([&](auto t) {
      printer << ::mlir::nvgpu::WarpgroupMatrixDescriptorType::getMnemonic();
t.print(printer);
      return ::mlir::success();
    })
    .Case<::mlir::nvgpu::WarpgroupAccumulatorType>([&](auto t) {
      printer << ::mlir::nvgpu::WarpgroupAccumulatorType::getMnemonic();
t.print(printer);
      return ::mlir::success();
    })
    .Default([](auto) { return ::mlir::failure(); });
}

namespace mlir {
namespace nvgpu {
} // namespace nvgpu
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::nvgpu::DeviceAsyncTokenType)
namespace mlir {
namespace nvgpu {
namespace detail {
struct MBarrierGroupTypeStorage : public ::mlir::TypeStorage {
  using KeyTy = std::tuple<Attribute, unsigned>;
  MBarrierGroupTypeStorage(Attribute memorySpace, unsigned num_barriers) : memorySpace(std::move(memorySpace)), num_barriers(std::move(num_barriers)) {}

  KeyTy getAsKey() const {
    return KeyTy(memorySpace, num_barriers);
  }

  bool operator==(const KeyTy &tblgenKey) const {
    return (memorySpace == std::get<0>(tblgenKey)) && (num_barriers == std::get<1>(tblgenKey));
  }

  static ::llvm::hash_code hashKey(const KeyTy &tblgenKey) {
    return ::llvm::hash_combine(std::get<0>(tblgenKey), std::get<1>(tblgenKey));
  }

  static MBarrierGroupTypeStorage *construct(::mlir::TypeStorageAllocator &allocator, KeyTy &&tblgenKey) {
    auto memorySpace = std::move(std::get<0>(tblgenKey));
    auto num_barriers = std::move(std::get<1>(tblgenKey));
    return new (allocator.allocate<MBarrierGroupTypeStorage>()) MBarrierGroupTypeStorage(std::move(memorySpace), std::move(num_barriers));
  }

  Attribute memorySpace;
  unsigned num_barriers;
};
} // namespace detail
MBarrierGroupType MBarrierGroupType::get(::mlir::MLIRContext *context, Attribute memorySpace, unsigned num_barriers) {
  return Base::get(context, std::move(memorySpace), std::move(num_barriers));
}

MBarrierGroupType MBarrierGroupType::get(::mlir::MLIRContext *context, Attribute memorySpace) {
  return Base::get(context, memorySpace, 1);
}

::mlir::Type MBarrierGroupType::parse(::mlir::AsmParser &odsParser) {
  ::mlir::Builder odsBuilder(odsParser.getContext());
  ::llvm::SMLoc odsLoc = odsParser.getCurrentLocation();
  (void) odsLoc;
  ::mlir::FailureOr<Attribute> _result_memorySpace;
  ::mlir::FailureOr<unsigned> _result_num_barriers;
  // Parse literal '<'
  if (odsParser.parseLess()) return {};
  // Parse parameter struct
  bool _seen_memorySpace = false;
  bool _seen_num_barriers = false;
  {
    const auto _loop_body = [&](::llvm::StringRef _paramKey) -> bool {
      // Parse literal '='
      if (odsParser.parseEqual()) return {};
      if (!_seen_memorySpace && _paramKey == "memorySpace") {
        _seen_memorySpace = true;

        // Parse variable 'memorySpace'
        _result_memorySpace = ::mlir::FieldParser<Attribute>::parse(odsParser);
        if (::mlir::failed(_result_memorySpace)) {
          odsParser.emitError(odsParser.getCurrentLocation(), "failed to parse NVGPU_MBarrierGroup parameter 'memorySpace' which is to be a `Attribute`");
          return {};
        }
      } else if (!_seen_num_barriers && _paramKey == "num_barriers") {
        _seen_num_barriers = true;

        // Parse variable 'num_barriers'
        _result_num_barriers = ::mlir::FieldParser<unsigned>::parse(odsParser);
        if (::mlir::failed(_result_num_barriers)) {
          odsParser.emitError(odsParser.getCurrentLocation(), "failed to parse NVGPU_MBarrierGroup parameter 'num_barriers' which is to be a `unsigned`");
          return {};
        }
      } else {
        odsParser.emitError(odsParser.getCurrentLocation(), "duplicate or unknown struct parameter name: ") << _paramKey;
        return {};
      }
      return true;
    };
    do {
      ::llvm::StringRef _paramKey;
      if (odsParser.parseKeyword(&_paramKey)) {
        odsParser.emitError(odsParser.getCurrentLocation(),
                           "expected a parameter name in struct");
        return {};
      }
      if (!_loop_body(_paramKey)) return {};
    } while(!odsParser.parseOptionalComma());
    if (!_seen_memorySpace) {
      odsParser.emitError(odsParser.getCurrentLocation(), "struct is missing required parameter: ") << "memorySpace";
      return {};
    }
  }
  // Parse literal '>'
  if (odsParser.parseGreater()) return {};
  assert(::mlir::succeeded(_result_memorySpace));
  return MBarrierGroupType::get(odsParser.getContext(),
      Attribute((*_result_memorySpace)),
      unsigned((_result_num_barriers.value_or(1))));
}

void MBarrierGroupType::print(::mlir::AsmPrinter &odsPrinter) const {
  ::mlir::Builder odsBuilder(getContext());
  odsPrinter << "<";
  {
    bool _firstPrinted = true;
    if (!_firstPrinted) odsPrinter << ", ";
    _firstPrinted = false;
    odsPrinter << "memorySpace = ";
    odsPrinter.printStrippedAttrOrType(getMemorySpace());
    if (!(getNumBarriers() == 1)) {
      if (!_firstPrinted) odsPrinter << ", ";
      _firstPrinted = false;
      odsPrinter << "num_barriers = ";
      if (!(getNumBarriers() == 1)) {
        odsPrinter.printStrippedAttrOrType(getNumBarriers());
      }
    }
  }
  odsPrinter << ">";
}

Attribute MBarrierGroupType::getMemorySpace() const {
  return getImpl()->memorySpace;
}

unsigned MBarrierGroupType::getNumBarriers() const {
  return getImpl()->num_barriers;
}

} // namespace nvgpu
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::nvgpu::MBarrierGroupType)
namespace mlir {
namespace nvgpu {
} // namespace nvgpu
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::nvgpu::MBarrierTokenType)
namespace mlir {
namespace nvgpu {
namespace detail {
struct TensorMapDescriptorTypeStorage : public ::mlir::TypeStorage {
  using KeyTy = std::tuple<MemRefType, ::mlir::nvgpu::TensorMapSwizzleKind, ::mlir::nvgpu::TensorMapL2PromoKind, ::mlir::nvgpu::TensorMapOOBKind, ::mlir::nvgpu::TensorMapInterleaveKind>;
  TensorMapDescriptorTypeStorage(MemRefType tensor, ::mlir::nvgpu::TensorMapSwizzleKind swizzle, ::mlir::nvgpu::TensorMapL2PromoKind l2promo, ::mlir::nvgpu::TensorMapOOBKind oob, ::mlir::nvgpu::TensorMapInterleaveKind interleave) : tensor(std::move(tensor)), swizzle(std::move(swizzle)), l2promo(std::move(l2promo)), oob(std::move(oob)), interleave(std::move(interleave)) {}

  KeyTy getAsKey() const {
    return KeyTy(tensor, swizzle, l2promo, oob, interleave);
  }

  bool operator==(const KeyTy &tblgenKey) const {
    return (tensor == std::get<0>(tblgenKey)) && (swizzle == std::get<1>(tblgenKey)) && (l2promo == std::get<2>(tblgenKey)) && (oob == std::get<3>(tblgenKey)) && (interleave == std::get<4>(tblgenKey));
  }

  static ::llvm::hash_code hashKey(const KeyTy &tblgenKey) {
    return ::llvm::hash_combine(std::get<0>(tblgenKey), std::get<1>(tblgenKey), std::get<2>(tblgenKey), std::get<3>(tblgenKey), std::get<4>(tblgenKey));
  }

  static TensorMapDescriptorTypeStorage *construct(::mlir::TypeStorageAllocator &allocator, KeyTy &&tblgenKey) {
    auto tensor = std::move(std::get<0>(tblgenKey));
    auto swizzle = std::move(std::get<1>(tblgenKey));
    auto l2promo = std::move(std::get<2>(tblgenKey));
    auto oob = std::move(std::get<3>(tblgenKey));
    auto interleave = std::move(std::get<4>(tblgenKey));
    return new (allocator.allocate<TensorMapDescriptorTypeStorage>()) TensorMapDescriptorTypeStorage(std::move(tensor), std::move(swizzle), std::move(l2promo), std::move(oob), std::move(interleave));
  }

  MemRefType tensor;
  ::mlir::nvgpu::TensorMapSwizzleKind swizzle;
  ::mlir::nvgpu::TensorMapL2PromoKind l2promo;
  ::mlir::nvgpu::TensorMapOOBKind oob;
  ::mlir::nvgpu::TensorMapInterleaveKind interleave;
};
} // namespace detail
TensorMapDescriptorType TensorMapDescriptorType::get(::mlir::MLIRContext *context, MemRefType tensor, ::mlir::nvgpu::TensorMapSwizzleKind swizzle, ::mlir::nvgpu::TensorMapL2PromoKind l2promo, ::mlir::nvgpu::TensorMapOOBKind oob, ::mlir::nvgpu::TensorMapInterleaveKind interleave) {
  return Base::get(context, std::move(tensor), std::move(swizzle), std::move(l2promo), std::move(oob), std::move(interleave));
}

::mlir::Type TensorMapDescriptorType::parse(::mlir::AsmParser &odsParser) {
  ::mlir::Builder odsBuilder(odsParser.getContext());
  ::llvm::SMLoc odsLoc = odsParser.getCurrentLocation();
  (void) odsLoc;
  ::mlir::FailureOr<MemRefType> _result_tensor;
  ::mlir::FailureOr<::mlir::nvgpu::TensorMapSwizzleKind> _result_swizzle;
  ::mlir::FailureOr<::mlir::nvgpu::TensorMapL2PromoKind> _result_l2promo;
  ::mlir::FailureOr<::mlir::nvgpu::TensorMapOOBKind> _result_oob;
  ::mlir::FailureOr<::mlir::nvgpu::TensorMapInterleaveKind> _result_interleave;
  // Parse literal '<'
  if (odsParser.parseLess()) return {};
  // Parse parameter struct
  bool _seen_tensor = false;
  bool _seen_swizzle = false;
  bool _seen_l2promo = false;
  bool _seen_oob = false;
  bool _seen_interleave = false;
  {
    const auto _loop_body = [&](::llvm::StringRef _paramKey) -> bool {
      // Parse literal '='
      if (odsParser.parseEqual()) return {};
      if (!_seen_tensor && _paramKey == "tensor") {
        _seen_tensor = true;

        // Parse variable 'tensor'
        _result_tensor = ::mlir::FieldParser<MemRefType>::parse(odsParser);
        if (::mlir::failed(_result_tensor)) {
          odsParser.emitError(odsParser.getCurrentLocation(), "failed to parse NVGPU_TensorMapDescriptor parameter 'tensor' which is to be a `MemRefType`");
          return {};
        }
      } else if (!_seen_swizzle && _paramKey == "swizzle") {
        _seen_swizzle = true;

        // Parse variable 'swizzle'
        _result_swizzle = [&]() -> ::mlir::FailureOr<::mlir::nvgpu::TensorMapSwizzleKind> {
            auto loc = odsParser.getCurrentLocation();
            ::llvm::StringRef enumKeyword;
            if (::mlir::failed(odsParser.parseKeyword(&enumKeyword)))
              return ::mlir::failure();
            auto maybeEnum = ::mlir::nvgpu::symbolizeTensorMapSwizzleKind(enumKeyword);
            if (maybeEnum)
              return *maybeEnum;
            return {(::llvm::LogicalResult)(odsParser.emitError(loc) << "expected " << "::mlir::nvgpu::TensorMapSwizzleKind" << " to be one of: " << "none" << ", " << "swizzle_32b" << ", " << "swizzle_64b" << ", " << "swizzle_128b")};
          }();
        if (::mlir::failed(_result_swizzle)) {
          odsParser.emitError(odsParser.getCurrentLocation(), "failed to parse NVGPU_TensorMapDescriptor parameter 'swizzle' which is to be a `::mlir::nvgpu::TensorMapSwizzleKind`");
          return {};
        }
      } else if (!_seen_l2promo && _paramKey == "l2promo") {
        _seen_l2promo = true;

        // Parse variable 'l2promo'
        _result_l2promo = [&]() -> ::mlir::FailureOr<::mlir::nvgpu::TensorMapL2PromoKind> {
            auto loc = odsParser.getCurrentLocation();
            ::llvm::StringRef enumKeyword;
            if (::mlir::failed(odsParser.parseKeyword(&enumKeyword)))
              return ::mlir::failure();
            auto maybeEnum = ::mlir::nvgpu::symbolizeTensorMapL2PromoKind(enumKeyword);
            if (maybeEnum)
              return *maybeEnum;
            return {(::llvm::LogicalResult)(odsParser.emitError(loc) << "expected " << "::mlir::nvgpu::TensorMapL2PromoKind" << " to be one of: " << "none" << ", " << "l2promo_64b" << ", " << "l2promo_128b" << ", " << "l2promo_256b")};
          }();
        if (::mlir::failed(_result_l2promo)) {
          odsParser.emitError(odsParser.getCurrentLocation(), "failed to parse NVGPU_TensorMapDescriptor parameter 'l2promo' which is to be a `::mlir::nvgpu::TensorMapL2PromoKind`");
          return {};
        }
      } else if (!_seen_oob && _paramKey == "oob") {
        _seen_oob = true;

        // Parse variable 'oob'
        _result_oob = [&]() -> ::mlir::FailureOr<::mlir::nvgpu::TensorMapOOBKind> {
            auto loc = odsParser.getCurrentLocation();
            ::llvm::StringRef enumKeyword;
            if (::mlir::failed(odsParser.parseKeyword(&enumKeyword)))
              return ::mlir::failure();
            auto maybeEnum = ::mlir::nvgpu::symbolizeTensorMapOOBKind(enumKeyword);
            if (maybeEnum)
              return *maybeEnum;
            return {(::llvm::LogicalResult)(odsParser.emitError(loc) << "expected " << "::mlir::nvgpu::TensorMapOOBKind" << " to be one of: " << "zero" << ", " << "nan")};
          }();
        if (::mlir::failed(_result_oob)) {
          odsParser.emitError(odsParser.getCurrentLocation(), "failed to parse NVGPU_TensorMapDescriptor parameter 'oob' which is to be a `::mlir::nvgpu::TensorMapOOBKind`");
          return {};
        }
      } else if (!_seen_interleave && _paramKey == "interleave") {
        _seen_interleave = true;

        // Parse variable 'interleave'
        _result_interleave = [&]() -> ::mlir::FailureOr<::mlir::nvgpu::TensorMapInterleaveKind> {
            auto loc = odsParser.getCurrentLocation();
            ::llvm::StringRef enumKeyword;
            if (::mlir::failed(odsParser.parseKeyword(&enumKeyword)))
              return ::mlir::failure();
            auto maybeEnum = ::mlir::nvgpu::symbolizeTensorMapInterleaveKind(enumKeyword);
            if (maybeEnum)
              return *maybeEnum;
            return {(::llvm::LogicalResult)(odsParser.emitError(loc) << "expected " << "::mlir::nvgpu::TensorMapInterleaveKind" << " to be one of: " << "none" << ", " << "interleave_16b" << ", " << "interleave_32b")};
          }();
        if (::mlir::failed(_result_interleave)) {
          odsParser.emitError(odsParser.getCurrentLocation(), "failed to parse NVGPU_TensorMapDescriptor parameter 'interleave' which is to be a `::mlir::nvgpu::TensorMapInterleaveKind`");
          return {};
        }
      } else {
        odsParser.emitError(odsParser.getCurrentLocation(), "duplicate or unknown struct parameter name: ") << _paramKey;
        return {};
      }
      return true;
    };
    for (unsigned odsStructIndex = 0; odsStructIndex < 5; ++odsStructIndex) {
      ::llvm::StringRef _paramKey;
      if (odsParser.parseKeyword(&_paramKey)) {
        odsParser.emitError(odsParser.getCurrentLocation(),
                           "expected a parameter name in struct");
        return {};
      }
      if (!_loop_body(_paramKey)) return {};
      if ((odsStructIndex != 5 - 1) && odsParser.parseComma())
        return {};
    }
  }
  // Parse literal '>'
  if (odsParser.parseGreater()) return {};
  assert(::mlir::succeeded(_result_tensor));
  assert(::mlir::succeeded(_result_swizzle));
  assert(::mlir::succeeded(_result_l2promo));
  assert(::mlir::succeeded(_result_oob));
  assert(::mlir::succeeded(_result_interleave));
  return TensorMapDescriptorType::get(odsParser.getContext(),
      MemRefType((*_result_tensor)),
      ::mlir::nvgpu::TensorMapSwizzleKind((*_result_swizzle)),
      ::mlir::nvgpu::TensorMapL2PromoKind((*_result_l2promo)),
      ::mlir::nvgpu::TensorMapOOBKind((*_result_oob)),
      ::mlir::nvgpu::TensorMapInterleaveKind((*_result_interleave)));
}

void TensorMapDescriptorType::print(::mlir::AsmPrinter &odsPrinter) const {
  ::mlir::Builder odsBuilder(getContext());
  odsPrinter << "<";
  {
    bool _firstPrinted = true;
    if (!_firstPrinted) odsPrinter << ", ";
    _firstPrinted = false;
    odsPrinter << "tensor = ";
    odsPrinter.printStrippedAttrOrType(getTensor());
    if (!_firstPrinted) odsPrinter << ", ";
    _firstPrinted = false;
    odsPrinter << "swizzle = ";
    odsPrinter << stringifyTensorMapSwizzleKind(getSwizzle());
    if (!_firstPrinted) odsPrinter << ", ";
    _firstPrinted = false;
    odsPrinter << "l2promo = ";
    odsPrinter << stringifyTensorMapL2PromoKind(getL2promo());
    if (!_firstPrinted) odsPrinter << ", ";
    _firstPrinted = false;
    odsPrinter << "oob = ";
    odsPrinter << stringifyTensorMapOOBKind(getOob());
    if (!_firstPrinted) odsPrinter << ", ";
    _firstPrinted = false;
    odsPrinter << "interleave = ";
    odsPrinter << stringifyTensorMapInterleaveKind(getInterleave());
  }
  odsPrinter << ">";
}

MemRefType TensorMapDescriptorType::getTensor() const {
  return getImpl()->tensor;
}

::mlir::nvgpu::TensorMapSwizzleKind TensorMapDescriptorType::getSwizzle() const {
  return getImpl()->swizzle;
}

::mlir::nvgpu::TensorMapL2PromoKind TensorMapDescriptorType::getL2promo() const {
  return getImpl()->l2promo;
}

::mlir::nvgpu::TensorMapOOBKind TensorMapDescriptorType::getOob() const {
  return getImpl()->oob;
}

::mlir::nvgpu::TensorMapInterleaveKind TensorMapDescriptorType::getInterleave() const {
  return getImpl()->interleave;
}

} // namespace nvgpu
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::nvgpu::TensorMapDescriptorType)
namespace mlir {
namespace nvgpu {
namespace detail {
struct WarpgroupMatrixDescriptorTypeStorage : public ::mlir::TypeStorage {
  using KeyTy = std::tuple<MemRefType>;
  WarpgroupMatrixDescriptorTypeStorage(MemRefType tensor) : tensor(std::move(tensor)) {}

  KeyTy getAsKey() const {
    return KeyTy(tensor);
  }

  bool operator==(const KeyTy &tblgenKey) const {
    return (tensor == std::get<0>(tblgenKey));
  }

  static ::llvm::hash_code hashKey(const KeyTy &tblgenKey) {
    return ::llvm::hash_combine(std::get<0>(tblgenKey));
  }

  static WarpgroupMatrixDescriptorTypeStorage *construct(::mlir::TypeStorageAllocator &allocator, KeyTy &&tblgenKey) {
    auto tensor = std::move(std::get<0>(tblgenKey));
    return new (allocator.allocate<WarpgroupMatrixDescriptorTypeStorage>()) WarpgroupMatrixDescriptorTypeStorage(std::move(tensor));
  }

  MemRefType tensor;
};
} // namespace detail
WarpgroupMatrixDescriptorType WarpgroupMatrixDescriptorType::get(::mlir::MLIRContext *context, MemRefType tensor) {
  return Base::get(context, std::move(tensor));
}

::mlir::Type WarpgroupMatrixDescriptorType::parse(::mlir::AsmParser &odsParser) {
  ::mlir::Builder odsBuilder(odsParser.getContext());
  ::llvm::SMLoc odsLoc = odsParser.getCurrentLocation();
  (void) odsLoc;
  ::mlir::FailureOr<MemRefType> _result_tensor;
  // Parse literal '<'
  if (odsParser.parseLess()) return {};
  // Parse parameter struct
  bool _seen_tensor = false;
  {
    const auto _loop_body = [&](::llvm::StringRef _paramKey) -> bool {
      // Parse literal '='
      if (odsParser.parseEqual()) return {};
      if (!_seen_tensor && _paramKey == "tensor") {
        _seen_tensor = true;

        // Parse variable 'tensor'
        _result_tensor = ::mlir::FieldParser<MemRefType>::parse(odsParser);
        if (::mlir::failed(_result_tensor)) {
          odsParser.emitError(odsParser.getCurrentLocation(), "failed to parse NVGPU_WarpgroupMatrixDescriptor parameter 'tensor' which is to be a `MemRefType`");
          return {};
        }
      } else {
        odsParser.emitError(odsParser.getCurrentLocation(), "duplicate or unknown struct parameter name: ") << _paramKey;
        return {};
      }
      return true;
    };
    for (unsigned odsStructIndex = 0; odsStructIndex < 1; ++odsStructIndex) {
      ::llvm::StringRef _paramKey;
      if (odsParser.parseKeyword(&_paramKey)) {
        odsParser.emitError(odsParser.getCurrentLocation(),
                           "expected a parameter name in struct");
        return {};
      }
      if (!_loop_body(_paramKey)) return {};
      if ((odsStructIndex != 1 - 1) && odsParser.parseComma())
        return {};
    }
  }
  // Parse literal '>'
  if (odsParser.parseGreater()) return {};
  assert(::mlir::succeeded(_result_tensor));
  return WarpgroupMatrixDescriptorType::get(odsParser.getContext(),
      MemRefType((*_result_tensor)));
}

void WarpgroupMatrixDescriptorType::print(::mlir::AsmPrinter &odsPrinter) const {
  ::mlir::Builder odsBuilder(getContext());
  odsPrinter << "<";
  {
    bool _firstPrinted = true;
    if (!_firstPrinted) odsPrinter << ", ";
    _firstPrinted = false;
    odsPrinter << "tensor = ";
    odsPrinter.printStrippedAttrOrType(getTensor());
  }
  odsPrinter << ">";
}

MemRefType WarpgroupMatrixDescriptorType::getTensor() const {
  return getImpl()->tensor;
}

} // namespace nvgpu
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::nvgpu::WarpgroupMatrixDescriptorType)
namespace mlir {
namespace nvgpu {
namespace detail {
struct WarpgroupAccumulatorTypeStorage : public ::mlir::TypeStorage {
  using KeyTy = std::tuple<VectorType>;
  WarpgroupAccumulatorTypeStorage(VectorType fragmented) : fragmented(std::move(fragmented)) {}

  KeyTy getAsKey() const {
    return KeyTy(fragmented);
  }

  bool operator==(const KeyTy &tblgenKey) const {
    return (fragmented == std::get<0>(tblgenKey));
  }

  static ::llvm::hash_code hashKey(const KeyTy &tblgenKey) {
    return ::llvm::hash_combine(std::get<0>(tblgenKey));
  }

  static WarpgroupAccumulatorTypeStorage *construct(::mlir::TypeStorageAllocator &allocator, KeyTy &&tblgenKey) {
    auto fragmented = std::move(std::get<0>(tblgenKey));
    return new (allocator.allocate<WarpgroupAccumulatorTypeStorage>()) WarpgroupAccumulatorTypeStorage(std::move(fragmented));
  }

  VectorType fragmented;
};
} // namespace detail
WarpgroupAccumulatorType WarpgroupAccumulatorType::get(::mlir::MLIRContext *context, VectorType fragmented) {
  return Base::get(context, std::move(fragmented));
}

::mlir::Type WarpgroupAccumulatorType::parse(::mlir::AsmParser &odsParser) {
  ::mlir::Builder odsBuilder(odsParser.getContext());
  ::llvm::SMLoc odsLoc = odsParser.getCurrentLocation();
  (void) odsLoc;
  ::mlir::FailureOr<VectorType> _result_fragmented;
  // Parse literal '<'
  if (odsParser.parseLess()) return {};
  // Parse parameter struct
  bool _seen_fragmented = false;
  {
    const auto _loop_body = [&](::llvm::StringRef _paramKey) -> bool {
      // Parse literal '='
      if (odsParser.parseEqual()) return {};
      if (!_seen_fragmented && _paramKey == "fragmented") {
        _seen_fragmented = true;

        // Parse variable 'fragmented'
        _result_fragmented = ::mlir::FieldParser<VectorType>::parse(odsParser);
        if (::mlir::failed(_result_fragmented)) {
          odsParser.emitError(odsParser.getCurrentLocation(), "failed to parse NVGPU_WarpgroupAccumulator parameter 'fragmented' which is to be a `VectorType`");
          return {};
        }
      } else {
        odsParser.emitError(odsParser.getCurrentLocation(), "duplicate or unknown struct parameter name: ") << _paramKey;
        return {};
      }
      return true;
    };
    for (unsigned odsStructIndex = 0; odsStructIndex < 1; ++odsStructIndex) {
      ::llvm::StringRef _paramKey;
      if (odsParser.parseKeyword(&_paramKey)) {
        odsParser.emitError(odsParser.getCurrentLocation(),
                           "expected a parameter name in struct");
        return {};
      }
      if (!_loop_body(_paramKey)) return {};
      if ((odsStructIndex != 1 - 1) && odsParser.parseComma())
        return {};
    }
  }
  // Parse literal '>'
  if (odsParser.parseGreater()) return {};
  assert(::mlir::succeeded(_result_fragmented));
  return WarpgroupAccumulatorType::get(odsParser.getContext(),
      VectorType((*_result_fragmented)));
}

void WarpgroupAccumulatorType::print(::mlir::AsmPrinter &odsPrinter) const {
  ::mlir::Builder odsBuilder(getContext());
  odsPrinter << "<";
  {
    bool _firstPrinted = true;
    if (!_firstPrinted) odsPrinter << ", ";
    _firstPrinted = false;
    odsPrinter << "fragmented = ";
    odsPrinter.printStrippedAttrOrType(getFragmented());
  }
  odsPrinter << ">";
}

VectorType WarpgroupAccumulatorType::getFragmented() const {
  return getImpl()->fragmented;
}

} // namespace nvgpu
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::nvgpu::WarpgroupAccumulatorType)
namespace mlir {
namespace nvgpu {

/// Parse a type registered to this dialect.
::mlir::Type NVGPUDialect::parseType(::mlir::DialectAsmParser &parser) const {
  ::llvm::SMLoc typeLoc = parser.getCurrentLocation();
  ::llvm::StringRef mnemonic;
  ::mlir::Type genType;
  auto parseResult = generatedTypeParser(parser, &mnemonic, genType);
  if (parseResult.has_value())
    return genType;
  
  parser.emitError(typeLoc) << "unknown  type `"
      << mnemonic << "` in dialect `" << getNamespace() << "`";
  return {};
}
/// Print a type registered to this dialect.
void NVGPUDialect::printType(::mlir::Type type,
                    ::mlir::DialectAsmPrinter &printer) const {
  if (::mlir::succeeded(generatedTypePrinter(type, printer)))
    return;
  
}
} // namespace nvgpu
} // namespace mlir

#endif  // GET_TYPEDEF_CLASSES

