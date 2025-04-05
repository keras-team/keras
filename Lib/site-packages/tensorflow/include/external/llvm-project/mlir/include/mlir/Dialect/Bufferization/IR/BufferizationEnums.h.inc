/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Enum Utility Declarations                                                  *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|* From: BufferizationEnums.td                                                *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

namespace mlir {
namespace bufferization {
// option for map layout
enum class LayoutMapOption : uint32_t {
  InferLayoutMap = 0,
  IdentityLayoutMap = 1,
  FullyDynamicLayoutMap = 2,
};

::std::optional<LayoutMapOption> symbolizeLayoutMapOption(uint32_t);
::llvm::StringRef stringifyLayoutMapOption(LayoutMapOption);
::std::optional<LayoutMapOption> symbolizeLayoutMapOption(::llvm::StringRef);
inline constexpr unsigned getMaxEnumValForLayoutMapOption() {
  return 2;
}


inline ::llvm::StringRef stringifyEnum(LayoutMapOption enumValue) {
  return stringifyLayoutMapOption(enumValue);
}

template <typename EnumType>
::std::optional<EnumType> symbolizeEnum(::llvm::StringRef);

template <>
inline ::std::optional<LayoutMapOption> symbolizeEnum<LayoutMapOption>(::llvm::StringRef str) {
  return symbolizeLayoutMapOption(str);
}

class LayoutMapOptionAttr : public ::mlir::IntegerAttr {
public:
  using ValueType = LayoutMapOption;
  using ::mlir::IntegerAttr::IntegerAttr;
  static bool classof(::mlir::Attribute attr);
  static LayoutMapOptionAttr get(::mlir::MLIRContext *context, LayoutMapOption val);
  LayoutMapOption getValue() const;
};
} // namespace bufferization
} // namespace mlir

namespace mlir {
template <typename T, typename>
struct FieldParser;

template<>
struct FieldParser<::mlir::bufferization::LayoutMapOption, ::mlir::bufferization::LayoutMapOption> {
  template <typename ParserT>
  static FailureOr<::mlir::bufferization::LayoutMapOption> parse(ParserT &parser) {
    // Parse the keyword/string containing the enum.
    std::string enumKeyword;
    auto loc = parser.getCurrentLocation();
    if (failed(parser.parseOptionalKeywordOrString(&enumKeyword)))
      return parser.emitError(loc, "expected keyword for option for map layout");

    // Symbolize the keyword.
    if (::std::optional<::mlir::bufferization::LayoutMapOption> attr = ::mlir::bufferization::symbolizeEnum<::mlir::bufferization::LayoutMapOption>(enumKeyword))
      return *attr;
    return parser.emitError(loc, "invalid option for map layout specification: ") << enumKeyword;
  }
};
} // namespace mlir

namespace llvm {
inline ::llvm::raw_ostream &operator<<(::llvm::raw_ostream &p, ::mlir::bufferization::LayoutMapOption value) {
  auto valueStr = stringifyEnum(value);
  return p << valueStr;
}
} // namespace llvm

namespace llvm {
template<> struct DenseMapInfo<::mlir::bufferization::LayoutMapOption> {
  using StorageInfo = ::llvm::DenseMapInfo<uint32_t>;

  static inline ::mlir::bufferization::LayoutMapOption getEmptyKey() {
    return static_cast<::mlir::bufferization::LayoutMapOption>(StorageInfo::getEmptyKey());
  }

  static inline ::mlir::bufferization::LayoutMapOption getTombstoneKey() {
    return static_cast<::mlir::bufferization::LayoutMapOption>(StorageInfo::getTombstoneKey());
  }

  static unsigned getHashValue(const ::mlir::bufferization::LayoutMapOption &val) {
    return StorageInfo::getHashValue(static_cast<uint32_t>(val));
  }

  static bool isEqual(const ::mlir::bufferization::LayoutMapOption &lhs, const ::mlir::bufferization::LayoutMapOption &rhs) {
    return lhs == rhs;
  }
};
}

