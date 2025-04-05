/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Enum Utility Declarations                                                  *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|* From: enums.td                                                             *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

namespace mlir {
namespace sdy {
// propagation direction enum
enum class PropagationDirection : uint32_t {
  NONE = 0,
  FORWARD = 1,
  BACKWARD = 2,
  BOTH = 3,
};

::std::optional<PropagationDirection> symbolizePropagationDirection(uint32_t);
::llvm::StringRef stringifyPropagationDirection(PropagationDirection);
::std::optional<PropagationDirection> symbolizePropagationDirection(::llvm::StringRef);
inline constexpr unsigned getMaxEnumValForPropagationDirection() {
  return 3;
}


inline ::llvm::StringRef stringifyEnum(PropagationDirection enumValue) {
  return stringifyPropagationDirection(enumValue);
}

template <typename EnumType>
::std::optional<EnumType> symbolizeEnum(::llvm::StringRef);

template <>
inline ::std::optional<PropagationDirection> symbolizeEnum<PropagationDirection>(::llvm::StringRef str) {
  return symbolizePropagationDirection(str);
}

class PropagationDirectionAttr : public ::mlir::IntegerAttr {
public:
  using ValueType = PropagationDirection;
  using ::mlir::IntegerAttr::IntegerAttr;
  static bool classof(::mlir::Attribute attr);
  static PropagationDirectionAttr get(::mlir::MLIRContext *context, PropagationDirection val);
  PropagationDirection getValue() const;
};
} // namespace sdy
} // namespace mlir

namespace mlir {
template <typename T, typename>
struct FieldParser;

template<>
struct FieldParser<::mlir::sdy::PropagationDirection, ::mlir::sdy::PropagationDirection> {
  template <typename ParserT>
  static FailureOr<::mlir::sdy::PropagationDirection> parse(ParserT &parser) {
    // Parse the keyword/string containing the enum.
    std::string enumKeyword;
    auto loc = parser.getCurrentLocation();
    if (failed(parser.parseOptionalKeywordOrString(&enumKeyword)))
      return parser.emitError(loc, "expected keyword for propagation direction enum");

    // Symbolize the keyword.
    if (::std::optional<::mlir::sdy::PropagationDirection> attr = ::mlir::sdy::symbolizeEnum<::mlir::sdy::PropagationDirection>(enumKeyword))
      return *attr;
    return parser.emitError(loc, "invalid propagation direction enum specification: ") << enumKeyword;
  }
};
} // namespace mlir

namespace llvm {
inline ::llvm::raw_ostream &operator<<(::llvm::raw_ostream &p, ::mlir::sdy::PropagationDirection value) {
  auto valueStr = stringifyEnum(value);
  return p << valueStr;
}
} // namespace llvm

namespace llvm {
template<> struct DenseMapInfo<::mlir::sdy::PropagationDirection> {
  using StorageInfo = ::llvm::DenseMapInfo<uint32_t>;

  static inline ::mlir::sdy::PropagationDirection getEmptyKey() {
    return static_cast<::mlir::sdy::PropagationDirection>(StorageInfo::getEmptyKey());
  }

  static inline ::mlir::sdy::PropagationDirection getTombstoneKey() {
    return static_cast<::mlir::sdy::PropagationDirection>(StorageInfo::getTombstoneKey());
  }

  static unsigned getHashValue(const ::mlir::sdy::PropagationDirection &val) {
    return StorageInfo::getHashValue(static_cast<uint32_t>(val));
  }

  static bool isEqual(const ::mlir::sdy::PropagationDirection &lhs, const ::mlir::sdy::PropagationDirection &rhs) {
    return lhs == rhs;
  }
};
}

