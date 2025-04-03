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
namespace NVVM {
class ReduxKindAttr;
class SharedSpaceAttr;
class ProxyKindAttr;
class MemScopeKindAttr;
class SetMaxRegisterActionAttr;
class ShflKindAttr;
class LoadCacheModifierKindAttr;
class MMAB1OpAttr;
class MMAIntOverflowAttr;
class MMAShapeAttr;
class MMALayoutAttr;
class MMATypesAttr;
class MMAFragAttr;
class WGMMAScaleInAttr;
class WGMMAScaleOutAttr;
class WGMMATypesAttr;
class NVVMTargetAttr;
namespace detail {
struct ReduxKindAttrStorage;
} // namespace detail
class ReduxKindAttr : public ::mlir::Attribute::AttrBase<ReduxKindAttr, ::mlir::Attribute, detail::ReduxKindAttrStorage> {
public:
  using Base::Base;
  static constexpr ::llvm::StringLiteral name = "nvvm.redux_kind";
  static constexpr ::llvm::StringLiteral dialectName = "nvvm";
  static ReduxKindAttr get(::mlir::MLIRContext *context, ::mlir::NVVM::ReduxKind value);
  static constexpr ::llvm::StringLiteral getMnemonic() {
    return {"redux_kind"};
  }

  static ::mlir::Attribute parse(::mlir::AsmParser &odsParser, ::mlir::Type odsType);
  void print(::mlir::AsmPrinter &odsPrinter) const;
  ::mlir::NVVM::ReduxKind getValue() const;
};
namespace detail {
struct SharedSpaceAttrStorage;
} // namespace detail
class SharedSpaceAttr : public ::mlir::Attribute::AttrBase<SharedSpaceAttr, ::mlir::Attribute, detail::SharedSpaceAttrStorage> {
public:
  using Base::Base;
  static constexpr ::llvm::StringLiteral name = "nvvm.shared_space";
  static constexpr ::llvm::StringLiteral dialectName = "nvvm";
  static SharedSpaceAttr get(::mlir::MLIRContext *context, ::mlir::NVVM::SharedSpace value);
  static constexpr ::llvm::StringLiteral getMnemonic() {
    return {"shared_space"};
  }

  static ::mlir::Attribute parse(::mlir::AsmParser &odsParser, ::mlir::Type odsType);
  void print(::mlir::AsmPrinter &odsPrinter) const;
  ::mlir::NVVM::SharedSpace getValue() const;
};
namespace detail {
struct ProxyKindAttrStorage;
} // namespace detail
class ProxyKindAttr : public ::mlir::Attribute::AttrBase<ProxyKindAttr, ::mlir::Attribute, detail::ProxyKindAttrStorage> {
public:
  using Base::Base;
  static constexpr ::llvm::StringLiteral name = "nvvm.proxy_kind";
  static constexpr ::llvm::StringLiteral dialectName = "nvvm";
  static ProxyKindAttr get(::mlir::MLIRContext *context, ::mlir::NVVM::ProxyKind value);
  static constexpr ::llvm::StringLiteral getMnemonic() {
    return {"proxy_kind"};
  }

  static ::mlir::Attribute parse(::mlir::AsmParser &odsParser, ::mlir::Type odsType);
  void print(::mlir::AsmPrinter &odsPrinter) const;
  ::mlir::NVVM::ProxyKind getValue() const;
};
namespace detail {
struct MemScopeKindAttrStorage;
} // namespace detail
class MemScopeKindAttr : public ::mlir::Attribute::AttrBase<MemScopeKindAttr, ::mlir::Attribute, detail::MemScopeKindAttrStorage> {
public:
  using Base::Base;
  static constexpr ::llvm::StringLiteral name = "nvvm.mem_scope";
  static constexpr ::llvm::StringLiteral dialectName = "nvvm";
  static MemScopeKindAttr get(::mlir::MLIRContext *context, ::mlir::NVVM::MemScopeKind value);
  static constexpr ::llvm::StringLiteral getMnemonic() {
    return {"mem_scope"};
  }

  static ::mlir::Attribute parse(::mlir::AsmParser &odsParser, ::mlir::Type odsType);
  void print(::mlir::AsmPrinter &odsPrinter) const;
  ::mlir::NVVM::MemScopeKind getValue() const;
};
namespace detail {
struct SetMaxRegisterActionAttrStorage;
} // namespace detail
class SetMaxRegisterActionAttr : public ::mlir::Attribute::AttrBase<SetMaxRegisterActionAttr, ::mlir::Attribute, detail::SetMaxRegisterActionAttrStorage> {
public:
  using Base::Base;
  static constexpr ::llvm::StringLiteral name = "nvvm.action";
  static constexpr ::llvm::StringLiteral dialectName = "nvvm";
  static SetMaxRegisterActionAttr get(::mlir::MLIRContext *context, ::mlir::NVVM::SetMaxRegisterAction value);
  static constexpr ::llvm::StringLiteral getMnemonic() {
    return {"action"};
  }

  static ::mlir::Attribute parse(::mlir::AsmParser &odsParser, ::mlir::Type odsType);
  void print(::mlir::AsmPrinter &odsPrinter) const;
  ::mlir::NVVM::SetMaxRegisterAction getValue() const;
};
namespace detail {
struct ShflKindAttrStorage;
} // namespace detail
class ShflKindAttr : public ::mlir::Attribute::AttrBase<ShflKindAttr, ::mlir::Attribute, detail::ShflKindAttrStorage> {
public:
  using Base::Base;
  static constexpr ::llvm::StringLiteral name = "nvvm.shfl_kind";
  static constexpr ::llvm::StringLiteral dialectName = "nvvm";
  static ShflKindAttr get(::mlir::MLIRContext *context, ::mlir::NVVM::ShflKind value);
  static constexpr ::llvm::StringLiteral getMnemonic() {
    return {"shfl_kind"};
  }

  static ::mlir::Attribute parse(::mlir::AsmParser &odsParser, ::mlir::Type odsType);
  void print(::mlir::AsmPrinter &odsPrinter) const;
  ::mlir::NVVM::ShflKind getValue() const;
};
namespace detail {
struct LoadCacheModifierKindAttrStorage;
} // namespace detail
class LoadCacheModifierKindAttr : public ::mlir::Attribute::AttrBase<LoadCacheModifierKindAttr, ::mlir::Attribute, detail::LoadCacheModifierKindAttrStorage> {
public:
  using Base::Base;
  static constexpr ::llvm::StringLiteral name = "nvvm.load_cache_modifier";
  static constexpr ::llvm::StringLiteral dialectName = "nvvm";
  static LoadCacheModifierKindAttr get(::mlir::MLIRContext *context, ::mlir::NVVM::LoadCacheModifierKind value);
  static constexpr ::llvm::StringLiteral getMnemonic() {
    return {"load_cache_modifier"};
  }

  static ::mlir::Attribute parse(::mlir::AsmParser &odsParser, ::mlir::Type odsType);
  void print(::mlir::AsmPrinter &odsPrinter) const;
  ::mlir::NVVM::LoadCacheModifierKind getValue() const;
};
namespace detail {
struct MMAB1OpAttrStorage;
} // namespace detail
class MMAB1OpAttr : public ::mlir::Attribute::AttrBase<MMAB1OpAttr, ::mlir::Attribute, detail::MMAB1OpAttrStorage> {
public:
  using Base::Base;
  static constexpr ::llvm::StringLiteral name = "nvvm.mma_b1op";
  static constexpr ::llvm::StringLiteral dialectName = "nvvm";
  static MMAB1OpAttr get(::mlir::MLIRContext *context, ::mlir::NVVM::MMAB1Op value);
  static constexpr ::llvm::StringLiteral getMnemonic() {
    return {"mma_b1op"};
  }

  static ::mlir::Attribute parse(::mlir::AsmParser &odsParser, ::mlir::Type odsType);
  void print(::mlir::AsmPrinter &odsPrinter) const;
  ::mlir::NVVM::MMAB1Op getValue() const;
};
namespace detail {
struct MMAIntOverflowAttrStorage;
} // namespace detail
class MMAIntOverflowAttr : public ::mlir::Attribute::AttrBase<MMAIntOverflowAttr, ::mlir::Attribute, detail::MMAIntOverflowAttrStorage> {
public:
  using Base::Base;
  static constexpr ::llvm::StringLiteral name = "nvvm.mma_int_overflow";
  static constexpr ::llvm::StringLiteral dialectName = "nvvm";
  static MMAIntOverflowAttr get(::mlir::MLIRContext *context, ::mlir::NVVM::MMAIntOverflow value);
  static constexpr ::llvm::StringLiteral getMnemonic() {
    return {"mma_int_overflow"};
  }

  static ::mlir::Attribute parse(::mlir::AsmParser &odsParser, ::mlir::Type odsType);
  void print(::mlir::AsmPrinter &odsPrinter) const;
  ::mlir::NVVM::MMAIntOverflow getValue() const;
};
namespace detail {
struct MMAShapeAttrStorage;
} // namespace detail
class MMAShapeAttr : public ::mlir::Attribute::AttrBase<MMAShapeAttr, ::mlir::Attribute, detail::MMAShapeAttrStorage> {
public:
  using Base::Base;
  static constexpr ::llvm::StringLiteral name = "nvvm.shape";
  static constexpr ::llvm::StringLiteral dialectName = "nvvm";
  static MMAShapeAttr get(::mlir::MLIRContext *context, int m, int n, int k);
  static constexpr ::llvm::StringLiteral getMnemonic() {
    return {"shape"};
  }

  static ::mlir::Attribute parse(::mlir::AsmParser &odsParser, ::mlir::Type odsType);
  void print(::mlir::AsmPrinter &odsPrinter) const;
  int getM() const;
  int getN() const;
  int getK() const;
};
namespace detail {
struct MMALayoutAttrStorage;
} // namespace detail
class MMALayoutAttr : public ::mlir::Attribute::AttrBase<MMALayoutAttr, ::mlir::Attribute, detail::MMALayoutAttrStorage> {
public:
  using Base::Base;
  static constexpr ::llvm::StringLiteral name = "nvvm.mma_layout";
  static constexpr ::llvm::StringLiteral dialectName = "nvvm";
  static MMALayoutAttr get(::mlir::MLIRContext *context, ::mlir::NVVM::MMALayout value);
  static constexpr ::llvm::StringLiteral getMnemonic() {
    return {"mma_layout"};
  }

  static ::mlir::Attribute parse(::mlir::AsmParser &odsParser, ::mlir::Type odsType);
  void print(::mlir::AsmPrinter &odsPrinter) const;
  ::mlir::NVVM::MMALayout getValue() const;
};
namespace detail {
struct MMATypesAttrStorage;
} // namespace detail
class MMATypesAttr : public ::mlir::Attribute::AttrBase<MMATypesAttr, ::mlir::Attribute, detail::MMATypesAttrStorage> {
public:
  using Base::Base;
  static constexpr ::llvm::StringLiteral name = "nvvm.mma_type";
  static constexpr ::llvm::StringLiteral dialectName = "nvvm";
  static MMATypesAttr get(::mlir::MLIRContext *context, ::mlir::NVVM::MMATypes value);
  static constexpr ::llvm::StringLiteral getMnemonic() {
    return {"mma_type"};
  }

  static ::mlir::Attribute parse(::mlir::AsmParser &odsParser, ::mlir::Type odsType);
  void print(::mlir::AsmPrinter &odsPrinter) const;
  ::mlir::NVVM::MMATypes getValue() const;
};
namespace detail {
struct MMAFragAttrStorage;
} // namespace detail
class MMAFragAttr : public ::mlir::Attribute::AttrBase<MMAFragAttr, ::mlir::Attribute, detail::MMAFragAttrStorage> {
public:
  using Base::Base;
  static constexpr ::llvm::StringLiteral name = "nvvm.mma_frag";
  static constexpr ::llvm::StringLiteral dialectName = "nvvm";
  static MMAFragAttr get(::mlir::MLIRContext *context, ::mlir::NVVM::MMAFrag value);
  static constexpr ::llvm::StringLiteral getMnemonic() {
    return {"mma_frag"};
  }

  static ::mlir::Attribute parse(::mlir::AsmParser &odsParser, ::mlir::Type odsType);
  void print(::mlir::AsmPrinter &odsPrinter) const;
  ::mlir::NVVM::MMAFrag getValue() const;
};
namespace detail {
struct WGMMAScaleInAttrStorage;
} // namespace detail
class WGMMAScaleInAttr : public ::mlir::Attribute::AttrBase<WGMMAScaleInAttr, ::mlir::Attribute, detail::WGMMAScaleInAttrStorage> {
public:
  using Base::Base;
  static constexpr ::llvm::StringLiteral name = "nvvm.wgmma_scale_in";
  static constexpr ::llvm::StringLiteral dialectName = "nvvm";
  static WGMMAScaleInAttr get(::mlir::MLIRContext *context, ::mlir::NVVM::WGMMAScaleIn value);
  static constexpr ::llvm::StringLiteral getMnemonic() {
    return {"wgmma_scale_in"};
  }

  static ::mlir::Attribute parse(::mlir::AsmParser &odsParser, ::mlir::Type odsType);
  void print(::mlir::AsmPrinter &odsPrinter) const;
  ::mlir::NVVM::WGMMAScaleIn getValue() const;
};
namespace detail {
struct WGMMAScaleOutAttrStorage;
} // namespace detail
class WGMMAScaleOutAttr : public ::mlir::Attribute::AttrBase<WGMMAScaleOutAttr, ::mlir::Attribute, detail::WGMMAScaleOutAttrStorage> {
public:
  using Base::Base;
  static constexpr ::llvm::StringLiteral name = "nvvm.wgmma_scale_out";
  static constexpr ::llvm::StringLiteral dialectName = "nvvm";
  static WGMMAScaleOutAttr get(::mlir::MLIRContext *context, ::mlir::NVVM::WGMMAScaleOut value);
  static constexpr ::llvm::StringLiteral getMnemonic() {
    return {"wgmma_scale_out"};
  }

  static ::mlir::Attribute parse(::mlir::AsmParser &odsParser, ::mlir::Type odsType);
  void print(::mlir::AsmPrinter &odsPrinter) const;
  ::mlir::NVVM::WGMMAScaleOut getValue() const;
};
namespace detail {
struct WGMMATypesAttrStorage;
} // namespace detail
class WGMMATypesAttr : public ::mlir::Attribute::AttrBase<WGMMATypesAttr, ::mlir::Attribute, detail::WGMMATypesAttrStorage> {
public:
  using Base::Base;
  static constexpr ::llvm::StringLiteral name = "nvvm.wgmma_type";
  static constexpr ::llvm::StringLiteral dialectName = "nvvm";
  static WGMMATypesAttr get(::mlir::MLIRContext *context, ::mlir::NVVM::WGMMATypes value);
  static constexpr ::llvm::StringLiteral getMnemonic() {
    return {"wgmma_type"};
  }

  static ::mlir::Attribute parse(::mlir::AsmParser &odsParser, ::mlir::Type odsType);
  void print(::mlir::AsmPrinter &odsPrinter) const;
  ::mlir::NVVM::WGMMATypes getValue() const;
};
namespace detail {
struct NVVMTargetAttrStorage;
} // namespace detail
class NVVMTargetAttr : public ::mlir::Attribute::AttrBase<NVVMTargetAttr, ::mlir::Attribute, detail::NVVMTargetAttrStorage> {
public:
  using Base::Base;
  bool hasFlag(StringRef flag) const;
  bool hasFastMath() const;
  bool hasFtz() const;
  static constexpr ::llvm::StringLiteral name = "nvvm.target";
  static constexpr ::llvm::StringLiteral dialectName = "nvvm";
  using Base::getChecked;
  static NVVMTargetAttr get(::mlir::MLIRContext *context, int optLevel = 2, StringRef triple = "nvptx64-nvidia-cuda", StringRef chip = "sm_50", StringRef features = "+ptx60", DictionaryAttr targetFlags = nullptr, ArrayAttr linkFiles = nullptr);
  static NVVMTargetAttr getChecked(::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError, ::mlir::MLIRContext *context, int optLevel = 2, StringRef triple = "nvptx64-nvidia-cuda", StringRef chip = "sm_50", StringRef features = "+ptx60", DictionaryAttr targetFlags = nullptr, ArrayAttr linkFiles = nullptr);
  static ::llvm::LogicalResult verify(::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError, int O, ::llvm::StringRef triple, ::llvm::StringRef chip, ::llvm::StringRef features, DictionaryAttr flags, ArrayAttr link);
  static ::llvm::LogicalResult verifyInvariants(::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError, int O, ::llvm::StringRef triple, ::llvm::StringRef chip, ::llvm::StringRef features, DictionaryAttr flags, ArrayAttr link);
  static constexpr ::llvm::StringLiteral getMnemonic() {
    return {"target"};
  }

  static ::mlir::Attribute parse(::mlir::AsmParser &odsParser, ::mlir::Type odsType);
  void print(::mlir::AsmPrinter &odsPrinter) const;
  int getO() const;
  ::llvm::StringRef getTriple() const;
  ::llvm::StringRef getChip() const;
  ::llvm::StringRef getFeatures() const;
  DictionaryAttr getFlags() const;
  ArrayAttr getLink() const;
};
} // namespace NVVM
} // namespace mlir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::NVVM::ReduxKindAttr)
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::NVVM::SharedSpaceAttr)
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::NVVM::ProxyKindAttr)
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::NVVM::MemScopeKindAttr)
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::NVVM::SetMaxRegisterActionAttr)
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::NVVM::ShflKindAttr)
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::NVVM::LoadCacheModifierKindAttr)
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::NVVM::MMAB1OpAttr)
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::NVVM::MMAIntOverflowAttr)
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::NVVM::MMAShapeAttr)
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::NVVM::MMALayoutAttr)
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::NVVM::MMATypesAttr)
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::NVVM::MMAFragAttr)
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::NVVM::WGMMAScaleInAttr)
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::NVVM::WGMMAScaleOutAttr)
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::NVVM::WGMMATypesAttr)
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::NVVM::NVVMTargetAttr)

#endif  // GET_ATTRDEF_CLASSES

