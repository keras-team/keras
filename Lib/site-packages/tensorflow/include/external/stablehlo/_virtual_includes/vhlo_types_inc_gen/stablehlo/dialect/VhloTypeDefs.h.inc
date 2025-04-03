/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* TypeDef Declarations                                                       *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

#ifdef GET_TYPEDEF_CLASSES
#undef GET_TYPEDEF_CLASSES


namespace mlir {
class AsmParser;
class AsmPrinter;
} // namespace mlir
namespace mlir {
namespace vhlo {
class BooleanV1Type;
class ComplexV1Type;
class FloatBF16V1Type;
class FloatF16V1Type;
class FloatF32V1Type;
class FloatF64V1Type;
class FloatF8E3M4V1Type;
class FloatF8E4M3V1Type;
class FloatF8E4M3FNV1Type;
class FloatF8E5M2V1Type;
class FloatF8E4M3FNUZV1Type;
class FloatF8E4M3B11FNUZV1Type;
class FloatF8E5M2FNUZV1Type;
class FloatTF32V1Type;
class FunctionV1Type;
class IndexV1Type;
class IntegerSI2V1Type;
class IntegerSI4V1Type;
class IntegerSI8V1Type;
class IntegerSI16V1Type;
class IntegerSI32V1Type;
class IntegerSI64V1Type;
class IntegerUI2V1Type;
class IntegerUI4V1Type;
class IntegerUI8V1Type;
class IntegerUI16V1Type;
class IntegerUI32V1Type;
class IntegerUI64V1Type;
class NoneV1Type;
class RankedTensorV1Type;
class TokenV1Type;
class TupleV1Type;
class UniformQuantizedV1Type;
class UniformQuantizedPerAxisV1Type;
class UnrankedTensorV1Type;
class WitnessV1Type;
class BooleanV1Type : public ::mlir::Type::TypeBase<BooleanV1Type, ::mlir::Type, ::mlir::TypeStorage, ::mlir::vhlo::VersionedTypeInterface::Trait> {
public:
  using Base::Base;
  mlir::vhlo::Version getMinVersion() {
    return mlir::vhlo::Version(0, 9, 0);
  }
  mlir::vhlo::Version getMaxVersion() {
     return mlir::vhlo::Version::getCurrentVersion(); 
  }
  static constexpr ::llvm::StringLiteral name = "vhlo.bool_v1";
  static constexpr ::llvm::StringLiteral dialectName = "vhlo";
  static constexpr ::llvm::StringLiteral getMnemonic() {
    return {"bool_v1"};
  }

};
namespace detail {
struct ComplexV1TypeStorage;
} // namespace detail
class ComplexV1Type : public ::mlir::Type::TypeBase<ComplexV1Type, ::mlir::Type, detail::ComplexV1TypeStorage, ::mlir::vhlo::VersionedTypeInterface::Trait> {
public:
  using Base::Base;
  mlir::vhlo::Version getMinVersion() {
    return mlir::vhlo::Version(0, 9, 0);
  }
  mlir::vhlo::Version getMaxVersion() {
     return mlir::vhlo::Version::getCurrentVersion(); 
  }
  static constexpr ::llvm::StringLiteral name = "vhlo.complex_v1";
  static constexpr ::llvm::StringLiteral dialectName = "vhlo";
  using Base::getChecked;
  static ComplexV1Type get(::mlir::MLIRContext *context, Type elementType);
  static ComplexV1Type getChecked(::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError, ::mlir::MLIRContext *context, Type elementType);
  static ::llvm::LogicalResult verify(::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError, Type elementType);
  static ::llvm::LogicalResult verifyInvariants(::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError, Type elementType);
  static constexpr ::llvm::StringLiteral getMnemonic() {
    return {"complex_v1"};
  }

  static ::mlir::Type parse(::mlir::AsmParser &odsParser);
  void print(::mlir::AsmPrinter &odsPrinter) const;
  Type getElementType() const;
};
class FloatBF16V1Type : public ::mlir::Type::TypeBase<FloatBF16V1Type, ::mlir::Type, ::mlir::TypeStorage, ::mlir::vhlo::VersionedTypeInterface::Trait> {
public:
  using Base::Base;
  mlir::vhlo::Version getMinVersion() {
    return mlir::vhlo::Version(0, 9, 0);
  }
  mlir::vhlo::Version getMaxVersion() {
     return mlir::vhlo::Version::getCurrentVersion(); 
  }
  static constexpr ::llvm::StringLiteral name = "vhlo.bf16_v1";
  static constexpr ::llvm::StringLiteral dialectName = "vhlo";
  static constexpr ::llvm::StringLiteral getMnemonic() {
    return {"bf16_v1"};
  }

};
class FloatF16V1Type : public ::mlir::Type::TypeBase<FloatF16V1Type, ::mlir::Type, ::mlir::TypeStorage, ::mlir::vhlo::VersionedTypeInterface::Trait> {
public:
  using Base::Base;
  mlir::vhlo::Version getMinVersion() {
    return mlir::vhlo::Version(0, 9, 0);
  }
  mlir::vhlo::Version getMaxVersion() {
     return mlir::vhlo::Version::getCurrentVersion(); 
  }
  static constexpr ::llvm::StringLiteral name = "vhlo.f16_v1";
  static constexpr ::llvm::StringLiteral dialectName = "vhlo";
  static constexpr ::llvm::StringLiteral getMnemonic() {
    return {"f16_v1"};
  }

};
class FloatF32V1Type : public ::mlir::Type::TypeBase<FloatF32V1Type, ::mlir::Type, ::mlir::TypeStorage, ::mlir::vhlo::VersionedTypeInterface::Trait> {
public:
  using Base::Base;
  mlir::vhlo::Version getMinVersion() {
    return mlir::vhlo::Version(0, 9, 0);
  }
  mlir::vhlo::Version getMaxVersion() {
     return mlir::vhlo::Version::getCurrentVersion(); 
  }
  static constexpr ::llvm::StringLiteral name = "vhlo.f32_v1";
  static constexpr ::llvm::StringLiteral dialectName = "vhlo";
  static constexpr ::llvm::StringLiteral getMnemonic() {
    return {"f32_v1"};
  }

};
class FloatF64V1Type : public ::mlir::Type::TypeBase<FloatF64V1Type, ::mlir::Type, ::mlir::TypeStorage, ::mlir::vhlo::VersionedTypeInterface::Trait> {
public:
  using Base::Base;
  mlir::vhlo::Version getMinVersion() {
    return mlir::vhlo::Version(0, 9, 0);
  }
  mlir::vhlo::Version getMaxVersion() {
     return mlir::vhlo::Version::getCurrentVersion(); 
  }
  static constexpr ::llvm::StringLiteral name = "vhlo.f64_v1";
  static constexpr ::llvm::StringLiteral dialectName = "vhlo";
  static constexpr ::llvm::StringLiteral getMnemonic() {
    return {"f64_v1"};
  }

};
class FloatF8E3M4V1Type : public ::mlir::Type::TypeBase<FloatF8E3M4V1Type, ::mlir::Type, ::mlir::TypeStorage, ::mlir::vhlo::VersionedTypeInterface::Trait> {
public:
  using Base::Base;
  mlir::vhlo::Version getMinVersion() {
    return mlir::vhlo::Version(1, 7, 0);
  }
  mlir::vhlo::Version getMaxVersion() {
     return mlir::vhlo::Version::getCurrentVersion(); 
  }
  static constexpr ::llvm::StringLiteral name = "vhlo.f8E3M4_v1";
  static constexpr ::llvm::StringLiteral dialectName = "vhlo";
  static constexpr ::llvm::StringLiteral getMnemonic() {
    return {"f8E3M4_v1"};
  }

};
class FloatF8E4M3V1Type : public ::mlir::Type::TypeBase<FloatF8E4M3V1Type, ::mlir::Type, ::mlir::TypeStorage, ::mlir::vhlo::VersionedTypeInterface::Trait> {
public:
  using Base::Base;
  mlir::vhlo::Version getMinVersion() {
    return mlir::vhlo::Version(1, 7, 0);
  }
  mlir::vhlo::Version getMaxVersion() {
     return mlir::vhlo::Version::getCurrentVersion(); 
  }
  static constexpr ::llvm::StringLiteral name = "vhlo.f8E4M3_v1";
  static constexpr ::llvm::StringLiteral dialectName = "vhlo";
  static constexpr ::llvm::StringLiteral getMnemonic() {
    return {"f8E4M3_v1"};
  }

};
class FloatF8E4M3FNV1Type : public ::mlir::Type::TypeBase<FloatF8E4M3FNV1Type, ::mlir::Type, ::mlir::TypeStorage, ::mlir::vhlo::VersionedTypeInterface::Trait> {
public:
  using Base::Base;
  mlir::vhlo::Version getMinVersion() {
    return mlir::vhlo::Version(0, 9, 0);
  }
  mlir::vhlo::Version getMaxVersion() {
     return mlir::vhlo::Version::getCurrentVersion(); 
  }
  static constexpr ::llvm::StringLiteral name = "vhlo.f8E4M3FN_v1";
  static constexpr ::llvm::StringLiteral dialectName = "vhlo";
  static constexpr ::llvm::StringLiteral getMnemonic() {
    return {"f8E4M3FN_v1"};
  }

};
class FloatF8E5M2V1Type : public ::mlir::Type::TypeBase<FloatF8E5M2V1Type, ::mlir::Type, ::mlir::TypeStorage, ::mlir::vhlo::VersionedTypeInterface::Trait> {
public:
  using Base::Base;
  mlir::vhlo::Version getMinVersion() {
    return mlir::vhlo::Version(0, 9, 0);
  }
  mlir::vhlo::Version getMaxVersion() {
     return mlir::vhlo::Version::getCurrentVersion(); 
  }
  static constexpr ::llvm::StringLiteral name = "vhlo.f8E5M2_v1";
  static constexpr ::llvm::StringLiteral dialectName = "vhlo";
  static constexpr ::llvm::StringLiteral getMnemonic() {
    return {"f8E5M2_v1"};
  }

};
class FloatF8E4M3FNUZV1Type : public ::mlir::Type::TypeBase<FloatF8E4M3FNUZV1Type, ::mlir::Type, ::mlir::TypeStorage, ::mlir::vhlo::VersionedTypeInterface::Trait> {
public:
  using Base::Base;
  mlir::vhlo::Version getMinVersion() {
    return mlir::vhlo::Version(0, 10, 0);
  }
  mlir::vhlo::Version getMaxVersion() {
     return mlir::vhlo::Version::getCurrentVersion(); 
  }
  static constexpr ::llvm::StringLiteral name = "vhlo.f8E4M3FNUZ_v1";
  static constexpr ::llvm::StringLiteral dialectName = "vhlo";
  static constexpr ::llvm::StringLiteral getMnemonic() {
    return {"f8E4M3FNUZ_v1"};
  }

};
class FloatF8E4M3B11FNUZV1Type : public ::mlir::Type::TypeBase<FloatF8E4M3B11FNUZV1Type, ::mlir::Type, ::mlir::TypeStorage, ::mlir::vhlo::VersionedTypeInterface::Trait> {
public:
  using Base::Base;
  mlir::vhlo::Version getMinVersion() {
    return mlir::vhlo::Version(0, 11, 0);
  }
  mlir::vhlo::Version getMaxVersion() {
     return mlir::vhlo::Version::getCurrentVersion(); 
  }
  static constexpr ::llvm::StringLiteral name = "vhlo.f8E4M3B11FNUZ_v1";
  static constexpr ::llvm::StringLiteral dialectName = "vhlo";
  static constexpr ::llvm::StringLiteral getMnemonic() {
    return {"f8E4M3B11FNUZ_v1"};
  }

};
class FloatF8E5M2FNUZV1Type : public ::mlir::Type::TypeBase<FloatF8E5M2FNUZV1Type, ::mlir::Type, ::mlir::TypeStorage, ::mlir::vhlo::VersionedTypeInterface::Trait> {
public:
  using Base::Base;
  mlir::vhlo::Version getMinVersion() {
    return mlir::vhlo::Version(0, 10, 0);
  }
  mlir::vhlo::Version getMaxVersion() {
     return mlir::vhlo::Version::getCurrentVersion(); 
  }
  static constexpr ::llvm::StringLiteral name = "vhlo.f8E5M2FNUZ_v1";
  static constexpr ::llvm::StringLiteral dialectName = "vhlo";
  static constexpr ::llvm::StringLiteral getMnemonic() {
    return {"f8E5M2FNUZ_v1"};
  }

};
class FloatTF32V1Type : public ::mlir::Type::TypeBase<FloatTF32V1Type, ::mlir::Type, ::mlir::TypeStorage, ::mlir::vhlo::VersionedTypeInterface::Trait> {
public:
  using Base::Base;
  mlir::vhlo::Version getMinVersion() {
    return mlir::vhlo::Version(1, 6, 0);
  }
  mlir::vhlo::Version getMaxVersion() {
     return mlir::vhlo::Version::getCurrentVersion(); 
  }
  static constexpr ::llvm::StringLiteral name = "vhlo.tf31_v1";
  static constexpr ::llvm::StringLiteral dialectName = "vhlo";
  static constexpr ::llvm::StringLiteral getMnemonic() {
    return {"tf31_v1"};
  }

};
namespace detail {
struct FunctionV1TypeStorage;
} // namespace detail
class FunctionV1Type : public ::mlir::Type::TypeBase<FunctionV1Type, ::mlir::Type, detail::FunctionV1TypeStorage, ::mlir::vhlo::VersionedTypeInterface::Trait> {
public:
  using Base::Base;
  mlir::vhlo::Version getMinVersion() {
    return mlir::vhlo::Version(0, 9, 0);
  }
  mlir::vhlo::Version getMaxVersion() {
     return mlir::vhlo::Version::getCurrentVersion(); 
  }
  static constexpr ::llvm::StringLiteral name = "vhlo.func_v1";
  static constexpr ::llvm::StringLiteral dialectName = "vhlo";
  using Base::getChecked;
  static FunctionV1Type get(::mlir::MLIRContext *context, ::llvm::ArrayRef<::mlir::Type> inputs, ::llvm::ArrayRef<mlir::Type> outputs);
  static FunctionV1Type getChecked(::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError, ::mlir::MLIRContext *context, ::llvm::ArrayRef<::mlir::Type> inputs, ::llvm::ArrayRef<mlir::Type> outputs);
  static ::llvm::LogicalResult verify(::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError, ::llvm::ArrayRef<::mlir::Type> inputs, ::llvm::ArrayRef<mlir::Type> outputs);
  static ::llvm::LogicalResult verifyInvariants(::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError, ::llvm::ArrayRef<::mlir::Type> inputs, ::llvm::ArrayRef<mlir::Type> outputs);
  static constexpr ::llvm::StringLiteral getMnemonic() {
    return {"func_v1"};
  }

  static ::mlir::Type parse(::mlir::AsmParser &odsParser);
  void print(::mlir::AsmPrinter &odsPrinter) const;
  ::llvm::ArrayRef<::mlir::Type> getInputs() const;
  ::llvm::ArrayRef<mlir::Type> getOutputs() const;
};
class IndexV1Type : public ::mlir::Type::TypeBase<IndexV1Type, ::mlir::Type, ::mlir::TypeStorage, ::mlir::vhlo::VersionedTypeInterface::Trait> {
public:
  using Base::Base;
  mlir::vhlo::Version getMinVersion() {
    return mlir::vhlo::Version(0, 9, 0);
  }
  mlir::vhlo::Version getMaxVersion() {
     return mlir::vhlo::Version::getCurrentVersion(); 
  }
  static constexpr ::llvm::StringLiteral name = "vhlo.index_v1";
  static constexpr ::llvm::StringLiteral dialectName = "vhlo";
  static constexpr ::llvm::StringLiteral getMnemonic() {
    return {"index_v1"};
  }

};
class IntegerSI2V1Type : public ::mlir::Type::TypeBase<IntegerSI2V1Type, ::mlir::Type, ::mlir::TypeStorage, ::mlir::vhlo::VersionedTypeInterface::Trait> {
public:
  using Base::Base;
  mlir::vhlo::Version getMinVersion() {
    return mlir::vhlo::Version(1, 2, 0);
  }
  mlir::vhlo::Version getMaxVersion() {
     return mlir::vhlo::Version::getCurrentVersion(); 
  }
  static constexpr ::llvm::StringLiteral name = "vhlo.i2_v1";
  static constexpr ::llvm::StringLiteral dialectName = "vhlo";
  static constexpr ::llvm::StringLiteral getMnemonic() {
    return {"i2_v1"};
  }

};
class IntegerSI4V1Type : public ::mlir::Type::TypeBase<IntegerSI4V1Type, ::mlir::Type, ::mlir::TypeStorage, ::mlir::vhlo::VersionedTypeInterface::Trait> {
public:
  using Base::Base;
  mlir::vhlo::Version getMinVersion() {
    return mlir::vhlo::Version(0, 9, 0);
  }
  mlir::vhlo::Version getMaxVersion() {
     return mlir::vhlo::Version::getCurrentVersion(); 
  }
  static constexpr ::llvm::StringLiteral name = "vhlo.i4_v1";
  static constexpr ::llvm::StringLiteral dialectName = "vhlo";
  static constexpr ::llvm::StringLiteral getMnemonic() {
    return {"i4_v1"};
  }

};
class IntegerSI8V1Type : public ::mlir::Type::TypeBase<IntegerSI8V1Type, ::mlir::Type, ::mlir::TypeStorage, ::mlir::vhlo::VersionedTypeInterface::Trait> {
public:
  using Base::Base;
  mlir::vhlo::Version getMinVersion() {
    return mlir::vhlo::Version(0, 9, 0);
  }
  mlir::vhlo::Version getMaxVersion() {
     return mlir::vhlo::Version::getCurrentVersion(); 
  }
  static constexpr ::llvm::StringLiteral name = "vhlo.i8_v1";
  static constexpr ::llvm::StringLiteral dialectName = "vhlo";
  static constexpr ::llvm::StringLiteral getMnemonic() {
    return {"i8_v1"};
  }

};
class IntegerSI16V1Type : public ::mlir::Type::TypeBase<IntegerSI16V1Type, ::mlir::Type, ::mlir::TypeStorage, ::mlir::vhlo::VersionedTypeInterface::Trait> {
public:
  using Base::Base;
  mlir::vhlo::Version getMinVersion() {
    return mlir::vhlo::Version(0, 9, 0);
  }
  mlir::vhlo::Version getMaxVersion() {
     return mlir::vhlo::Version::getCurrentVersion(); 
  }
  static constexpr ::llvm::StringLiteral name = "vhlo.i16_v1";
  static constexpr ::llvm::StringLiteral dialectName = "vhlo";
  static constexpr ::llvm::StringLiteral getMnemonic() {
    return {"i16_v1"};
  }

};
class IntegerSI32V1Type : public ::mlir::Type::TypeBase<IntegerSI32V1Type, ::mlir::Type, ::mlir::TypeStorage, ::mlir::vhlo::VersionedTypeInterface::Trait> {
public:
  using Base::Base;
  mlir::vhlo::Version getMinVersion() {
    return mlir::vhlo::Version(0, 9, 0);
  }
  mlir::vhlo::Version getMaxVersion() {
     return mlir::vhlo::Version::getCurrentVersion(); 
  }
  static constexpr ::llvm::StringLiteral name = "vhlo.i32_v1";
  static constexpr ::llvm::StringLiteral dialectName = "vhlo";
  static constexpr ::llvm::StringLiteral getMnemonic() {
    return {"i32_v1"};
  }

};
class IntegerSI64V1Type : public ::mlir::Type::TypeBase<IntegerSI64V1Type, ::mlir::Type, ::mlir::TypeStorage, ::mlir::vhlo::VersionedTypeInterface::Trait> {
public:
  using Base::Base;
  mlir::vhlo::Version getMinVersion() {
    return mlir::vhlo::Version(0, 9, 0);
  }
  mlir::vhlo::Version getMaxVersion() {
     return mlir::vhlo::Version::getCurrentVersion(); 
  }
  static constexpr ::llvm::StringLiteral name = "vhlo.i64_v1";
  static constexpr ::llvm::StringLiteral dialectName = "vhlo";
  static constexpr ::llvm::StringLiteral getMnemonic() {
    return {"i64_v1"};
  }

};
class IntegerUI2V1Type : public ::mlir::Type::TypeBase<IntegerUI2V1Type, ::mlir::Type, ::mlir::TypeStorage, ::mlir::vhlo::VersionedTypeInterface::Trait> {
public:
  using Base::Base;
  mlir::vhlo::Version getMinVersion() {
    return mlir::vhlo::Version(1, 2, 0);
  }
  mlir::vhlo::Version getMaxVersion() {
     return mlir::vhlo::Version::getCurrentVersion(); 
  }
  static constexpr ::llvm::StringLiteral name = "vhlo.ui2_v1";
  static constexpr ::llvm::StringLiteral dialectName = "vhlo";
  static constexpr ::llvm::StringLiteral getMnemonic() {
    return {"ui2_v1"};
  }

};
class IntegerUI4V1Type : public ::mlir::Type::TypeBase<IntegerUI4V1Type, ::mlir::Type, ::mlir::TypeStorage, ::mlir::vhlo::VersionedTypeInterface::Trait> {
public:
  using Base::Base;
  mlir::vhlo::Version getMinVersion() {
    return mlir::vhlo::Version(0, 9, 0);
  }
  mlir::vhlo::Version getMaxVersion() {
     return mlir::vhlo::Version::getCurrentVersion(); 
  }
  static constexpr ::llvm::StringLiteral name = "vhlo.ui4_v1";
  static constexpr ::llvm::StringLiteral dialectName = "vhlo";
  static constexpr ::llvm::StringLiteral getMnemonic() {
    return {"ui4_v1"};
  }

};
class IntegerUI8V1Type : public ::mlir::Type::TypeBase<IntegerUI8V1Type, ::mlir::Type, ::mlir::TypeStorage, ::mlir::vhlo::VersionedTypeInterface::Trait> {
public:
  using Base::Base;
  mlir::vhlo::Version getMinVersion() {
    return mlir::vhlo::Version(0, 9, 0);
  }
  mlir::vhlo::Version getMaxVersion() {
     return mlir::vhlo::Version::getCurrentVersion(); 
  }
  static constexpr ::llvm::StringLiteral name = "vhlo.ui8_v1";
  static constexpr ::llvm::StringLiteral dialectName = "vhlo";
  static constexpr ::llvm::StringLiteral getMnemonic() {
    return {"ui8_v1"};
  }

};
class IntegerUI16V1Type : public ::mlir::Type::TypeBase<IntegerUI16V1Type, ::mlir::Type, ::mlir::TypeStorage, ::mlir::vhlo::VersionedTypeInterface::Trait> {
public:
  using Base::Base;
  mlir::vhlo::Version getMinVersion() {
    return mlir::vhlo::Version(0, 9, 0);
  }
  mlir::vhlo::Version getMaxVersion() {
     return mlir::vhlo::Version::getCurrentVersion(); 
  }
  static constexpr ::llvm::StringLiteral name = "vhlo.ui16_v1";
  static constexpr ::llvm::StringLiteral dialectName = "vhlo";
  static constexpr ::llvm::StringLiteral getMnemonic() {
    return {"ui16_v1"};
  }

};
class IntegerUI32V1Type : public ::mlir::Type::TypeBase<IntegerUI32V1Type, ::mlir::Type, ::mlir::TypeStorage, ::mlir::vhlo::VersionedTypeInterface::Trait> {
public:
  using Base::Base;
  mlir::vhlo::Version getMinVersion() {
    return mlir::vhlo::Version(0, 9, 0);
  }
  mlir::vhlo::Version getMaxVersion() {
     return mlir::vhlo::Version::getCurrentVersion(); 
  }
  static constexpr ::llvm::StringLiteral name = "vhlo.ui32_v1";
  static constexpr ::llvm::StringLiteral dialectName = "vhlo";
  static constexpr ::llvm::StringLiteral getMnemonic() {
    return {"ui32_v1"};
  }

};
class IntegerUI64V1Type : public ::mlir::Type::TypeBase<IntegerUI64V1Type, ::mlir::Type, ::mlir::TypeStorage, ::mlir::vhlo::VersionedTypeInterface::Trait> {
public:
  using Base::Base;
  mlir::vhlo::Version getMinVersion() {
    return mlir::vhlo::Version(0, 9, 0);
  }
  mlir::vhlo::Version getMaxVersion() {
     return mlir::vhlo::Version::getCurrentVersion(); 
  }
  static constexpr ::llvm::StringLiteral name = "vhlo.ui64_v1";
  static constexpr ::llvm::StringLiteral dialectName = "vhlo";
  static constexpr ::llvm::StringLiteral getMnemonic() {
    return {"ui64_v1"};
  }

};
class NoneV1Type : public ::mlir::Type::TypeBase<NoneV1Type, ::mlir::Type, ::mlir::TypeStorage, ::mlir::vhlo::VersionedTypeInterface::Trait> {
public:
  using Base::Base;
  mlir::vhlo::Version getMinVersion() {
    return mlir::vhlo::Version(1, 6, 0);
  }
  mlir::vhlo::Version getMaxVersion() {
     return mlir::vhlo::Version::getCurrentVersion(); 
  }
  static constexpr ::llvm::StringLiteral name = "vhlo.none_v1";
  static constexpr ::llvm::StringLiteral dialectName = "vhlo";
  static constexpr ::llvm::StringLiteral getMnemonic() {
    return {"none_v1"};
  }

};
namespace detail {
struct RankedTensorV1TypeStorage;
} // namespace detail
class RankedTensorV1Type : public ::mlir::Type::TypeBase<RankedTensorV1Type, ::mlir::Type, detail::RankedTensorV1TypeStorage, ::mlir::vhlo::VersionedTypeInterface::Trait> {
public:
  using Base::Base;
  mlir::vhlo::Version getMinVersion() {
    return mlir::vhlo::Version(0, 9, 0);
  }
  mlir::vhlo::Version getMaxVersion() {
     return mlir::vhlo::Version::getCurrentVersion(); 
  }
  static constexpr ::llvm::StringLiteral name = "vhlo.tensor_v1";
  static constexpr ::llvm::StringLiteral dialectName = "vhlo";
  using Base::getChecked;
  static RankedTensorV1Type get(::mlir::MLIRContext *context, ::llvm::ArrayRef<int64_t> shape, ::mlir::Type elementType, ::mlir::Attribute encoding);
  static RankedTensorV1Type getChecked(::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError, ::mlir::MLIRContext *context, ::llvm::ArrayRef<int64_t> shape, ::mlir::Type elementType, ::mlir::Attribute encoding);
  static ::llvm::LogicalResult verify(::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError, ::llvm::ArrayRef<int64_t> shape, ::mlir::Type elementType, ::mlir::Attribute encoding);
  static ::llvm::LogicalResult verifyInvariants(::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError, ::llvm::ArrayRef<int64_t> shape, ::mlir::Type elementType, ::mlir::Attribute encoding);
  static constexpr ::llvm::StringLiteral getMnemonic() {
    return {"tensor_v1"};
  }

  static ::mlir::Type parse(::mlir::AsmParser &odsParser);
  void print(::mlir::AsmPrinter &odsPrinter) const;
  ::llvm::ArrayRef<int64_t> getShape() const;
  ::mlir::Type getElementType() const;
  ::mlir::Attribute getEncoding() const;
};
class TokenV1Type : public ::mlir::Type::TypeBase<TokenV1Type, ::mlir::Type, ::mlir::TypeStorage, ::mlir::vhlo::VersionedTypeInterface::Trait> {
public:
  using Base::Base;
  mlir::vhlo::Version getMinVersion() {
    return mlir::vhlo::Version(0, 9, 0);
  }
  mlir::vhlo::Version getMaxVersion() {
     return mlir::vhlo::Version::getCurrentVersion(); 
  }
  static constexpr ::llvm::StringLiteral name = "vhlo.token_v1";
  static constexpr ::llvm::StringLiteral dialectName = "vhlo";
  static constexpr ::llvm::StringLiteral getMnemonic() {
    return {"token_v1"};
  }

};
namespace detail {
struct TupleV1TypeStorage;
} // namespace detail
class TupleV1Type : public ::mlir::Type::TypeBase<TupleV1Type, ::mlir::Type, detail::TupleV1TypeStorage, ::mlir::vhlo::VersionedTypeInterface::Trait> {
public:
  using Base::Base;
  mlir::vhlo::Version getMinVersion() {
    return mlir::vhlo::Version(0, 9, 0);
  }
  mlir::vhlo::Version getMaxVersion() {
     return mlir::vhlo::Version::getCurrentVersion(); 
  }
  static constexpr ::llvm::StringLiteral name = "vhlo.tuple_v1";
  static constexpr ::llvm::StringLiteral dialectName = "vhlo";
  using Base::getChecked;
  static TupleV1Type get(::mlir::MLIRContext *context, ::llvm::ArrayRef<::mlir::Type> types);
  static TupleV1Type getChecked(::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError, ::mlir::MLIRContext *context, ::llvm::ArrayRef<::mlir::Type> types);
  static ::llvm::LogicalResult verify(::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError, ::llvm::ArrayRef<::mlir::Type> types);
  static ::llvm::LogicalResult verifyInvariants(::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError, ::llvm::ArrayRef<::mlir::Type> types);
  static constexpr ::llvm::StringLiteral getMnemonic() {
    return {"tuple_v1"};
  }

  static ::mlir::Type parse(::mlir::AsmParser &odsParser);
  void print(::mlir::AsmPrinter &odsPrinter) const;
  ::llvm::ArrayRef<::mlir::Type> getTypes() const;
};
namespace detail {
struct UniformQuantizedV1TypeStorage;
} // namespace detail
class UniformQuantizedV1Type : public ::mlir::Type::TypeBase<UniformQuantizedV1Type, ::mlir::Type, detail::UniformQuantizedV1TypeStorage, ::mlir::vhlo::VersionedTypeInterface::Trait> {
public:
  using Base::Base;
  mlir::vhlo::Version getMinVersion() {
    return mlir::vhlo::Version(0, 9, 0);
  }
  mlir::vhlo::Version getMaxVersion() {
     return mlir::vhlo::Version::getCurrentVersion(); 
  }
  static constexpr ::llvm::StringLiteral name = "vhlo.quant_v1";
  static constexpr ::llvm::StringLiteral dialectName = "vhlo";
  using Base::getChecked;
  static UniformQuantizedV1Type get(::mlir::MLIRContext *context, unsigned flags, ::mlir::Type storageType, ::mlir::Type expressedType, ::llvm::APFloat scale, int64_t zeroPoint, int64_t storageTypeMin, int64_t storageTypeMax);
  static UniformQuantizedV1Type getChecked(::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError, ::mlir::MLIRContext *context, unsigned flags, ::mlir::Type storageType, ::mlir::Type expressedType, ::llvm::APFloat scale, int64_t zeroPoint, int64_t storageTypeMin, int64_t storageTypeMax);
  static ::llvm::LogicalResult verify(::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError, unsigned flags, ::mlir::Type storageType, ::mlir::Type expressedType, ::llvm::APFloat scale, int64_t zeroPoint, int64_t storageTypeMin, int64_t storageTypeMax);
  static ::llvm::LogicalResult verifyInvariants(::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError, unsigned flags, ::mlir::Type storageType, ::mlir::Type expressedType, ::llvm::APFloat scale, int64_t zeroPoint, int64_t storageTypeMin, int64_t storageTypeMax);
  static constexpr ::llvm::StringLiteral getMnemonic() {
    return {"quant_v1"};
  }

  static ::mlir::Type parse(::mlir::AsmParser &odsParser);
  void print(::mlir::AsmPrinter &odsPrinter) const;
  unsigned getFlags() const;
  ::mlir::Type getStorageType() const;
  ::mlir::Type getExpressedType() const;
  ::llvm::APFloat getScale() const;
  int64_t getZeroPoint() const;
  int64_t getStorageTypeMin() const;
  int64_t getStorageTypeMax() const;
};
namespace detail {
struct UniformQuantizedPerAxisV1TypeStorage;
} // namespace detail
class UniformQuantizedPerAxisV1Type : public ::mlir::Type::TypeBase<UniformQuantizedPerAxisV1Type, ::mlir::Type, detail::UniformQuantizedPerAxisV1TypeStorage, ::mlir::vhlo::VersionedTypeInterface::Trait> {
public:
  using Base::Base;
  mlir::vhlo::Version getMinVersion() {
    return mlir::vhlo::Version(0, 18, 0);
  }
  mlir::vhlo::Version getMaxVersion() {
     return mlir::vhlo::Version::getCurrentVersion(); 
  }
  static constexpr ::llvm::StringLiteral name = "vhlo.quant_per_axis_v1";
  static constexpr ::llvm::StringLiteral dialectName = "vhlo";
  using Base::getChecked;
  static UniformQuantizedPerAxisV1Type get(::mlir::MLIRContext *context, unsigned flags, ::mlir::Type storageType, ::mlir::Type expressedType, int32_t quantizedDimension, ::llvm::ArrayRef<::llvm::APFloat> scales, ::llvm::ArrayRef<int64_t> zeroPoints, int64_t storageTypeMin, int64_t storageTypeMax);
  static UniformQuantizedPerAxisV1Type getChecked(::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError, ::mlir::MLIRContext *context, unsigned flags, ::mlir::Type storageType, ::mlir::Type expressedType, int32_t quantizedDimension, ::llvm::ArrayRef<::llvm::APFloat> scales, ::llvm::ArrayRef<int64_t> zeroPoints, int64_t storageTypeMin, int64_t storageTypeMax);
  static ::llvm::LogicalResult verify(::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError, unsigned flags, ::mlir::Type storageType, ::mlir::Type expressedType, int32_t quantizedDimension, ::llvm::ArrayRef<::llvm::APFloat> scales, ::llvm::ArrayRef<int64_t> zeroPoints, int64_t storageTypeMin, int64_t storageTypeMax);
  static ::llvm::LogicalResult verifyInvariants(::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError, unsigned flags, ::mlir::Type storageType, ::mlir::Type expressedType, int32_t quantizedDimension, ::llvm::ArrayRef<::llvm::APFloat> scales, ::llvm::ArrayRef<int64_t> zeroPoints, int64_t storageTypeMin, int64_t storageTypeMax);
  static constexpr ::llvm::StringLiteral getMnemonic() {
    return {"quant_per_axis_v1"};
  }

  static ::mlir::Type parse(::mlir::AsmParser &odsParser);
  void print(::mlir::AsmPrinter &odsPrinter) const;
  unsigned getFlags() const;
  ::mlir::Type getStorageType() const;
  ::mlir::Type getExpressedType() const;
  int32_t getQuantizedDimension() const;
  ::llvm::ArrayRef<::llvm::APFloat> getScales() const;
  ::llvm::ArrayRef<int64_t> getZeroPoints() const;
  int64_t getStorageTypeMin() const;
  int64_t getStorageTypeMax() const;
};
namespace detail {
struct UnrankedTensorV1TypeStorage;
} // namespace detail
class UnrankedTensorV1Type : public ::mlir::Type::TypeBase<UnrankedTensorV1Type, ::mlir::Type, detail::UnrankedTensorV1TypeStorage, ::mlir::vhlo::VersionedTypeInterface::Trait> {
public:
  using Base::Base;
  mlir::vhlo::Version getMinVersion() {
    return mlir::vhlo::Version(0, 9, 0);
  }
  mlir::vhlo::Version getMaxVersion() {
     return mlir::vhlo::Version::getCurrentVersion(); 
  }
  static constexpr ::llvm::StringLiteral name = "vhlo.unranked_tensor_v1";
  static constexpr ::llvm::StringLiteral dialectName = "vhlo";
  using Base::getChecked;
  static UnrankedTensorV1Type get(::mlir::MLIRContext *context, ::mlir::Type elementType);
  static UnrankedTensorV1Type getChecked(::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError, ::mlir::MLIRContext *context, ::mlir::Type elementType);
  static ::llvm::LogicalResult verify(::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError, ::mlir::Type elementType);
  static ::llvm::LogicalResult verifyInvariants(::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError, ::mlir::Type elementType);
  static constexpr ::llvm::StringLiteral getMnemonic() {
    return {"unranked_tensor_v1"};
  }

  static ::mlir::Type parse(::mlir::AsmParser &odsParser);
  void print(::mlir::AsmPrinter &odsPrinter) const;
  ::mlir::Type getElementType() const;
};
class WitnessV1Type : public ::mlir::Type::TypeBase<WitnessV1Type, ::mlir::Type, ::mlir::TypeStorage, ::mlir::vhlo::VersionedTypeInterface::Trait> {
public:
  using Base::Base;
  mlir::vhlo::Version getMinVersion() {
    return mlir::vhlo::Version(0, 9, 0);
  }
  mlir::vhlo::Version getMaxVersion() {
     return mlir::vhlo::Version::getCurrentVersion(); 
  }
  static constexpr ::llvm::StringLiteral name = "vhlo.witness_v1";
  static constexpr ::llvm::StringLiteral dialectName = "vhlo";
  static constexpr ::llvm::StringLiteral getMnemonic() {
    return {"witness_v1"};
  }

};
} // namespace vhlo
} // namespace mlir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::vhlo::BooleanV1Type)
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::vhlo::ComplexV1Type)
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::vhlo::FloatBF16V1Type)
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::vhlo::FloatF16V1Type)
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::vhlo::FloatF32V1Type)
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::vhlo::FloatF64V1Type)
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::vhlo::FloatF8E3M4V1Type)
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::vhlo::FloatF8E4M3V1Type)
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::vhlo::FloatF8E4M3FNV1Type)
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::vhlo::FloatF8E5M2V1Type)
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::vhlo::FloatF8E4M3FNUZV1Type)
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::vhlo::FloatF8E4M3B11FNUZV1Type)
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::vhlo::FloatF8E5M2FNUZV1Type)
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::vhlo::FloatTF32V1Type)
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::vhlo::FunctionV1Type)
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::vhlo::IndexV1Type)
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::vhlo::IntegerSI2V1Type)
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::vhlo::IntegerSI4V1Type)
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::vhlo::IntegerSI8V1Type)
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::vhlo::IntegerSI16V1Type)
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::vhlo::IntegerSI32V1Type)
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::vhlo::IntegerSI64V1Type)
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::vhlo::IntegerUI2V1Type)
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::vhlo::IntegerUI4V1Type)
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::vhlo::IntegerUI8V1Type)
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::vhlo::IntegerUI16V1Type)
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::vhlo::IntegerUI32V1Type)
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::vhlo::IntegerUI64V1Type)
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::vhlo::NoneV1Type)
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::vhlo::RankedTensorV1Type)
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::vhlo::TokenV1Type)
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::vhlo::TupleV1Type)
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::vhlo::UniformQuantizedV1Type)
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::vhlo::UniformQuantizedPerAxisV1Type)
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::vhlo::UnrankedTensorV1Type)
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::vhlo::WitnessV1Type)

#endif  // GET_TYPEDEF_CLASSES

