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
namespace nvgpu {
class DeviceAsyncTokenType;
class MBarrierGroupType;
class MBarrierTokenType;
class TensorMapDescriptorType;
class WarpgroupMatrixDescriptorType;
class WarpgroupAccumulatorType;
class DeviceAsyncTokenType : public ::mlir::Type::TypeBase<DeviceAsyncTokenType, ::mlir::Type, ::mlir::TypeStorage> {
public:
  using Base::Base;
  static constexpr ::llvm::StringLiteral name = "nvgpu.device.async.token";
  static constexpr ::llvm::StringLiteral dialectName = "nvgpu";
  static constexpr ::llvm::StringLiteral getMnemonic() {
    return {"device.async.token"};
  }

};
namespace detail {
struct MBarrierGroupTypeStorage;
} // namespace detail
class MBarrierGroupType : public ::mlir::Type::TypeBase<MBarrierGroupType, ::mlir::Type, detail::MBarrierGroupTypeStorage> {
public:
  using Base::Base;
  static constexpr ::llvm::StringLiteral name = "nvgpu.mbarrier.group";
  static constexpr ::llvm::StringLiteral dialectName = "nvgpu";
  static MBarrierGroupType get(::mlir::MLIRContext *context, Attribute memorySpace, unsigned num_barriers);
  static MBarrierGroupType get(::mlir::MLIRContext *context, Attribute memorySpace);
  static constexpr ::llvm::StringLiteral getMnemonic() {
    return {"mbarrier.group"};
  }

  static ::mlir::Type parse(::mlir::AsmParser &odsParser);
  void print(::mlir::AsmPrinter &odsPrinter) const;
  Attribute getMemorySpace() const;
  unsigned getNumBarriers() const;
};
class MBarrierTokenType : public ::mlir::Type::TypeBase<MBarrierTokenType, ::mlir::Type, ::mlir::TypeStorage> {
public:
  using Base::Base;
  static constexpr ::llvm::StringLiteral name = "nvgpu.mbarrier.token";
  static constexpr ::llvm::StringLiteral dialectName = "nvgpu";
  static constexpr ::llvm::StringLiteral getMnemonic() {
    return {"mbarrier.token"};
  }

};
namespace detail {
struct TensorMapDescriptorTypeStorage;
} // namespace detail
class TensorMapDescriptorType : public ::mlir::Type::TypeBase<TensorMapDescriptorType, ::mlir::Type, detail::TensorMapDescriptorTypeStorage> {
public:
  using Base::Base;
  static constexpr ::llvm::StringLiteral name = "nvgpu.tensormap.descriptor";
  static constexpr ::llvm::StringLiteral dialectName = "nvgpu";
  static TensorMapDescriptorType get(::mlir::MLIRContext *context, MemRefType tensor, ::mlir::nvgpu::TensorMapSwizzleKind swizzle, ::mlir::nvgpu::TensorMapL2PromoKind l2promo, ::mlir::nvgpu::TensorMapOOBKind oob, ::mlir::nvgpu::TensorMapInterleaveKind interleave);
  static constexpr ::llvm::StringLiteral getMnemonic() {
    return {"tensormap.descriptor"};
  }

  static ::mlir::Type parse(::mlir::AsmParser &odsParser);
  void print(::mlir::AsmPrinter &odsPrinter) const;
  MemRefType getTensor() const;
  ::mlir::nvgpu::TensorMapSwizzleKind getSwizzle() const;
  ::mlir::nvgpu::TensorMapL2PromoKind getL2promo() const;
  ::mlir::nvgpu::TensorMapOOBKind getOob() const;
  ::mlir::nvgpu::TensorMapInterleaveKind getInterleave() const;
};
namespace detail {
struct WarpgroupMatrixDescriptorTypeStorage;
} // namespace detail
class WarpgroupMatrixDescriptorType : public ::mlir::Type::TypeBase<WarpgroupMatrixDescriptorType, ::mlir::Type, detail::WarpgroupMatrixDescriptorTypeStorage> {
public:
  using Base::Base;
  static constexpr ::llvm::StringLiteral name = "nvgpu.warpgroup.descriptor";
  static constexpr ::llvm::StringLiteral dialectName = "nvgpu";
  static WarpgroupMatrixDescriptorType get(::mlir::MLIRContext *context, MemRefType tensor);
  static constexpr ::llvm::StringLiteral getMnemonic() {
    return {"warpgroup.descriptor"};
  }

  static ::mlir::Type parse(::mlir::AsmParser &odsParser);
  void print(::mlir::AsmPrinter &odsPrinter) const;
  MemRefType getTensor() const;
};
namespace detail {
struct WarpgroupAccumulatorTypeStorage;
} // namespace detail
class WarpgroupAccumulatorType : public ::mlir::Type::TypeBase<WarpgroupAccumulatorType, ::mlir::Type, detail::WarpgroupAccumulatorTypeStorage> {
public:
  using Base::Base;
  static constexpr ::llvm::StringLiteral name = "nvgpu.warpgroup.accumulator";
  static constexpr ::llvm::StringLiteral dialectName = "nvgpu";
  static WarpgroupAccumulatorType get(::mlir::MLIRContext *context, VectorType fragmented);
  static constexpr ::llvm::StringLiteral getMnemonic() {
    return {"warpgroup.accumulator"};
  }

  static ::mlir::Type parse(::mlir::AsmParser &odsParser);
  void print(::mlir::AsmPrinter &odsPrinter) const;
  VectorType getFragmented() const;
};
} // namespace nvgpu
} // namespace mlir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::nvgpu::DeviceAsyncTokenType)
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::nvgpu::MBarrierGroupType)
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::nvgpu::MBarrierTokenType)
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::nvgpu::TensorMapDescriptorType)
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::nvgpu::WarpgroupMatrixDescriptorType)
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::nvgpu::WarpgroupAccumulatorType)

#endif  // GET_TYPEDEF_CLASSES

