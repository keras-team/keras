/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Enum Utility Definitions                                                   *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|* From: GPUOps.td                                                            *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

namespace mlir {
namespace gpu {
::llvm::StringRef stringifyAddressSpace(AddressSpace val) {
  switch (val) {
    case AddressSpace::Global: return "global";
    case AddressSpace::Workgroup: return "workgroup";
    case AddressSpace::Private: return "private";
  }
  return "";
}

::std::optional<AddressSpace> symbolizeAddressSpace(::llvm::StringRef str) {
  return ::llvm::StringSwitch<::std::optional<AddressSpace>>(str)
      .Case("global", AddressSpace::Global)
      .Case("workgroup", AddressSpace::Workgroup)
      .Case("private", AddressSpace::Private)
      .Default(::std::nullopt);
}
::std::optional<AddressSpace> symbolizeAddressSpace(uint32_t value) {
  switch (value) {
  case 1: return AddressSpace::Global;
  case 2: return AddressSpace::Workgroup;
  case 3: return AddressSpace::Private;
  default: return ::std::nullopt;
  }
}

} // namespace gpu
} // namespace mlir

namespace mlir {
namespace gpu {
::llvm::StringRef stringifyAllReduceOperation(AllReduceOperation val) {
  switch (val) {
    case AllReduceOperation::ADD: return "add";
    case AllReduceOperation::MUL: return "mul";
    case AllReduceOperation::MINUI: return "minui";
    case AllReduceOperation::MINSI: return "minsi";
    case AllReduceOperation::MINNUMF: return "minnumf";
    case AllReduceOperation::MAXUI: return "maxui";
    case AllReduceOperation::MAXSI: return "maxsi";
    case AllReduceOperation::MAXNUMF: return "maxnumf";
    case AllReduceOperation::AND: return "and";
    case AllReduceOperation::OR: return "or";
    case AllReduceOperation::XOR: return "xor";
    case AllReduceOperation::MINIMUMF: return "minimumf";
    case AllReduceOperation::MAXIMUMF: return "maximumf";
  }
  return "";
}

::std::optional<AllReduceOperation> symbolizeAllReduceOperation(::llvm::StringRef str) {
  return ::llvm::StringSwitch<::std::optional<AllReduceOperation>>(str)
      .Case("add", AllReduceOperation::ADD)
      .Case("mul", AllReduceOperation::MUL)
      .Case("minui", AllReduceOperation::MINUI)
      .Case("minsi", AllReduceOperation::MINSI)
      .Case("minnumf", AllReduceOperation::MINNUMF)
      .Case("maxui", AllReduceOperation::MAXUI)
      .Case("maxsi", AllReduceOperation::MAXSI)
      .Case("maxnumf", AllReduceOperation::MAXNUMF)
      .Case("and", AllReduceOperation::AND)
      .Case("or", AllReduceOperation::OR)
      .Case("xor", AllReduceOperation::XOR)
      .Case("minimumf", AllReduceOperation::MINIMUMF)
      .Case("maximumf", AllReduceOperation::MAXIMUMF)
      .Default(::std::nullopt);
}
::std::optional<AllReduceOperation> symbolizeAllReduceOperation(uint32_t value) {
  switch (value) {
  case 0: return AllReduceOperation::ADD;
  case 1: return AllReduceOperation::MUL;
  case 2: return AllReduceOperation::MINUI;
  case 3: return AllReduceOperation::MINSI;
  case 4: return AllReduceOperation::MINNUMF;
  case 5: return AllReduceOperation::MAXUI;
  case 6: return AllReduceOperation::MAXSI;
  case 7: return AllReduceOperation::MAXNUMF;
  case 8: return AllReduceOperation::AND;
  case 9: return AllReduceOperation::OR;
  case 10: return AllReduceOperation::XOR;
  case 11: return AllReduceOperation::MINIMUMF;
  case 12: return AllReduceOperation::MAXIMUMF;
  default: return ::std::nullopt;
  }
}

} // namespace gpu
} // namespace mlir

namespace mlir {
namespace gpu {
::llvm::StringRef stringifyCompilationTarget(CompilationTarget val) {
  switch (val) {
    case CompilationTarget::Offload: return "offload";
    case CompilationTarget::Assembly: return "assembly";
    case CompilationTarget::Binary: return "bin";
    case CompilationTarget::Fatbin: return "fatbin";
  }
  return "";
}

::std::optional<CompilationTarget> symbolizeCompilationTarget(::llvm::StringRef str) {
  return ::llvm::StringSwitch<::std::optional<CompilationTarget>>(str)
      .Case("offload", CompilationTarget::Offload)
      .Case("assembly", CompilationTarget::Assembly)
      .Case("bin", CompilationTarget::Binary)
      .Case("fatbin", CompilationTarget::Fatbin)
      .Default(::std::nullopt);
}
::std::optional<CompilationTarget> symbolizeCompilationTarget(uint32_t value) {
  switch (value) {
  case 1: return CompilationTarget::Offload;
  case 2: return CompilationTarget::Assembly;
  case 3: return CompilationTarget::Binary;
  case 4: return CompilationTarget::Fatbin;
  default: return ::std::nullopt;
  }
}

} // namespace gpu
} // namespace mlir

namespace mlir {
namespace gpu {
::llvm::StringRef stringifyDimension(Dimension val) {
  switch (val) {
    case Dimension::x: return "x";
    case Dimension::y: return "y";
    case Dimension::z: return "z";
  }
  return "";
}

::std::optional<Dimension> symbolizeDimension(::llvm::StringRef str) {
  return ::llvm::StringSwitch<::std::optional<Dimension>>(str)
      .Case("x", Dimension::x)
      .Case("y", Dimension::y)
      .Case("z", Dimension::z)
      .Default(::std::nullopt);
}
::std::optional<Dimension> symbolizeDimension(uint32_t value) {
  switch (value) {
  case 0: return Dimension::x;
  case 1: return Dimension::y;
  case 2: return Dimension::z;
  default: return ::std::nullopt;
  }
}

} // namespace gpu
} // namespace mlir

namespace mlir {
namespace gpu {
::llvm::StringRef stringifyPrune2To4SpMatFlag(Prune2To4SpMatFlag val) {
  switch (val) {
    case Prune2To4SpMatFlag::NONE: return "NONE";
    case Prune2To4SpMatFlag::PRUNE_ONLY: return "PRUNE_ONLY";
    case Prune2To4SpMatFlag::PRUNE_AND_CHECK: return "PRUNE_AND_CHECK";
  }
  return "";
}

::std::optional<Prune2To4SpMatFlag> symbolizePrune2To4SpMatFlag(::llvm::StringRef str) {
  return ::llvm::StringSwitch<::std::optional<Prune2To4SpMatFlag>>(str)
      .Case("NONE", Prune2To4SpMatFlag::NONE)
      .Case("PRUNE_ONLY", Prune2To4SpMatFlag::PRUNE_ONLY)
      .Case("PRUNE_AND_CHECK", Prune2To4SpMatFlag::PRUNE_AND_CHECK)
      .Default(::std::nullopt);
}
::std::optional<Prune2To4SpMatFlag> symbolizePrune2To4SpMatFlag(uint32_t value) {
  switch (value) {
  case 0: return Prune2To4SpMatFlag::NONE;
  case 1: return Prune2To4SpMatFlag::PRUNE_ONLY;
  case 2: return Prune2To4SpMatFlag::PRUNE_AND_CHECK;
  default: return ::std::nullopt;
  }
}

} // namespace gpu
} // namespace mlir

namespace mlir {
namespace gpu {
::llvm::StringRef stringifyShuffleMode(ShuffleMode val) {
  switch (val) {
    case ShuffleMode::XOR: return "xor";
    case ShuffleMode::UP: return "up";
    case ShuffleMode::DOWN: return "down";
    case ShuffleMode::IDX: return "idx";
  }
  return "";
}

::std::optional<ShuffleMode> symbolizeShuffleMode(::llvm::StringRef str) {
  return ::llvm::StringSwitch<::std::optional<ShuffleMode>>(str)
      .Case("xor", ShuffleMode::XOR)
      .Case("up", ShuffleMode::UP)
      .Case("down", ShuffleMode::DOWN)
      .Case("idx", ShuffleMode::IDX)
      .Default(::std::nullopt);
}
::std::optional<ShuffleMode> symbolizeShuffleMode(uint32_t value) {
  switch (value) {
  case 0: return ShuffleMode::XOR;
  case 2: return ShuffleMode::UP;
  case 1: return ShuffleMode::DOWN;
  case 3: return ShuffleMode::IDX;
  default: return ::std::nullopt;
  }
}

} // namespace gpu
} // namespace mlir

namespace mlir {
namespace gpu {
::llvm::StringRef stringifySpGEMMWorkEstimationOrComputeKind(SpGEMMWorkEstimationOrComputeKind val) {
  switch (val) {
    case SpGEMMWorkEstimationOrComputeKind::WORK_ESTIMATION: return "WORK_ESTIMATION";
    case SpGEMMWorkEstimationOrComputeKind::COMPUTE: return "COMPUTE";
  }
  return "";
}

::std::optional<SpGEMMWorkEstimationOrComputeKind> symbolizeSpGEMMWorkEstimationOrComputeKind(::llvm::StringRef str) {
  return ::llvm::StringSwitch<::std::optional<SpGEMMWorkEstimationOrComputeKind>>(str)
      .Case("WORK_ESTIMATION", SpGEMMWorkEstimationOrComputeKind::WORK_ESTIMATION)
      .Case("COMPUTE", SpGEMMWorkEstimationOrComputeKind::COMPUTE)
      .Default(::std::nullopt);
}
::std::optional<SpGEMMWorkEstimationOrComputeKind> symbolizeSpGEMMWorkEstimationOrComputeKind(uint32_t value) {
  switch (value) {
  case 0: return SpGEMMWorkEstimationOrComputeKind::WORK_ESTIMATION;
  case 1: return SpGEMMWorkEstimationOrComputeKind::COMPUTE;
  default: return ::std::nullopt;
  }
}

} // namespace gpu
} // namespace mlir

namespace mlir {
namespace gpu {
::llvm::StringRef stringifyTransposeMode(TransposeMode val) {
  switch (val) {
    case TransposeMode::NON_TRANSPOSE: return "NON_TRANSPOSE";
    case TransposeMode::TRANSPOSE: return "TRANSPOSE";
    case TransposeMode::CONJUGATE_TRANSPOSE: return "CONJUGATE_TRANSPOSE";
  }
  return "";
}

::std::optional<TransposeMode> symbolizeTransposeMode(::llvm::StringRef str) {
  return ::llvm::StringSwitch<::std::optional<TransposeMode>>(str)
      .Case("NON_TRANSPOSE", TransposeMode::NON_TRANSPOSE)
      .Case("TRANSPOSE", TransposeMode::TRANSPOSE)
      .Case("CONJUGATE_TRANSPOSE", TransposeMode::CONJUGATE_TRANSPOSE)
      .Default(::std::nullopt);
}
::std::optional<TransposeMode> symbolizeTransposeMode(uint32_t value) {
  switch (value) {
  case 0: return TransposeMode::NON_TRANSPOSE;
  case 1: return TransposeMode::TRANSPOSE;
  case 2: return TransposeMode::CONJUGATE_TRANSPOSE;
  default: return ::std::nullopt;
  }
}

} // namespace gpu
} // namespace mlir

namespace mlir {
namespace gpu {
::llvm::StringRef stringifyMMAElementwiseOp(MMAElementwiseOp val) {
  switch (val) {
    case MMAElementwiseOp::ADDF: return "addf";
    case MMAElementwiseOp::MULF: return "mulf";
    case MMAElementwiseOp::SUBF: return "subf";
    case MMAElementwiseOp::MAXF: return "maxf";
    case MMAElementwiseOp::MINF: return "minf";
    case MMAElementwiseOp::DIVF: return "divf";
    case MMAElementwiseOp::ADDI: return "addi";
    case MMAElementwiseOp::MULI: return "muli";
    case MMAElementwiseOp::SUBI: return "subi";
    case MMAElementwiseOp::DIVS: return "divs";
    case MMAElementwiseOp::DIVU: return "divu";
    case MMAElementwiseOp::NEGATEF: return "negatef";
    case MMAElementwiseOp::NEGATES: return "negates";
    case MMAElementwiseOp::EXTF: return "extf";
  }
  return "";
}

::std::optional<MMAElementwiseOp> symbolizeMMAElementwiseOp(::llvm::StringRef str) {
  return ::llvm::StringSwitch<::std::optional<MMAElementwiseOp>>(str)
      .Case("addf", MMAElementwiseOp::ADDF)
      .Case("mulf", MMAElementwiseOp::MULF)
      .Case("subf", MMAElementwiseOp::SUBF)
      .Case("maxf", MMAElementwiseOp::MAXF)
      .Case("minf", MMAElementwiseOp::MINF)
      .Case("divf", MMAElementwiseOp::DIVF)
      .Case("addi", MMAElementwiseOp::ADDI)
      .Case("muli", MMAElementwiseOp::MULI)
      .Case("subi", MMAElementwiseOp::SUBI)
      .Case("divs", MMAElementwiseOp::DIVS)
      .Case("divu", MMAElementwiseOp::DIVU)
      .Case("negatef", MMAElementwiseOp::NEGATEF)
      .Case("negates", MMAElementwiseOp::NEGATES)
      .Case("extf", MMAElementwiseOp::EXTF)
      .Default(::std::nullopt);
}
::std::optional<MMAElementwiseOp> symbolizeMMAElementwiseOp(uint32_t value) {
  switch (value) {
  case 0: return MMAElementwiseOp::ADDF;
  case 1: return MMAElementwiseOp::MULF;
  case 2: return MMAElementwiseOp::SUBF;
  case 3: return MMAElementwiseOp::MAXF;
  case 4: return MMAElementwiseOp::MINF;
  case 5: return MMAElementwiseOp::DIVF;
  case 6: return MMAElementwiseOp::ADDI;
  case 7: return MMAElementwiseOp::MULI;
  case 8: return MMAElementwiseOp::SUBI;
  case 9: return MMAElementwiseOp::DIVS;
  case 10: return MMAElementwiseOp::DIVU;
  case 11: return MMAElementwiseOp::NEGATEF;
  case 12: return MMAElementwiseOp::NEGATES;
  case 13: return MMAElementwiseOp::EXTF;
  default: return ::std::nullopt;
  }
}

} // namespace gpu
} // namespace mlir

namespace mlir {
namespace gpu {
::llvm::StringRef stringifyMappingId(MappingId val) {
  switch (val) {
    case MappingId::DimX: return "x";
    case MappingId::DimY: return "y";
    case MappingId::DimZ: return "z";
    case MappingId::LinearDim0: return "linear_dim_0";
    case MappingId::LinearDim1: return "linear_dim_1";
    case MappingId::LinearDim2: return "linear_dim_2";
    case MappingId::LinearDim3: return "linear_dim_3";
    case MappingId::LinearDim4: return "linear_dim_4";
    case MappingId::LinearDim5: return "linear_dim_5";
    case MappingId::LinearDim6: return "linear_dim_6";
    case MappingId::LinearDim7: return "linear_dim_7";
    case MappingId::LinearDim8: return "linear_dim_8";
    case MappingId::LinearDim9: return "linear_dim_9";
  }
  return "";
}

::std::optional<MappingId> symbolizeMappingId(::llvm::StringRef str) {
  return ::llvm::StringSwitch<::std::optional<MappingId>>(str)
      .Case("x", MappingId::DimX)
      .Case("y", MappingId::DimY)
      .Case("z", MappingId::DimZ)
      .Case("linear_dim_0", MappingId::LinearDim0)
      .Case("linear_dim_1", MappingId::LinearDim1)
      .Case("linear_dim_2", MappingId::LinearDim2)
      .Case("linear_dim_3", MappingId::LinearDim3)
      .Case("linear_dim_4", MappingId::LinearDim4)
      .Case("linear_dim_5", MappingId::LinearDim5)
      .Case("linear_dim_6", MappingId::LinearDim6)
      .Case("linear_dim_7", MappingId::LinearDim7)
      .Case("linear_dim_8", MappingId::LinearDim8)
      .Case("linear_dim_9", MappingId::LinearDim9)
      .Default(::std::nullopt);
}
::std::optional<MappingId> symbolizeMappingId(uint64_t value) {
  switch (value) {
  case 0: return MappingId::DimX;
  case 1: return MappingId::DimY;
  case 2: return MappingId::DimZ;
  case 3: return MappingId::LinearDim0;
  case 4: return MappingId::LinearDim1;
  case 5: return MappingId::LinearDim2;
  case 6: return MappingId::LinearDim3;
  case 7: return MappingId::LinearDim4;
  case 8: return MappingId::LinearDim5;
  case 9: return MappingId::LinearDim6;
  case 10: return MappingId::LinearDim7;
  case 11: return MappingId::LinearDim8;
  case 12: return MappingId::LinearDim9;
  default: return ::std::nullopt;
  }
}

bool MappingIdAttr::classof(::mlir::Attribute attr) {
  return (((::llvm::isa<::mlir::IntegerAttr>(attr))) && ((::llvm::cast<::mlir::IntegerAttr>(attr).getType().isSignlessInteger(64)))) && (((::llvm::cast<::mlir::IntegerAttr>(attr).getInt() == 0)) || ((::llvm::cast<::mlir::IntegerAttr>(attr).getInt() == 1)) || ((::llvm::cast<::mlir::IntegerAttr>(attr).getInt() == 2)) || ((::llvm::cast<::mlir::IntegerAttr>(attr).getInt() == 3)) || ((::llvm::cast<::mlir::IntegerAttr>(attr).getInt() == 4)) || ((::llvm::cast<::mlir::IntegerAttr>(attr).getInt() == 5)) || ((::llvm::cast<::mlir::IntegerAttr>(attr).getInt() == 6)) || ((::llvm::cast<::mlir::IntegerAttr>(attr).getInt() == 7)) || ((::llvm::cast<::mlir::IntegerAttr>(attr).getInt() == 8)) || ((::llvm::cast<::mlir::IntegerAttr>(attr).getInt() == 9)) || ((::llvm::cast<::mlir::IntegerAttr>(attr).getInt() == 10)) || ((::llvm::cast<::mlir::IntegerAttr>(attr).getInt() == 11)) || ((::llvm::cast<::mlir::IntegerAttr>(attr).getInt() == 12)));
}
MappingIdAttr MappingIdAttr::get(::mlir::MLIRContext *context, MappingId val) {
  ::mlir::IntegerType intType = ::mlir::IntegerType::get(context, 64);
  ::mlir::IntegerAttr baseAttr = ::mlir::IntegerAttr::get(intType, static_cast<uint64_t>(val));
  return ::llvm::cast<MappingIdAttr>(baseAttr);
}
MappingId MappingIdAttr::getValue() const {
  return static_cast<MappingId>(::mlir::IntegerAttr::getInt());
}
} // namespace gpu
} // namespace mlir

namespace mlir {
namespace gpu {
::llvm::StringRef stringifyProcessor(Processor val) {
  switch (val) {
    case Processor::BlockX: return "block_x";
    case Processor::BlockY: return "block_y";
    case Processor::BlockZ: return "block_z";
    case Processor::ThreadX: return "thread_x";
    case Processor::ThreadY: return "thread_y";
    case Processor::ThreadZ: return "thread_z";
    case Processor::Sequential: return "sequential";
  }
  return "";
}

::std::optional<Processor> symbolizeProcessor(::llvm::StringRef str) {
  return ::llvm::StringSwitch<::std::optional<Processor>>(str)
      .Case("block_x", Processor::BlockX)
      .Case("block_y", Processor::BlockY)
      .Case("block_z", Processor::BlockZ)
      .Case("thread_x", Processor::ThreadX)
      .Case("thread_y", Processor::ThreadY)
      .Case("thread_z", Processor::ThreadZ)
      .Case("sequential", Processor::Sequential)
      .Default(::std::nullopt);
}
::std::optional<Processor> symbolizeProcessor(uint64_t value) {
  switch (value) {
  case 0: return Processor::BlockX;
  case 1: return Processor::BlockY;
  case 2: return Processor::BlockZ;
  case 3: return Processor::ThreadX;
  case 4: return Processor::ThreadY;
  case 5: return Processor::ThreadZ;
  case 6: return Processor::Sequential;
  default: return ::std::nullopt;
  }
}

bool ProcessorAttr::classof(::mlir::Attribute attr) {
  return (((::llvm::isa<::mlir::IntegerAttr>(attr))) && ((::llvm::cast<::mlir::IntegerAttr>(attr).getType().isSignlessInteger(64)))) && (((::llvm::cast<::mlir::IntegerAttr>(attr).getInt() == 0)) || ((::llvm::cast<::mlir::IntegerAttr>(attr).getInt() == 1)) || ((::llvm::cast<::mlir::IntegerAttr>(attr).getInt() == 2)) || ((::llvm::cast<::mlir::IntegerAttr>(attr).getInt() == 3)) || ((::llvm::cast<::mlir::IntegerAttr>(attr).getInt() == 4)) || ((::llvm::cast<::mlir::IntegerAttr>(attr).getInt() == 5)) || ((::llvm::cast<::mlir::IntegerAttr>(attr).getInt() == 6)));
}
ProcessorAttr ProcessorAttr::get(::mlir::MLIRContext *context, Processor val) {
  ::mlir::IntegerType intType = ::mlir::IntegerType::get(context, 64);
  ::mlir::IntegerAttr baseAttr = ::mlir::IntegerAttr::get(intType, static_cast<uint64_t>(val));
  return ::llvm::cast<ProcessorAttr>(baseAttr);
}
Processor ProcessorAttr::getValue() const {
  return static_cast<Processor>(::mlir::IntegerAttr::getInt());
}
} // namespace gpu
} // namespace mlir

