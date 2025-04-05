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
namespace gpu {
class AddressSpaceAttr;
class KernelMetadataAttr;
class KernelTableAttr;
class ObjectAttr;
class SelectObjectAttr;
class GPUBlockMappingAttr;
class GPUWarpgroupMappingAttr;
class GPUWarpMappingAttr;
class GPUThreadMappingAttr;
class GPUMemorySpaceMappingAttr;
class ParallelLoopDimMappingAttr;
class DimensionAttr;
class AllReduceOperationAttr;
class ShuffleModeAttr;
class MMAElementwiseOpAttr;
class Prune2To4SpMatFlagAttr;
class TransposeModeAttr;
class SpGEMMWorkEstimationOrComputeKindAttr;
namespace detail {
struct AddressSpaceAttrStorage;
} // namespace detail
class AddressSpaceAttr : public ::mlir::Attribute::AttrBase<AddressSpaceAttr, ::mlir::Attribute, detail::AddressSpaceAttrStorage> {
public:
  using Base::Base;
  static constexpr ::llvm::StringLiteral name = "gpu.address_space";
  static constexpr ::llvm::StringLiteral dialectName = "gpu";
  static AddressSpaceAttr get(::mlir::MLIRContext *context, ::mlir::gpu::AddressSpace value);
  static constexpr ::llvm::StringLiteral getMnemonic() {
    return {"address_space"};
  }

  static ::mlir::Attribute parse(::mlir::AsmParser &odsParser, ::mlir::Type odsType);
  void print(::mlir::AsmPrinter &odsPrinter) const;
  ::mlir::gpu::AddressSpace getValue() const;
};
namespace detail {
struct KernelMetadataAttrStorage;
} // namespace detail
class KernelMetadataAttr : public ::mlir::Attribute::AttrBase<KernelMetadataAttr, ::mlir::Attribute, detail::KernelMetadataAttrStorage> {
public:
  using Base::Base;
  /// Compare two kernels based on the name.
  bool operator<(const KernelMetadataAttr& other) const {
    return getName().getValue() < other.getName().getValue();
  }

  /// Returns the metadata attribute corresponding to `key` or `nullptr`
  /// if missing.
  Attribute getAttr(StringRef key) const {
    DictionaryAttr attrs = getMetadata();
    return attrs ? attrs.get(key) : nullptr;
  }
  template <typename ConcreteAttr>
  ConcreteAttr getAttr(StringRef key) const {
    return llvm::dyn_cast_or_null<ConcreteAttr>(getAttr(key));
  }
  Attribute getAttr(StringAttr key) const {
    DictionaryAttr attrs = getMetadata();
    return attrs ? attrs.get(key) : nullptr;
  }
  template <typename ConcreteAttr>
  ConcreteAttr getAttr(StringAttr key) const {
    return llvm::dyn_cast_or_null<ConcreteAttr>(getAttr(key));
  }

  /// Returns the attribute dictionary at position `index`.
  DictionaryAttr getArgAttrDict(unsigned index) {
    ArrayAttr argArray = getArgAttrs();
    return argArray ? llvm::cast<DictionaryAttr>(argArray[index]) : nullptr;
  }

  /// Return the specified attribute, if present, for the argument at 'index',
  /// null otherwise.
  Attribute getArgAttr(unsigned index, StringAttr name) {
    DictionaryAttr argDict = getArgAttrDict(index);
    return argDict ? argDict.get(name) : nullptr;
  }
  Attribute getArgAttr(unsigned index, StringRef name) {
    DictionaryAttr argDict = getArgAttrDict(index);
    return argDict ? argDict.get(name) : nullptr;
  }

  /// Returns a new KernelMetadataAttr that contains `attrs` in the metadata dictionary.
  KernelMetadataAttr appendMetadata(ArrayRef<NamedAttribute> attrs) const;
  static constexpr ::llvm::StringLiteral name = "gpu.kernel_metadata";
  static constexpr ::llvm::StringLiteral dialectName = "gpu";
  using Base::getChecked;
  static KernelMetadataAttr get(::mlir::MLIRContext *context, StringAttr name, Type function_type, ArrayAttr arg_attrs, DictionaryAttr metadata);
  static KernelMetadataAttr getChecked(::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError, ::mlir::MLIRContext *context, StringAttr name, Type function_type, ArrayAttr arg_attrs, DictionaryAttr metadata);
  static KernelMetadataAttr get(StringAttr name, Type functionType, ArrayAttr argAttrs = nullptr, DictionaryAttr metadata = nullptr);
  static KernelMetadataAttr getChecked(::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError, StringAttr name, Type functionType, ArrayAttr argAttrs = nullptr, DictionaryAttr metadata = nullptr);
  static KernelMetadataAttr get(FunctionOpInterface kernel, DictionaryAttr metadata = nullptr);
  static KernelMetadataAttr getChecked(::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError, FunctionOpInterface kernel, DictionaryAttr metadata = nullptr);
  static ::llvm::LogicalResult verify(::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError, StringAttr name, Type function_type, ArrayAttr arg_attrs, DictionaryAttr metadata);
  static ::llvm::LogicalResult verifyInvariants(::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError, StringAttr name, Type function_type, ArrayAttr arg_attrs, DictionaryAttr metadata);
  static constexpr ::llvm::StringLiteral getMnemonic() {
    return {"kernel_metadata"};
  }

  static ::mlir::Attribute parse(::mlir::AsmParser &odsParser, ::mlir::Type odsType);
  void print(::mlir::AsmPrinter &odsPrinter) const;
  StringAttr getName() const;
  Type getFunctionType() const;
  ArrayAttr getArgAttrs() const;
  DictionaryAttr getMetadata() const;
};
namespace detail {
struct KernelTableAttrStorage;
} // namespace detail
class KernelTableAttr : public ::mlir::Attribute::AttrBase<KernelTableAttr, ::mlir::Attribute, detail::KernelTableAttrStorage> {
public:
  using Base::Base;
  llvm::ArrayRef<KernelMetadataAttr>::iterator begin() const {
    return getKernelTable().begin();
  }
  llvm::ArrayRef<KernelMetadataAttr>::iterator end() const {
    return getKernelTable().end();
  }
  size_t size() const {
    return getKernelTable().size();
  }
  bool empty() const {
    return getKernelTable().empty();
  }

  /// Returns the kernel with name `key` or `nullptr` if not present.
  KernelMetadataAttr lookup(StringRef key) const;
  KernelMetadataAttr lookup(StringAttr key) const;
  static constexpr ::llvm::StringLiteral name = "gpu.kernel_table";
  static constexpr ::llvm::StringLiteral dialectName = "gpu";
  using Base::getChecked;
  static KernelTableAttr get(::mlir::MLIRContext *context, ArrayRef<KernelMetadataAttr> kernels, bool isSorted = false);
  static KernelTableAttr getChecked(::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError, ::mlir::MLIRContext *context, ArrayRef<KernelMetadataAttr> kernels, bool isSorted = false);
  static ::llvm::LogicalResult verify(::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError, ::llvm::ArrayRef<KernelMetadataAttr> kernel_table);
  static ::llvm::LogicalResult verifyInvariants(::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError, ::llvm::ArrayRef<KernelMetadataAttr> kernel_table);
  static constexpr ::llvm::StringLiteral getMnemonic() {
    return {"kernel_table"};
  }

  static ::mlir::Attribute parse(::mlir::AsmParser &odsParser, ::mlir::Type odsType);
  void print(::mlir::AsmPrinter &odsPrinter) const;
  ::llvm::ArrayRef<KernelMetadataAttr> getKernelTable() const;
};
namespace detail {
struct ObjectAttrStorage;
} // namespace detail
class ObjectAttr : public ::mlir::Attribute::AttrBase<ObjectAttr, ::mlir::Attribute, detail::ObjectAttrStorage> {
public:
  using Base::Base;
  static constexpr ::llvm::StringLiteral name = "gpu.object";
  static constexpr ::llvm::StringLiteral dialectName = "gpu";
  using Base::getChecked;
  static ObjectAttr get(::mlir::MLIRContext *context, Attribute target, CompilationTarget format, StringAttr object, DictionaryAttr properties, KernelTableAttr kernels);
  static ObjectAttr getChecked(::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError, ::mlir::MLIRContext *context, Attribute target, CompilationTarget format, StringAttr object, DictionaryAttr properties, KernelTableAttr kernels);
  static ObjectAttr get(Attribute target, CompilationTarget format, StringAttr object, DictionaryAttr properties = nullptr, KernelTableAttr kernels = nullptr);
  static ObjectAttr getChecked(::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError, Attribute target, CompilationTarget format, StringAttr object, DictionaryAttr properties = nullptr, KernelTableAttr kernels = nullptr);
  static ::llvm::LogicalResult verify(::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError, Attribute target, CompilationTarget format, StringAttr object, DictionaryAttr properties, KernelTableAttr kernels);
  static ::llvm::LogicalResult verifyInvariants(::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError, Attribute target, CompilationTarget format, StringAttr object, DictionaryAttr properties, KernelTableAttr kernels);
  static constexpr ::llvm::StringLiteral getMnemonic() {
    return {"object"};
  }

  static ::mlir::Attribute parse(::mlir::AsmParser &odsParser, ::mlir::Type odsType);
  void print(::mlir::AsmPrinter &odsPrinter) const;
  Attribute getTarget() const;
  CompilationTarget getFormat() const;
  StringAttr getObject() const;
  DictionaryAttr getProperties() const;
  KernelTableAttr getKernels() const;
};
namespace detail {
struct SelectObjectAttrStorage;
} // namespace detail
class SelectObjectAttr : public ::mlir::Attribute::AttrBase<SelectObjectAttr, ::mlir::Attribute, detail::SelectObjectAttrStorage, ::mlir::gpu::OffloadingTranslationAttrTrait> {
public:
  using Base::Base;
  static constexpr ::llvm::StringLiteral name = "gpu.select_object";
  static constexpr ::llvm::StringLiteral dialectName = "gpu";
  using Base::getChecked;
  static SelectObjectAttr get(::mlir::MLIRContext *context, Attribute target);
  static SelectObjectAttr getChecked(::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError, ::mlir::MLIRContext *context, Attribute target);
  static ::llvm::LogicalResult verify(::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError, Attribute target);
  static ::llvm::LogicalResult verifyInvariants(::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError, Attribute target);
  static constexpr ::llvm::StringLiteral getMnemonic() {
    return {"select_object"};
  }

  static ::mlir::Attribute parse(::mlir::AsmParser &odsParser, ::mlir::Type odsType);
  void print(::mlir::AsmPrinter &odsPrinter) const;
  Attribute getTarget() const;
};
namespace detail {
struct GPUBlockMappingAttrStorage;
} // namespace detail
class GPUBlockMappingAttr : public ::mlir::Attribute::AttrBase<GPUBlockMappingAttr, ::mlir::Attribute, detail::GPUBlockMappingAttrStorage, ::mlir::DeviceMappingAttrInterface::Trait> {
public:
  using Base::Base;
  static constexpr ::llvm::StringLiteral name = "gpu.block";
  static constexpr ::llvm::StringLiteral dialectName = "gpu";
  static GPUBlockMappingAttr get(::mlir::MLIRContext *context, ::mlir::gpu::MappingId block);
  static constexpr ::llvm::StringLiteral getMnemonic() {
    return {"block"};
  }

  static ::mlir::Attribute parse(::mlir::AsmParser &odsParser, ::mlir::Type odsType);
  void print(::mlir::AsmPrinter &odsPrinter) const;
  ::mlir::gpu::MappingId getBlock() const;
  int64_t getMappingId() const;
  bool isLinearMapping() const;
  int64_t getRelativeIndex() const;
};
namespace detail {
struct GPUWarpgroupMappingAttrStorage;
} // namespace detail
class GPUWarpgroupMappingAttr : public ::mlir::Attribute::AttrBase<GPUWarpgroupMappingAttr, ::mlir::Attribute, detail::GPUWarpgroupMappingAttrStorage, ::mlir::DeviceMappingAttrInterface::Trait> {
public:
  using Base::Base;
  static constexpr ::llvm::StringLiteral name = "gpu.warpgroup";
  static constexpr ::llvm::StringLiteral dialectName = "gpu";
  static GPUWarpgroupMappingAttr get(::mlir::MLIRContext *context, ::mlir::gpu::MappingId warpgroup);
  static constexpr ::llvm::StringLiteral getMnemonic() {
    return {"warpgroup"};
  }

  static ::mlir::Attribute parse(::mlir::AsmParser &odsParser, ::mlir::Type odsType);
  void print(::mlir::AsmPrinter &odsPrinter) const;
  ::mlir::gpu::MappingId getWarpgroup() const;
  int64_t getMappingId() const;
  bool isLinearMapping() const;
  int64_t getRelativeIndex() const;
};
namespace detail {
struct GPUWarpMappingAttrStorage;
} // namespace detail
class GPUWarpMappingAttr : public ::mlir::Attribute::AttrBase<GPUWarpMappingAttr, ::mlir::Attribute, detail::GPUWarpMappingAttrStorage, ::mlir::DeviceMappingAttrInterface::Trait> {
public:
  using Base::Base;
  static constexpr ::llvm::StringLiteral name = "gpu.warp";
  static constexpr ::llvm::StringLiteral dialectName = "gpu";
  static GPUWarpMappingAttr get(::mlir::MLIRContext *context, ::mlir::gpu::MappingId warp);
  static constexpr ::llvm::StringLiteral getMnemonic() {
    return {"warp"};
  }

  static ::mlir::Attribute parse(::mlir::AsmParser &odsParser, ::mlir::Type odsType);
  void print(::mlir::AsmPrinter &odsPrinter) const;
  ::mlir::gpu::MappingId getWarp() const;
  int64_t getMappingId() const;
  bool isLinearMapping() const;
  int64_t getRelativeIndex() const;
};
namespace detail {
struct GPUThreadMappingAttrStorage;
} // namespace detail
class GPUThreadMappingAttr : public ::mlir::Attribute::AttrBase<GPUThreadMappingAttr, ::mlir::Attribute, detail::GPUThreadMappingAttrStorage, ::mlir::DeviceMappingAttrInterface::Trait> {
public:
  using Base::Base;
  static constexpr ::llvm::StringLiteral name = "gpu.thread";
  static constexpr ::llvm::StringLiteral dialectName = "gpu";
  static GPUThreadMappingAttr get(::mlir::MLIRContext *context, ::mlir::gpu::MappingId thread);
  static constexpr ::llvm::StringLiteral getMnemonic() {
    return {"thread"};
  }

  static ::mlir::Attribute parse(::mlir::AsmParser &odsParser, ::mlir::Type odsType);
  void print(::mlir::AsmPrinter &odsPrinter) const;
  ::mlir::gpu::MappingId getThread() const;
  int64_t getMappingId() const;
  bool isLinearMapping() const;
  int64_t getRelativeIndex() const;
};
namespace detail {
struct GPUMemorySpaceMappingAttrStorage;
} // namespace detail
class GPUMemorySpaceMappingAttr : public ::mlir::Attribute::AttrBase<GPUMemorySpaceMappingAttr, ::mlir::Attribute, detail::GPUMemorySpaceMappingAttrStorage, ::mlir::DeviceMappingAttrInterface::Trait> {
public:
  using Base::Base;
  static constexpr ::llvm::StringLiteral name = "gpu.memory_space";
  static constexpr ::llvm::StringLiteral dialectName = "gpu";
  static GPUMemorySpaceMappingAttr get(::mlir::MLIRContext *context, ::mlir::gpu::AddressSpace address_space);
  static constexpr ::llvm::StringLiteral getMnemonic() {
    return {"memory_space"};
  }

  static ::mlir::Attribute parse(::mlir::AsmParser &odsParser, ::mlir::Type odsType);
  void print(::mlir::AsmPrinter &odsPrinter) const;
  ::mlir::gpu::AddressSpace getAddressSpace() const;
  int64_t getMappingId() const;
  bool isLinearMapping() const;
  int64_t getRelativeIndex() const;
};
namespace detail {
struct ParallelLoopDimMappingAttrStorage;
} // namespace detail
class ParallelLoopDimMappingAttr : public ::mlir::Attribute::AttrBase<ParallelLoopDimMappingAttr, ::mlir::Attribute, detail::ParallelLoopDimMappingAttrStorage> {
public:
  using Base::Base;
  static constexpr ::llvm::StringLiteral name = "gpu.loop_dim_map";
  static constexpr ::llvm::StringLiteral dialectName = "gpu";
  static ParallelLoopDimMappingAttr get(::mlir::MLIRContext *context, ::mlir::gpu::Processor processor, AffineMap map, AffineMap bound);
  static constexpr ::llvm::StringLiteral getMnemonic() {
    return {"loop_dim_map"};
  }

  static ::mlir::Attribute parse(::mlir::AsmParser &odsParser, ::mlir::Type odsType);
  void print(::mlir::AsmPrinter &odsPrinter) const;
  ::mlir::gpu::Processor getProcessor() const;
  AffineMap getMap() const;
  AffineMap getBound() const;
};
namespace detail {
struct DimensionAttrStorage;
} // namespace detail
class DimensionAttr : public ::mlir::Attribute::AttrBase<DimensionAttr, ::mlir::Attribute, detail::DimensionAttrStorage> {
public:
  using Base::Base;
  static constexpr ::llvm::StringLiteral name = "gpu.dim";
  static constexpr ::llvm::StringLiteral dialectName = "gpu";
  static DimensionAttr get(::mlir::MLIRContext *context, ::mlir::gpu::Dimension value);
  static constexpr ::llvm::StringLiteral getMnemonic() {
    return {"dim"};
  }

  static ::mlir::Attribute parse(::mlir::AsmParser &odsParser, ::mlir::Type odsType);
  void print(::mlir::AsmPrinter &odsPrinter) const;
  ::mlir::gpu::Dimension getValue() const;
};
namespace detail {
struct AllReduceOperationAttrStorage;
} // namespace detail
class AllReduceOperationAttr : public ::mlir::Attribute::AttrBase<AllReduceOperationAttr, ::mlir::Attribute, detail::AllReduceOperationAttrStorage> {
public:
  using Base::Base;
  static constexpr ::llvm::StringLiteral name = "gpu.all_reduce_op";
  static constexpr ::llvm::StringLiteral dialectName = "gpu";
  static AllReduceOperationAttr get(::mlir::MLIRContext *context, ::mlir::gpu::AllReduceOperation value);
  static constexpr ::llvm::StringLiteral getMnemonic() {
    return {"all_reduce_op"};
  }

  static ::mlir::Attribute parse(::mlir::AsmParser &odsParser, ::mlir::Type odsType);
  void print(::mlir::AsmPrinter &odsPrinter) const;
  ::mlir::gpu::AllReduceOperation getValue() const;
};
namespace detail {
struct ShuffleModeAttrStorage;
} // namespace detail
class ShuffleModeAttr : public ::mlir::Attribute::AttrBase<ShuffleModeAttr, ::mlir::Attribute, detail::ShuffleModeAttrStorage> {
public:
  using Base::Base;
  static constexpr ::llvm::StringLiteral name = "gpu.shuffle_mode";
  static constexpr ::llvm::StringLiteral dialectName = "gpu";
  static ShuffleModeAttr get(::mlir::MLIRContext *context, ::mlir::gpu::ShuffleMode value);
  static constexpr ::llvm::StringLiteral getMnemonic() {
    return {"shuffle_mode"};
  }

  static ::mlir::Attribute parse(::mlir::AsmParser &odsParser, ::mlir::Type odsType);
  void print(::mlir::AsmPrinter &odsPrinter) const;
  ::mlir::gpu::ShuffleMode getValue() const;
};
namespace detail {
struct MMAElementwiseOpAttrStorage;
} // namespace detail
class MMAElementwiseOpAttr : public ::mlir::Attribute::AttrBase<MMAElementwiseOpAttr, ::mlir::Attribute, detail::MMAElementwiseOpAttrStorage> {
public:
  using Base::Base;
  static constexpr ::llvm::StringLiteral name = "gpu.mma_element_wise";
  static constexpr ::llvm::StringLiteral dialectName = "gpu";
  static MMAElementwiseOpAttr get(::mlir::MLIRContext *context, ::mlir::gpu::MMAElementwiseOp value);
  static constexpr ::llvm::StringLiteral getMnemonic() {
    return {"mma_element_wise"};
  }

  static ::mlir::Attribute parse(::mlir::AsmParser &odsParser, ::mlir::Type odsType);
  void print(::mlir::AsmPrinter &odsPrinter) const;
  ::mlir::gpu::MMAElementwiseOp getValue() const;
};
namespace detail {
struct Prune2To4SpMatFlagAttrStorage;
} // namespace detail
class Prune2To4SpMatFlagAttr : public ::mlir::Attribute::AttrBase<Prune2To4SpMatFlagAttr, ::mlir::Attribute, detail::Prune2To4SpMatFlagAttrStorage> {
public:
  using Base::Base;
  static constexpr ::llvm::StringLiteral name = "gpu.prune_2to4_spmat_flag";
  static constexpr ::llvm::StringLiteral dialectName = "gpu";
  static Prune2To4SpMatFlagAttr get(::mlir::MLIRContext *context, ::mlir::gpu::Prune2To4SpMatFlag value);
  static constexpr ::llvm::StringLiteral getMnemonic() {
    return {"prune_2to4_spmat_flag"};
  }

  static ::mlir::Attribute parse(::mlir::AsmParser &odsParser, ::mlir::Type odsType);
  void print(::mlir::AsmPrinter &odsPrinter) const;
  ::mlir::gpu::Prune2To4SpMatFlag getValue() const;
};
namespace detail {
struct TransposeModeAttrStorage;
} // namespace detail
class TransposeModeAttr : public ::mlir::Attribute::AttrBase<TransposeModeAttr, ::mlir::Attribute, detail::TransposeModeAttrStorage> {
public:
  using Base::Base;
  static constexpr ::llvm::StringLiteral name = "gpu.mat_transpose_mode";
  static constexpr ::llvm::StringLiteral dialectName = "gpu";
  static TransposeModeAttr get(::mlir::MLIRContext *context, ::mlir::gpu::TransposeMode value);
  static constexpr ::llvm::StringLiteral getMnemonic() {
    return {"mat_transpose_mode"};
  }

  static ::mlir::Attribute parse(::mlir::AsmParser &odsParser, ::mlir::Type odsType);
  void print(::mlir::AsmPrinter &odsPrinter) const;
  ::mlir::gpu::TransposeMode getValue() const;
};
namespace detail {
struct SpGEMMWorkEstimationOrComputeKindAttrStorage;
} // namespace detail
class SpGEMMWorkEstimationOrComputeKindAttr : public ::mlir::Attribute::AttrBase<SpGEMMWorkEstimationOrComputeKindAttr, ::mlir::Attribute, detail::SpGEMMWorkEstimationOrComputeKindAttrStorage> {
public:
  using Base::Base;
  static constexpr ::llvm::StringLiteral name = "gpu.spgemm_work_estimation_or_compute_kind";
  static constexpr ::llvm::StringLiteral dialectName = "gpu";
  static SpGEMMWorkEstimationOrComputeKindAttr get(::mlir::MLIRContext *context, ::mlir::gpu::SpGEMMWorkEstimationOrComputeKind value);
  static constexpr ::llvm::StringLiteral getMnemonic() {
    return {"spgemm_work_estimation_or_compute_kind"};
  }

  static ::mlir::Attribute parse(::mlir::AsmParser &odsParser, ::mlir::Type odsType);
  void print(::mlir::AsmPrinter &odsPrinter) const;
  ::mlir::gpu::SpGEMMWorkEstimationOrComputeKind getValue() const;
};
} // namespace gpu
} // namespace mlir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::gpu::AddressSpaceAttr)
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::gpu::KernelMetadataAttr)
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::gpu::KernelTableAttr)
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::gpu::ObjectAttr)
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::gpu::SelectObjectAttr)
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::gpu::GPUBlockMappingAttr)
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::gpu::GPUWarpgroupMappingAttr)
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::gpu::GPUWarpMappingAttr)
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::gpu::GPUThreadMappingAttr)
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::gpu::GPUMemorySpaceMappingAttr)
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::gpu::ParallelLoopDimMappingAttr)
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::gpu::DimensionAttr)
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::gpu::AllReduceOperationAttr)
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::gpu::ShuffleModeAttr)
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::gpu::MMAElementwiseOpAttr)
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::gpu::Prune2To4SpMatFlagAttr)
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::gpu::TransposeModeAttr)
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::gpu::SpGEMMWorkEstimationOrComputeKindAttr)

#endif  // GET_ATTRDEF_CLASSES

