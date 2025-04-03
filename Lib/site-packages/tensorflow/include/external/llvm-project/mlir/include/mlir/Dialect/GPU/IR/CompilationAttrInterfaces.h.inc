/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Interface Declarations                                                     *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

namespace mlir {
namespace gpu {
class TargetAttrInterface;
namespace detail {
struct TargetAttrInterfaceInterfaceTraits {
  struct Concept {
    /// The methods defined by the interface.
    std::optional<::mlir::SmallVector<char, 0>> (*serializeToObject)(const Concept *impl, ::mlir::Attribute , ::mlir::Operation*, const ::mlir::gpu::TargetOptions&);
    ::mlir::Attribute (*createObject)(const Concept *impl, ::mlir::Attribute , ::mlir::Operation *, const ::llvm::SmallVector<char, 0> &, const ::mlir::gpu::TargetOptions &);
  };
  template<typename ConcreteAttr>
  class Model : public Concept {
  public:
    using Interface = ::mlir::gpu::TargetAttrInterface;
    Model() : Concept{serializeToObject, createObject} {}

    static inline std::optional<::mlir::SmallVector<char, 0>> serializeToObject(const Concept *impl, ::mlir::Attribute tablegen_opaque_val, ::mlir::Operation* module, const ::mlir::gpu::TargetOptions& options);
    static inline ::mlir::Attribute createObject(const Concept *impl, ::mlir::Attribute tablegen_opaque_val, ::mlir::Operation * module, const ::llvm::SmallVector<char, 0> & object, const ::mlir::gpu::TargetOptions & options);
  };
  template<typename ConcreteAttr>
  class FallbackModel : public Concept {
  public:
    using Interface = ::mlir::gpu::TargetAttrInterface;
    FallbackModel() : Concept{serializeToObject, createObject} {}

    static inline std::optional<::mlir::SmallVector<char, 0>> serializeToObject(const Concept *impl, ::mlir::Attribute tablegen_opaque_val, ::mlir::Operation* module, const ::mlir::gpu::TargetOptions& options);
    static inline ::mlir::Attribute createObject(const Concept *impl, ::mlir::Attribute tablegen_opaque_val, ::mlir::Operation * module, const ::llvm::SmallVector<char, 0> & object, const ::mlir::gpu::TargetOptions & options);
  };
  template<typename ConcreteModel, typename ConcreteAttr>
  class ExternalModel : public FallbackModel<ConcreteModel> {
  public:
    using ConcreteEntity = ConcreteAttr;
  };
};
template <typename ConcreteAttr>
struct TargetAttrInterfaceTrait;

} // namespace detail
class TargetAttrInterface : public ::mlir::AttributeInterface<TargetAttrInterface, detail::TargetAttrInterfaceInterfaceTraits> {
public:
  using ::mlir::AttributeInterface<TargetAttrInterface, detail::TargetAttrInterfaceInterfaceTraits>::AttributeInterface;
  template <typename ConcreteAttr>
  struct Trait : public detail::TargetAttrInterfaceTrait<ConcreteAttr> {};
  /// Serializes a GPU module to a string containing a representation of the
  /// module.
  /// 
  /// If serialization fails then the method should return `std::nullopt`.
  /// 
  /// The `module` parameter must be a GPU Module Op. The `options` parameter
  /// is meant to be used for passing additional options that are not in the
  /// attribute.
  std::optional<::mlir::SmallVector<char, 0>> serializeToObject(::mlir::Operation* module, const ::mlir::gpu::TargetOptions& options) const;
  /// Creates a GPU object attribute from a binary string.
  /// 
  /// The `module` parameter must be a `GPUModuleOp` and can be used to
  /// retrieve additional information like the list of kernels in the binary.
  /// The `object` parameter is a binary string. The `options` parameter is
  /// meant to be used for passing additional options that are not in the
  /// attribute.
  ::mlir::Attribute createObject(::mlir::Operation * module, const ::llvm::SmallVector<char, 0> & object, const ::mlir::gpu::TargetOptions & options) const;
};
namespace detail {
  template <typename ConcreteAttr>
  struct TargetAttrInterfaceTrait : public ::mlir::AttributeInterface<TargetAttrInterface, detail::TargetAttrInterfaceInterfaceTraits>::Trait<ConcreteAttr> {
  };
}// namespace detail
} // namespace gpu
} // namespace mlir
namespace mlir {
namespace gpu {
class OffloadingLLVMTranslationAttrInterface;
namespace detail {
struct OffloadingLLVMTranslationAttrInterfaceInterfaceTraits {
  struct Concept {
    /// The methods defined by the interface.
    ::llvm::LogicalResult (*embedBinary)(const Concept *impl, ::mlir::Attribute , ::mlir::Operation*, ::llvm::IRBuilderBase&, ::mlir::LLVM::ModuleTranslation&);
    ::llvm::LogicalResult (*launchKernel)(const Concept *impl, ::mlir::Attribute , ::mlir::Operation*, ::mlir::Operation*, ::llvm::IRBuilderBase&, ::mlir::LLVM::ModuleTranslation&);
  };
  template<typename ConcreteAttr>
  class Model : public Concept {
  public:
    using Interface = ::mlir::gpu::OffloadingLLVMTranslationAttrInterface;
    Model() : Concept{embedBinary, launchKernel} {}

    static inline ::llvm::LogicalResult embedBinary(const Concept *impl, ::mlir::Attribute tablegen_opaque_val, ::mlir::Operation* binaryOp, ::llvm::IRBuilderBase& hostBuilder, ::mlir::LLVM::ModuleTranslation& hostModuleTranslation);
    static inline ::llvm::LogicalResult launchKernel(const Concept *impl, ::mlir::Attribute tablegen_opaque_val, ::mlir::Operation* launchFunc, ::mlir::Operation* binaryOp, ::llvm::IRBuilderBase& hostBuilder, ::mlir::LLVM::ModuleTranslation& hostModuleTranslation);
  };
  template<typename ConcreteAttr>
  class FallbackModel : public Concept {
  public:
    using Interface = ::mlir::gpu::OffloadingLLVMTranslationAttrInterface;
    FallbackModel() : Concept{embedBinary, launchKernel} {}

    static inline ::llvm::LogicalResult embedBinary(const Concept *impl, ::mlir::Attribute tablegen_opaque_val, ::mlir::Operation* binaryOp, ::llvm::IRBuilderBase& hostBuilder, ::mlir::LLVM::ModuleTranslation& hostModuleTranslation);
    static inline ::llvm::LogicalResult launchKernel(const Concept *impl, ::mlir::Attribute tablegen_opaque_val, ::mlir::Operation* launchFunc, ::mlir::Operation* binaryOp, ::llvm::IRBuilderBase& hostBuilder, ::mlir::LLVM::ModuleTranslation& hostModuleTranslation);
  };
  template<typename ConcreteModel, typename ConcreteAttr>
  class ExternalModel : public FallbackModel<ConcreteModel> {
  public:
    using ConcreteEntity = ConcreteAttr;
  };
};
template <typename ConcreteAttr>
struct OffloadingLLVMTranslationAttrInterfaceTrait;

} // namespace detail
class OffloadingLLVMTranslationAttrInterface : public ::mlir::AttributeInterface<OffloadingLLVMTranslationAttrInterface, detail::OffloadingLLVMTranslationAttrInterfaceInterfaceTraits> {
public:
  using ::mlir::AttributeInterface<OffloadingLLVMTranslationAttrInterface, detail::OffloadingLLVMTranslationAttrInterfaceInterfaceTraits>::AttributeInterface;
  template <typename ConcreteAttr>
  struct Trait : public detail::OffloadingLLVMTranslationAttrInterfaceTrait<ConcreteAttr> {};
  /// Translates a `gpu.binary` Op into a sequence of LLVM IR target-specific
  /// instructions, embedding the binary into a host LLVM module.
  /// 
  /// The LLVM translation mechanism invokes this function when translating a
  /// `gpu.binary`.
  /// 
  /// The first argument has to be a GPU binary operation.
  /// If the function fails at any point, it must return `failure`.
  ::llvm::LogicalResult embedBinary(::mlir::Operation* binaryOp, ::llvm::IRBuilderBase& hostBuilder, ::mlir::LLVM::ModuleTranslation& hostModuleTranslation) const;
  /// Translates a `gpu.launch_func` op into a sequence of LLVM IR
  /// target-specific instructions, resulting in a kernel launch on host IR.
  /// 
  /// The LLVM translation mechanism invokes this function when translating a
  /// `gpu.launch_func` operation; it searches the appropriate binary and uses
  /// its offloading handler.
  /// 
  /// The first two arguments must be GPU launch and binary operations,
  /// respectively. If the function fails at any point, it must return
  /// `failure`.
  ::llvm::LogicalResult launchKernel(::mlir::Operation* launchFunc, ::mlir::Operation* binaryOp, ::llvm::IRBuilderBase& hostBuilder, ::mlir::LLVM::ModuleTranslation& hostModuleTranslation) const;
};
namespace detail {
  template <typename ConcreteAttr>
  struct OffloadingLLVMTranslationAttrInterfaceTrait : public ::mlir::AttributeInterface<OffloadingLLVMTranslationAttrInterface, detail::OffloadingLLVMTranslationAttrInterfaceInterfaceTraits>::Trait<ConcreteAttr> {
  };
}// namespace detail
} // namespace gpu
} // namespace mlir
namespace mlir {
namespace gpu {
template<typename ConcreteAttr>
std::optional<::mlir::SmallVector<char, 0>> detail::TargetAttrInterfaceInterfaceTraits::Model<ConcreteAttr>::serializeToObject(const Concept *impl, ::mlir::Attribute tablegen_opaque_val, ::mlir::Operation* module, const ::mlir::gpu::TargetOptions& options) {
  return (::llvm::cast<ConcreteAttr>(tablegen_opaque_val)).serializeToObject(module, options);
}
template<typename ConcreteAttr>
::mlir::Attribute detail::TargetAttrInterfaceInterfaceTraits::Model<ConcreteAttr>::createObject(const Concept *impl, ::mlir::Attribute tablegen_opaque_val, ::mlir::Operation * module, const ::llvm::SmallVector<char, 0> & object, const ::mlir::gpu::TargetOptions & options) {
  return (::llvm::cast<ConcreteAttr>(tablegen_opaque_val)).createObject(module, object, options);
}
template<typename ConcreteAttr>
std::optional<::mlir::SmallVector<char, 0>> detail::TargetAttrInterfaceInterfaceTraits::FallbackModel<ConcreteAttr>::serializeToObject(const Concept *impl, ::mlir::Attribute tablegen_opaque_val, ::mlir::Operation* module, const ::mlir::gpu::TargetOptions& options) {
  return static_cast<const ConcreteAttr *>(impl)->serializeToObject(tablegen_opaque_val, module, options);
}
template<typename ConcreteAttr>
::mlir::Attribute detail::TargetAttrInterfaceInterfaceTraits::FallbackModel<ConcreteAttr>::createObject(const Concept *impl, ::mlir::Attribute tablegen_opaque_val, ::mlir::Operation * module, const ::llvm::SmallVector<char, 0> & object, const ::mlir::gpu::TargetOptions & options) {
  return static_cast<const ConcreteAttr *>(impl)->createObject(tablegen_opaque_val, module, object, options);
}
} // namespace gpu
} // namespace mlir
namespace mlir {
namespace gpu {
template<typename ConcreteAttr>
::llvm::LogicalResult detail::OffloadingLLVMTranslationAttrInterfaceInterfaceTraits::Model<ConcreteAttr>::embedBinary(const Concept *impl, ::mlir::Attribute tablegen_opaque_val, ::mlir::Operation* binaryOp, ::llvm::IRBuilderBase& hostBuilder, ::mlir::LLVM::ModuleTranslation& hostModuleTranslation) {
  return (::llvm::cast<ConcreteAttr>(tablegen_opaque_val)).embedBinary(binaryOp, hostBuilder, hostModuleTranslation);
}
template<typename ConcreteAttr>
::llvm::LogicalResult detail::OffloadingLLVMTranslationAttrInterfaceInterfaceTraits::Model<ConcreteAttr>::launchKernel(const Concept *impl, ::mlir::Attribute tablegen_opaque_val, ::mlir::Operation* launchFunc, ::mlir::Operation* binaryOp, ::llvm::IRBuilderBase& hostBuilder, ::mlir::LLVM::ModuleTranslation& hostModuleTranslation) {
  return (::llvm::cast<ConcreteAttr>(tablegen_opaque_val)).launchKernel(launchFunc, binaryOp, hostBuilder, hostModuleTranslation);
}
template<typename ConcreteAttr>
::llvm::LogicalResult detail::OffloadingLLVMTranslationAttrInterfaceInterfaceTraits::FallbackModel<ConcreteAttr>::embedBinary(const Concept *impl, ::mlir::Attribute tablegen_opaque_val, ::mlir::Operation* binaryOp, ::llvm::IRBuilderBase& hostBuilder, ::mlir::LLVM::ModuleTranslation& hostModuleTranslation) {
  return static_cast<const ConcreteAttr *>(impl)->embedBinary(tablegen_opaque_val, binaryOp, hostBuilder, hostModuleTranslation);
}
template<typename ConcreteAttr>
::llvm::LogicalResult detail::OffloadingLLVMTranslationAttrInterfaceInterfaceTraits::FallbackModel<ConcreteAttr>::launchKernel(const Concept *impl, ::mlir::Attribute tablegen_opaque_val, ::mlir::Operation* launchFunc, ::mlir::Operation* binaryOp, ::llvm::IRBuilderBase& hostBuilder, ::mlir::LLVM::ModuleTranslation& hostModuleTranslation) {
  return static_cast<const ConcreteAttr *>(impl)->launchKernel(tablegen_opaque_val, launchFunc, binaryOp, hostBuilder, hostModuleTranslation);
}
} // namespace gpu
} // namespace mlir
