/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Interface Declarations                                                     *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

class LayoutSensitiveInterface;
namespace detail {
struct LayoutSensitiveInterfaceInterfaceTraits {
  struct Concept {
    /// The methods defined by the interface.
    StringRef (*getDataFormat)(const Concept *impl, ::mlir::Operation *);
    StringRef (*data_format)(const Concept *impl, ::mlir::Operation *);
    SmallVector<unsigned, 4> (*GetLayoutDependentArgs)(const Concept *impl, ::mlir::Operation *);
    SmallVector<unsigned, 4> (*GetLayoutDependentResults)(const Concept *impl, ::mlir::Operation *);
    StringRef (*GetOptimalLayout)(const Concept *impl, ::mlir::Operation *, const RuntimeDevices&);
    LogicalResult (*UpdateDataFormat)(const Concept *impl, ::mlir::Operation *, StringRef);
  };
  template<typename ConcreteOp>
  class Model : public Concept {
  public:
    using Interface = LayoutSensitiveInterface;
    Model() : Concept{getDataFormat, data_format, GetLayoutDependentArgs, GetLayoutDependentResults, GetOptimalLayout, UpdateDataFormat} {}

    static inline StringRef getDataFormat(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline StringRef data_format(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline SmallVector<unsigned, 4> GetLayoutDependentArgs(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline SmallVector<unsigned, 4> GetLayoutDependentResults(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline StringRef GetOptimalLayout(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, const RuntimeDevices& devices);
    static inline LogicalResult UpdateDataFormat(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, StringRef data_format);
  };
  template<typename ConcreteOp>
  class FallbackModel : public Concept {
  public:
    using Interface = LayoutSensitiveInterface;
    FallbackModel() : Concept{getDataFormat, data_format, GetLayoutDependentArgs, GetLayoutDependentResults, GetOptimalLayout, UpdateDataFormat} {}

    static inline StringRef getDataFormat(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline StringRef data_format(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline SmallVector<unsigned, 4> GetLayoutDependentArgs(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline SmallVector<unsigned, 4> GetLayoutDependentResults(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline StringRef GetOptimalLayout(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, const RuntimeDevices& devices);
    static inline LogicalResult UpdateDataFormat(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, StringRef data_format);
  };
  template<typename ConcreteModel, typename ConcreteOp>
  class ExternalModel : public FallbackModel<ConcreteModel> {
  public:
    using ConcreteEntity = ConcreteOp;
  };
};
template <typename ConcreteOp>
struct LayoutSensitiveInterfaceTrait;

} // namespace detail
class LayoutSensitiveInterface : public ::mlir::OpInterface<LayoutSensitiveInterface, detail::LayoutSensitiveInterfaceInterfaceTraits> {
public:
  using ::mlir::OpInterface<LayoutSensitiveInterface, detail::LayoutSensitiveInterfaceInterfaceTraits>::OpInterface;
  template <typename ConcreteOp>
  struct Trait : public detail::LayoutSensitiveInterfaceTrait<ConcreteOp> {};
  /// Returns current operation data format (data layout).
  StringRef getDataFormat();
  /// Deprecated method that returns current operation data format (data layout).
  StringRef data_format();
  /// Returns indices of layout dependent arguments.
  SmallVector<unsigned, 4> GetLayoutDependentArgs();
  /// Returns indices of layout dependent results.
  SmallVector<unsigned, 4> GetLayoutDependentResults();
  /// Returns the optimal data layout based on the available devices.
  StringRef GetOptimalLayout(const RuntimeDevices& devices);
  /// Updates operation attributes and operands to account for the updated
  ///         data format. If data format is not supported, must return failure.
  LogicalResult UpdateDataFormat(StringRef data_format);
};
namespace detail {
  template <typename ConcreteOp>
  struct LayoutSensitiveInterfaceTrait : public ::mlir::OpInterface<LayoutSensitiveInterface, detail::LayoutSensitiveInterfaceInterfaceTraits>::Trait<ConcreteOp> {
    static ::llvm::LogicalResult verifyTrait(::mlir::Operation *op) {
      return VerifyLayoutSensitiveInterface(op);
    }
  };
}// namespace detail
class FoldOperandsTransposeInterface;
namespace detail {
struct FoldOperandsTransposeInterfaceInterfaceTraits {
  struct Concept {
    /// The methods defined by the interface.
    SmallVector<unsigned, 4> (*GetLayoutDependentArgs)(const Concept *impl, ::mlir::Operation *);
    SmallVector<unsigned, 4> (*GetLayoutDependentResults)(const Concept *impl, ::mlir::Operation *);
    LogicalResult (*FoldOperandsPermutation)(const Concept *impl, ::mlir::Operation *, ArrayRef<int64_t>);
  };
  template<typename ConcreteOp>
  class Model : public Concept {
  public:
    using Interface = FoldOperandsTransposeInterface;
    Model() : Concept{GetLayoutDependentArgs, GetLayoutDependentResults, FoldOperandsPermutation} {}

    static inline SmallVector<unsigned, 4> GetLayoutDependentArgs(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline SmallVector<unsigned, 4> GetLayoutDependentResults(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline LogicalResult FoldOperandsPermutation(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, ArrayRef<int64_t> permutation);
  };
  template<typename ConcreteOp>
  class FallbackModel : public Concept {
  public:
    using Interface = FoldOperandsTransposeInterface;
    FallbackModel() : Concept{GetLayoutDependentArgs, GetLayoutDependentResults, FoldOperandsPermutation} {}

    static inline SmallVector<unsigned, 4> GetLayoutDependentArgs(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline SmallVector<unsigned, 4> GetLayoutDependentResults(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline LogicalResult FoldOperandsPermutation(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, ArrayRef<int64_t> permutation);
  };
  template<typename ConcreteModel, typename ConcreteOp>
  class ExternalModel : public FallbackModel<ConcreteModel> {
  public:
    using ConcreteEntity = ConcreteOp;
  };
};
template <typename ConcreteOp>
struct FoldOperandsTransposeInterfaceTrait;

} // namespace detail
class FoldOperandsTransposeInterface : public ::mlir::OpInterface<FoldOperandsTransposeInterface, detail::FoldOperandsTransposeInterfaceInterfaceTraits> {
public:
  using ::mlir::OpInterface<FoldOperandsTransposeInterface, detail::FoldOperandsTransposeInterfaceInterfaceTraits>::OpInterface;
  template <typename ConcreteOp>
  struct Trait : public detail::FoldOperandsTransposeInterfaceTrait<ConcreteOp> {};
  /// Returns indices of layout dependent arguments.
  SmallVector<unsigned, 4> GetLayoutDependentArgs();
  /// Returns indices of layout dependent results.
  SmallVector<unsigned, 4> GetLayoutDependentResults();
  /// Updates operation attributes and operands to account for the folded
  ///         permutation. If folding of permutation is not possible, must return
  ///         failure.
  LogicalResult FoldOperandsPermutation(ArrayRef<int64_t> permutation);
};
namespace detail {
  template <typename ConcreteOp>
  struct FoldOperandsTransposeInterfaceTrait : public ::mlir::OpInterface<FoldOperandsTransposeInterface, detail::FoldOperandsTransposeInterfaceInterfaceTraits>::Trait<ConcreteOp> {
    static ::llvm::LogicalResult verifyTrait(::mlir::Operation *op) {
      return VerifyFoldOperandsTransposeInterface(op);
    }
  };
}// namespace detail
class ResourceHandleAllocatorInterface;
namespace detail {
struct ResourceHandleAllocatorInterfaceInterfaceTraits {
  struct Concept {
    /// The methods defined by the interface.
    llvm::SmallVector<ResourceHandleValueAndId, 4> (*GetResourceHandleValueAndIdList)(const Concept *impl, ::mlir::Operation *, llvm::SmallDenseMap<ResourceHandle, int64_t>&, int64_t&);
  };
  template<typename ConcreteOp>
  class Model : public Concept {
  public:
    using Interface = ResourceHandleAllocatorInterface;
    Model() : Concept{GetResourceHandleValueAndIdList} {}

    static inline llvm::SmallVector<ResourceHandleValueAndId, 4> GetResourceHandleValueAndIdList(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, llvm::SmallDenseMap<ResourceHandle, int64_t>& resource_handle_id_map, int64_t& next_id);
  };
  template<typename ConcreteOp>
  class FallbackModel : public Concept {
  public:
    using Interface = ResourceHandleAllocatorInterface;
    FallbackModel() : Concept{GetResourceHandleValueAndIdList} {}

    static inline llvm::SmallVector<ResourceHandleValueAndId, 4> GetResourceHandleValueAndIdList(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, llvm::SmallDenseMap<ResourceHandle, int64_t>& resource_handle_id_map, int64_t& next_id);
  };
  template<typename ConcreteModel, typename ConcreteOp>
  class ExternalModel : public FallbackModel<ConcreteModel> {
  public:
    using ConcreteEntity = ConcreteOp;
  };
};
template <typename ConcreteOp>
struct ResourceHandleAllocatorInterfaceTrait;

} // namespace detail
class ResourceHandleAllocatorInterface : public ::mlir::OpInterface<ResourceHandleAllocatorInterface, detail::ResourceHandleAllocatorInterfaceInterfaceTraits> {
public:
  using ::mlir::OpInterface<ResourceHandleAllocatorInterface, detail::ResourceHandleAllocatorInterfaceInterfaceTraits>::OpInterface;
  template <typename ConcreteOp>
  struct Trait : public detail::ResourceHandleAllocatorInterfaceTrait<ConcreteOp> {};
  /// Returns resource handle values and the unique ids associated with
  ///                  the resource handles for this op. The handles should be created
  ///                  for only the resource tensors in the results of the op. If a
  ///                  resource handle is reused, then an existing id will be
  ///                  returned. The order of the resource handles in the returned
  ///                  vector are the order of those resources in the results of the
  ///                  op.
  llvm::SmallVector<ResourceHandleValueAndId, 4> GetResourceHandleValueAndIdList(llvm::SmallDenseMap<ResourceHandle, int64_t>& resource_handle_id_map, int64_t& next_id);
};
namespace detail {
  template <typename ConcreteOp>
  struct ResourceHandleAllocatorInterfaceTrait : public ::mlir::OpInterface<ResourceHandleAllocatorInterface, detail::ResourceHandleAllocatorInterfaceInterfaceTraits>::Trait<ConcreteOp> {
  };
}// namespace detail
class GetResourceInstanceInterface;
namespace detail {
struct GetResourceInstanceInterfaceInterfaceTraits {
  struct Concept {
    /// The methods defined by the interface.
    std::optional<std::string> (*GetResourceInstanceStr)(const Concept *impl, ::mlir::Operation *);
  };
  template<typename ConcreteOp>
  class Model : public Concept {
  public:
    using Interface = GetResourceInstanceInterface;
    Model() : Concept{GetResourceInstanceStr} {}

    static inline std::optional<std::string> GetResourceInstanceStr(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
  };
  template<typename ConcreteOp>
  class FallbackModel : public Concept {
  public:
    using Interface = GetResourceInstanceInterface;
    FallbackModel() : Concept{GetResourceInstanceStr} {}

    static inline std::optional<std::string> GetResourceInstanceStr(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
  };
  template<typename ConcreteModel, typename ConcreteOp>
  class ExternalModel : public FallbackModel<ConcreteModel> {
  public:
    using ConcreteEntity = ConcreteOp;
  };
};
template <typename ConcreteOp>
struct GetResourceInstanceInterfaceTrait;

} // namespace detail
class GetResourceInstanceInterface : public ::mlir::OpInterface<GetResourceInstanceInterface, detail::GetResourceInstanceInterfaceInterfaceTraits> {
public:
  using ::mlir::OpInterface<GetResourceInstanceInterface, detail::GetResourceInstanceInterfaceInterfaceTraits>::OpInterface;
  template <typename ConcreteOp>
  struct Trait : public detail::GetResourceInstanceInterfaceTrait<ConcreteOp> {};
  /// Returns a string corresponding to the resource instance
  ///                  accessed by this op, or `std::nullopt` if the resource should
  ///                  be ignored. The implementation must guarantee that the
  ///                  mapping between resource instances and strings is bijective,
  ///                  i.e., two op instances should return the same string if and
  ///                  only if they access the same resource. The interface should
  ///                  only be used for ops that access exactly one op-based resource
  ///                  (see `tf_op_base.td` for details).
  std::optional<std::string> GetResourceInstanceStr();
};
namespace detail {
  template <typename ConcreteOp>
  struct GetResourceInstanceInterfaceTrait : public ::mlir::OpInterface<GetResourceInstanceInterface, detail::GetResourceInstanceInterfaceInterfaceTraits>::Trait<ConcreteOp> {
  };
}// namespace detail
template<typename ConcreteOp>
StringRef detail::LayoutSensitiveInterfaceInterfaceTraits::Model<ConcreteOp>::getDataFormat(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getDataFormat();
}
template<typename ConcreteOp>
StringRef detail::LayoutSensitiveInterfaceInterfaceTraits::Model<ConcreteOp>::data_format(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getDataFormat();
}
template<typename ConcreteOp>
SmallVector<unsigned, 4> detail::LayoutSensitiveInterfaceInterfaceTraits::Model<ConcreteOp>::GetLayoutDependentArgs(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).GetLayoutDependentArgs();
}
template<typename ConcreteOp>
SmallVector<unsigned, 4> detail::LayoutSensitiveInterfaceInterfaceTraits::Model<ConcreteOp>::GetLayoutDependentResults(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).GetLayoutDependentResults();
}
template<typename ConcreteOp>
StringRef detail::LayoutSensitiveInterfaceInterfaceTraits::Model<ConcreteOp>::GetOptimalLayout(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, const RuntimeDevices& devices) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).GetOptimalLayout(devices);
}
template<typename ConcreteOp>
LogicalResult detail::LayoutSensitiveInterfaceInterfaceTraits::Model<ConcreteOp>::UpdateDataFormat(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, StringRef data_format) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).UpdateDataFormat(data_format);
}
template<typename ConcreteOp>
StringRef detail::LayoutSensitiveInterfaceInterfaceTraits::FallbackModel<ConcreteOp>::getDataFormat(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return static_cast<const ConcreteOp *>(impl)->getDataFormat(tablegen_opaque_val);
}
template<typename ConcreteOp>
StringRef detail::LayoutSensitiveInterfaceInterfaceTraits::FallbackModel<ConcreteOp>::data_format(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return static_cast<const ConcreteOp *>(impl)->data_format(tablegen_opaque_val);
}
template<typename ConcreteOp>
SmallVector<unsigned, 4> detail::LayoutSensitiveInterfaceInterfaceTraits::FallbackModel<ConcreteOp>::GetLayoutDependentArgs(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return static_cast<const ConcreteOp *>(impl)->GetLayoutDependentArgs(tablegen_opaque_val);
}
template<typename ConcreteOp>
SmallVector<unsigned, 4> detail::LayoutSensitiveInterfaceInterfaceTraits::FallbackModel<ConcreteOp>::GetLayoutDependentResults(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return static_cast<const ConcreteOp *>(impl)->GetLayoutDependentResults(tablegen_opaque_val);
}
template<typename ConcreteOp>
StringRef detail::LayoutSensitiveInterfaceInterfaceTraits::FallbackModel<ConcreteOp>::GetOptimalLayout(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, const RuntimeDevices& devices) {
  return static_cast<const ConcreteOp *>(impl)->GetOptimalLayout(tablegen_opaque_val, devices);
}
template<typename ConcreteOp>
LogicalResult detail::LayoutSensitiveInterfaceInterfaceTraits::FallbackModel<ConcreteOp>::UpdateDataFormat(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, StringRef data_format) {
  return static_cast<const ConcreteOp *>(impl)->UpdateDataFormat(tablegen_opaque_val, data_format);
}
template<typename ConcreteOp>
SmallVector<unsigned, 4> detail::FoldOperandsTransposeInterfaceInterfaceTraits::Model<ConcreteOp>::GetLayoutDependentArgs(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).GetLayoutDependentArgs();
}
template<typename ConcreteOp>
SmallVector<unsigned, 4> detail::FoldOperandsTransposeInterfaceInterfaceTraits::Model<ConcreteOp>::GetLayoutDependentResults(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).GetLayoutDependentResults();
}
template<typename ConcreteOp>
LogicalResult detail::FoldOperandsTransposeInterfaceInterfaceTraits::Model<ConcreteOp>::FoldOperandsPermutation(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, ArrayRef<int64_t> permutation) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).FoldOperandsPermutation(permutation);
}
template<typename ConcreteOp>
SmallVector<unsigned, 4> detail::FoldOperandsTransposeInterfaceInterfaceTraits::FallbackModel<ConcreteOp>::GetLayoutDependentArgs(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return static_cast<const ConcreteOp *>(impl)->GetLayoutDependentArgs(tablegen_opaque_val);
}
template<typename ConcreteOp>
SmallVector<unsigned, 4> detail::FoldOperandsTransposeInterfaceInterfaceTraits::FallbackModel<ConcreteOp>::GetLayoutDependentResults(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return static_cast<const ConcreteOp *>(impl)->GetLayoutDependentResults(tablegen_opaque_val);
}
template<typename ConcreteOp>
LogicalResult detail::FoldOperandsTransposeInterfaceInterfaceTraits::FallbackModel<ConcreteOp>::FoldOperandsPermutation(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, ArrayRef<int64_t> permutation) {
  return static_cast<const ConcreteOp *>(impl)->FoldOperandsPermutation(tablegen_opaque_val, permutation);
}
template<typename ConcreteOp>
llvm::SmallVector<ResourceHandleValueAndId, 4> detail::ResourceHandleAllocatorInterfaceInterfaceTraits::Model<ConcreteOp>::GetResourceHandleValueAndIdList(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, llvm::SmallDenseMap<ResourceHandle, int64_t>& resource_handle_id_map, int64_t& next_id) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).GetResourceHandleValueAndIdList(resource_handle_id_map, next_id);
}
template<typename ConcreteOp>
llvm::SmallVector<ResourceHandleValueAndId, 4> detail::ResourceHandleAllocatorInterfaceInterfaceTraits::FallbackModel<ConcreteOp>::GetResourceHandleValueAndIdList(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, llvm::SmallDenseMap<ResourceHandle, int64_t>& resource_handle_id_map, int64_t& next_id) {
  return static_cast<const ConcreteOp *>(impl)->GetResourceHandleValueAndIdList(tablegen_opaque_val, resource_handle_id_map, next_id);
}
template<typename ConcreteOp>
std::optional<std::string> detail::GetResourceInstanceInterfaceInterfaceTraits::Model<ConcreteOp>::GetResourceInstanceStr(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).GetResourceInstanceStr();
}
template<typename ConcreteOp>
std::optional<std::string> detail::GetResourceInstanceInterfaceInterfaceTraits::FallbackModel<ConcreteOp>::GetResourceInstanceStr(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return static_cast<const ConcreteOp *>(impl)->GetResourceInstanceStr(tablegen_opaque_val);
}
