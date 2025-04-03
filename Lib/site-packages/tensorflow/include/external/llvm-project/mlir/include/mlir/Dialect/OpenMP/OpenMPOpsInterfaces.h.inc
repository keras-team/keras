/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Interface Declarations                                                     *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

namespace mlir {
namespace omp {
class OutlineableOpenMPOpInterface;
namespace detail {
struct OutlineableOpenMPOpInterfaceInterfaceTraits {
  struct Concept {
    /// The methods defined by the interface.
    ::mlir::Block*(*getAllocaBlock)(const Concept *impl, ::mlir::Operation *);
  };
  template<typename ConcreteOp>
  class Model : public Concept {
  public:
    using Interface = ::mlir::omp::OutlineableOpenMPOpInterface;
    Model() : Concept{getAllocaBlock} {}

    static inline ::mlir::Block*getAllocaBlock(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
  };
  template<typename ConcreteOp>
  class FallbackModel : public Concept {
  public:
    using Interface = ::mlir::omp::OutlineableOpenMPOpInterface;
    FallbackModel() : Concept{getAllocaBlock} {}

    static inline ::mlir::Block*getAllocaBlock(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
  };
  template<typename ConcreteModel, typename ConcreteOp>
  class ExternalModel : public FallbackModel<ConcreteModel> {
  public:
    using ConcreteEntity = ConcreteOp;
  };
};
template <typename ConcreteOp>
struct OutlineableOpenMPOpInterfaceTrait;

} // namespace detail
class OutlineableOpenMPOpInterface : public ::mlir::OpInterface<OutlineableOpenMPOpInterface, detail::OutlineableOpenMPOpInterfaceInterfaceTraits> {
public:
  using ::mlir::OpInterface<OutlineableOpenMPOpInterface, detail::OutlineableOpenMPOpInterfaceInterfaceTraits>::OpInterface;
  template <typename ConcreteOp>
  struct Trait : public detail::OutlineableOpenMPOpInterfaceTrait<ConcreteOp> {};
  /// Get alloca block
  ::mlir::Block*getAllocaBlock();
};
namespace detail {
  template <typename ConcreteOp>
  struct OutlineableOpenMPOpInterfaceTrait : public ::mlir::OpInterface<OutlineableOpenMPOpInterface, detail::OutlineableOpenMPOpInterfaceInterfaceTraits>::Trait<ConcreteOp> {
  };
}// namespace detail
} // namespace omp
} // namespace mlir
namespace mlir {
namespace omp {
class MapClauseOwningOpInterface;
namespace detail {
struct MapClauseOwningOpInterfaceInterfaceTraits {
  struct Concept {
    /// The methods defined by the interface.
    ::mlir::OperandRange (*getMapVars)(const Concept *impl, ::mlir::Operation *);
    ::mlir::MutableOperandRange (*getMapVarsMutable)(const Concept *impl, ::mlir::Operation *);
  };
  template<typename ConcreteOp>
  class Model : public Concept {
  public:
    using Interface = ::mlir::omp::MapClauseOwningOpInterface;
    Model() : Concept{getMapVars, getMapVarsMutable} {}

    static inline ::mlir::OperandRange getMapVars(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline ::mlir::MutableOperandRange getMapVarsMutable(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
  };
  template<typename ConcreteOp>
  class FallbackModel : public Concept {
  public:
    using Interface = ::mlir::omp::MapClauseOwningOpInterface;
    FallbackModel() : Concept{getMapVars, getMapVarsMutable} {}

    static inline ::mlir::OperandRange getMapVars(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline ::mlir::MutableOperandRange getMapVarsMutable(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
  };
  template<typename ConcreteModel, typename ConcreteOp>
  class ExternalModel : public FallbackModel<ConcreteModel> {
  public:
    using ConcreteEntity = ConcreteOp;
  };
};
template <typename ConcreteOp>
struct MapClauseOwningOpInterfaceTrait;

} // namespace detail
class MapClauseOwningOpInterface : public ::mlir::OpInterface<MapClauseOwningOpInterface, detail::MapClauseOwningOpInterfaceInterfaceTraits> {
public:
  using ::mlir::OpInterface<MapClauseOwningOpInterface, detail::MapClauseOwningOpInterfaceInterfaceTraits>::OpInterface;
  template <typename ConcreteOp>
  struct Trait : public detail::MapClauseOwningOpInterfaceTrait<ConcreteOp> {};
  /// Get map operands
  ::mlir::OperandRange getMapVars();
  /// Get mutable map operands
  ::mlir::MutableOperandRange getMapVarsMutable();
};
namespace detail {
  template <typename ConcreteOp>
  struct MapClauseOwningOpInterfaceTrait : public ::mlir::OpInterface<MapClauseOwningOpInterface, detail::MapClauseOwningOpInterfaceInterfaceTraits>::Trait<ConcreteOp> {
  };
}// namespace detail
} // namespace omp
} // namespace mlir
namespace mlir {
namespace omp {
class ReductionClauseInterface;
namespace detail {
struct ReductionClauseInterfaceInterfaceTraits {
  struct Concept {
    /// The methods defined by the interface.
    ::mlir::SmallVector<::mlir::Value> (*getAllReductionVars)(const Concept *impl, ::mlir::Operation *);
  };
  template<typename ConcreteOp>
  class Model : public Concept {
  public:
    using Interface = ::mlir::omp::ReductionClauseInterface;
    Model() : Concept{getAllReductionVars} {}

    static inline ::mlir::SmallVector<::mlir::Value> getAllReductionVars(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
  };
  template<typename ConcreteOp>
  class FallbackModel : public Concept {
  public:
    using Interface = ::mlir::omp::ReductionClauseInterface;
    FallbackModel() : Concept{getAllReductionVars} {}

    static inline ::mlir::SmallVector<::mlir::Value> getAllReductionVars(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
  };
  template<typename ConcreteModel, typename ConcreteOp>
  class ExternalModel : public FallbackModel<ConcreteModel> {
  public:
    using ConcreteEntity = ConcreteOp;
    ::mlir::SmallVector<::mlir::Value> getAllReductionVars(::mlir::Operation *tablegen_opaque_val) const;
  };
};
template <typename ConcreteOp>
struct ReductionClauseInterfaceTrait;

} // namespace detail
class ReductionClauseInterface : public ::mlir::OpInterface<ReductionClauseInterface, detail::ReductionClauseInterfaceInterfaceTraits> {
public:
  using ::mlir::OpInterface<ReductionClauseInterface, detail::ReductionClauseInterfaceInterfaceTraits>::OpInterface;
  template <typename ConcreteOp>
  struct Trait : public detail::ReductionClauseInterfaceTrait<ConcreteOp> {};
  /// Get reduction vars
  ::mlir::SmallVector<::mlir::Value> getAllReductionVars();
};
namespace detail {
  template <typename ConcreteOp>
  struct ReductionClauseInterfaceTrait : public ::mlir::OpInterface<ReductionClauseInterface, detail::ReductionClauseInterfaceInterfaceTraits>::Trait<ConcreteOp> {
    /// Get reduction vars
    ::mlir::SmallVector<::mlir::Value> getAllReductionVars() {
      return (*static_cast<ConcreteOp *>(this)).getReductionVars();
    }
  };
}// namespace detail
} // namespace omp
} // namespace mlir
namespace mlir {
namespace omp {
class LoopWrapperInterface;
namespace detail {
struct LoopWrapperInterfaceInterfaceTraits {
  struct Concept {
    /// The methods defined by the interface.
    ::mlir::omp::LoopWrapperInterface (*getNestedWrapper)(const Concept *impl, ::mlir::Operation *);
    ::mlir::Operation *(*getWrappedLoop)(const Concept *impl, ::mlir::Operation *);
  };
  template<typename ConcreteOp>
  class Model : public Concept {
  public:
    using Interface = ::mlir::omp::LoopWrapperInterface;
    Model() : Concept{getNestedWrapper, getWrappedLoop} {}

    static inline ::mlir::omp::LoopWrapperInterface getNestedWrapper(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline ::mlir::Operation *getWrappedLoop(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
  };
  template<typename ConcreteOp>
  class FallbackModel : public Concept {
  public:
    using Interface = ::mlir::omp::LoopWrapperInterface;
    FallbackModel() : Concept{getNestedWrapper, getWrappedLoop} {}

    static inline ::mlir::omp::LoopWrapperInterface getNestedWrapper(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline ::mlir::Operation *getWrappedLoop(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
  };
  template<typename ConcreteModel, typename ConcreteOp>
  class ExternalModel : public FallbackModel<ConcreteModel> {
  public:
    using ConcreteEntity = ConcreteOp;
    ::mlir::omp::LoopWrapperInterface getNestedWrapper(::mlir::Operation *tablegen_opaque_val) const;
    ::mlir::Operation *getWrappedLoop(::mlir::Operation *tablegen_opaque_val) const;
  };
};
template <typename ConcreteOp>
struct LoopWrapperInterfaceTrait;

} // namespace detail
class LoopWrapperInterface : public ::mlir::OpInterface<LoopWrapperInterface, detail::LoopWrapperInterfaceInterfaceTraits> {
public:
  using ::mlir::OpInterface<LoopWrapperInterface, detail::LoopWrapperInterfaceInterfaceTraits>::OpInterface;
  template <typename ConcreteOp>
  struct Trait : public detail::LoopWrapperInterfaceTrait<ConcreteOp> {};
  /// If there is another loop wrapper immediately nested inside, return that
  /// operation. Assumes this operation is a valid loop wrapper.
  ::mlir::omp::LoopWrapperInterface getNestedWrapper();
  /// Return the loop nest nested directly or indirectly inside of this loop
  /// wrapper. Assumes this operation is a valid loop wrapper.
  ::mlir::Operation *getWrappedLoop();
};
namespace detail {
  template <typename ConcreteOp>
  struct LoopWrapperInterfaceTrait : public ::mlir::OpInterface<LoopWrapperInterface, detail::LoopWrapperInterfaceInterfaceTraits>::Trait<ConcreteOp> {
    /// If there is another loop wrapper immediately nested inside, return that
    /// operation. Assumes this operation is a valid loop wrapper.
    ::mlir::omp::LoopWrapperInterface getNestedWrapper() {
      Operation *nested = &*(*static_cast<ConcreteOp *>(this))->getRegion(0).op_begin();
        return ::llvm::dyn_cast<LoopWrapperInterface>(nested);
    }
    /// Return the loop nest nested directly or indirectly inside of this loop
    /// wrapper. Assumes this operation is a valid loop wrapper.
    ::mlir::Operation *getWrappedLoop() {
      if (LoopWrapperInterface nested = (*static_cast<ConcreteOp *>(this)).getNestedWrapper())
          return nested.getWrappedLoop();
        return &*(*static_cast<ConcreteOp *>(this))->getRegion(0).op_begin();
    }
  };
}// namespace detail
} // namespace omp
} // namespace mlir
namespace mlir {
namespace omp {
class ComposableOpInterface;
namespace detail {
struct ComposableOpInterfaceInterfaceTraits {
  struct Concept {
    /// The methods defined by the interface.
    bool (*isComposite)(const Concept *impl, ::mlir::Operation *);
    void (*setComposite)(const Concept *impl, ::mlir::Operation *, bool);
  };
  template<typename ConcreteOp>
  class Model : public Concept {
  public:
    using Interface = ::mlir::omp::ComposableOpInterface;
    Model() : Concept{isComposite, setComposite} {}

    static inline bool isComposite(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline void setComposite(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, bool val);
  };
  template<typename ConcreteOp>
  class FallbackModel : public Concept {
  public:
    using Interface = ::mlir::omp::ComposableOpInterface;
    FallbackModel() : Concept{isComposite, setComposite} {}

    static inline bool isComposite(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline void setComposite(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, bool val);
  };
  template<typename ConcreteModel, typename ConcreteOp>
  class ExternalModel : public FallbackModel<ConcreteModel> {
  public:
    using ConcreteEntity = ConcreteOp;
    bool isComposite(::mlir::Operation *tablegen_opaque_val) const;
    void setComposite(::mlir::Operation *tablegen_opaque_val, bool val) const;
  };
};
template <typename ConcreteOp>
struct ComposableOpInterfaceTrait;

} // namespace detail
class ComposableOpInterface : public ::mlir::OpInterface<ComposableOpInterface, detail::ComposableOpInterfaceInterfaceTraits> {
public:
  using ::mlir::OpInterface<ComposableOpInterface, detail::ComposableOpInterfaceInterfaceTraits>::OpInterface;
  template <typename ConcreteOp>
  struct Trait : public detail::ComposableOpInterfaceTrait<ConcreteOp> {};
  /// Check whether the operation is representing a leaf of a composite OpenMP
  /// construct.
  bool isComposite();
  /// Mark the operation as part of an OpenMP composite construct.
  void setComposite(bool val);
};
namespace detail {
  template <typename ConcreteOp>
  struct ComposableOpInterfaceTrait : public ::mlir::OpInterface<ComposableOpInterface, detail::ComposableOpInterfaceInterfaceTraits>::Trait<ConcreteOp> {
    /// Check whether the operation is representing a leaf of a composite OpenMP
    /// construct.
    bool isComposite() {
      return (*static_cast<ConcreteOp *>(this))->hasAttr("omp.composite");
    }
    /// Mark the operation as part of an OpenMP composite construct.
    void setComposite(bool val) {
      if (val)
          (*static_cast<ConcreteOp *>(this))->setDiscardableAttr("omp.composite", mlir::UnitAttr::get((*static_cast<ConcreteOp *>(this))->getContext()));
        else
          (*static_cast<ConcreteOp *>(this))->removeDiscardableAttr("omp.composite");
    }
  };
}// namespace detail
} // namespace omp
} // namespace mlir
namespace mlir {
namespace omp {
class DeclareTargetInterface;
namespace detail {
struct DeclareTargetInterfaceInterfaceTraits {
  struct Concept {
    /// The methods defined by the interface.
    void (*setDeclareTarget)(const Concept *impl, ::mlir::Operation *, mlir::omp::DeclareTargetDeviceType, mlir::omp::DeclareTargetCaptureClause);
    bool (*isDeclareTarget)(const Concept *impl, ::mlir::Operation *);
    mlir::omp::DeclareTargetDeviceType (*getDeclareTargetDeviceType)(const Concept *impl, ::mlir::Operation *);
    mlir::omp::DeclareTargetCaptureClause (*getDeclareTargetCaptureClause)(const Concept *impl, ::mlir::Operation *);
  };
  template<typename ConcreteOp>
  class Model : public Concept {
  public:
    using Interface = ::mlir::omp::DeclareTargetInterface;
    Model() : Concept{setDeclareTarget, isDeclareTarget, getDeclareTargetDeviceType, getDeclareTargetCaptureClause} {}

    static inline void setDeclareTarget(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, mlir::omp::DeclareTargetDeviceType deviceType, mlir::omp::DeclareTargetCaptureClause captureClause);
    static inline bool isDeclareTarget(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline mlir::omp::DeclareTargetDeviceType getDeclareTargetDeviceType(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline mlir::omp::DeclareTargetCaptureClause getDeclareTargetCaptureClause(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
  };
  template<typename ConcreteOp>
  class FallbackModel : public Concept {
  public:
    using Interface = ::mlir::omp::DeclareTargetInterface;
    FallbackModel() : Concept{setDeclareTarget, isDeclareTarget, getDeclareTargetDeviceType, getDeclareTargetCaptureClause} {}

    static inline void setDeclareTarget(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, mlir::omp::DeclareTargetDeviceType deviceType, mlir::omp::DeclareTargetCaptureClause captureClause);
    static inline bool isDeclareTarget(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline mlir::omp::DeclareTargetDeviceType getDeclareTargetDeviceType(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline mlir::omp::DeclareTargetCaptureClause getDeclareTargetCaptureClause(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
  };
  template<typename ConcreteModel, typename ConcreteOp>
  class ExternalModel : public FallbackModel<ConcreteModel> {
  public:
    using ConcreteEntity = ConcreteOp;
    void setDeclareTarget(::mlir::Operation *tablegen_opaque_val, mlir::omp::DeclareTargetDeviceType deviceType, mlir::omp::DeclareTargetCaptureClause captureClause) const;
    bool isDeclareTarget(::mlir::Operation *tablegen_opaque_val) const;
    mlir::omp::DeclareTargetDeviceType getDeclareTargetDeviceType(::mlir::Operation *tablegen_opaque_val) const;
    mlir::omp::DeclareTargetCaptureClause getDeclareTargetCaptureClause(::mlir::Operation *tablegen_opaque_val) const;
  };
};
template <typename ConcreteOp>
struct DeclareTargetInterfaceTrait;

} // namespace detail
class DeclareTargetInterface : public ::mlir::OpInterface<DeclareTargetInterface, detail::DeclareTargetInterfaceInterfaceTraits> {
public:
  using ::mlir::OpInterface<DeclareTargetInterface, detail::DeclareTargetInterfaceInterfaceTraits>::OpInterface;
  template <typename ConcreteOp>
  struct Trait : public detail::DeclareTargetInterfaceTrait<ConcreteOp> {};
  /// Set the declare target attribute on the current operation with the
  /// specified attribute arguments.
  void setDeclareTarget(mlir::omp::DeclareTargetDeviceType deviceType, mlir::omp::DeclareTargetCaptureClause captureClause);
  /// Checks if the declare target attribute has been applied and exists on the
  /// current operation. Returns true if it exists on it, otherwise returns
  /// false.
  bool isDeclareTarget();
  /// Returns the DeclareTargetDeviceType segment of the DeclareTarget attribute if it
  /// exists on the current operation. Otherwise it returns null.
  mlir::omp::DeclareTargetDeviceType getDeclareTargetDeviceType();
  /// Returns the DeclareTargetCaptureClause segment of the DeclareTarget attribute if it
  /// exists on the current operation. Otherwise it returns null.
  mlir::omp::DeclareTargetCaptureClause getDeclareTargetCaptureClause();
};
namespace detail {
  template <typename ConcreteOp>
  struct DeclareTargetInterfaceTrait : public ::mlir::OpInterface<DeclareTargetInterface, detail::DeclareTargetInterfaceInterfaceTraits>::Trait<ConcreteOp> {
    /// Set the declare target attribute on the current operation with the
    /// specified attribute arguments.
    void setDeclareTarget(mlir::omp::DeclareTargetDeviceType deviceType, mlir::omp::DeclareTargetCaptureClause captureClause) {
      (*static_cast<ConcreteOp *>(this))->setAttr("omp.declare_target",
                  mlir::omp::DeclareTargetAttr::get(
                      (*static_cast<ConcreteOp *>(this))->getContext(),
                      mlir::omp::DeclareTargetDeviceTypeAttr::get(
                          (*static_cast<ConcreteOp *>(this))->getContext(), deviceType),
                      mlir::omp::DeclareTargetCaptureClauseAttr::get(
                          (*static_cast<ConcreteOp *>(this))->getContext(), captureClause)));
    }
    /// Checks if the declare target attribute has been applied and exists on the
    /// current operation. Returns true if it exists on it, otherwise returns
    /// false.
    bool isDeclareTarget() {
      return (*static_cast<ConcreteOp *>(this))->hasAttr("omp.declare_target");
    }
    /// Returns the DeclareTargetDeviceType segment of the DeclareTarget attribute if it
    /// exists on the current operation. Otherwise it returns null.
    mlir::omp::DeclareTargetDeviceType getDeclareTargetDeviceType() {
      if (mlir::Attribute dTar = (*static_cast<ConcreteOp *>(this))->getAttr("omp.declare_target"))
          if (auto dAttr = llvm::dyn_cast_or_null<mlir::omp::DeclareTargetAttr>(dTar))
            return dAttr.getDeviceType().getValue();
        return {};
    }
    /// Returns the DeclareTargetCaptureClause segment of the DeclareTarget attribute if it
    /// exists on the current operation. Otherwise it returns null.
    mlir::omp::DeclareTargetCaptureClause getDeclareTargetCaptureClause() {
      if (mlir::Attribute dTar = (*static_cast<ConcreteOp *>(this))->getAttr("omp.declare_target"))
          if (auto dAttr = llvm::dyn_cast_or_null<mlir::omp::DeclareTargetAttr>(dTar))
            return dAttr.getCaptureClause().getValue();
        return {};
    }
  };
}// namespace detail
} // namespace omp
} // namespace mlir
namespace mlir {
namespace omp {
class OffloadModuleInterface;
namespace detail {
struct OffloadModuleInterfaceInterfaceTraits {
  struct Concept {
    /// The methods defined by the interface.
    void (*setIsTargetDevice)(const Concept *impl, ::mlir::Operation *, bool);
    bool (*getIsTargetDevice)(const Concept *impl, ::mlir::Operation *);
    void (*setIsGPU)(const Concept *impl, ::mlir::Operation *, bool);
    bool (*getIsGPU)(const Concept *impl, ::mlir::Operation *);
    mlir::omp::FlagsAttr (*getFlags)(const Concept *impl, ::mlir::Operation *);
    void (*setFlags)(const Concept *impl, ::mlir::Operation *, uint32_t, bool, bool, bool, bool, uint32_t, bool);
    void (*setHostIRFilePath)(const Concept *impl, ::mlir::Operation *, std::string);
    llvm::StringRef (*getHostIRFilePath)(const Concept *impl, ::mlir::Operation *);
    ::mlir::omp::ClauseRequires (*getRequires)(const Concept *impl, ::mlir::Operation *);
    void (*setRequires)(const Concept *impl, ::mlir::Operation *, ::mlir::omp::ClauseRequires);
    ::llvm::ArrayRef<::mlir::Attribute> (*getTargetTriples)(const Concept *impl, ::mlir::Operation *);
    void (*setTargetTriples)(const Concept *impl, ::mlir::Operation *, ::llvm::ArrayRef<::std::string>);
  };
  template<typename ConcreteOp>
  class Model : public Concept {
  public:
    using Interface = ::mlir::omp::OffloadModuleInterface;
    Model() : Concept{setIsTargetDevice, getIsTargetDevice, setIsGPU, getIsGPU, getFlags, setFlags, setHostIRFilePath, getHostIRFilePath, getRequires, setRequires, getTargetTriples, setTargetTriples} {}

    static inline void setIsTargetDevice(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, bool isTargetDevice);
    static inline bool getIsTargetDevice(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline void setIsGPU(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, bool isGPU);
    static inline bool getIsGPU(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline mlir::omp::FlagsAttr getFlags(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline void setFlags(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, uint32_t debugKind, bool assumeTeamsOversubscription, bool assumeThreadsOversubscription, bool assumeNoThreadState, bool assumeNoNestedParallelism, uint32_t openmpDeviceVersion, bool noGPULib);
    static inline void setHostIRFilePath(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, std::string hostIRFilePath);
    static inline llvm::StringRef getHostIRFilePath(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline ::mlir::omp::ClauseRequires getRequires(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline void setRequires(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, ::mlir::omp::ClauseRequires clauses);
    static inline ::llvm::ArrayRef<::mlir::Attribute> getTargetTriples(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline void setTargetTriples(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, ::llvm::ArrayRef<::std::string> targetTriples);
  };
  template<typename ConcreteOp>
  class FallbackModel : public Concept {
  public:
    using Interface = ::mlir::omp::OffloadModuleInterface;
    FallbackModel() : Concept{setIsTargetDevice, getIsTargetDevice, setIsGPU, getIsGPU, getFlags, setFlags, setHostIRFilePath, getHostIRFilePath, getRequires, setRequires, getTargetTriples, setTargetTriples} {}

    static inline void setIsTargetDevice(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, bool isTargetDevice);
    static inline bool getIsTargetDevice(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline void setIsGPU(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, bool isGPU);
    static inline bool getIsGPU(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline mlir::omp::FlagsAttr getFlags(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline void setFlags(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, uint32_t debugKind, bool assumeTeamsOversubscription, bool assumeThreadsOversubscription, bool assumeNoThreadState, bool assumeNoNestedParallelism, uint32_t openmpDeviceVersion, bool noGPULib);
    static inline void setHostIRFilePath(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, std::string hostIRFilePath);
    static inline llvm::StringRef getHostIRFilePath(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline ::mlir::omp::ClauseRequires getRequires(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline void setRequires(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, ::mlir::omp::ClauseRequires clauses);
    static inline ::llvm::ArrayRef<::mlir::Attribute> getTargetTriples(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline void setTargetTriples(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, ::llvm::ArrayRef<::std::string> targetTriples);
  };
  template<typename ConcreteModel, typename ConcreteOp>
  class ExternalModel : public FallbackModel<ConcreteModel> {
  public:
    using ConcreteEntity = ConcreteOp;
    void setIsTargetDevice(::mlir::Operation *tablegen_opaque_val, bool isTargetDevice) const;
    bool getIsTargetDevice(::mlir::Operation *tablegen_opaque_val) const;
    void setIsGPU(::mlir::Operation *tablegen_opaque_val, bool isGPU) const;
    bool getIsGPU(::mlir::Operation *tablegen_opaque_val) const;
    mlir::omp::FlagsAttr getFlags(::mlir::Operation *tablegen_opaque_val) const;
    void setFlags(::mlir::Operation *tablegen_opaque_val, uint32_t debugKind, bool assumeTeamsOversubscription, bool assumeThreadsOversubscription, bool assumeNoThreadState, bool assumeNoNestedParallelism, uint32_t openmpDeviceVersion, bool noGPULib) const;
    void setHostIRFilePath(::mlir::Operation *tablegen_opaque_val, std::string hostIRFilePath) const;
    llvm::StringRef getHostIRFilePath(::mlir::Operation *tablegen_opaque_val) const;
    ::mlir::omp::ClauseRequires getRequires(::mlir::Operation *tablegen_opaque_val) const;
    void setRequires(::mlir::Operation *tablegen_opaque_val, ::mlir::omp::ClauseRequires clauses) const;
    ::llvm::ArrayRef<::mlir::Attribute> getTargetTriples(::mlir::Operation *tablegen_opaque_val) const;
    void setTargetTriples(::mlir::Operation *tablegen_opaque_val, ::llvm::ArrayRef<::std::string> targetTriples) const;
  };
};
template <typename ConcreteOp>
struct OffloadModuleInterfaceTrait;

} // namespace detail
class OffloadModuleInterface : public ::mlir::OpInterface<OffloadModuleInterface, detail::OffloadModuleInterfaceInterfaceTraits> {
public:
  using ::mlir::OpInterface<OffloadModuleInterface, detail::OffloadModuleInterfaceInterfaceTraits>::OpInterface;
  template <typename ConcreteOp>
  struct Trait : public detail::OffloadModuleInterfaceTrait<ConcreteOp> {};
  /// Set the attribute on the current module with the specified boolean
  /// argument.
  void setIsTargetDevice(bool isTargetDevice);
  /// Get the attribute on the current module if it exists and
  /// return its value, if it doesn't exist it returns false by default.
  bool getIsTargetDevice();
  /// Set the attribute on the current module with the specified boolean
  /// argument.
  void setIsGPU(bool isGPU);
  /// Get the attribute on the current module if it exists and
  /// return its value, if it doesn't exist it returns false by default.
  bool getIsGPU();
  /// Get the FlagsAttr attribute on the current module if it exists
  /// and return the attribute, if it doesn't exit it returns a nullptr
  mlir::omp::FlagsAttr getFlags();
  /// Apply an omp.FlagsAttr to a module with the specified values
  /// for the flags
  void setFlags(uint32_t debugKind, bool assumeTeamsOversubscription, bool assumeThreadsOversubscription, bool assumeNoThreadState, bool assumeNoNestedParallelism, uint32_t openmpDeviceVersion, bool noGPULib);
  /// Set a StringAttr on the current module containing the host IR file path. This
  /// file path is used in two-phase compilation during the device phase to generate
  /// device side LLVM IR when lowering MLIR.
  void setHostIRFilePath(std::string hostIRFilePath);
  /// Find the host-ir file path StringAttr from the current module if it exists and
  /// return its contained value, if it doesn't exist it returns an empty string. This
  /// file path is used in two-phase compilation during the device phase to generate
  /// device side LLVM IR when lowering MLIR.
  llvm::StringRef getHostIRFilePath();
  /// Get the omp.requires attribute on the operator if it's present and
  /// return its value. If it doesn't exist, return `ClauseRequires::none` by
  /// default.
  ::mlir::omp::ClauseRequires getRequires();
  /// Set the omp.requires attribute on the operator to the specified clauses.
  void setRequires(::mlir::omp::ClauseRequires clauses);
  /// Get the omp.target_triples attribute on the operator if it's present and
  /// return its value. If it doesn't exist, return an empty array by default.
  ::llvm::ArrayRef<::mlir::Attribute> getTargetTriples();
  /// Set the omp.target_triples attribute on the operation.
  void setTargetTriples(::llvm::ArrayRef<::std::string> targetTriples);
};
namespace detail {
  template <typename ConcreteOp>
  struct OffloadModuleInterfaceTrait : public ::mlir::OpInterface<OffloadModuleInterface, detail::OffloadModuleInterfaceInterfaceTraits>::Trait<ConcreteOp> {
    /// Set the attribute on the current module with the specified boolean
    /// argument.
    void setIsTargetDevice(bool isTargetDevice) {
      (*static_cast<ConcreteOp *>(this))->setAttr(
          mlir::StringAttr::get((*static_cast<ConcreteOp *>(this))->getContext(), llvm::Twine{"omp.is_target_device"}),
            mlir::BoolAttr::get((*static_cast<ConcreteOp *>(this))->getContext(), isTargetDevice));
    }
    /// Get the attribute on the current module if it exists and
    /// return its value, if it doesn't exist it returns false by default.
    bool getIsTargetDevice() {
      if (Attribute isTargetDevice = (*static_cast<ConcreteOp *>(this))->getAttr("omp.is_target_device"))
          if (::llvm::isa<mlir::BoolAttr>(isTargetDevice))
           return ::llvm::dyn_cast<BoolAttr>(isTargetDevice).getValue();
        return false;
    }
    /// Set the attribute on the current module with the specified boolean
    /// argument.
    void setIsGPU(bool isGPU) {
      (*static_cast<ConcreteOp *>(this))->setAttr(
          mlir::StringAttr::get((*static_cast<ConcreteOp *>(this))->getContext(), "omp.is_gpu"),
            mlir::BoolAttr::get((*static_cast<ConcreteOp *>(this))->getContext(), isGPU));
    }
    /// Get the attribute on the current module if it exists and
    /// return its value, if it doesn't exist it returns false by default.
    bool getIsGPU() {
      if (Attribute isTargetCGAttr = (*static_cast<ConcreteOp *>(this))->getAttr("omp.is_gpu"))
          if (auto isTargetCGVal = ::llvm::dyn_cast<BoolAttr>(isTargetCGAttr))
           return isTargetCGVal.getValue();
        return false;
    }
    /// Get the FlagsAttr attribute on the current module if it exists
    /// and return the attribute, if it doesn't exit it returns a nullptr
    mlir::omp::FlagsAttr getFlags() {
      if (Attribute flags = (*static_cast<ConcreteOp *>(this))->getAttr("omp.flags"))
          return ::llvm::dyn_cast_or_null<mlir::omp::FlagsAttr>(flags);
        return nullptr;
    }
    /// Apply an omp.FlagsAttr to a module with the specified values
    /// for the flags
    void setFlags(uint32_t debugKind, bool assumeTeamsOversubscription, bool assumeThreadsOversubscription, bool assumeNoThreadState, bool assumeNoNestedParallelism, uint32_t openmpDeviceVersion, bool noGPULib) {
      (*static_cast<ConcreteOp *>(this))->setAttr(("omp." + mlir::omp::FlagsAttr::getMnemonic()).str(),
                  mlir::omp::FlagsAttr::get((*static_cast<ConcreteOp *>(this))->getContext(), debugKind,
                      assumeTeamsOversubscription, assumeThreadsOversubscription,
                      assumeNoThreadState, assumeNoNestedParallelism, noGPULib, openmpDeviceVersion));
    }
    /// Set a StringAttr on the current module containing the host IR file path. This
    /// file path is used in two-phase compilation during the device phase to generate
    /// device side LLVM IR when lowering MLIR.
    void setHostIRFilePath(std::string hostIRFilePath) {
      (*static_cast<ConcreteOp *>(this))->setAttr(
          mlir::StringAttr::get((*static_cast<ConcreteOp *>(this))->getContext(), llvm::Twine{"omp.host_ir_filepath"}),
            mlir::StringAttr::get((*static_cast<ConcreteOp *>(this))->getContext(), hostIRFilePath));
    }
    /// Find the host-ir file path StringAttr from the current module if it exists and
    /// return its contained value, if it doesn't exist it returns an empty string. This
    /// file path is used in two-phase compilation during the device phase to generate
    /// device side LLVM IR when lowering MLIR.
    llvm::StringRef getHostIRFilePath() {
      if (Attribute filepath = (*static_cast<ConcreteOp *>(this))->getAttr("omp.host_ir_filepath"))
          if (::llvm::isa<mlir::StringAttr>(filepath))
            return ::llvm::dyn_cast<mlir::StringAttr>(filepath).getValue();
        return {};
    }
    /// Get the omp.requires attribute on the operator if it's present and
    /// return its value. If it doesn't exist, return `ClauseRequires::none` by
    /// default.
    ::mlir::omp::ClauseRequires getRequires() {
      if (Attribute requiresAttr = (*static_cast<ConcreteOp *>(this))->getAttr("omp.requires"))
          if (auto requiresVal = ::llvm::dyn_cast<mlir::omp::ClauseRequiresAttr>(requiresAttr))
            return requiresVal.getValue();
        return mlir::omp::ClauseRequires::none;
    }
    /// Set the omp.requires attribute on the operator to the specified clauses.
    void setRequires(::mlir::omp::ClauseRequires clauses) {
      (*static_cast<ConcreteOp *>(this))->setAttr(mlir::StringAttr::get((*static_cast<ConcreteOp *>(this))->getContext(), "omp.requires"),
          mlir::omp::ClauseRequiresAttr::get((*static_cast<ConcreteOp *>(this))->getContext(), clauses));
    }
    /// Get the omp.target_triples attribute on the operator if it's present and
    /// return its value. If it doesn't exist, return an empty array by default.
    ::llvm::ArrayRef<::mlir::Attribute> getTargetTriples() {
      if (Attribute triplesAttr = (*static_cast<ConcreteOp *>(this))->getAttr("omp.target_triples"))
          if (auto triples = ::llvm::dyn_cast<::mlir::ArrayAttr>(triplesAttr))
            return triples.getValue();
        return {};
    }
    /// Set the omp.target_triples attribute on the operation.
    void setTargetTriples(::llvm::ArrayRef<::std::string> targetTriples) {
      auto names = ::llvm::to_vector(::llvm::map_range(
            targetTriples, [&](::std::string str) -> ::mlir::Attribute {
              return mlir::StringAttr::get((*static_cast<ConcreteOp *>(this))->getContext(), str);
            }));
        (*static_cast<ConcreteOp *>(this))->setAttr(
            ::mlir::StringAttr::get((*static_cast<ConcreteOp *>(this))->getContext(), "omp.target_triples"),
            ::mlir::ArrayAttr::get((*static_cast<ConcreteOp *>(this))->getContext(), names));
    }
  };
}// namespace detail
} // namespace omp
} // namespace mlir
namespace mlir {
namespace omp {
template<typename ConcreteOp>
::mlir::Block*detail::OutlineableOpenMPOpInterfaceInterfaceTraits::Model<ConcreteOp>::getAllocaBlock(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return &(llvm::cast<ConcreteOp>(tablegen_opaque_val)).getRegion().front();
}
template<typename ConcreteOp>
::mlir::Block*detail::OutlineableOpenMPOpInterfaceInterfaceTraits::FallbackModel<ConcreteOp>::getAllocaBlock(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return static_cast<const ConcreteOp *>(impl)->getAllocaBlock(tablegen_opaque_val);
}
} // namespace omp
} // namespace mlir
namespace mlir {
namespace omp {
template<typename ConcreteOp>
::mlir::OperandRange detail::MapClauseOwningOpInterfaceInterfaceTraits::Model<ConcreteOp>::getMapVars(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getMapVars();
}
template<typename ConcreteOp>
::mlir::MutableOperandRange detail::MapClauseOwningOpInterfaceInterfaceTraits::Model<ConcreteOp>::getMapVarsMutable(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getMapVarsMutable();
}
template<typename ConcreteOp>
::mlir::OperandRange detail::MapClauseOwningOpInterfaceInterfaceTraits::FallbackModel<ConcreteOp>::getMapVars(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return static_cast<const ConcreteOp *>(impl)->getMapVars(tablegen_opaque_val);
}
template<typename ConcreteOp>
::mlir::MutableOperandRange detail::MapClauseOwningOpInterfaceInterfaceTraits::FallbackModel<ConcreteOp>::getMapVarsMutable(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return static_cast<const ConcreteOp *>(impl)->getMapVarsMutable(tablegen_opaque_val);
}
} // namespace omp
} // namespace mlir
namespace mlir {
namespace omp {
template<typename ConcreteOp>
::mlir::SmallVector<::mlir::Value> detail::ReductionClauseInterfaceInterfaceTraits::Model<ConcreteOp>::getAllReductionVars(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getAllReductionVars();
}
template<typename ConcreteOp>
::mlir::SmallVector<::mlir::Value> detail::ReductionClauseInterfaceInterfaceTraits::FallbackModel<ConcreteOp>::getAllReductionVars(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return static_cast<const ConcreteOp *>(impl)->getAllReductionVars(tablegen_opaque_val);
}
template<typename ConcreteModel, typename ConcreteOp>
::mlir::SmallVector<::mlir::Value> detail::ReductionClauseInterfaceInterfaceTraits::ExternalModel<ConcreteModel, ConcreteOp>::getAllReductionVars(::mlir::Operation *tablegen_opaque_val) const {
return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getReductionVars();
}
} // namespace omp
} // namespace mlir
namespace mlir {
namespace omp {
template<typename ConcreteOp>
::mlir::omp::LoopWrapperInterface detail::LoopWrapperInterfaceInterfaceTraits::Model<ConcreteOp>::getNestedWrapper(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getNestedWrapper();
}
template<typename ConcreteOp>
::mlir::Operation *detail::LoopWrapperInterfaceInterfaceTraits::Model<ConcreteOp>::getWrappedLoop(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getWrappedLoop();
}
template<typename ConcreteOp>
::mlir::omp::LoopWrapperInterface detail::LoopWrapperInterfaceInterfaceTraits::FallbackModel<ConcreteOp>::getNestedWrapper(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return static_cast<const ConcreteOp *>(impl)->getNestedWrapper(tablegen_opaque_val);
}
template<typename ConcreteOp>
::mlir::Operation *detail::LoopWrapperInterfaceInterfaceTraits::FallbackModel<ConcreteOp>::getWrappedLoop(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return static_cast<const ConcreteOp *>(impl)->getWrappedLoop(tablegen_opaque_val);
}
template<typename ConcreteModel, typename ConcreteOp>
::mlir::omp::LoopWrapperInterface detail::LoopWrapperInterfaceInterfaceTraits::ExternalModel<ConcreteModel, ConcreteOp>::getNestedWrapper(::mlir::Operation *tablegen_opaque_val) const {
Operation *nested = &*(llvm::cast<ConcreteOp>(tablegen_opaque_val))->getRegion(0).op_begin();
        return ::llvm::dyn_cast<LoopWrapperInterface>(nested);
}
template<typename ConcreteModel, typename ConcreteOp>
::mlir::Operation *detail::LoopWrapperInterfaceInterfaceTraits::ExternalModel<ConcreteModel, ConcreteOp>::getWrappedLoop(::mlir::Operation *tablegen_opaque_val) const {
if (LoopWrapperInterface nested = (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getNestedWrapper())
          return nested.getWrappedLoop();
        return &*(llvm::cast<ConcreteOp>(tablegen_opaque_val))->getRegion(0).op_begin();
}
} // namespace omp
} // namespace mlir
namespace mlir {
namespace omp {
template<typename ConcreteOp>
bool detail::ComposableOpInterfaceInterfaceTraits::Model<ConcreteOp>::isComposite(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).isComposite();
}
template<typename ConcreteOp>
void detail::ComposableOpInterfaceInterfaceTraits::Model<ConcreteOp>::setComposite(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, bool val) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).setComposite(val);
}
template<typename ConcreteOp>
bool detail::ComposableOpInterfaceInterfaceTraits::FallbackModel<ConcreteOp>::isComposite(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return static_cast<const ConcreteOp *>(impl)->isComposite(tablegen_opaque_val);
}
template<typename ConcreteOp>
void detail::ComposableOpInterfaceInterfaceTraits::FallbackModel<ConcreteOp>::setComposite(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, bool val) {
  return static_cast<const ConcreteOp *>(impl)->setComposite(tablegen_opaque_val, val);
}
template<typename ConcreteModel, typename ConcreteOp>
bool detail::ComposableOpInterfaceInterfaceTraits::ExternalModel<ConcreteModel, ConcreteOp>::isComposite(::mlir::Operation *tablegen_opaque_val) const {
return (llvm::cast<ConcreteOp>(tablegen_opaque_val))->hasAttr("omp.composite");
}
template<typename ConcreteModel, typename ConcreteOp>
void detail::ComposableOpInterfaceInterfaceTraits::ExternalModel<ConcreteModel, ConcreteOp>::setComposite(::mlir::Operation *tablegen_opaque_val, bool val) const {
if (val)
          (llvm::cast<ConcreteOp>(tablegen_opaque_val))->setDiscardableAttr("omp.composite", mlir::UnitAttr::get((llvm::cast<ConcreteOp>(tablegen_opaque_val))->getContext()));
        else
          (llvm::cast<ConcreteOp>(tablegen_opaque_val))->removeDiscardableAttr("omp.composite");
}
} // namespace omp
} // namespace mlir
namespace mlir {
namespace omp {
template<typename ConcreteOp>
void detail::DeclareTargetInterfaceInterfaceTraits::Model<ConcreteOp>::setDeclareTarget(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, mlir::omp::DeclareTargetDeviceType deviceType, mlir::omp::DeclareTargetCaptureClause captureClause) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).setDeclareTarget(deviceType, captureClause);
}
template<typename ConcreteOp>
bool detail::DeclareTargetInterfaceInterfaceTraits::Model<ConcreteOp>::isDeclareTarget(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).isDeclareTarget();
}
template<typename ConcreteOp>
mlir::omp::DeclareTargetDeviceType detail::DeclareTargetInterfaceInterfaceTraits::Model<ConcreteOp>::getDeclareTargetDeviceType(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getDeclareTargetDeviceType();
}
template<typename ConcreteOp>
mlir::omp::DeclareTargetCaptureClause detail::DeclareTargetInterfaceInterfaceTraits::Model<ConcreteOp>::getDeclareTargetCaptureClause(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getDeclareTargetCaptureClause();
}
template<typename ConcreteOp>
void detail::DeclareTargetInterfaceInterfaceTraits::FallbackModel<ConcreteOp>::setDeclareTarget(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, mlir::omp::DeclareTargetDeviceType deviceType, mlir::omp::DeclareTargetCaptureClause captureClause) {
  return static_cast<const ConcreteOp *>(impl)->setDeclareTarget(tablegen_opaque_val, deviceType, captureClause);
}
template<typename ConcreteOp>
bool detail::DeclareTargetInterfaceInterfaceTraits::FallbackModel<ConcreteOp>::isDeclareTarget(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return static_cast<const ConcreteOp *>(impl)->isDeclareTarget(tablegen_opaque_val);
}
template<typename ConcreteOp>
mlir::omp::DeclareTargetDeviceType detail::DeclareTargetInterfaceInterfaceTraits::FallbackModel<ConcreteOp>::getDeclareTargetDeviceType(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return static_cast<const ConcreteOp *>(impl)->getDeclareTargetDeviceType(tablegen_opaque_val);
}
template<typename ConcreteOp>
mlir::omp::DeclareTargetCaptureClause detail::DeclareTargetInterfaceInterfaceTraits::FallbackModel<ConcreteOp>::getDeclareTargetCaptureClause(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return static_cast<const ConcreteOp *>(impl)->getDeclareTargetCaptureClause(tablegen_opaque_val);
}
template<typename ConcreteModel, typename ConcreteOp>
void detail::DeclareTargetInterfaceInterfaceTraits::ExternalModel<ConcreteModel, ConcreteOp>::setDeclareTarget(::mlir::Operation *tablegen_opaque_val, mlir::omp::DeclareTargetDeviceType deviceType, mlir::omp::DeclareTargetCaptureClause captureClause) const {
(llvm::cast<ConcreteOp>(tablegen_opaque_val))->setAttr("omp.declare_target",
                  mlir::omp::DeclareTargetAttr::get(
                      (llvm::cast<ConcreteOp>(tablegen_opaque_val))->getContext(),
                      mlir::omp::DeclareTargetDeviceTypeAttr::get(
                          (llvm::cast<ConcreteOp>(tablegen_opaque_val))->getContext(), deviceType),
                      mlir::omp::DeclareTargetCaptureClauseAttr::get(
                          (llvm::cast<ConcreteOp>(tablegen_opaque_val))->getContext(), captureClause)));
}
template<typename ConcreteModel, typename ConcreteOp>
bool detail::DeclareTargetInterfaceInterfaceTraits::ExternalModel<ConcreteModel, ConcreteOp>::isDeclareTarget(::mlir::Operation *tablegen_opaque_val) const {
return (llvm::cast<ConcreteOp>(tablegen_opaque_val))->hasAttr("omp.declare_target");
}
template<typename ConcreteModel, typename ConcreteOp>
mlir::omp::DeclareTargetDeviceType detail::DeclareTargetInterfaceInterfaceTraits::ExternalModel<ConcreteModel, ConcreteOp>::getDeclareTargetDeviceType(::mlir::Operation *tablegen_opaque_val) const {
if (mlir::Attribute dTar = (llvm::cast<ConcreteOp>(tablegen_opaque_val))->getAttr("omp.declare_target"))
          if (auto dAttr = llvm::dyn_cast_or_null<mlir::omp::DeclareTargetAttr>(dTar))
            return dAttr.getDeviceType().getValue();
        return {};
}
template<typename ConcreteModel, typename ConcreteOp>
mlir::omp::DeclareTargetCaptureClause detail::DeclareTargetInterfaceInterfaceTraits::ExternalModel<ConcreteModel, ConcreteOp>::getDeclareTargetCaptureClause(::mlir::Operation *tablegen_opaque_val) const {
if (mlir::Attribute dTar = (llvm::cast<ConcreteOp>(tablegen_opaque_val))->getAttr("omp.declare_target"))
          if (auto dAttr = llvm::dyn_cast_or_null<mlir::omp::DeclareTargetAttr>(dTar))
            return dAttr.getCaptureClause().getValue();
        return {};
}
} // namespace omp
} // namespace mlir
namespace mlir {
namespace omp {
template<typename ConcreteOp>
void detail::OffloadModuleInterfaceInterfaceTraits::Model<ConcreteOp>::setIsTargetDevice(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, bool isTargetDevice) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).setIsTargetDevice(isTargetDevice);
}
template<typename ConcreteOp>
bool detail::OffloadModuleInterfaceInterfaceTraits::Model<ConcreteOp>::getIsTargetDevice(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getIsTargetDevice();
}
template<typename ConcreteOp>
void detail::OffloadModuleInterfaceInterfaceTraits::Model<ConcreteOp>::setIsGPU(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, bool isGPU) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).setIsGPU(isGPU);
}
template<typename ConcreteOp>
bool detail::OffloadModuleInterfaceInterfaceTraits::Model<ConcreteOp>::getIsGPU(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getIsGPU();
}
template<typename ConcreteOp>
mlir::omp::FlagsAttr detail::OffloadModuleInterfaceInterfaceTraits::Model<ConcreteOp>::getFlags(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getFlags();
}
template<typename ConcreteOp>
void detail::OffloadModuleInterfaceInterfaceTraits::Model<ConcreteOp>::setFlags(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, uint32_t debugKind, bool assumeTeamsOversubscription, bool assumeThreadsOversubscription, bool assumeNoThreadState, bool assumeNoNestedParallelism, uint32_t openmpDeviceVersion, bool noGPULib) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).setFlags(debugKind, assumeTeamsOversubscription, assumeThreadsOversubscription, assumeNoThreadState, assumeNoNestedParallelism, openmpDeviceVersion, noGPULib);
}
template<typename ConcreteOp>
void detail::OffloadModuleInterfaceInterfaceTraits::Model<ConcreteOp>::setHostIRFilePath(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, std::string hostIRFilePath) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).setHostIRFilePath(hostIRFilePath);
}
template<typename ConcreteOp>
llvm::StringRef detail::OffloadModuleInterfaceInterfaceTraits::Model<ConcreteOp>::getHostIRFilePath(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getHostIRFilePath();
}
template<typename ConcreteOp>
::mlir::omp::ClauseRequires detail::OffloadModuleInterfaceInterfaceTraits::Model<ConcreteOp>::getRequires(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getRequires();
}
template<typename ConcreteOp>
void detail::OffloadModuleInterfaceInterfaceTraits::Model<ConcreteOp>::setRequires(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, ::mlir::omp::ClauseRequires clauses) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).setRequires(clauses);
}
template<typename ConcreteOp>
::llvm::ArrayRef<::mlir::Attribute> detail::OffloadModuleInterfaceInterfaceTraits::Model<ConcreteOp>::getTargetTriples(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getTargetTriples();
}
template<typename ConcreteOp>
void detail::OffloadModuleInterfaceInterfaceTraits::Model<ConcreteOp>::setTargetTriples(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, ::llvm::ArrayRef<::std::string> targetTriples) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).setTargetTriples(targetTriples);
}
template<typename ConcreteOp>
void detail::OffloadModuleInterfaceInterfaceTraits::FallbackModel<ConcreteOp>::setIsTargetDevice(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, bool isTargetDevice) {
  return static_cast<const ConcreteOp *>(impl)->setIsTargetDevice(tablegen_opaque_val, isTargetDevice);
}
template<typename ConcreteOp>
bool detail::OffloadModuleInterfaceInterfaceTraits::FallbackModel<ConcreteOp>::getIsTargetDevice(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return static_cast<const ConcreteOp *>(impl)->getIsTargetDevice(tablegen_opaque_val);
}
template<typename ConcreteOp>
void detail::OffloadModuleInterfaceInterfaceTraits::FallbackModel<ConcreteOp>::setIsGPU(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, bool isGPU) {
  return static_cast<const ConcreteOp *>(impl)->setIsGPU(tablegen_opaque_val, isGPU);
}
template<typename ConcreteOp>
bool detail::OffloadModuleInterfaceInterfaceTraits::FallbackModel<ConcreteOp>::getIsGPU(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return static_cast<const ConcreteOp *>(impl)->getIsGPU(tablegen_opaque_val);
}
template<typename ConcreteOp>
mlir::omp::FlagsAttr detail::OffloadModuleInterfaceInterfaceTraits::FallbackModel<ConcreteOp>::getFlags(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return static_cast<const ConcreteOp *>(impl)->getFlags(tablegen_opaque_val);
}
template<typename ConcreteOp>
void detail::OffloadModuleInterfaceInterfaceTraits::FallbackModel<ConcreteOp>::setFlags(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, uint32_t debugKind, bool assumeTeamsOversubscription, bool assumeThreadsOversubscription, bool assumeNoThreadState, bool assumeNoNestedParallelism, uint32_t openmpDeviceVersion, bool noGPULib) {
  return static_cast<const ConcreteOp *>(impl)->setFlags(tablegen_opaque_val, debugKind, assumeTeamsOversubscription, assumeThreadsOversubscription, assumeNoThreadState, assumeNoNestedParallelism, openmpDeviceVersion, noGPULib);
}
template<typename ConcreteOp>
void detail::OffloadModuleInterfaceInterfaceTraits::FallbackModel<ConcreteOp>::setHostIRFilePath(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, std::string hostIRFilePath) {
  return static_cast<const ConcreteOp *>(impl)->setHostIRFilePath(tablegen_opaque_val, hostIRFilePath);
}
template<typename ConcreteOp>
llvm::StringRef detail::OffloadModuleInterfaceInterfaceTraits::FallbackModel<ConcreteOp>::getHostIRFilePath(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return static_cast<const ConcreteOp *>(impl)->getHostIRFilePath(tablegen_opaque_val);
}
template<typename ConcreteOp>
::mlir::omp::ClauseRequires detail::OffloadModuleInterfaceInterfaceTraits::FallbackModel<ConcreteOp>::getRequires(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return static_cast<const ConcreteOp *>(impl)->getRequires(tablegen_opaque_val);
}
template<typename ConcreteOp>
void detail::OffloadModuleInterfaceInterfaceTraits::FallbackModel<ConcreteOp>::setRequires(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, ::mlir::omp::ClauseRequires clauses) {
  return static_cast<const ConcreteOp *>(impl)->setRequires(tablegen_opaque_val, clauses);
}
template<typename ConcreteOp>
::llvm::ArrayRef<::mlir::Attribute> detail::OffloadModuleInterfaceInterfaceTraits::FallbackModel<ConcreteOp>::getTargetTriples(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return static_cast<const ConcreteOp *>(impl)->getTargetTriples(tablegen_opaque_val);
}
template<typename ConcreteOp>
void detail::OffloadModuleInterfaceInterfaceTraits::FallbackModel<ConcreteOp>::setTargetTriples(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, ::llvm::ArrayRef<::std::string> targetTriples) {
  return static_cast<const ConcreteOp *>(impl)->setTargetTriples(tablegen_opaque_val, targetTriples);
}
template<typename ConcreteModel, typename ConcreteOp>
void detail::OffloadModuleInterfaceInterfaceTraits::ExternalModel<ConcreteModel, ConcreteOp>::setIsTargetDevice(::mlir::Operation *tablegen_opaque_val, bool isTargetDevice) const {
(llvm::cast<ConcreteOp>(tablegen_opaque_val))->setAttr(
          mlir::StringAttr::get((llvm::cast<ConcreteOp>(tablegen_opaque_val))->getContext(), llvm::Twine{"omp.is_target_device"}),
            mlir::BoolAttr::get((llvm::cast<ConcreteOp>(tablegen_opaque_val))->getContext(), isTargetDevice));
}
template<typename ConcreteModel, typename ConcreteOp>
bool detail::OffloadModuleInterfaceInterfaceTraits::ExternalModel<ConcreteModel, ConcreteOp>::getIsTargetDevice(::mlir::Operation *tablegen_opaque_val) const {
if (Attribute isTargetDevice = (llvm::cast<ConcreteOp>(tablegen_opaque_val))->getAttr("omp.is_target_device"))
          if (::llvm::isa<mlir::BoolAttr>(isTargetDevice))
           return ::llvm::dyn_cast<BoolAttr>(isTargetDevice).getValue();
        return false;
}
template<typename ConcreteModel, typename ConcreteOp>
void detail::OffloadModuleInterfaceInterfaceTraits::ExternalModel<ConcreteModel, ConcreteOp>::setIsGPU(::mlir::Operation *tablegen_opaque_val, bool isGPU) const {
(llvm::cast<ConcreteOp>(tablegen_opaque_val))->setAttr(
          mlir::StringAttr::get((llvm::cast<ConcreteOp>(tablegen_opaque_val))->getContext(), "omp.is_gpu"),
            mlir::BoolAttr::get((llvm::cast<ConcreteOp>(tablegen_opaque_val))->getContext(), isGPU));
}
template<typename ConcreteModel, typename ConcreteOp>
bool detail::OffloadModuleInterfaceInterfaceTraits::ExternalModel<ConcreteModel, ConcreteOp>::getIsGPU(::mlir::Operation *tablegen_opaque_val) const {
if (Attribute isTargetCGAttr = (llvm::cast<ConcreteOp>(tablegen_opaque_val))->getAttr("omp.is_gpu"))
          if (auto isTargetCGVal = ::llvm::dyn_cast<BoolAttr>(isTargetCGAttr))
           return isTargetCGVal.getValue();
        return false;
}
template<typename ConcreteModel, typename ConcreteOp>
mlir::omp::FlagsAttr detail::OffloadModuleInterfaceInterfaceTraits::ExternalModel<ConcreteModel, ConcreteOp>::getFlags(::mlir::Operation *tablegen_opaque_val) const {
if (Attribute flags = (llvm::cast<ConcreteOp>(tablegen_opaque_val))->getAttr("omp.flags"))
          return ::llvm::dyn_cast_or_null<mlir::omp::FlagsAttr>(flags);
        return nullptr;
}
template<typename ConcreteModel, typename ConcreteOp>
void detail::OffloadModuleInterfaceInterfaceTraits::ExternalModel<ConcreteModel, ConcreteOp>::setFlags(::mlir::Operation *tablegen_opaque_val, uint32_t debugKind, bool assumeTeamsOversubscription, bool assumeThreadsOversubscription, bool assumeNoThreadState, bool assumeNoNestedParallelism, uint32_t openmpDeviceVersion, bool noGPULib) const {
(llvm::cast<ConcreteOp>(tablegen_opaque_val))->setAttr(("omp." + mlir::omp::FlagsAttr::getMnemonic()).str(),
                  mlir::omp::FlagsAttr::get((llvm::cast<ConcreteOp>(tablegen_opaque_val))->getContext(), debugKind,
                      assumeTeamsOversubscription, assumeThreadsOversubscription,
                      assumeNoThreadState, assumeNoNestedParallelism, noGPULib, openmpDeviceVersion));
}
template<typename ConcreteModel, typename ConcreteOp>
void detail::OffloadModuleInterfaceInterfaceTraits::ExternalModel<ConcreteModel, ConcreteOp>::setHostIRFilePath(::mlir::Operation *tablegen_opaque_val, std::string hostIRFilePath) const {
(llvm::cast<ConcreteOp>(tablegen_opaque_val))->setAttr(
          mlir::StringAttr::get((llvm::cast<ConcreteOp>(tablegen_opaque_val))->getContext(), llvm::Twine{"omp.host_ir_filepath"}),
            mlir::StringAttr::get((llvm::cast<ConcreteOp>(tablegen_opaque_val))->getContext(), hostIRFilePath));
}
template<typename ConcreteModel, typename ConcreteOp>
llvm::StringRef detail::OffloadModuleInterfaceInterfaceTraits::ExternalModel<ConcreteModel, ConcreteOp>::getHostIRFilePath(::mlir::Operation *tablegen_opaque_val) const {
if (Attribute filepath = (llvm::cast<ConcreteOp>(tablegen_opaque_val))->getAttr("omp.host_ir_filepath"))
          if (::llvm::isa<mlir::StringAttr>(filepath))
            return ::llvm::dyn_cast<mlir::StringAttr>(filepath).getValue();
        return {};
}
template<typename ConcreteModel, typename ConcreteOp>
::mlir::omp::ClauseRequires detail::OffloadModuleInterfaceInterfaceTraits::ExternalModel<ConcreteModel, ConcreteOp>::getRequires(::mlir::Operation *tablegen_opaque_val) const {
if (Attribute requiresAttr = (llvm::cast<ConcreteOp>(tablegen_opaque_val))->getAttr("omp.requires"))
          if (auto requiresVal = ::llvm::dyn_cast<mlir::omp::ClauseRequiresAttr>(requiresAttr))
            return requiresVal.getValue();
        return mlir::omp::ClauseRequires::none;
}
template<typename ConcreteModel, typename ConcreteOp>
void detail::OffloadModuleInterfaceInterfaceTraits::ExternalModel<ConcreteModel, ConcreteOp>::setRequires(::mlir::Operation *tablegen_opaque_val, ::mlir::omp::ClauseRequires clauses) const {
(llvm::cast<ConcreteOp>(tablegen_opaque_val))->setAttr(mlir::StringAttr::get((llvm::cast<ConcreteOp>(tablegen_opaque_val))->getContext(), "omp.requires"),
          mlir::omp::ClauseRequiresAttr::get((llvm::cast<ConcreteOp>(tablegen_opaque_val))->getContext(), clauses));
}
template<typename ConcreteModel, typename ConcreteOp>
::llvm::ArrayRef<::mlir::Attribute> detail::OffloadModuleInterfaceInterfaceTraits::ExternalModel<ConcreteModel, ConcreteOp>::getTargetTriples(::mlir::Operation *tablegen_opaque_val) const {
if (Attribute triplesAttr = (llvm::cast<ConcreteOp>(tablegen_opaque_val))->getAttr("omp.target_triples"))
          if (auto triples = ::llvm::dyn_cast<::mlir::ArrayAttr>(triplesAttr))
            return triples.getValue();
        return {};
}
template<typename ConcreteModel, typename ConcreteOp>
void detail::OffloadModuleInterfaceInterfaceTraits::ExternalModel<ConcreteModel, ConcreteOp>::setTargetTriples(::mlir::Operation *tablegen_opaque_val, ::llvm::ArrayRef<::std::string> targetTriples) const {
auto names = ::llvm::to_vector(::llvm::map_range(
            targetTriples, [&](::std::string str) -> ::mlir::Attribute {
              return mlir::StringAttr::get((llvm::cast<ConcreteOp>(tablegen_opaque_val))->getContext(), str);
            }));
        (llvm::cast<ConcreteOp>(tablegen_opaque_val))->setAttr(
            ::mlir::StringAttr::get((llvm::cast<ConcreteOp>(tablegen_opaque_val))->getContext(), "omp.target_triples"),
            ::mlir::ArrayAttr::get((llvm::cast<ConcreteOp>(tablegen_opaque_val))->getContext(), names));
}
} // namespace omp
} // namespace mlir
