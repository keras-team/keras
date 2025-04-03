/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Interface Declarations                                                     *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

namespace mlir {
class DLTIQueryInterface;
namespace detail {
struct DLTIQueryInterfaceInterfaceTraits {
  struct Concept {
    /// The methods defined by the interface.
    ::mlir::FailureOr<::mlir::Attribute> (*query)(const Concept *impl, ::mlir::Attribute , ::mlir::DataLayoutEntryKey);
  };
  template<typename ConcreteAttr>
  class Model : public Concept {
  public:
    using Interface = ::mlir::DLTIQueryInterface;
    Model() : Concept{query} {}

    static inline ::mlir::FailureOr<::mlir::Attribute> query(const Concept *impl, ::mlir::Attribute tablegen_opaque_val, ::mlir::DataLayoutEntryKey key);
  };
  template<typename ConcreteAttr>
  class FallbackModel : public Concept {
  public:
    using Interface = ::mlir::DLTIQueryInterface;
    FallbackModel() : Concept{query} {}

    static inline ::mlir::FailureOr<::mlir::Attribute> query(const Concept *impl, ::mlir::Attribute tablegen_opaque_val, ::mlir::DataLayoutEntryKey key);
  };
  template<typename ConcreteModel, typename ConcreteAttr>
  class ExternalModel : public FallbackModel<ConcreteModel> {
  public:
    using ConcreteEntity = ConcreteAttr;
  };
};
template <typename ConcreteAttr>
struct DLTIQueryInterfaceTrait;

} // namespace detail
class DLTIQueryInterface : public ::mlir::AttributeInterface<DLTIQueryInterface, detail::DLTIQueryInterfaceInterfaceTraits> {
public:
  using ::mlir::AttributeInterface<DLTIQueryInterface, detail::DLTIQueryInterfaceInterfaceTraits>::AttributeInterface;
  template <typename ConcreteAttr>
  struct Trait : public detail::DLTIQueryInterfaceTrait<ConcreteAttr> {};
  /// Returns the attribute associated with the key.
  ::mlir::FailureOr<::mlir::Attribute> query(::mlir::DataLayoutEntryKey key) const;
};
namespace detail {
  template <typename ConcreteAttr>
  struct DLTIQueryInterfaceTrait : public ::mlir::AttributeInterface<DLTIQueryInterface, detail::DLTIQueryInterfaceInterfaceTraits>::Trait<ConcreteAttr> {
  };
}// namespace detail
} // namespace mlir
namespace mlir {
class DataLayoutEntryInterface;
namespace detail {
struct DataLayoutEntryInterfaceInterfaceTraits {
  struct Concept {
    /// The methods defined by the interface.
    ::mlir::DataLayoutEntryKey (*getKey)(const Concept *impl, ::mlir::Attribute );
    ::mlir::Attribute (*getValue)(const Concept *impl, ::mlir::Attribute );
    ::llvm::LogicalResult (*verifyEntry)(const Concept *impl, ::mlir::Attribute , ::mlir::Location);
  };
  template<typename ConcreteAttr>
  class Model : public Concept {
  public:
    using Interface = ::mlir::DataLayoutEntryInterface;
    Model() : Concept{getKey, getValue, verifyEntry} {}

    static inline ::mlir::DataLayoutEntryKey getKey(const Concept *impl, ::mlir::Attribute tablegen_opaque_val);
    static inline ::mlir::Attribute getValue(const Concept *impl, ::mlir::Attribute tablegen_opaque_val);
    static inline ::llvm::LogicalResult verifyEntry(const Concept *impl, ::mlir::Attribute tablegen_opaque_val, ::mlir::Location loc);
  };
  template<typename ConcreteAttr>
  class FallbackModel : public Concept {
  public:
    using Interface = ::mlir::DataLayoutEntryInterface;
    FallbackModel() : Concept{getKey, getValue, verifyEntry} {}

    static inline ::mlir::DataLayoutEntryKey getKey(const Concept *impl, ::mlir::Attribute tablegen_opaque_val);
    static inline ::mlir::Attribute getValue(const Concept *impl, ::mlir::Attribute tablegen_opaque_val);
    static inline ::llvm::LogicalResult verifyEntry(const Concept *impl, ::mlir::Attribute tablegen_opaque_val, ::mlir::Location loc);
  };
  template<typename ConcreteModel, typename ConcreteAttr>
  class ExternalModel : public FallbackModel<ConcreteModel> {
  public:
    using ConcreteEntity = ConcreteAttr;
    ::llvm::LogicalResult verifyEntry(::mlir::Attribute tablegen_opaque_val, ::mlir::Location loc) const;
  };
};
template <typename ConcreteAttr>
struct DataLayoutEntryInterfaceTrait;

} // namespace detail
class DataLayoutEntryInterface : public ::mlir::AttributeInterface<DataLayoutEntryInterface, detail::DataLayoutEntryInterfaceInterfaceTraits> {
public:
  using ::mlir::AttributeInterface<DataLayoutEntryInterface, detail::DataLayoutEntryInterfaceInterfaceTraits>::AttributeInterface;
  template <typename ConcreteAttr>
  struct Trait : public detail::DataLayoutEntryInterfaceTrait<ConcreteAttr> {};
  /// Returns the key of the this layout entry.
  ::mlir::DataLayoutEntryKey getKey() const;
  /// Returns the value of this layout entry.
  ::mlir::Attribute getValue() const;
  /// Checks that the entry is well-formed, reports errors at the provided location.
  ::llvm::LogicalResult verifyEntry(::mlir::Location loc) const;

    /// Returns `true` if the key of this entry is a type.
    bool isTypeEntry() {
      return getKey().is<::mlir::Type>();
    }
};
namespace detail {
  template <typename ConcreteAttr>
  struct DataLayoutEntryInterfaceTrait : public ::mlir::AttributeInterface<DataLayoutEntryInterface, detail::DataLayoutEntryInterfaceInterfaceTraits>::Trait<ConcreteAttr> {
    /// Checks that the entry is well-formed, reports errors at the provided location.
    ::llvm::LogicalResult verifyEntry(::mlir::Location loc) const {
      return ::mlir::success();
    }
  };
}// namespace detail
} // namespace mlir
namespace mlir {
class DataLayoutSpecInterface;
namespace detail {
struct DataLayoutSpecInterfaceInterfaceTraits {
  struct Concept {
    /// The methods defined by the interface.
    ::mlir::DataLayoutSpecInterface (*combineWith)(const Concept *impl, ::mlir::Attribute , ::llvm::ArrayRef<::mlir::DataLayoutSpecInterface>);
    ::mlir::DataLayoutEntryListRef (*getEntries)(const Concept *impl, ::mlir::Attribute );
    ::mlir::StringAttr (*getEndiannessIdentifier)(const Concept *impl, ::mlir::Attribute , ::mlir::MLIRContext *);
    ::mlir::StringAttr (*getAllocaMemorySpaceIdentifier)(const Concept *impl, ::mlir::Attribute , ::mlir::MLIRContext *);
    ::mlir::StringAttr (*getProgramMemorySpaceIdentifier)(const Concept *impl, ::mlir::Attribute , ::mlir::MLIRContext *);
    ::mlir::StringAttr (*getGlobalMemorySpaceIdentifier)(const Concept *impl, ::mlir::Attribute , ::mlir::MLIRContext *);
    ::mlir::StringAttr (*getStackAlignmentIdentifier)(const Concept *impl, ::mlir::Attribute , ::mlir::MLIRContext *);
    ::mlir::DataLayoutEntryList (*getSpecForType)(const Concept *impl, ::mlir::Attribute , ::mlir::TypeID);
    ::mlir::DataLayoutEntryInterface (*getSpecForIdentifier)(const Concept *impl, ::mlir::Attribute , ::mlir::StringAttr);
    ::llvm::LogicalResult (*verifySpec)(const Concept *impl, ::mlir::Attribute , ::mlir::Location);
    /// The base classes of this interface.
    const ::mlir::DLTIQueryInterface::Concept *implDLTIQueryInterface = nullptr;

    void initializeInterfaceConcept(::mlir::detail::InterfaceMap &interfaceMap) {
      implDLTIQueryInterface = interfaceMap.lookup<::mlir::DLTIQueryInterface>();
      assert(implDLTIQueryInterface && "`::mlir::DataLayoutSpecInterface` expected its base interface `::mlir::DLTIQueryInterface` to be registered");
    }
  };
  template<typename ConcreteAttr>
  class Model : public Concept {
  public:
    using Interface = ::mlir::DataLayoutSpecInterface;
    Model() : Concept{combineWith, getEntries, getEndiannessIdentifier, getAllocaMemorySpaceIdentifier, getProgramMemorySpaceIdentifier, getGlobalMemorySpaceIdentifier, getStackAlignmentIdentifier, getSpecForType, getSpecForIdentifier, verifySpec} {}

    static inline ::mlir::DataLayoutSpecInterface combineWith(const Concept *impl, ::mlir::Attribute tablegen_opaque_val, ::llvm::ArrayRef<::mlir::DataLayoutSpecInterface> specs);
    static inline ::mlir::DataLayoutEntryListRef getEntries(const Concept *impl, ::mlir::Attribute tablegen_opaque_val);
    static inline ::mlir::StringAttr getEndiannessIdentifier(const Concept *impl, ::mlir::Attribute tablegen_opaque_val, ::mlir::MLIRContext * context);
    static inline ::mlir::StringAttr getAllocaMemorySpaceIdentifier(const Concept *impl, ::mlir::Attribute tablegen_opaque_val, ::mlir::MLIRContext * context);
    static inline ::mlir::StringAttr getProgramMemorySpaceIdentifier(const Concept *impl, ::mlir::Attribute tablegen_opaque_val, ::mlir::MLIRContext * context);
    static inline ::mlir::StringAttr getGlobalMemorySpaceIdentifier(const Concept *impl, ::mlir::Attribute tablegen_opaque_val, ::mlir::MLIRContext * context);
    static inline ::mlir::StringAttr getStackAlignmentIdentifier(const Concept *impl, ::mlir::Attribute tablegen_opaque_val, ::mlir::MLIRContext * context);
    static inline ::mlir::DataLayoutEntryList getSpecForType(const Concept *impl, ::mlir::Attribute tablegen_opaque_val, ::mlir::TypeID type);
    static inline ::mlir::DataLayoutEntryInterface getSpecForIdentifier(const Concept *impl, ::mlir::Attribute tablegen_opaque_val, ::mlir::StringAttr identifier);
    static inline ::llvm::LogicalResult verifySpec(const Concept *impl, ::mlir::Attribute tablegen_opaque_val, ::mlir::Location loc);
  };
  template<typename ConcreteAttr>
  class FallbackModel : public Concept {
  public:
    using Interface = ::mlir::DataLayoutSpecInterface;
    FallbackModel() : Concept{combineWith, getEntries, getEndiannessIdentifier, getAllocaMemorySpaceIdentifier, getProgramMemorySpaceIdentifier, getGlobalMemorySpaceIdentifier, getStackAlignmentIdentifier, getSpecForType, getSpecForIdentifier, verifySpec} {}

    static inline ::mlir::DataLayoutSpecInterface combineWith(const Concept *impl, ::mlir::Attribute tablegen_opaque_val, ::llvm::ArrayRef<::mlir::DataLayoutSpecInterface> specs);
    static inline ::mlir::DataLayoutEntryListRef getEntries(const Concept *impl, ::mlir::Attribute tablegen_opaque_val);
    static inline ::mlir::StringAttr getEndiannessIdentifier(const Concept *impl, ::mlir::Attribute tablegen_opaque_val, ::mlir::MLIRContext * context);
    static inline ::mlir::StringAttr getAllocaMemorySpaceIdentifier(const Concept *impl, ::mlir::Attribute tablegen_opaque_val, ::mlir::MLIRContext * context);
    static inline ::mlir::StringAttr getProgramMemorySpaceIdentifier(const Concept *impl, ::mlir::Attribute tablegen_opaque_val, ::mlir::MLIRContext * context);
    static inline ::mlir::StringAttr getGlobalMemorySpaceIdentifier(const Concept *impl, ::mlir::Attribute tablegen_opaque_val, ::mlir::MLIRContext * context);
    static inline ::mlir::StringAttr getStackAlignmentIdentifier(const Concept *impl, ::mlir::Attribute tablegen_opaque_val, ::mlir::MLIRContext * context);
    static inline ::mlir::DataLayoutEntryList getSpecForType(const Concept *impl, ::mlir::Attribute tablegen_opaque_val, ::mlir::TypeID type);
    static inline ::mlir::DataLayoutEntryInterface getSpecForIdentifier(const Concept *impl, ::mlir::Attribute tablegen_opaque_val, ::mlir::StringAttr identifier);
    static inline ::llvm::LogicalResult verifySpec(const Concept *impl, ::mlir::Attribute tablegen_opaque_val, ::mlir::Location loc);
  };
  template<typename ConcreteModel, typename ConcreteAttr>
  class ExternalModel : public FallbackModel<ConcreteModel> {
  public:
    using ConcreteEntity = ConcreteAttr;
    ::mlir::DataLayoutEntryList getSpecForType(::mlir::Attribute tablegen_opaque_val, ::mlir::TypeID type) const;
    ::mlir::DataLayoutEntryInterface getSpecForIdentifier(::mlir::Attribute tablegen_opaque_val, ::mlir::StringAttr identifier) const;
    ::llvm::LogicalResult verifySpec(::mlir::Attribute tablegen_opaque_val, ::mlir::Location loc) const;
  };
};
template <typename ConcreteAttr>
struct DataLayoutSpecInterfaceTrait;

} // namespace detail
class DataLayoutSpecInterface : public ::mlir::AttributeInterface<DataLayoutSpecInterface, detail::DataLayoutSpecInterfaceInterfaceTraits> {
public:
  using ::mlir::AttributeInterface<DataLayoutSpecInterface, detail::DataLayoutSpecInterfaceInterfaceTraits>::AttributeInterface;
  template <typename ConcreteAttr>
  struct Trait : public detail::DataLayoutSpecInterfaceTrait<ConcreteAttr> {};
  /// Combines the current layout with the given list of layouts, provided from the outermost (oldest) to the innermost (newest). Returns null on failure.
  ::mlir::DataLayoutSpecInterface combineWith(::llvm::ArrayRef<::mlir::DataLayoutSpecInterface> specs) const;
  /// Returns the list of layout entries.
  ::mlir::DataLayoutEntryListRef getEntries() const;
  /// Returns the endianness identifier.
  ::mlir::StringAttr getEndiannessIdentifier(::mlir::MLIRContext * context) const;
  /// Returns the alloca memory space identifier.
  ::mlir::StringAttr getAllocaMemorySpaceIdentifier(::mlir::MLIRContext * context) const;
  /// Returns the program memory space identifier.
  ::mlir::StringAttr getProgramMemorySpaceIdentifier(::mlir::MLIRContext * context) const;
  /// Returns the global memory space identifier.
  ::mlir::StringAttr getGlobalMemorySpaceIdentifier(::mlir::MLIRContext * context) const;
  /// Returns the stack alignment identifier.
  ::mlir::StringAttr getStackAlignmentIdentifier(::mlir::MLIRContext * context) const;
  /// Returns a copy of the entries related to a specific type class regardles of type parameters.
  ::mlir::DataLayoutEntryList getSpecForType(::mlir::TypeID type) const;
  /// Returns the entry related to the given identifier, if present.
  ::mlir::DataLayoutEntryInterface getSpecForIdentifier(::mlir::StringAttr identifier) const;
  /// Verifies the validity of the specification and reports any errors at the given location.
  ::llvm::LogicalResult verifySpec(::mlir::Location loc) const;

    /// Returns a copy of the entries related to the type specified as template
    /// parameter.
    template <typename Ty>
    DataLayoutEntryList getSpecForType() {
      return getSpecForType(TypeID::get<Ty>());
    }

    /// Helper for default implementation of `DLTIQueryInterface`'s `query`.
    inline ::mlir::FailureOr<::mlir::Attribute>
    queryHelper(::mlir::DataLayoutEntryKey key) const {
      for (DataLayoutEntryInterface entry : getEntries())
        if (entry.getKey() == key)
            return entry.getValue();
      return ::mlir::failure();
    }

    /// Populates the given maps with lists of entries grouped by the type or
    /// identifier they are associated with. Users are not expected to call this
    /// method directly.
    void bucketEntriesByType(
        ::llvm::DenseMap<::mlir::TypeID, ::mlir::DataLayoutEntryList> &types,
        ::llvm::DenseMap<::mlir::StringAttr,
                         ::mlir::DataLayoutEntryInterface> &ids);
  //===----------------------------------------------------------------===//
  // Inherited from ::mlir::DLTIQueryInterface
  //===----------------------------------------------------------------===//

  operator ::mlir::DLTIQueryInterface () const {
    if (!*this) return nullptr;
    return ::mlir::DLTIQueryInterface(*this, getImpl()->implDLTIQueryInterface);
  }

  /// Returns the attribute associated with the key.
  ::mlir::FailureOr<::mlir::Attribute> query(::mlir::DataLayoutEntryKey key) const;
};
namespace detail {
  template <typename ConcreteAttr>
  struct DataLayoutSpecInterfaceTrait : public ::mlir::AttributeInterface<DataLayoutSpecInterface, detail::DataLayoutSpecInterfaceInterfaceTraits>::Trait<ConcreteAttr> {
    /// Returns a copy of the entries related to a specific type class regardles of type parameters.
    ::mlir::DataLayoutEntryList getSpecForType(::mlir::TypeID type) const {
      return ::mlir::detail::filterEntriesForType((*static_cast<const ConcreteAttr *>(this)).getEntries(), type);
    }
    /// Returns the entry related to the given identifier, if present.
    ::mlir::DataLayoutEntryInterface getSpecForIdentifier(::mlir::StringAttr identifier) const {
      return ::mlir::detail::filterEntryForIdentifier((*static_cast<const ConcreteAttr *>(this)).getEntries(),
                                                        identifier);
    }
    /// Verifies the validity of the specification and reports any errors at the given location.
    ::llvm::LogicalResult verifySpec(::mlir::Location loc) const {
      return ::mlir::detail::verifyDataLayoutSpec((*static_cast<const ConcreteAttr *>(this)), loc);
    }
  };
}// namespace detail
} // namespace mlir
namespace mlir {
class TargetDeviceSpecInterface;
namespace detail {
struct TargetDeviceSpecInterfaceInterfaceTraits {
  struct Concept {
    /// The methods defined by the interface.
    ::mlir::DataLayoutEntryListRef (*getEntries)(const Concept *impl, ::mlir::Attribute );
    ::mlir::DataLayoutEntryInterface (*getSpecForIdentifier)(const Concept *impl, ::mlir::Attribute , ::mlir::StringAttr);
    ::llvm::LogicalResult (*verifyEntry)(const Concept *impl, ::mlir::Attribute , ::mlir::Location);
    /// The base classes of this interface.
    const ::mlir::DLTIQueryInterface::Concept *implDLTIQueryInterface = nullptr;

    void initializeInterfaceConcept(::mlir::detail::InterfaceMap &interfaceMap) {
      implDLTIQueryInterface = interfaceMap.lookup<::mlir::DLTIQueryInterface>();
      assert(implDLTIQueryInterface && "`::mlir::TargetDeviceSpecInterface` expected its base interface `::mlir::DLTIQueryInterface` to be registered");
    }
  };
  template<typename ConcreteAttr>
  class Model : public Concept {
  public:
    using Interface = ::mlir::TargetDeviceSpecInterface;
    Model() : Concept{getEntries, getSpecForIdentifier, verifyEntry} {}

    static inline ::mlir::DataLayoutEntryListRef getEntries(const Concept *impl, ::mlir::Attribute tablegen_opaque_val);
    static inline ::mlir::DataLayoutEntryInterface getSpecForIdentifier(const Concept *impl, ::mlir::Attribute tablegen_opaque_val, ::mlir::StringAttr identifier);
    static inline ::llvm::LogicalResult verifyEntry(const Concept *impl, ::mlir::Attribute tablegen_opaque_val, ::mlir::Location loc);
  };
  template<typename ConcreteAttr>
  class FallbackModel : public Concept {
  public:
    using Interface = ::mlir::TargetDeviceSpecInterface;
    FallbackModel() : Concept{getEntries, getSpecForIdentifier, verifyEntry} {}

    static inline ::mlir::DataLayoutEntryListRef getEntries(const Concept *impl, ::mlir::Attribute tablegen_opaque_val);
    static inline ::mlir::DataLayoutEntryInterface getSpecForIdentifier(const Concept *impl, ::mlir::Attribute tablegen_opaque_val, ::mlir::StringAttr identifier);
    static inline ::llvm::LogicalResult verifyEntry(const Concept *impl, ::mlir::Attribute tablegen_opaque_val, ::mlir::Location loc);
  };
  template<typename ConcreteModel, typename ConcreteAttr>
  class ExternalModel : public FallbackModel<ConcreteModel> {
  public:
    using ConcreteEntity = ConcreteAttr;
    ::mlir::DataLayoutEntryInterface getSpecForIdentifier(::mlir::Attribute tablegen_opaque_val, ::mlir::StringAttr identifier) const;
    ::llvm::LogicalResult verifyEntry(::mlir::Attribute tablegen_opaque_val, ::mlir::Location loc) const;
  };
};
template <typename ConcreteAttr>
struct TargetDeviceSpecInterfaceTrait;

} // namespace detail
class TargetDeviceSpecInterface : public ::mlir::AttributeInterface<TargetDeviceSpecInterface, detail::TargetDeviceSpecInterfaceInterfaceTraits> {
public:
  using ::mlir::AttributeInterface<TargetDeviceSpecInterface, detail::TargetDeviceSpecInterfaceInterfaceTraits>::AttributeInterface;
  template <typename ConcreteAttr>
  struct Trait : public detail::TargetDeviceSpecInterfaceTrait<ConcreteAttr> {};
  /// Returns the list of layout entries.
  ::mlir::DataLayoutEntryListRef getEntries() const;
  /// Returns the entry related to the given identifier, if present.
  ::mlir::DataLayoutEntryInterface getSpecForIdentifier(::mlir::StringAttr identifier) const;
  /// Checks that the entry is well-formed, reports errors at the provided location.
  ::llvm::LogicalResult verifyEntry(::mlir::Location loc) const;

    /// Helper for default implementation of `DLTIQueryInterface`'s `query`.
    ::mlir::FailureOr<::mlir::Attribute>
    queryHelper(::mlir::DataLayoutEntryKey key) const {
      if (auto strKey = llvm::dyn_cast<StringAttr>(key))
        if (DataLayoutEntryInterface spec = getSpecForIdentifier(strKey))
          return spec.getValue();
      return ::mlir::failure();
    }
  //===----------------------------------------------------------------===//
  // Inherited from ::mlir::DLTIQueryInterface
  //===----------------------------------------------------------------===//

  operator ::mlir::DLTIQueryInterface () const {
    if (!*this) return nullptr;
    return ::mlir::DLTIQueryInterface(*this, getImpl()->implDLTIQueryInterface);
  }

  /// Returns the attribute associated with the key.
  ::mlir::FailureOr<::mlir::Attribute> query(::mlir::DataLayoutEntryKey key) const;
};
namespace detail {
  template <typename ConcreteAttr>
  struct TargetDeviceSpecInterfaceTrait : public ::mlir::AttributeInterface<TargetDeviceSpecInterface, detail::TargetDeviceSpecInterfaceInterfaceTraits>::Trait<ConcreteAttr> {
    /// Returns the entry related to the given identifier, if present.
    ::mlir::DataLayoutEntryInterface getSpecForIdentifier(::mlir::StringAttr identifier) const {
      return ::mlir::detail::filterEntryForIdentifier((*static_cast<const ConcreteAttr *>(this)).getEntries(),
                                                        identifier);
    }
    /// Checks that the entry is well-formed, reports errors at the provided location.
    ::llvm::LogicalResult verifyEntry(::mlir::Location loc) const {
      return ::mlir::success();
    }
  };
}// namespace detail
} // namespace mlir
namespace mlir {
class TargetSystemSpecInterface;
namespace detail {
struct TargetSystemSpecInterfaceInterfaceTraits {
  struct Concept {
    /// The methods defined by the interface.
    llvm::ArrayRef<DeviceIDTargetDeviceSpecPair> (*getEntries)(const Concept *impl, ::mlir::Attribute );
    std::optional<::mlir::TargetDeviceSpecInterface> (*getDeviceSpecForDeviceID)(const Concept *impl, ::mlir::Attribute , StringAttr);
    ::llvm::LogicalResult (*verifySpec)(const Concept *impl, ::mlir::Attribute , ::mlir::Location);
    /// The base classes of this interface.
    const ::mlir::DLTIQueryInterface::Concept *implDLTIQueryInterface = nullptr;

    void initializeInterfaceConcept(::mlir::detail::InterfaceMap &interfaceMap) {
      implDLTIQueryInterface = interfaceMap.lookup<::mlir::DLTIQueryInterface>();
      assert(implDLTIQueryInterface && "`::mlir::TargetSystemSpecInterface` expected its base interface `::mlir::DLTIQueryInterface` to be registered");
    }
  };
  template<typename ConcreteAttr>
  class Model : public Concept {
  public:
    using Interface = ::mlir::TargetSystemSpecInterface;
    Model() : Concept{getEntries, getDeviceSpecForDeviceID, verifySpec} {}

    static inline llvm::ArrayRef<DeviceIDTargetDeviceSpecPair> getEntries(const Concept *impl, ::mlir::Attribute tablegen_opaque_val);
    static inline std::optional<::mlir::TargetDeviceSpecInterface> getDeviceSpecForDeviceID(const Concept *impl, ::mlir::Attribute tablegen_opaque_val, StringAttr deviceID);
    static inline ::llvm::LogicalResult verifySpec(const Concept *impl, ::mlir::Attribute tablegen_opaque_val, ::mlir::Location loc);
  };
  template<typename ConcreteAttr>
  class FallbackModel : public Concept {
  public:
    using Interface = ::mlir::TargetSystemSpecInterface;
    FallbackModel() : Concept{getEntries, getDeviceSpecForDeviceID, verifySpec} {}

    static inline llvm::ArrayRef<DeviceIDTargetDeviceSpecPair> getEntries(const Concept *impl, ::mlir::Attribute tablegen_opaque_val);
    static inline std::optional<::mlir::TargetDeviceSpecInterface> getDeviceSpecForDeviceID(const Concept *impl, ::mlir::Attribute tablegen_opaque_val, StringAttr deviceID);
    static inline ::llvm::LogicalResult verifySpec(const Concept *impl, ::mlir::Attribute tablegen_opaque_val, ::mlir::Location loc);
  };
  template<typename ConcreteModel, typename ConcreteAttr>
  class ExternalModel : public FallbackModel<ConcreteModel> {
  public:
    using ConcreteEntity = ConcreteAttr;
    ::llvm::LogicalResult verifySpec(::mlir::Attribute tablegen_opaque_val, ::mlir::Location loc) const;
  };
};
template <typename ConcreteAttr>
struct TargetSystemSpecInterfaceTrait;

} // namespace detail
class TargetSystemSpecInterface : public ::mlir::AttributeInterface<TargetSystemSpecInterface, detail::TargetSystemSpecInterfaceInterfaceTraits> {
public:
  using ::mlir::AttributeInterface<TargetSystemSpecInterface, detail::TargetSystemSpecInterfaceInterfaceTraits>::AttributeInterface;
  template <typename ConcreteAttr>
  struct Trait : public detail::TargetSystemSpecInterfaceTrait<ConcreteAttr> {};
  /// Returns the list of layout entries.
  llvm::ArrayRef<DeviceIDTargetDeviceSpecPair> getEntries() const;
  /// Returns the device description spec for given device ID
  std::optional<::mlir::TargetDeviceSpecInterface> getDeviceSpecForDeviceID(StringAttr deviceID) const;
  /// Verifies the validity of the specification and reports any errors at the given location.
  ::llvm::LogicalResult verifySpec(::mlir::Location loc) const;

    using DeviceID = StringAttr;

    /// Helper for default implementation of `DLTIQueryInterface`'s `query`.
    ::mlir::FailureOr<::mlir::Attribute>
    queryHelper(::mlir::DataLayoutEntryKey key) const {
      if (auto strKey = llvm::dyn_cast<::mlir::StringAttr>(key))
        if (auto deviceSpec = getDeviceSpecForDeviceID(strKey))
          return *deviceSpec;
      return ::mlir::failure();
    }
  //===----------------------------------------------------------------===//
  // Inherited from ::mlir::DLTIQueryInterface
  //===----------------------------------------------------------------===//

  operator ::mlir::DLTIQueryInterface () const {
    if (!*this) return nullptr;
    return ::mlir::DLTIQueryInterface(*this, getImpl()->implDLTIQueryInterface);
  }

  /// Returns the attribute associated with the key.
  ::mlir::FailureOr<::mlir::Attribute> query(::mlir::DataLayoutEntryKey key) const;
};
namespace detail {
  template <typename ConcreteAttr>
  struct TargetSystemSpecInterfaceTrait : public ::mlir::AttributeInterface<TargetSystemSpecInterface, detail::TargetSystemSpecInterfaceInterfaceTraits>::Trait<ConcreteAttr> {
    /// Verifies the validity of the specification and reports any errors at the given location.
    ::llvm::LogicalResult verifySpec(::mlir::Location loc) const {
      return ::mlir::detail::verifyTargetSystemSpec((*static_cast<const ConcreteAttr *>(this)), loc);
    }
  };
}// namespace detail
} // namespace mlir
namespace mlir {
template<typename ConcreteAttr>
::mlir::FailureOr<::mlir::Attribute> detail::DLTIQueryInterfaceInterfaceTraits::Model<ConcreteAttr>::query(const Concept *impl, ::mlir::Attribute tablegen_opaque_val, ::mlir::DataLayoutEntryKey key) {
  return (::llvm::cast<ConcreteAttr>(tablegen_opaque_val)).query(key);
}
template<typename ConcreteAttr>
::mlir::FailureOr<::mlir::Attribute> detail::DLTIQueryInterfaceInterfaceTraits::FallbackModel<ConcreteAttr>::query(const Concept *impl, ::mlir::Attribute tablegen_opaque_val, ::mlir::DataLayoutEntryKey key) {
  return static_cast<const ConcreteAttr *>(impl)->query(tablegen_opaque_val, key);
}
} // namespace mlir
namespace mlir {
template<typename ConcreteAttr>
::mlir::DataLayoutEntryKey detail::DataLayoutEntryInterfaceInterfaceTraits::Model<ConcreteAttr>::getKey(const Concept *impl, ::mlir::Attribute tablegen_opaque_val) {
  return (::llvm::cast<ConcreteAttr>(tablegen_opaque_val)).getKey();
}
template<typename ConcreteAttr>
::mlir::Attribute detail::DataLayoutEntryInterfaceInterfaceTraits::Model<ConcreteAttr>::getValue(const Concept *impl, ::mlir::Attribute tablegen_opaque_val) {
  return (::llvm::cast<ConcreteAttr>(tablegen_opaque_val)).getValue();
}
template<typename ConcreteAttr>
::llvm::LogicalResult detail::DataLayoutEntryInterfaceInterfaceTraits::Model<ConcreteAttr>::verifyEntry(const Concept *impl, ::mlir::Attribute tablegen_opaque_val, ::mlir::Location loc) {
  return (::llvm::cast<ConcreteAttr>(tablegen_opaque_val)).verifyEntry(loc);
}
template<typename ConcreteAttr>
::mlir::DataLayoutEntryKey detail::DataLayoutEntryInterfaceInterfaceTraits::FallbackModel<ConcreteAttr>::getKey(const Concept *impl, ::mlir::Attribute tablegen_opaque_val) {
  return static_cast<const ConcreteAttr *>(impl)->getKey(tablegen_opaque_val);
}
template<typename ConcreteAttr>
::mlir::Attribute detail::DataLayoutEntryInterfaceInterfaceTraits::FallbackModel<ConcreteAttr>::getValue(const Concept *impl, ::mlir::Attribute tablegen_opaque_val) {
  return static_cast<const ConcreteAttr *>(impl)->getValue(tablegen_opaque_val);
}
template<typename ConcreteAttr>
::llvm::LogicalResult detail::DataLayoutEntryInterfaceInterfaceTraits::FallbackModel<ConcreteAttr>::verifyEntry(const Concept *impl, ::mlir::Attribute tablegen_opaque_val, ::mlir::Location loc) {
  return static_cast<const ConcreteAttr *>(impl)->verifyEntry(tablegen_opaque_val, loc);
}
template<typename ConcreteModel, typename ConcreteAttr>
::llvm::LogicalResult detail::DataLayoutEntryInterfaceInterfaceTraits::ExternalModel<ConcreteModel, ConcreteAttr>::verifyEntry(::mlir::Attribute tablegen_opaque_val, ::mlir::Location loc) const {
return ::mlir::success();
}
} // namespace mlir
namespace mlir {
template<typename ConcreteAttr>
::mlir::DataLayoutSpecInterface detail::DataLayoutSpecInterfaceInterfaceTraits::Model<ConcreteAttr>::combineWith(const Concept *impl, ::mlir::Attribute tablegen_opaque_val, ::llvm::ArrayRef<::mlir::DataLayoutSpecInterface> specs) {
  return (::llvm::cast<ConcreteAttr>(tablegen_opaque_val)).combineWith(specs);
}
template<typename ConcreteAttr>
::mlir::DataLayoutEntryListRef detail::DataLayoutSpecInterfaceInterfaceTraits::Model<ConcreteAttr>::getEntries(const Concept *impl, ::mlir::Attribute tablegen_opaque_val) {
  return (::llvm::cast<ConcreteAttr>(tablegen_opaque_val)).getEntries();
}
template<typename ConcreteAttr>
::mlir::StringAttr detail::DataLayoutSpecInterfaceInterfaceTraits::Model<ConcreteAttr>::getEndiannessIdentifier(const Concept *impl, ::mlir::Attribute tablegen_opaque_val, ::mlir::MLIRContext * context) {
  return (::llvm::cast<ConcreteAttr>(tablegen_opaque_val)).getEndiannessIdentifier(context);
}
template<typename ConcreteAttr>
::mlir::StringAttr detail::DataLayoutSpecInterfaceInterfaceTraits::Model<ConcreteAttr>::getAllocaMemorySpaceIdentifier(const Concept *impl, ::mlir::Attribute tablegen_opaque_val, ::mlir::MLIRContext * context) {
  return (::llvm::cast<ConcreteAttr>(tablegen_opaque_val)).getAllocaMemorySpaceIdentifier(context);
}
template<typename ConcreteAttr>
::mlir::StringAttr detail::DataLayoutSpecInterfaceInterfaceTraits::Model<ConcreteAttr>::getProgramMemorySpaceIdentifier(const Concept *impl, ::mlir::Attribute tablegen_opaque_val, ::mlir::MLIRContext * context) {
  return (::llvm::cast<ConcreteAttr>(tablegen_opaque_val)).getProgramMemorySpaceIdentifier(context);
}
template<typename ConcreteAttr>
::mlir::StringAttr detail::DataLayoutSpecInterfaceInterfaceTraits::Model<ConcreteAttr>::getGlobalMemorySpaceIdentifier(const Concept *impl, ::mlir::Attribute tablegen_opaque_val, ::mlir::MLIRContext * context) {
  return (::llvm::cast<ConcreteAttr>(tablegen_opaque_val)).getGlobalMemorySpaceIdentifier(context);
}
template<typename ConcreteAttr>
::mlir::StringAttr detail::DataLayoutSpecInterfaceInterfaceTraits::Model<ConcreteAttr>::getStackAlignmentIdentifier(const Concept *impl, ::mlir::Attribute tablegen_opaque_val, ::mlir::MLIRContext * context) {
  return (::llvm::cast<ConcreteAttr>(tablegen_opaque_val)).getStackAlignmentIdentifier(context);
}
template<typename ConcreteAttr>
::mlir::DataLayoutEntryList detail::DataLayoutSpecInterfaceInterfaceTraits::Model<ConcreteAttr>::getSpecForType(const Concept *impl, ::mlir::Attribute tablegen_opaque_val, ::mlir::TypeID type) {
  return (::llvm::cast<ConcreteAttr>(tablegen_opaque_val)).getSpecForType(type);
}
template<typename ConcreteAttr>
::mlir::DataLayoutEntryInterface detail::DataLayoutSpecInterfaceInterfaceTraits::Model<ConcreteAttr>::getSpecForIdentifier(const Concept *impl, ::mlir::Attribute tablegen_opaque_val, ::mlir::StringAttr identifier) {
  return (::llvm::cast<ConcreteAttr>(tablegen_opaque_val)).getSpecForIdentifier(identifier);
}
template<typename ConcreteAttr>
::llvm::LogicalResult detail::DataLayoutSpecInterfaceInterfaceTraits::Model<ConcreteAttr>::verifySpec(const Concept *impl, ::mlir::Attribute tablegen_opaque_val, ::mlir::Location loc) {
  return (::llvm::cast<ConcreteAttr>(tablegen_opaque_val)).verifySpec(loc);
}
template<typename ConcreteAttr>
::mlir::DataLayoutSpecInterface detail::DataLayoutSpecInterfaceInterfaceTraits::FallbackModel<ConcreteAttr>::combineWith(const Concept *impl, ::mlir::Attribute tablegen_opaque_val, ::llvm::ArrayRef<::mlir::DataLayoutSpecInterface> specs) {
  return static_cast<const ConcreteAttr *>(impl)->combineWith(tablegen_opaque_val, specs);
}
template<typename ConcreteAttr>
::mlir::DataLayoutEntryListRef detail::DataLayoutSpecInterfaceInterfaceTraits::FallbackModel<ConcreteAttr>::getEntries(const Concept *impl, ::mlir::Attribute tablegen_opaque_val) {
  return static_cast<const ConcreteAttr *>(impl)->getEntries(tablegen_opaque_val);
}
template<typename ConcreteAttr>
::mlir::StringAttr detail::DataLayoutSpecInterfaceInterfaceTraits::FallbackModel<ConcreteAttr>::getEndiannessIdentifier(const Concept *impl, ::mlir::Attribute tablegen_opaque_val, ::mlir::MLIRContext * context) {
  return static_cast<const ConcreteAttr *>(impl)->getEndiannessIdentifier(tablegen_opaque_val, context);
}
template<typename ConcreteAttr>
::mlir::StringAttr detail::DataLayoutSpecInterfaceInterfaceTraits::FallbackModel<ConcreteAttr>::getAllocaMemorySpaceIdentifier(const Concept *impl, ::mlir::Attribute tablegen_opaque_val, ::mlir::MLIRContext * context) {
  return static_cast<const ConcreteAttr *>(impl)->getAllocaMemorySpaceIdentifier(tablegen_opaque_val, context);
}
template<typename ConcreteAttr>
::mlir::StringAttr detail::DataLayoutSpecInterfaceInterfaceTraits::FallbackModel<ConcreteAttr>::getProgramMemorySpaceIdentifier(const Concept *impl, ::mlir::Attribute tablegen_opaque_val, ::mlir::MLIRContext * context) {
  return static_cast<const ConcreteAttr *>(impl)->getProgramMemorySpaceIdentifier(tablegen_opaque_val, context);
}
template<typename ConcreteAttr>
::mlir::StringAttr detail::DataLayoutSpecInterfaceInterfaceTraits::FallbackModel<ConcreteAttr>::getGlobalMemorySpaceIdentifier(const Concept *impl, ::mlir::Attribute tablegen_opaque_val, ::mlir::MLIRContext * context) {
  return static_cast<const ConcreteAttr *>(impl)->getGlobalMemorySpaceIdentifier(tablegen_opaque_val, context);
}
template<typename ConcreteAttr>
::mlir::StringAttr detail::DataLayoutSpecInterfaceInterfaceTraits::FallbackModel<ConcreteAttr>::getStackAlignmentIdentifier(const Concept *impl, ::mlir::Attribute tablegen_opaque_val, ::mlir::MLIRContext * context) {
  return static_cast<const ConcreteAttr *>(impl)->getStackAlignmentIdentifier(tablegen_opaque_val, context);
}
template<typename ConcreteAttr>
::mlir::DataLayoutEntryList detail::DataLayoutSpecInterfaceInterfaceTraits::FallbackModel<ConcreteAttr>::getSpecForType(const Concept *impl, ::mlir::Attribute tablegen_opaque_val, ::mlir::TypeID type) {
  return static_cast<const ConcreteAttr *>(impl)->getSpecForType(tablegen_opaque_val, type);
}
template<typename ConcreteAttr>
::mlir::DataLayoutEntryInterface detail::DataLayoutSpecInterfaceInterfaceTraits::FallbackModel<ConcreteAttr>::getSpecForIdentifier(const Concept *impl, ::mlir::Attribute tablegen_opaque_val, ::mlir::StringAttr identifier) {
  return static_cast<const ConcreteAttr *>(impl)->getSpecForIdentifier(tablegen_opaque_val, identifier);
}
template<typename ConcreteAttr>
::llvm::LogicalResult detail::DataLayoutSpecInterfaceInterfaceTraits::FallbackModel<ConcreteAttr>::verifySpec(const Concept *impl, ::mlir::Attribute tablegen_opaque_val, ::mlir::Location loc) {
  return static_cast<const ConcreteAttr *>(impl)->verifySpec(tablegen_opaque_val, loc);
}
template<typename ConcreteModel, typename ConcreteAttr>
::mlir::DataLayoutEntryList detail::DataLayoutSpecInterfaceInterfaceTraits::ExternalModel<ConcreteModel, ConcreteAttr>::getSpecForType(::mlir::Attribute tablegen_opaque_val, ::mlir::TypeID type) const {
return ::mlir::detail::filterEntriesForType((::llvm::cast<ConcreteAttr>(tablegen_opaque_val)).getEntries(), type);
}
template<typename ConcreteModel, typename ConcreteAttr>
::mlir::DataLayoutEntryInterface detail::DataLayoutSpecInterfaceInterfaceTraits::ExternalModel<ConcreteModel, ConcreteAttr>::getSpecForIdentifier(::mlir::Attribute tablegen_opaque_val, ::mlir::StringAttr identifier) const {
return ::mlir::detail::filterEntryForIdentifier((::llvm::cast<ConcreteAttr>(tablegen_opaque_val)).getEntries(),
                                                        identifier);
}
template<typename ConcreteModel, typename ConcreteAttr>
::llvm::LogicalResult detail::DataLayoutSpecInterfaceInterfaceTraits::ExternalModel<ConcreteModel, ConcreteAttr>::verifySpec(::mlir::Attribute tablegen_opaque_val, ::mlir::Location loc) const {
return ::mlir::detail::verifyDataLayoutSpec((::llvm::cast<ConcreteAttr>(tablegen_opaque_val)), loc);
}
} // namespace mlir
namespace mlir {
template<typename ConcreteAttr>
::mlir::DataLayoutEntryListRef detail::TargetDeviceSpecInterfaceInterfaceTraits::Model<ConcreteAttr>::getEntries(const Concept *impl, ::mlir::Attribute tablegen_opaque_val) {
  return (::llvm::cast<ConcreteAttr>(tablegen_opaque_val)).getEntries();
}
template<typename ConcreteAttr>
::mlir::DataLayoutEntryInterface detail::TargetDeviceSpecInterfaceInterfaceTraits::Model<ConcreteAttr>::getSpecForIdentifier(const Concept *impl, ::mlir::Attribute tablegen_opaque_val, ::mlir::StringAttr identifier) {
  return (::llvm::cast<ConcreteAttr>(tablegen_opaque_val)).getSpecForIdentifier(identifier);
}
template<typename ConcreteAttr>
::llvm::LogicalResult detail::TargetDeviceSpecInterfaceInterfaceTraits::Model<ConcreteAttr>::verifyEntry(const Concept *impl, ::mlir::Attribute tablegen_opaque_val, ::mlir::Location loc) {
  return (::llvm::cast<ConcreteAttr>(tablegen_opaque_val)).verifyEntry(loc);
}
template<typename ConcreteAttr>
::mlir::DataLayoutEntryListRef detail::TargetDeviceSpecInterfaceInterfaceTraits::FallbackModel<ConcreteAttr>::getEntries(const Concept *impl, ::mlir::Attribute tablegen_opaque_val) {
  return static_cast<const ConcreteAttr *>(impl)->getEntries(tablegen_opaque_val);
}
template<typename ConcreteAttr>
::mlir::DataLayoutEntryInterface detail::TargetDeviceSpecInterfaceInterfaceTraits::FallbackModel<ConcreteAttr>::getSpecForIdentifier(const Concept *impl, ::mlir::Attribute tablegen_opaque_val, ::mlir::StringAttr identifier) {
  return static_cast<const ConcreteAttr *>(impl)->getSpecForIdentifier(tablegen_opaque_val, identifier);
}
template<typename ConcreteAttr>
::llvm::LogicalResult detail::TargetDeviceSpecInterfaceInterfaceTraits::FallbackModel<ConcreteAttr>::verifyEntry(const Concept *impl, ::mlir::Attribute tablegen_opaque_val, ::mlir::Location loc) {
  return static_cast<const ConcreteAttr *>(impl)->verifyEntry(tablegen_opaque_val, loc);
}
template<typename ConcreteModel, typename ConcreteAttr>
::mlir::DataLayoutEntryInterface detail::TargetDeviceSpecInterfaceInterfaceTraits::ExternalModel<ConcreteModel, ConcreteAttr>::getSpecForIdentifier(::mlir::Attribute tablegen_opaque_val, ::mlir::StringAttr identifier) const {
return ::mlir::detail::filterEntryForIdentifier((::llvm::cast<ConcreteAttr>(tablegen_opaque_val)).getEntries(),
                                                        identifier);
}
template<typename ConcreteModel, typename ConcreteAttr>
::llvm::LogicalResult detail::TargetDeviceSpecInterfaceInterfaceTraits::ExternalModel<ConcreteModel, ConcreteAttr>::verifyEntry(::mlir::Attribute tablegen_opaque_val, ::mlir::Location loc) const {
return ::mlir::success();
}
} // namespace mlir
namespace mlir {
template<typename ConcreteAttr>
llvm::ArrayRef<DeviceIDTargetDeviceSpecPair> detail::TargetSystemSpecInterfaceInterfaceTraits::Model<ConcreteAttr>::getEntries(const Concept *impl, ::mlir::Attribute tablegen_opaque_val) {
  return (::llvm::cast<ConcreteAttr>(tablegen_opaque_val)).getEntries();
}
template<typename ConcreteAttr>
std::optional<::mlir::TargetDeviceSpecInterface> detail::TargetSystemSpecInterfaceInterfaceTraits::Model<ConcreteAttr>::getDeviceSpecForDeviceID(const Concept *impl, ::mlir::Attribute tablegen_opaque_val, StringAttr deviceID) {
  return (::llvm::cast<ConcreteAttr>(tablegen_opaque_val)).getDeviceSpecForDeviceID(deviceID);
}
template<typename ConcreteAttr>
::llvm::LogicalResult detail::TargetSystemSpecInterfaceInterfaceTraits::Model<ConcreteAttr>::verifySpec(const Concept *impl, ::mlir::Attribute tablegen_opaque_val, ::mlir::Location loc) {
  return (::llvm::cast<ConcreteAttr>(tablegen_opaque_val)).verifySpec(loc);
}
template<typename ConcreteAttr>
llvm::ArrayRef<DeviceIDTargetDeviceSpecPair> detail::TargetSystemSpecInterfaceInterfaceTraits::FallbackModel<ConcreteAttr>::getEntries(const Concept *impl, ::mlir::Attribute tablegen_opaque_val) {
  return static_cast<const ConcreteAttr *>(impl)->getEntries(tablegen_opaque_val);
}
template<typename ConcreteAttr>
std::optional<::mlir::TargetDeviceSpecInterface> detail::TargetSystemSpecInterfaceInterfaceTraits::FallbackModel<ConcreteAttr>::getDeviceSpecForDeviceID(const Concept *impl, ::mlir::Attribute tablegen_opaque_val, StringAttr deviceID) {
  return static_cast<const ConcreteAttr *>(impl)->getDeviceSpecForDeviceID(tablegen_opaque_val, deviceID);
}
template<typename ConcreteAttr>
::llvm::LogicalResult detail::TargetSystemSpecInterfaceInterfaceTraits::FallbackModel<ConcreteAttr>::verifySpec(const Concept *impl, ::mlir::Attribute tablegen_opaque_val, ::mlir::Location loc) {
  return static_cast<const ConcreteAttr *>(impl)->verifySpec(tablegen_opaque_val, loc);
}
template<typename ConcreteModel, typename ConcreteAttr>
::llvm::LogicalResult detail::TargetSystemSpecInterfaceInterfaceTraits::ExternalModel<ConcreteModel, ConcreteAttr>::verifySpec(::mlir::Attribute tablegen_opaque_val, ::mlir::Location loc) const {
return ::mlir::detail::verifyTargetSystemSpec((::llvm::cast<ConcreteAttr>(tablegen_opaque_val)), loc);
}
} // namespace mlir
