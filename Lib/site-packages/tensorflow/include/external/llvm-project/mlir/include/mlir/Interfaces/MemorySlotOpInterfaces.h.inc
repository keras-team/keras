/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Interface Declarations                                                     *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

namespace mlir {
class PromotableAllocationOpInterface;
namespace detail {
struct PromotableAllocationOpInterfaceInterfaceTraits {
  struct Concept {
    /// The methods defined by the interface.
    ::llvm::SmallVector<::mlir::MemorySlot> (*getPromotableSlots)(const Concept *impl, ::mlir::Operation *);
    ::mlir::Value (*getDefaultValue)(const Concept *impl, ::mlir::Operation *, const ::mlir::MemorySlot &, ::mlir::OpBuilder &);
    void (*handleBlockArgument)(const Concept *impl, ::mlir::Operation *, const ::mlir::MemorySlot &, ::mlir::BlockArgument, ::mlir::OpBuilder &);
    ::std::optional<::mlir::PromotableAllocationOpInterface> (*handlePromotionComplete)(const Concept *impl, ::mlir::Operation *, const ::mlir::MemorySlot &, ::mlir::Value, ::mlir::OpBuilder &);
  };
  template<typename ConcreteOp>
  class Model : public Concept {
  public:
    using Interface = ::mlir::PromotableAllocationOpInterface;
    Model() : Concept{getPromotableSlots, getDefaultValue, handleBlockArgument, handlePromotionComplete} {}

    static inline ::llvm::SmallVector<::mlir::MemorySlot> getPromotableSlots(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline ::mlir::Value getDefaultValue(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, const ::mlir::MemorySlot & slot, ::mlir::OpBuilder & builder);
    static inline void handleBlockArgument(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, const ::mlir::MemorySlot & slot, ::mlir::BlockArgument argument, ::mlir::OpBuilder & builder);
    static inline ::std::optional<::mlir::PromotableAllocationOpInterface> handlePromotionComplete(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, const ::mlir::MemorySlot & slot, ::mlir::Value defaultValue, ::mlir::OpBuilder & builder);
  };
  template<typename ConcreteOp>
  class FallbackModel : public Concept {
  public:
    using Interface = ::mlir::PromotableAllocationOpInterface;
    FallbackModel() : Concept{getPromotableSlots, getDefaultValue, handleBlockArgument, handlePromotionComplete} {}

    static inline ::llvm::SmallVector<::mlir::MemorySlot> getPromotableSlots(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline ::mlir::Value getDefaultValue(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, const ::mlir::MemorySlot & slot, ::mlir::OpBuilder & builder);
    static inline void handleBlockArgument(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, const ::mlir::MemorySlot & slot, ::mlir::BlockArgument argument, ::mlir::OpBuilder & builder);
    static inline ::std::optional<::mlir::PromotableAllocationOpInterface> handlePromotionComplete(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, const ::mlir::MemorySlot & slot, ::mlir::Value defaultValue, ::mlir::OpBuilder & builder);
  };
  template<typename ConcreteModel, typename ConcreteOp>
  class ExternalModel : public FallbackModel<ConcreteModel> {
  public:
    using ConcreteEntity = ConcreteOp;
  };
};
template <typename ConcreteOp>
struct PromotableAllocationOpInterfaceTrait;

} // namespace detail
class PromotableAllocationOpInterface : public ::mlir::OpInterface<PromotableAllocationOpInterface, detail::PromotableAllocationOpInterfaceInterfaceTraits> {
public:
  using ::mlir::OpInterface<PromotableAllocationOpInterface, detail::PromotableAllocationOpInterfaceInterfaceTraits>::OpInterface;
  template <typename ConcreteOp>
  struct Trait : public detail::PromotableAllocationOpInterfaceTrait<ConcreteOp> {};
  /// Returns a list of memory slots for which promotion should be attempted.
  /// This only considers the local semantics of the allocator, ignoring
  /// whether the slot pointer is properly used or not. This allocator is the
  /// "owner" of the returned slots, meaning no two allocators should return
  /// the same slot. The content of the memory slot must only be reachable
  /// using loads and stores to the provided slot pointer, no aliasing is
  /// allowed.
  /// 
  /// Promotion of the slot will lead to the slot pointer no longer being
  /// used, leaving the content of the memory slot unreachable.
  /// 
  /// No IR mutation is allowed in this method.
  ::llvm::SmallVector<::mlir::MemorySlot> getPromotableSlots();
  /// Provides the default Value of this memory slot. The provided Value
  /// will be used as the reaching definition of loads done before any store.
  /// This Value must outlive the promotion and dominate all the uses of this
  /// slot's pointer. The provided builder can be used to create the default
  /// value on the fly.
  /// 
  /// The builder is located at the beginning of the block where the slot
  /// pointer is defined.
  ::mlir::Value getDefaultValue(const ::mlir::MemorySlot & slot, ::mlir::OpBuilder & builder);
  /// Hook triggered for every new block argument added to a block.
  /// This will only be called for slots declared by this operation.
  /// 
  /// The builder is located at the beginning of the block on call. All IR
  /// mutations must happen through the builder.
  void handleBlockArgument(const ::mlir::MemorySlot & slot, ::mlir::BlockArgument argument, ::mlir::OpBuilder & builder);
  /// Hook triggered once the promotion of a slot is complete. This can
  /// also clean up the created default value if necessary.
  /// This will only be called for slots declared by this operation.
  /// 
  /// Must return a new promotable allocation op if this operation produced
  /// multiple promotable slots, nullopt otherwise.
  ::std::optional<::mlir::PromotableAllocationOpInterface> handlePromotionComplete(const ::mlir::MemorySlot & slot, ::mlir::Value defaultValue, ::mlir::OpBuilder & builder);
};
namespace detail {
  template <typename ConcreteOp>
  struct PromotableAllocationOpInterfaceTrait : public ::mlir::OpInterface<PromotableAllocationOpInterface, detail::PromotableAllocationOpInterfaceInterfaceTraits>::Trait<ConcreteOp> {
  };
}// namespace detail
} // namespace mlir
namespace mlir {
class PromotableMemOpInterface;
namespace detail {
struct PromotableMemOpInterfaceInterfaceTraits {
  struct Concept {
    /// The methods defined by the interface.
    bool (*loadsFrom)(const Concept *impl, ::mlir::Operation *, const ::mlir::MemorySlot &);
    bool (*storesTo)(const Concept *impl, ::mlir::Operation *, const ::mlir::MemorySlot &);
    ::mlir::Value (*getStored)(const Concept *impl, ::mlir::Operation *, const ::mlir::MemorySlot &, ::mlir::OpBuilder &, ::mlir::Value, const ::mlir::DataLayout &);
    bool (*canUsesBeRemoved)(const Concept *impl, ::mlir::Operation *, const ::mlir::MemorySlot &, const ::llvm::SmallPtrSetImpl<::mlir::OpOperand *> &, ::llvm::SmallVectorImpl<::mlir::OpOperand *> &, const ::mlir::DataLayout &);
    ::mlir::DeletionKind (*removeBlockingUses)(const Concept *impl, ::mlir::Operation *, const ::mlir::MemorySlot &, const ::llvm::SmallPtrSetImpl<mlir::OpOperand *> &, ::mlir::OpBuilder &, ::mlir::Value, const ::mlir::DataLayout &);
  };
  template<typename ConcreteOp>
  class Model : public Concept {
  public:
    using Interface = ::mlir::PromotableMemOpInterface;
    Model() : Concept{loadsFrom, storesTo, getStored, canUsesBeRemoved, removeBlockingUses} {}

    static inline bool loadsFrom(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, const ::mlir::MemorySlot & slot);
    static inline bool storesTo(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, const ::mlir::MemorySlot & slot);
    static inline ::mlir::Value getStored(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, const ::mlir::MemorySlot & slot, ::mlir::OpBuilder & builder, ::mlir::Value reachingDef, const ::mlir::DataLayout & dataLayout);
    static inline bool canUsesBeRemoved(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, const ::mlir::MemorySlot & slot, const ::llvm::SmallPtrSetImpl<::mlir::OpOperand *> & blockingUses, ::llvm::SmallVectorImpl<::mlir::OpOperand *> & newBlockingUses, const ::mlir::DataLayout & datalayout);
    static inline ::mlir::DeletionKind removeBlockingUses(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, const ::mlir::MemorySlot & slot, const ::llvm::SmallPtrSetImpl<mlir::OpOperand *> & blockingUses, ::mlir::OpBuilder & builder, ::mlir::Value reachingDefinition, const ::mlir::DataLayout & dataLayout);
  };
  template<typename ConcreteOp>
  class FallbackModel : public Concept {
  public:
    using Interface = ::mlir::PromotableMemOpInterface;
    FallbackModel() : Concept{loadsFrom, storesTo, getStored, canUsesBeRemoved, removeBlockingUses} {}

    static inline bool loadsFrom(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, const ::mlir::MemorySlot & slot);
    static inline bool storesTo(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, const ::mlir::MemorySlot & slot);
    static inline ::mlir::Value getStored(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, const ::mlir::MemorySlot & slot, ::mlir::OpBuilder & builder, ::mlir::Value reachingDef, const ::mlir::DataLayout & dataLayout);
    static inline bool canUsesBeRemoved(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, const ::mlir::MemorySlot & slot, const ::llvm::SmallPtrSetImpl<::mlir::OpOperand *> & blockingUses, ::llvm::SmallVectorImpl<::mlir::OpOperand *> & newBlockingUses, const ::mlir::DataLayout & datalayout);
    static inline ::mlir::DeletionKind removeBlockingUses(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, const ::mlir::MemorySlot & slot, const ::llvm::SmallPtrSetImpl<mlir::OpOperand *> & blockingUses, ::mlir::OpBuilder & builder, ::mlir::Value reachingDefinition, const ::mlir::DataLayout & dataLayout);
  };
  template<typename ConcreteModel, typename ConcreteOp>
  class ExternalModel : public FallbackModel<ConcreteModel> {
  public:
    using ConcreteEntity = ConcreteOp;
  };
};
template <typename ConcreteOp>
struct PromotableMemOpInterfaceTrait;

} // namespace detail
class PromotableMemOpInterface : public ::mlir::OpInterface<PromotableMemOpInterface, detail::PromotableMemOpInterfaceInterfaceTraits> {
public:
  using ::mlir::OpInterface<PromotableMemOpInterface, detail::PromotableMemOpInterfaceInterfaceTraits>::OpInterface;
  template <typename ConcreteOp>
  struct Trait : public detail::PromotableMemOpInterfaceTrait<ConcreteOp> {};
  /// Gets whether this operation loads from the specified slot.
  /// 
  /// No IR mutation is allowed in this method.
  bool loadsFrom(const ::mlir::MemorySlot & slot);
  /// Gets whether this operation stores to the specified slot.
  /// 
  /// No IR mutation is allowed in this method.
  bool storesTo(const ::mlir::MemorySlot & slot);
  /// Gets the value stored to the provided memory slot, or returns a null
  /// value if this operation does not store to this slot. An operation
  /// storing a value to a slot must always be able to provide the value it
  /// stores. This method is only called once per slot promotion, and only
  /// on operations that store to the slot according to the `storesTo` method.
  /// The returned value must dominate all operations dominated by the storing
  /// operation.
  /// 
  /// The builder is located immediately after the memory operation on call.
  /// No IR deletion is allowed in this method. IR mutations must not
  /// introduce new uses of the memory slot. Existing control flow must not
  /// be modified.
  ::mlir::Value getStored(const ::mlir::MemorySlot & slot, ::mlir::OpBuilder & builder, ::mlir::Value reachingDef, const ::mlir::DataLayout & dataLayout);
  /// Checks that this operation can be promoted to no longer use the provided
  /// blocking uses, in the context of promoting `slot`.
  /// 
  /// If the removal procedure of the use will require that other uses get
  /// removed, that dependency should be added to the `newBlockingUses`
  /// argument. Dependent uses must only be uses of results of this operation.
  /// 
  /// No IR mutation is allowed in this method.
  bool canUsesBeRemoved(const ::mlir::MemorySlot & slot, const ::llvm::SmallPtrSetImpl<::mlir::OpOperand *> & blockingUses, ::llvm::SmallVectorImpl<::mlir::OpOperand *> & newBlockingUses, const ::mlir::DataLayout & datalayout);
  /// Transforms IR to ensure that the current operation does not use the
  /// provided memory slot anymore. `reachingDefinition` contains the value
  /// currently stored in the provided memory slot, immediately before the
  /// current operation.
  /// 
  /// During the transformation, *no operation should be deleted*.
  /// The operation can only schedule its own deletion by returning the
  /// appropriate `DeletionKind`. The deletion must be legal assuming the
  /// blocking uses passed through the `newBlockingUses` list in
  /// `canUseBeRemoved` have been removed.
  /// 
  /// After calling this method, the blocking uses should have disappeared
  /// or this operation should have scheduled its own deletion.
  /// 
  /// This method will only be called after ensuring promotion is allowed via
  /// `canUseBeRemoved`. The requested blocking use removal may or may not
  /// have been done at the point of calling this method, but it will be done
  /// eventually.
  /// 
  /// The builder is located after the promotable operation on call.
  ::mlir::DeletionKind removeBlockingUses(const ::mlir::MemorySlot & slot, const ::llvm::SmallPtrSetImpl<mlir::OpOperand *> & blockingUses, ::mlir::OpBuilder & builder, ::mlir::Value reachingDefinition, const ::mlir::DataLayout & dataLayout);
};
namespace detail {
  template <typename ConcreteOp>
  struct PromotableMemOpInterfaceTrait : public ::mlir::OpInterface<PromotableMemOpInterface, detail::PromotableMemOpInterfaceInterfaceTraits>::Trait<ConcreteOp> {
  };
}// namespace detail
} // namespace mlir
namespace mlir {
class PromotableOpInterface;
namespace detail {
struct PromotableOpInterfaceInterfaceTraits {
  struct Concept {
    /// The methods defined by the interface.
    bool (*canUsesBeRemoved)(const Concept *impl, ::mlir::Operation *, const ::llvm::SmallPtrSetImpl<::mlir::OpOperand *> &, ::llvm::SmallVectorImpl<::mlir::OpOperand *> &, const ::mlir::DataLayout &);
    ::mlir::DeletionKind (*removeBlockingUses)(const Concept *impl, ::mlir::Operation *, const ::llvm::SmallPtrSetImpl<mlir::OpOperand *> &, ::mlir::OpBuilder &);
    bool (*requiresReplacedValues)(const Concept *impl, ::mlir::Operation *);
    void (*visitReplacedValues)(const Concept *impl, ::mlir::Operation *, ::llvm::ArrayRef<std::pair<::mlir::Operation*, ::mlir::Value>>, ::mlir::OpBuilder &);
  };
  template<typename ConcreteOp>
  class Model : public Concept {
  public:
    using Interface = ::mlir::PromotableOpInterface;
    Model() : Concept{canUsesBeRemoved, removeBlockingUses, requiresReplacedValues, visitReplacedValues} {}

    static inline bool canUsesBeRemoved(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, const ::llvm::SmallPtrSetImpl<::mlir::OpOperand *> & blockingUses, ::llvm::SmallVectorImpl<::mlir::OpOperand *> & newBlockingUses, const ::mlir::DataLayout & datalayout);
    static inline ::mlir::DeletionKind removeBlockingUses(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, const ::llvm::SmallPtrSetImpl<mlir::OpOperand *> & blockingUses, ::mlir::OpBuilder & builder);
    static inline bool requiresReplacedValues(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline void visitReplacedValues(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, ::llvm::ArrayRef<std::pair<::mlir::Operation*, ::mlir::Value>> mutatedDefs, ::mlir::OpBuilder & builder);
  };
  template<typename ConcreteOp>
  class FallbackModel : public Concept {
  public:
    using Interface = ::mlir::PromotableOpInterface;
    FallbackModel() : Concept{canUsesBeRemoved, removeBlockingUses, requiresReplacedValues, visitReplacedValues} {}

    static inline bool canUsesBeRemoved(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, const ::llvm::SmallPtrSetImpl<::mlir::OpOperand *> & blockingUses, ::llvm::SmallVectorImpl<::mlir::OpOperand *> & newBlockingUses, const ::mlir::DataLayout & datalayout);
    static inline ::mlir::DeletionKind removeBlockingUses(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, const ::llvm::SmallPtrSetImpl<mlir::OpOperand *> & blockingUses, ::mlir::OpBuilder & builder);
    static inline bool requiresReplacedValues(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline void visitReplacedValues(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, ::llvm::ArrayRef<std::pair<::mlir::Operation*, ::mlir::Value>> mutatedDefs, ::mlir::OpBuilder & builder);
  };
  template<typename ConcreteModel, typename ConcreteOp>
  class ExternalModel : public FallbackModel<ConcreteModel> {
  public:
    using ConcreteEntity = ConcreteOp;
    bool requiresReplacedValues(::mlir::Operation *tablegen_opaque_val) const;
    void visitReplacedValues(::mlir::Operation *tablegen_opaque_val, ::llvm::ArrayRef<std::pair<::mlir::Operation*, ::mlir::Value>> mutatedDefs, ::mlir::OpBuilder &builder) const;
  };
};
template <typename ConcreteOp>
struct PromotableOpInterfaceTrait;

} // namespace detail
class PromotableOpInterface : public ::mlir::OpInterface<PromotableOpInterface, detail::PromotableOpInterfaceInterfaceTraits> {
public:
  using ::mlir::OpInterface<PromotableOpInterface, detail::PromotableOpInterfaceInterfaceTraits>::OpInterface;
  template <typename ConcreteOp>
  struct Trait : public detail::PromotableOpInterfaceTrait<ConcreteOp> {};
  /// Checks that this operation can be promoted to no longer use the provided
  /// blocking uses, in order to allow optimization.
  /// 
  /// If the removal procedure of the use will require that other uses get
  /// removed, that dependency should be added to the `newBlockingUses`
  /// argument. Dependent uses must only be uses of results of this operation.
  /// 
  /// No IR mutation is allowed in this method.
  bool canUsesBeRemoved(const ::llvm::SmallPtrSetImpl<::mlir::OpOperand *> & blockingUses, ::llvm::SmallVectorImpl<::mlir::OpOperand *> & newBlockingUses, const ::mlir::DataLayout & datalayout);
  /// Transforms IR to ensure that the current operation does not use the
  /// provided blocking uses anymore. In contrast to
  /// `PromotableMemOpInterface`, operations implementing this interface
  /// must not need access to the reaching definition of the content of the
  /// slot.
  /// 
  /// During the transformation, *no operation should be deleted*.
  /// The operation can only schedule its own deletion by returning the
  /// appropriate `DeletionKind`. The deletion must be legal assuming the
  /// blocking uses passed through the `newBlockingUses` list in
  /// `canUseBeRemoved` have been removed.
  /// 
  /// After calling this method, the blocking uses should have disappeared
  /// or this operation should have scheduled its own deletion.
  /// 
  /// This method will only be called after ensuring promotion is allowed via
  /// `canUseBeRemoved`. The requested blocking use removal may or may not
  /// have been done at the point of calling this method, but it will be done
  /// eventually.
  /// 
  /// The builder is located after the promotable operation on call.
  ::mlir::DeletionKind removeBlockingUses(const ::llvm::SmallPtrSetImpl<mlir::OpOperand *> & blockingUses, ::mlir::OpBuilder & builder);
  /// This method allows the promoted operation to visit the SSA values used
  /// in place of the memory slot once the promotion process of the memory
  /// slot is complete.
  /// 
  /// If this method returns true, the `visitReplacedValues` method on this
  /// operation will be called after the main mutation stage finishes
  /// (i.e., after all ops have been processed with `removeBlockingUses`).
  /// 
  /// Operations should only the replaced values if the intended
  /// transformation applies to all the replaced values. Furthermore, replaced
  /// values must not be deleted.
  bool requiresReplacedValues();
  /// Transforms the IR using the SSA values that replaced the memory slot.
  /// 
  /// This method will only be called after all blocking uses have been
  /// scheduled for removal and if `requiresReplacedValues` returned
  /// true.
  /// 
  /// The builder is located after the promotable operation on call. During
  /// the transformation, *no operation should be deleted*.
  void visitReplacedValues(::llvm::ArrayRef<std::pair<::mlir::Operation*, ::mlir::Value>> mutatedDefs, ::mlir::OpBuilder & builder);
};
namespace detail {
  template <typename ConcreteOp>
  struct PromotableOpInterfaceTrait : public ::mlir::OpInterface<PromotableOpInterface, detail::PromotableOpInterfaceInterfaceTraits>::Trait<ConcreteOp> {
    /// This method allows the promoted operation to visit the SSA values used
    /// in place of the memory slot once the promotion process of the memory
    /// slot is complete.
    /// 
    /// If this method returns true, the `visitReplacedValues` method on this
    /// operation will be called after the main mutation stage finishes
    /// (i.e., after all ops have been processed with `removeBlockingUses`).
    /// 
    /// Operations should only the replaced values if the intended
    /// transformation applies to all the replaced values. Furthermore, replaced
    /// values must not be deleted.
    bool requiresReplacedValues() {
      return false;
    }
    /// Transforms the IR using the SSA values that replaced the memory slot.
    /// 
    /// This method will only be called after all blocking uses have been
    /// scheduled for removal and if `requiresReplacedValues` returned
    /// true.
    /// 
    /// The builder is located after the promotable operation on call. During
    /// the transformation, *no operation should be deleted*.
    void visitReplacedValues(::llvm::ArrayRef<std::pair<::mlir::Operation*, ::mlir::Value>> mutatedDefs, ::mlir::OpBuilder & builder) {
      return;
    }
  };
}// namespace detail
} // namespace mlir
namespace mlir {
class DestructurableAllocationOpInterface;
namespace detail {
struct DestructurableAllocationOpInterfaceInterfaceTraits {
  struct Concept {
    /// The methods defined by the interface.
    ::llvm::SmallVector<::mlir::DestructurableMemorySlot> (*getDestructurableSlots)(const Concept *impl, ::mlir::Operation *);
    ::llvm::DenseMap<::mlir::Attribute, ::mlir::MemorySlot> (*destructure)(const Concept *impl, ::mlir::Operation *, const ::mlir::DestructurableMemorySlot &, const ::llvm::SmallPtrSetImpl<::mlir::Attribute> &, ::mlir::OpBuilder &, ::mlir::SmallVectorImpl<::mlir::DestructurableAllocationOpInterface> &);
    ::std::optional<::mlir::DestructurableAllocationOpInterface> (*handleDestructuringComplete)(const Concept *impl, ::mlir::Operation *, const ::mlir::DestructurableMemorySlot &, ::mlir::OpBuilder &);
  };
  template<typename ConcreteOp>
  class Model : public Concept {
  public:
    using Interface = ::mlir::DestructurableAllocationOpInterface;
    Model() : Concept{getDestructurableSlots, destructure, handleDestructuringComplete} {}

    static inline ::llvm::SmallVector<::mlir::DestructurableMemorySlot> getDestructurableSlots(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline ::llvm::DenseMap<::mlir::Attribute, ::mlir::MemorySlot> destructure(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, const ::mlir::DestructurableMemorySlot & slot, const ::llvm::SmallPtrSetImpl<::mlir::Attribute> & usedIndices, ::mlir::OpBuilder & builder, ::mlir::SmallVectorImpl<::mlir::DestructurableAllocationOpInterface> & newAllocators);
    static inline ::std::optional<::mlir::DestructurableAllocationOpInterface> handleDestructuringComplete(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, const ::mlir::DestructurableMemorySlot & slot, ::mlir::OpBuilder & builder);
  };
  template<typename ConcreteOp>
  class FallbackModel : public Concept {
  public:
    using Interface = ::mlir::DestructurableAllocationOpInterface;
    FallbackModel() : Concept{getDestructurableSlots, destructure, handleDestructuringComplete} {}

    static inline ::llvm::SmallVector<::mlir::DestructurableMemorySlot> getDestructurableSlots(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline ::llvm::DenseMap<::mlir::Attribute, ::mlir::MemorySlot> destructure(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, const ::mlir::DestructurableMemorySlot & slot, const ::llvm::SmallPtrSetImpl<::mlir::Attribute> & usedIndices, ::mlir::OpBuilder & builder, ::mlir::SmallVectorImpl<::mlir::DestructurableAllocationOpInterface> & newAllocators);
    static inline ::std::optional<::mlir::DestructurableAllocationOpInterface> handleDestructuringComplete(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, const ::mlir::DestructurableMemorySlot & slot, ::mlir::OpBuilder & builder);
  };
  template<typename ConcreteModel, typename ConcreteOp>
  class ExternalModel : public FallbackModel<ConcreteModel> {
  public:
    using ConcreteEntity = ConcreteOp;
  };
};
template <typename ConcreteOp>
struct DestructurableAllocationOpInterfaceTrait;

} // namespace detail
class DestructurableAllocationOpInterface : public ::mlir::OpInterface<DestructurableAllocationOpInterface, detail::DestructurableAllocationOpInterfaceInterfaceTraits> {
public:
  using ::mlir::OpInterface<DestructurableAllocationOpInterface, detail::DestructurableAllocationOpInterfaceInterfaceTraits>::OpInterface;
  template <typename ConcreteOp>
  struct Trait : public detail::DestructurableAllocationOpInterfaceTrait<ConcreteOp> {};
  /// Returns the list of slots for which destructuring should be attempted,
  /// specifying in which way the slot should be destructured into subslots.
  /// The subslots are indexed by attributes. This computes the type of the
  /// pointer for each subslot to be generated. The type of the memory slot
  /// must implement `DestructurableTypeInterface`.
  /// 
  /// No IR mutation is allowed in this method.
  ::llvm::SmallVector<::mlir::DestructurableMemorySlot> getDestructurableSlots();
  /// Destructures this slot into multiple subslots. The newly generated slots
  /// may belong to a different allocator. The original slot must still exist
  /// at the end of this call. Only generates subslots for the indices found in
  /// `usedIndices` since all other subslots are unused.
  /// 
  /// The builder is located at the beginning of the block where the slot
  /// pointer is defined.
  ::llvm::DenseMap<::mlir::Attribute, ::mlir::MemorySlot> destructure(const ::mlir::DestructurableMemorySlot & slot, const ::llvm::SmallPtrSetImpl<::mlir::Attribute> & usedIndices, ::mlir::OpBuilder & builder, ::mlir::SmallVectorImpl<::mlir::DestructurableAllocationOpInterface> & newAllocators);
  /// Hook triggered once the destructuring of a slot is complete, meaning the
  /// original slot is no longer being refered to and could be deleted.
  /// This will only be called for slots declared by this operation.
  /// 
  /// Must return a new destructurable allocation op if this hook creates
  /// a new destructurable op, nullopt otherwise.
  ::std::optional<::mlir::DestructurableAllocationOpInterface> handleDestructuringComplete(const ::mlir::DestructurableMemorySlot & slot, ::mlir::OpBuilder & builder);
};
namespace detail {
  template <typename ConcreteOp>
  struct DestructurableAllocationOpInterfaceTrait : public ::mlir::OpInterface<DestructurableAllocationOpInterface, detail::DestructurableAllocationOpInterfaceInterfaceTraits>::Trait<ConcreteOp> {
  };
}// namespace detail
} // namespace mlir
namespace mlir {
class SafeMemorySlotAccessOpInterface;
namespace detail {
struct SafeMemorySlotAccessOpInterfaceInterfaceTraits {
  struct Concept {
    /// The methods defined by the interface.
    ::llvm::LogicalResult (*ensureOnlySafeAccesses)(const Concept *impl, ::mlir::Operation *, const ::mlir::MemorySlot &, ::mlir::SmallVectorImpl<::mlir::MemorySlot> &, const ::mlir::DataLayout &);
  };
  template<typename ConcreteOp>
  class Model : public Concept {
  public:
    using Interface = ::mlir::SafeMemorySlotAccessOpInterface;
    Model() : Concept{ensureOnlySafeAccesses} {}

    static inline ::llvm::LogicalResult ensureOnlySafeAccesses(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, const ::mlir::MemorySlot & slot, ::mlir::SmallVectorImpl<::mlir::MemorySlot> & mustBeSafelyUsed, const ::mlir::DataLayout & dataLayout);
  };
  template<typename ConcreteOp>
  class FallbackModel : public Concept {
  public:
    using Interface = ::mlir::SafeMemorySlotAccessOpInterface;
    FallbackModel() : Concept{ensureOnlySafeAccesses} {}

    static inline ::llvm::LogicalResult ensureOnlySafeAccesses(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, const ::mlir::MemorySlot & slot, ::mlir::SmallVectorImpl<::mlir::MemorySlot> & mustBeSafelyUsed, const ::mlir::DataLayout & dataLayout);
  };
  template<typename ConcreteModel, typename ConcreteOp>
  class ExternalModel : public FallbackModel<ConcreteModel> {
  public:
    using ConcreteEntity = ConcreteOp;
  };
};
template <typename ConcreteOp>
struct SafeMemorySlotAccessOpInterfaceTrait;

} // namespace detail
class SafeMemorySlotAccessOpInterface : public ::mlir::OpInterface<SafeMemorySlotAccessOpInterface, detail::SafeMemorySlotAccessOpInterfaceInterfaceTraits> {
public:
  using ::mlir::OpInterface<SafeMemorySlotAccessOpInterface, detail::SafeMemorySlotAccessOpInterfaceInterfaceTraits>::OpInterface;
  template <typename ConcreteOp>
  struct Trait : public detail::SafeMemorySlotAccessOpInterfaceTrait<ConcreteOp> {};
  /// Returns whether all accesses in this operation to the provided slot are
  /// done in a safe manner. To be safe, the access most only access the slot
  /// inside the bounds that its type implies.
  /// 
  /// If the safety of the accesses depends on the safety of the accesses to
  /// further memory slots, the result of this method will be conditioned to
  /// the safety of the accesses to the slots added by this method to
  /// `mustBeSafelyUsed`.
  /// 
  /// No IR mutation is allowed in this method.
  ::llvm::LogicalResult ensureOnlySafeAccesses(const ::mlir::MemorySlot & slot, ::mlir::SmallVectorImpl<::mlir::MemorySlot> & mustBeSafelyUsed, const ::mlir::DataLayout & dataLayout);
};
namespace detail {
  template <typename ConcreteOp>
  struct SafeMemorySlotAccessOpInterfaceTrait : public ::mlir::OpInterface<SafeMemorySlotAccessOpInterface, detail::SafeMemorySlotAccessOpInterfaceInterfaceTraits>::Trait<ConcreteOp> {
  };
}// namespace detail
} // namespace mlir
namespace mlir {
class DestructurableAccessorOpInterface;
namespace detail {
struct DestructurableAccessorOpInterfaceInterfaceTraits {
  struct Concept {
    /// The methods defined by the interface.
    bool (*canRewire)(const Concept *impl, ::mlir::Operation *, const ::mlir::DestructurableMemorySlot &, ::llvm::SmallPtrSetImpl<::mlir::Attribute> &, ::mlir::SmallVectorImpl<::mlir::MemorySlot> &, const ::mlir::DataLayout &);
    ::mlir::DeletionKind (*rewire)(const Concept *impl, ::mlir::Operation *, const ::mlir::DestructurableMemorySlot &, ::llvm::DenseMap<::mlir::Attribute, ::mlir::MemorySlot> &, ::mlir::OpBuilder &, const ::mlir::DataLayout &);
  };
  template<typename ConcreteOp>
  class Model : public Concept {
  public:
    using Interface = ::mlir::DestructurableAccessorOpInterface;
    Model() : Concept{canRewire, rewire} {}

    static inline bool canRewire(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, const ::mlir::DestructurableMemorySlot & slot, ::llvm::SmallPtrSetImpl<::mlir::Attribute> & usedIndices, ::mlir::SmallVectorImpl<::mlir::MemorySlot> & mustBeSafelyUsed, const ::mlir::DataLayout & dataLayout);
    static inline ::mlir::DeletionKind rewire(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, const ::mlir::DestructurableMemorySlot & slot, ::llvm::DenseMap<::mlir::Attribute, ::mlir::MemorySlot> & subslots, ::mlir::OpBuilder & builder, const ::mlir::DataLayout & dataLayout);
  };
  template<typename ConcreteOp>
  class FallbackModel : public Concept {
  public:
    using Interface = ::mlir::DestructurableAccessorOpInterface;
    FallbackModel() : Concept{canRewire, rewire} {}

    static inline bool canRewire(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, const ::mlir::DestructurableMemorySlot & slot, ::llvm::SmallPtrSetImpl<::mlir::Attribute> & usedIndices, ::mlir::SmallVectorImpl<::mlir::MemorySlot> & mustBeSafelyUsed, const ::mlir::DataLayout & dataLayout);
    static inline ::mlir::DeletionKind rewire(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, const ::mlir::DestructurableMemorySlot & slot, ::llvm::DenseMap<::mlir::Attribute, ::mlir::MemorySlot> & subslots, ::mlir::OpBuilder & builder, const ::mlir::DataLayout & dataLayout);
  };
  template<typename ConcreteModel, typename ConcreteOp>
  class ExternalModel : public FallbackModel<ConcreteModel> {
  public:
    using ConcreteEntity = ConcreteOp;
  };
};
template <typename ConcreteOp>
struct DestructurableAccessorOpInterfaceTrait;

} // namespace detail
class DestructurableAccessorOpInterface : public ::mlir::OpInterface<DestructurableAccessorOpInterface, detail::DestructurableAccessorOpInterfaceInterfaceTraits> {
public:
  using ::mlir::OpInterface<DestructurableAccessorOpInterface, detail::DestructurableAccessorOpInterfaceInterfaceTraits>::OpInterface;
  template <typename ConcreteOp>
  struct Trait : public detail::DestructurableAccessorOpInterfaceTrait<ConcreteOp> {};
  /// For a given destructurable memory slot, returns whether this operation can
  /// rewire its uses of the slot to use the slots generated after
  /// destructuring. This may involve creating new operations.
  /// 
  /// This method must also register the indices it will access within the
  /// `usedIndices` set. If the accessor generates new slots mapping to
  /// subelements, they must be registered in `mustBeSafelyUsed` to ensure
  /// they are used in a safe manner.
  /// 
  /// No IR mutation is allowed in this method.
  bool canRewire(const ::mlir::DestructurableMemorySlot & slot, ::llvm::SmallPtrSetImpl<::mlir::Attribute> & usedIndices, ::mlir::SmallVectorImpl<::mlir::MemorySlot> & mustBeSafelyUsed, const ::mlir::DataLayout & dataLayout);
  /// Rewires the use of a slot to the generated subslots, without deleting
  /// any operation. Returns whether the accessor should be deleted.
  /// 
  /// Deletion of operations is not allowed, only the accessor can be
  /// scheduled for deletion by returning the appropriate value.
  ::mlir::DeletionKind rewire(const ::mlir::DestructurableMemorySlot & slot, ::llvm::DenseMap<::mlir::Attribute, ::mlir::MemorySlot> & subslots, ::mlir::OpBuilder & builder, const ::mlir::DataLayout & dataLayout);
};
namespace detail {
  template <typename ConcreteOp>
  struct DestructurableAccessorOpInterfaceTrait : public ::mlir::OpInterface<DestructurableAccessorOpInterface, detail::DestructurableAccessorOpInterfaceInterfaceTraits>::Trait<ConcreteOp> {
  };
}// namespace detail
} // namespace mlir
namespace mlir {
template<typename ConcreteOp>
::llvm::SmallVector<::mlir::MemorySlot> detail::PromotableAllocationOpInterfaceInterfaceTraits::Model<ConcreteOp>::getPromotableSlots(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getPromotableSlots();
}
template<typename ConcreteOp>
::mlir::Value detail::PromotableAllocationOpInterfaceInterfaceTraits::Model<ConcreteOp>::getDefaultValue(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, const ::mlir::MemorySlot & slot, ::mlir::OpBuilder & builder) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getDefaultValue(slot, builder);
}
template<typename ConcreteOp>
void detail::PromotableAllocationOpInterfaceInterfaceTraits::Model<ConcreteOp>::handleBlockArgument(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, const ::mlir::MemorySlot & slot, ::mlir::BlockArgument argument, ::mlir::OpBuilder & builder) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).handleBlockArgument(slot, argument, builder);
}
template<typename ConcreteOp>
::std::optional<::mlir::PromotableAllocationOpInterface> detail::PromotableAllocationOpInterfaceInterfaceTraits::Model<ConcreteOp>::handlePromotionComplete(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, const ::mlir::MemorySlot & slot, ::mlir::Value defaultValue, ::mlir::OpBuilder & builder) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).handlePromotionComplete(slot, defaultValue, builder);
}
template<typename ConcreteOp>
::llvm::SmallVector<::mlir::MemorySlot> detail::PromotableAllocationOpInterfaceInterfaceTraits::FallbackModel<ConcreteOp>::getPromotableSlots(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return static_cast<const ConcreteOp *>(impl)->getPromotableSlots(tablegen_opaque_val);
}
template<typename ConcreteOp>
::mlir::Value detail::PromotableAllocationOpInterfaceInterfaceTraits::FallbackModel<ConcreteOp>::getDefaultValue(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, const ::mlir::MemorySlot & slot, ::mlir::OpBuilder & builder) {
  return static_cast<const ConcreteOp *>(impl)->getDefaultValue(tablegen_opaque_val, slot, builder);
}
template<typename ConcreteOp>
void detail::PromotableAllocationOpInterfaceInterfaceTraits::FallbackModel<ConcreteOp>::handleBlockArgument(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, const ::mlir::MemorySlot & slot, ::mlir::BlockArgument argument, ::mlir::OpBuilder & builder) {
  return static_cast<const ConcreteOp *>(impl)->handleBlockArgument(tablegen_opaque_val, slot, argument, builder);
}
template<typename ConcreteOp>
::std::optional<::mlir::PromotableAllocationOpInterface> detail::PromotableAllocationOpInterfaceInterfaceTraits::FallbackModel<ConcreteOp>::handlePromotionComplete(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, const ::mlir::MemorySlot & slot, ::mlir::Value defaultValue, ::mlir::OpBuilder & builder) {
  return static_cast<const ConcreteOp *>(impl)->handlePromotionComplete(tablegen_opaque_val, slot, defaultValue, builder);
}
} // namespace mlir
namespace mlir {
template<typename ConcreteOp>
bool detail::PromotableMemOpInterfaceInterfaceTraits::Model<ConcreteOp>::loadsFrom(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, const ::mlir::MemorySlot & slot) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).loadsFrom(slot);
}
template<typename ConcreteOp>
bool detail::PromotableMemOpInterfaceInterfaceTraits::Model<ConcreteOp>::storesTo(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, const ::mlir::MemorySlot & slot) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).storesTo(slot);
}
template<typename ConcreteOp>
::mlir::Value detail::PromotableMemOpInterfaceInterfaceTraits::Model<ConcreteOp>::getStored(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, const ::mlir::MemorySlot & slot, ::mlir::OpBuilder & builder, ::mlir::Value reachingDef, const ::mlir::DataLayout & dataLayout) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getStored(slot, builder, reachingDef, dataLayout);
}
template<typename ConcreteOp>
bool detail::PromotableMemOpInterfaceInterfaceTraits::Model<ConcreteOp>::canUsesBeRemoved(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, const ::mlir::MemorySlot & slot, const ::llvm::SmallPtrSetImpl<::mlir::OpOperand *> & blockingUses, ::llvm::SmallVectorImpl<::mlir::OpOperand *> & newBlockingUses, const ::mlir::DataLayout & datalayout) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).canUsesBeRemoved(slot, blockingUses, newBlockingUses, datalayout);
}
template<typename ConcreteOp>
::mlir::DeletionKind detail::PromotableMemOpInterfaceInterfaceTraits::Model<ConcreteOp>::removeBlockingUses(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, const ::mlir::MemorySlot & slot, const ::llvm::SmallPtrSetImpl<mlir::OpOperand *> & blockingUses, ::mlir::OpBuilder & builder, ::mlir::Value reachingDefinition, const ::mlir::DataLayout & dataLayout) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).removeBlockingUses(slot, blockingUses, builder, reachingDefinition, dataLayout);
}
template<typename ConcreteOp>
bool detail::PromotableMemOpInterfaceInterfaceTraits::FallbackModel<ConcreteOp>::loadsFrom(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, const ::mlir::MemorySlot & slot) {
  return static_cast<const ConcreteOp *>(impl)->loadsFrom(tablegen_opaque_val, slot);
}
template<typename ConcreteOp>
bool detail::PromotableMemOpInterfaceInterfaceTraits::FallbackModel<ConcreteOp>::storesTo(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, const ::mlir::MemorySlot & slot) {
  return static_cast<const ConcreteOp *>(impl)->storesTo(tablegen_opaque_val, slot);
}
template<typename ConcreteOp>
::mlir::Value detail::PromotableMemOpInterfaceInterfaceTraits::FallbackModel<ConcreteOp>::getStored(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, const ::mlir::MemorySlot & slot, ::mlir::OpBuilder & builder, ::mlir::Value reachingDef, const ::mlir::DataLayout & dataLayout) {
  return static_cast<const ConcreteOp *>(impl)->getStored(tablegen_opaque_val, slot, builder, reachingDef, dataLayout);
}
template<typename ConcreteOp>
bool detail::PromotableMemOpInterfaceInterfaceTraits::FallbackModel<ConcreteOp>::canUsesBeRemoved(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, const ::mlir::MemorySlot & slot, const ::llvm::SmallPtrSetImpl<::mlir::OpOperand *> & blockingUses, ::llvm::SmallVectorImpl<::mlir::OpOperand *> & newBlockingUses, const ::mlir::DataLayout & datalayout) {
  return static_cast<const ConcreteOp *>(impl)->canUsesBeRemoved(tablegen_opaque_val, slot, blockingUses, newBlockingUses, datalayout);
}
template<typename ConcreteOp>
::mlir::DeletionKind detail::PromotableMemOpInterfaceInterfaceTraits::FallbackModel<ConcreteOp>::removeBlockingUses(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, const ::mlir::MemorySlot & slot, const ::llvm::SmallPtrSetImpl<mlir::OpOperand *> & blockingUses, ::mlir::OpBuilder & builder, ::mlir::Value reachingDefinition, const ::mlir::DataLayout & dataLayout) {
  return static_cast<const ConcreteOp *>(impl)->removeBlockingUses(tablegen_opaque_val, slot, blockingUses, builder, reachingDefinition, dataLayout);
}
} // namespace mlir
namespace mlir {
template<typename ConcreteOp>
bool detail::PromotableOpInterfaceInterfaceTraits::Model<ConcreteOp>::canUsesBeRemoved(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, const ::llvm::SmallPtrSetImpl<::mlir::OpOperand *> & blockingUses, ::llvm::SmallVectorImpl<::mlir::OpOperand *> & newBlockingUses, const ::mlir::DataLayout & datalayout) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).canUsesBeRemoved(blockingUses, newBlockingUses, datalayout);
}
template<typename ConcreteOp>
::mlir::DeletionKind detail::PromotableOpInterfaceInterfaceTraits::Model<ConcreteOp>::removeBlockingUses(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, const ::llvm::SmallPtrSetImpl<mlir::OpOperand *> & blockingUses, ::mlir::OpBuilder & builder) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).removeBlockingUses(blockingUses, builder);
}
template<typename ConcreteOp>
bool detail::PromotableOpInterfaceInterfaceTraits::Model<ConcreteOp>::requiresReplacedValues(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).requiresReplacedValues();
}
template<typename ConcreteOp>
void detail::PromotableOpInterfaceInterfaceTraits::Model<ConcreteOp>::visitReplacedValues(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, ::llvm::ArrayRef<std::pair<::mlir::Operation*, ::mlir::Value>> mutatedDefs, ::mlir::OpBuilder & builder) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).visitReplacedValues(mutatedDefs, builder);
}
template<typename ConcreteOp>
bool detail::PromotableOpInterfaceInterfaceTraits::FallbackModel<ConcreteOp>::canUsesBeRemoved(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, const ::llvm::SmallPtrSetImpl<::mlir::OpOperand *> & blockingUses, ::llvm::SmallVectorImpl<::mlir::OpOperand *> & newBlockingUses, const ::mlir::DataLayout & datalayout) {
  return static_cast<const ConcreteOp *>(impl)->canUsesBeRemoved(tablegen_opaque_val, blockingUses, newBlockingUses, datalayout);
}
template<typename ConcreteOp>
::mlir::DeletionKind detail::PromotableOpInterfaceInterfaceTraits::FallbackModel<ConcreteOp>::removeBlockingUses(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, const ::llvm::SmallPtrSetImpl<mlir::OpOperand *> & blockingUses, ::mlir::OpBuilder & builder) {
  return static_cast<const ConcreteOp *>(impl)->removeBlockingUses(tablegen_opaque_val, blockingUses, builder);
}
template<typename ConcreteOp>
bool detail::PromotableOpInterfaceInterfaceTraits::FallbackModel<ConcreteOp>::requiresReplacedValues(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return static_cast<const ConcreteOp *>(impl)->requiresReplacedValues(tablegen_opaque_val);
}
template<typename ConcreteOp>
void detail::PromotableOpInterfaceInterfaceTraits::FallbackModel<ConcreteOp>::visitReplacedValues(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, ::llvm::ArrayRef<std::pair<::mlir::Operation*, ::mlir::Value>> mutatedDefs, ::mlir::OpBuilder & builder) {
  return static_cast<const ConcreteOp *>(impl)->visitReplacedValues(tablegen_opaque_val, mutatedDefs, builder);
}
template<typename ConcreteModel, typename ConcreteOp>
bool detail::PromotableOpInterfaceInterfaceTraits::ExternalModel<ConcreteModel, ConcreteOp>::requiresReplacedValues(::mlir::Operation *tablegen_opaque_val) const {
return false;
}
template<typename ConcreteModel, typename ConcreteOp>
void detail::PromotableOpInterfaceInterfaceTraits::ExternalModel<ConcreteModel, ConcreteOp>::visitReplacedValues(::mlir::Operation *tablegen_opaque_val, ::llvm::ArrayRef<std::pair<::mlir::Operation*, ::mlir::Value>> mutatedDefs, ::mlir::OpBuilder &builder) const {
return;
}
} // namespace mlir
namespace mlir {
template<typename ConcreteOp>
::llvm::SmallVector<::mlir::DestructurableMemorySlot> detail::DestructurableAllocationOpInterfaceInterfaceTraits::Model<ConcreteOp>::getDestructurableSlots(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getDestructurableSlots();
}
template<typename ConcreteOp>
::llvm::DenseMap<::mlir::Attribute, ::mlir::MemorySlot> detail::DestructurableAllocationOpInterfaceInterfaceTraits::Model<ConcreteOp>::destructure(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, const ::mlir::DestructurableMemorySlot & slot, const ::llvm::SmallPtrSetImpl<::mlir::Attribute> & usedIndices, ::mlir::OpBuilder & builder, ::mlir::SmallVectorImpl<::mlir::DestructurableAllocationOpInterface> & newAllocators) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).destructure(slot, usedIndices, builder, newAllocators);
}
template<typename ConcreteOp>
::std::optional<::mlir::DestructurableAllocationOpInterface> detail::DestructurableAllocationOpInterfaceInterfaceTraits::Model<ConcreteOp>::handleDestructuringComplete(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, const ::mlir::DestructurableMemorySlot & slot, ::mlir::OpBuilder & builder) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).handleDestructuringComplete(slot, builder);
}
template<typename ConcreteOp>
::llvm::SmallVector<::mlir::DestructurableMemorySlot> detail::DestructurableAllocationOpInterfaceInterfaceTraits::FallbackModel<ConcreteOp>::getDestructurableSlots(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return static_cast<const ConcreteOp *>(impl)->getDestructurableSlots(tablegen_opaque_val);
}
template<typename ConcreteOp>
::llvm::DenseMap<::mlir::Attribute, ::mlir::MemorySlot> detail::DestructurableAllocationOpInterfaceInterfaceTraits::FallbackModel<ConcreteOp>::destructure(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, const ::mlir::DestructurableMemorySlot & slot, const ::llvm::SmallPtrSetImpl<::mlir::Attribute> & usedIndices, ::mlir::OpBuilder & builder, ::mlir::SmallVectorImpl<::mlir::DestructurableAllocationOpInterface> & newAllocators) {
  return static_cast<const ConcreteOp *>(impl)->destructure(tablegen_opaque_val, slot, usedIndices, builder, newAllocators);
}
template<typename ConcreteOp>
::std::optional<::mlir::DestructurableAllocationOpInterface> detail::DestructurableAllocationOpInterfaceInterfaceTraits::FallbackModel<ConcreteOp>::handleDestructuringComplete(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, const ::mlir::DestructurableMemorySlot & slot, ::mlir::OpBuilder & builder) {
  return static_cast<const ConcreteOp *>(impl)->handleDestructuringComplete(tablegen_opaque_val, slot, builder);
}
} // namespace mlir
namespace mlir {
template<typename ConcreteOp>
::llvm::LogicalResult detail::SafeMemorySlotAccessOpInterfaceInterfaceTraits::Model<ConcreteOp>::ensureOnlySafeAccesses(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, const ::mlir::MemorySlot & slot, ::mlir::SmallVectorImpl<::mlir::MemorySlot> & mustBeSafelyUsed, const ::mlir::DataLayout & dataLayout) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).ensureOnlySafeAccesses(slot, mustBeSafelyUsed, dataLayout);
}
template<typename ConcreteOp>
::llvm::LogicalResult detail::SafeMemorySlotAccessOpInterfaceInterfaceTraits::FallbackModel<ConcreteOp>::ensureOnlySafeAccesses(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, const ::mlir::MemorySlot & slot, ::mlir::SmallVectorImpl<::mlir::MemorySlot> & mustBeSafelyUsed, const ::mlir::DataLayout & dataLayout) {
  return static_cast<const ConcreteOp *>(impl)->ensureOnlySafeAccesses(tablegen_opaque_val, slot, mustBeSafelyUsed, dataLayout);
}
} // namespace mlir
namespace mlir {
template<typename ConcreteOp>
bool detail::DestructurableAccessorOpInterfaceInterfaceTraits::Model<ConcreteOp>::canRewire(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, const ::mlir::DestructurableMemorySlot & slot, ::llvm::SmallPtrSetImpl<::mlir::Attribute> & usedIndices, ::mlir::SmallVectorImpl<::mlir::MemorySlot> & mustBeSafelyUsed, const ::mlir::DataLayout & dataLayout) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).canRewire(slot, usedIndices, mustBeSafelyUsed, dataLayout);
}
template<typename ConcreteOp>
::mlir::DeletionKind detail::DestructurableAccessorOpInterfaceInterfaceTraits::Model<ConcreteOp>::rewire(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, const ::mlir::DestructurableMemorySlot & slot, ::llvm::DenseMap<::mlir::Attribute, ::mlir::MemorySlot> & subslots, ::mlir::OpBuilder & builder, const ::mlir::DataLayout & dataLayout) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).rewire(slot, subslots, builder, dataLayout);
}
template<typename ConcreteOp>
bool detail::DestructurableAccessorOpInterfaceInterfaceTraits::FallbackModel<ConcreteOp>::canRewire(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, const ::mlir::DestructurableMemorySlot & slot, ::llvm::SmallPtrSetImpl<::mlir::Attribute> & usedIndices, ::mlir::SmallVectorImpl<::mlir::MemorySlot> & mustBeSafelyUsed, const ::mlir::DataLayout & dataLayout) {
  return static_cast<const ConcreteOp *>(impl)->canRewire(tablegen_opaque_val, slot, usedIndices, mustBeSafelyUsed, dataLayout);
}
template<typename ConcreteOp>
::mlir::DeletionKind detail::DestructurableAccessorOpInterfaceInterfaceTraits::FallbackModel<ConcreteOp>::rewire(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, const ::mlir::DestructurableMemorySlot & slot, ::llvm::DenseMap<::mlir::Attribute, ::mlir::MemorySlot> & subslots, ::mlir::OpBuilder & builder, const ::mlir::DataLayout & dataLayout) {
  return static_cast<const ConcreteOp *>(impl)->rewire(tablegen_opaque_val, slot, subslots, builder, dataLayout);
}
} // namespace mlir
