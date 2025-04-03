//===- SideEffectInterfaces.h - SideEffect in MLIR --------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains traits, interfaces, and utilities for defining and
// querying the side effects of an operation.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_INTERFACES_SIDEEFFECTINTERFACES_H
#define MLIR_INTERFACES_SIDEEFFECTINTERFACES_H

#include "mlir/IR/OpDefinition.h"

namespace mlir {
namespace SideEffects {
//===----------------------------------------------------------------------===//
// Effects
//===----------------------------------------------------------------------===//

/// This class represents a base class for a specific effect type.
class Effect {
public:
  /// This base class is used for derived effects that are non-parametric.
  template <typename DerivedEffect, typename BaseEffect = Effect>
  class Base : public BaseEffect {
  public:
    using BaseT = Base<DerivedEffect>;

    /// Return the unique identifier for the base effects class.
    static TypeID getEffectID() { return TypeID::get<DerivedEffect>(); }

    /// 'classof' used to support llvm style cast functionality.
    static bool classof(const ::mlir::SideEffects::Effect *effect) {
      return effect->getEffectID() == BaseT::getEffectID();
    }

    /// Returns a unique instance for the derived effect class.
    static DerivedEffect *get() {
      return BaseEffect::template get<DerivedEffect>();
    }
    using BaseEffect::get;

  protected:
    Base() : BaseEffect(BaseT::getEffectID()) {}
  };

  /// Return the unique identifier for the base effects class.
  TypeID getEffectID() const { return id; }

  /// Returns a unique instance for the given effect class.
  template <typename DerivedEffect>
  static DerivedEffect *get() {
    static_assert(std::is_base_of<Effect, DerivedEffect>::value,
                  "expected DerivedEffect to inherit from Effect");

    static DerivedEffect instance;
    return &instance;
  }

protected:
  Effect(TypeID id) : id(id) {}

private:
  /// The id of the derived effect class.
  TypeID id;
};

//===----------------------------------------------------------------------===//
// Resources
//===----------------------------------------------------------------------===//

/// This class represents a specific resource that an effect applies to. This
/// class represents an abstract interface for a given resource.
class Resource {
public:
  virtual ~Resource() = default;

  /// This base class is used for derived effects that are non-parametric.
  template <typename DerivedResource, typename BaseResource = Resource>
  class Base : public BaseResource {
  public:
    using BaseT = Base<DerivedResource>;

    /// Returns a unique instance for the given effect class.
    static DerivedResource *get() {
      static DerivedResource instance;
      return &instance;
    }

    /// Return the unique identifier for the base resource class.
    static TypeID getResourceID() { return TypeID::get<DerivedResource>(); }

    /// 'classof' used to support llvm style cast functionality.
    static bool classof(const Resource *resource) {
      return resource->getResourceID() == BaseT::getResourceID();
    }

  protected:
    Base() : BaseResource(BaseT::getResourceID()){};
  };

  /// Return the unique identifier for the base resource class.
  TypeID getResourceID() const { return id; }

  /// Return a string name of the resource.
  virtual StringRef getName() = 0;

protected:
  Resource(TypeID id) : id(id) {}

private:
  /// The id of the derived resource class.
  TypeID id;
};

/// A conservative default resource kind.
struct DefaultResource : public Resource::Base<DefaultResource> {
  StringRef getName() final { return "<Default>"; }
};

/// An automatic allocation-scope resource that is valid in the context of a
/// parent AutomaticAllocationScope trait.
struct AutomaticAllocationScopeResource
    : public Resource::Base<AutomaticAllocationScopeResource> {
  StringRef getName() final { return "AutomaticAllocationScope"; }
};

/// This class represents a specific instance of an effect. It contains the
/// effect being applied, a resource that corresponds to where the effect is
/// applied, and an optional symbol reference or value(either operand, result,
/// or region entry argument) that the effect is applied to, and an optional
/// parameters attribute further specifying the details of the effect.
template <typename EffectT>
class EffectInstance {
public:
  EffectInstance(EffectT *effect, Resource *resource = DefaultResource::get())
      : effect(effect), resource(resource), stage(0),
        effectOnFullRegion(false) {}
  EffectInstance(EffectT *effect, int stage, bool effectOnFullRegion,
                 Resource *resource = DefaultResource::get())
      : effect(effect), resource(resource), stage(stage),
        effectOnFullRegion(effectOnFullRegion) {}
  template <typename T,
            std::enable_if_t<
                llvm::is_one_of<T, OpOperand *, OpResult, BlockArgument>::value,
                bool> = true>
  EffectInstance(EffectT *effect, T value,
                 Resource *resource = DefaultResource::get())
      : effect(effect), resource(resource), value(value), stage(0),
        effectOnFullRegion(false) {}
  template <typename T,
            std::enable_if_t<
                llvm::is_one_of<T, OpOperand *, OpResult, BlockArgument>::value,
                bool> = true>
  EffectInstance(EffectT *effect, T value, int stage, bool effectOnFullRegion,
                 Resource *resource = DefaultResource::get())
      : effect(effect), resource(resource), value(value), stage(stage),
        effectOnFullRegion(effectOnFullRegion) {}
  EffectInstance(EffectT *effect, SymbolRefAttr symbol,
                 Resource *resource = DefaultResource::get())
      : effect(effect), resource(resource), value(symbol), stage(0),
        effectOnFullRegion(false) {}
  EffectInstance(EffectT *effect, SymbolRefAttr symbol, int stage,
                 bool effectOnFullRegion,
                 Resource *resource = DefaultResource::get())
      : effect(effect), resource(resource), value(symbol), stage(stage),
        effectOnFullRegion(effectOnFullRegion) {}
  EffectInstance(EffectT *effect, Attribute parameters,
                 Resource *resource = DefaultResource::get())
      : effect(effect), resource(resource), parameters(parameters), stage(0),
        effectOnFullRegion(false) {}
  EffectInstance(EffectT *effect, Attribute parameters, int stage,
                 bool effectOnFullRegion,
                 Resource *resource = DefaultResource::get())
      : effect(effect), resource(resource), parameters(parameters),
        stage(stage), effectOnFullRegion(effectOnFullRegion) {}
  template <typename T,
            std::enable_if_t<
                llvm::is_one_of<T, OpOperand *, OpResult, BlockArgument>::value,
                bool> = true>
  EffectInstance(EffectT *effect, T value, Attribute parameters,
                 Resource *resource = DefaultResource::get())
      : effect(effect), resource(resource), value(value),
        parameters(parameters), stage(0), effectOnFullRegion(false) {}
  template <typename T,
            std::enable_if_t<
                llvm::is_one_of<T, OpOperand *, OpResult, BlockArgument>::value,
                bool> = true>
  EffectInstance(EffectT *effect, T value, Attribute parameters, int stage,
                 bool effectOnFullRegion,
                 Resource *resource = DefaultResource::get())
      : effect(effect), resource(resource), value(value),
        parameters(parameters), stage(stage),
        effectOnFullRegion(effectOnFullRegion) {}
  EffectInstance(EffectT *effect, SymbolRefAttr symbol, Attribute parameters,
                 Resource *resource = DefaultResource::get())
      : effect(effect), resource(resource), value(symbol),
        parameters(parameters), stage(0), effectOnFullRegion(false) {}
  EffectInstance(EffectT *effect, SymbolRefAttr symbol, Attribute parameters,
                 int stage, bool effectOnFullRegion,
                 Resource *resource = DefaultResource::get())
      : effect(effect), resource(resource), value(symbol),
        parameters(parameters), stage(stage),
        effectOnFullRegion(effectOnFullRegion) {}

  /// Return the effect being applied.
  EffectT *getEffect() const { return effect; }

  /// Return the value the effect is applied on, or nullptr if there isn't a
  /// known value being affected.
  Value getValue() const {
    if (!value || llvm::isa_and_present<SymbolRefAttr>(value)) {
      return Value();
    }
    if (OpOperand *operand = llvm::dyn_cast_if_present<OpOperand *>(value)) {
      return operand->get();
    }
    if (OpResult result = llvm::dyn_cast_if_present<OpResult>(value)) {
      return result;
    }
    return cast_if_present<BlockArgument>(value);
  }

  /// Returns the OpOperand effect is applied on, or nullptr if there isn't a
  /// known value being effected.
  template <typename T,
            std::enable_if_t<
                llvm::is_one_of<T, OpOperand *, OpResult, BlockArgument>::value,
                bool> = true>
  T getEffectValue() const {
    return value ? dyn_cast_if_present<T>(value) : nullptr;
  }

  /// Return the symbol reference the effect is applied on, or nullptr if there
  /// isn't a known smbol being affected.
  SymbolRefAttr getSymbolRef() const {
    return value ? llvm::dyn_cast_if_present<SymbolRefAttr>(value)
                 : SymbolRefAttr();
  }

  /// Return the resource that the effect applies to.
  Resource *getResource() const { return resource; }

  /// Return the parameters of the effect, if any.
  Attribute getParameters() const { return parameters; }

  /// Return the effect happen stage.
  int getStage() const { return stage; }

  /// Return if this side effect act on every single value of resource.
  bool getEffectOnFullRegion() const { return effectOnFullRegion; }

private:
  /// The specific effect being applied.
  EffectT *effect;

  /// The resource that the given value resides in.
  Resource *resource;

  /// The Symbol, OpOperand, OpResult or BlockArgument that the effect applies
  /// to. This is optionally null.
  PointerUnion<SymbolRefAttr, OpOperand *, OpResult, BlockArgument> value;

  /// Additional parameters of the effect instance. An attribute is used for
  /// type-safe structured storage and context-based uniquing. Concrete effects
  /// can use this at their convenience. This is optionally null.
  Attribute parameters;

  // The stage side effect happen. Side effect with a lower stage
  // number happen earlier than those with a higher stage number
  int stage;

  // Does this side effect act on every single value of resource.
  bool effectOnFullRegion;
};
} // namespace SideEffects

namespace Speculation {
/// This enum is returned from the `getSpeculatability` method in the
/// `ConditionallySpeculatable` op interface.
enum class Speculatability {
  /// The Operation in question cannot be speculatively executed.  This could be
  /// because it may invoke undefined behavior or have other side effects.
  NotSpeculatable,

  // The Operation in question can be speculatively executed.  It does not have
  // any side effects or undefined behavior.
  Speculatable,

  // The Operation in question can be speculatively executed if all the
  // operations in all attached regions can also be speculatively executed.
  RecursivelySpeculatable,
};

constexpr auto NotSpeculatable = Speculatability::NotSpeculatable;
constexpr auto Speculatable = Speculatability::Speculatable;
constexpr auto RecursivelySpeculatable =
    Speculatability::RecursivelySpeculatable;
} // namespace Speculation

//===----------------------------------------------------------------------===//
// SideEffect Traits
//===----------------------------------------------------------------------===//

namespace OpTrait {
/// This trait indicates that the memory effects of an operation includes the
/// effects of operations nested within its regions. If the operation has no
/// derived effects interfaces, the operation itself can be assumed to have no
/// memory effects.
template <typename ConcreteType>
class HasRecursiveMemoryEffects
    : public TraitBase<ConcreteType, HasRecursiveMemoryEffects> {};

/// This trait marks an op (which must be tagged as implementing the
/// ConditionallySpeculatable interface) as being recursively speculatable.
/// This means that said op can be speculated only if all the instructions in
/// all the regions attached to the op can be speculated.
template <typename ConcreteType>
struct RecursivelySpeculatableImplTrait
    : public TraitBase<ConcreteType, RecursivelySpeculatableImplTrait> {

  Speculation::Speculatability getSpeculatability() {
    return Speculation::RecursivelySpeculatable;
  }
};

/// This trait marks an op (which must be tagged as implementing the
/// ConditionallySpeculatable interface) as being always speculatable.
template <typename ConcreteType>
struct AlwaysSpeculatableImplTrait
    : public TraitBase<ConcreteType, AlwaysSpeculatableImplTrait> {

  Speculation::Speculatability getSpeculatability() {
    return Speculation::Speculatable;
  }
};
} // namespace OpTrait

//===----------------------------------------------------------------------===//
// Operation Memory-Effect Modeling
//===----------------------------------------------------------------------===//

namespace MemoryEffects {
/// This class represents the base class used for memory effects.
struct Effect : public SideEffects::Effect {
  using SideEffects::Effect::Effect;

  /// A base class for memory effects that provides helper utilities.
  template <typename DerivedEffect>
  using Base = SideEffects::Effect::Base<DerivedEffect, Effect>;

  static bool classof(const SideEffects::Effect *effect);
};
using EffectInstance = SideEffects::EffectInstance<Effect>;

/// The following effect indicates that the operation allocates from some
/// resource. An 'allocate' effect implies only allocation of the resource, and
/// not any visible mutation or dereference.
struct Allocate : public Effect::Base<Allocate> {};

/// The following effect indicates that the operation frees some resource that
/// has been allocated. An 'allocate' effect implies only de-allocation of the
/// resource, and not any visible allocation, mutation or dereference.
struct Free : public Effect::Base<Free> {};

/// The following effect indicates that the operation reads from some resource.
/// A 'read' effect implies only dereferencing of the resource, and not any
/// visible mutation.
struct Read : public Effect::Base<Read> {};

/// The following effect indicates that the operation writes to some resource. A
/// 'write' effect implies only mutating a resource, and not any visible
/// dereference or read.
struct Write : public Effect::Base<Write> {};
} // namespace MemoryEffects

//===----------------------------------------------------------------------===//
// SideEffect Utilities
//===----------------------------------------------------------------------===//

/// Returns true if `op` has only an effect of type `EffectTy`.
template <typename EffectTy>
bool hasSingleEffect(Operation *op);

/// Returns true if `op` has only an effect of type `EffectTy` (and of no other
/// type) on `value`.
template <typename EffectTy>
bool hasSingleEffect(Operation *op, Value value);

/// Returns true if `op` has only an effect of type `EffectTy` (and of no other
/// type) on `value` of type `ValueTy`.
template <typename ValueTy, typename EffectTy>
bool hasSingleEffect(Operation *op, ValueTy value);

/// Returns true if `op` has an effect of type `EffectTy`.
template <typename... EffectTys>
bool hasEffect(Operation *op);

/// Returns true if `op` has an effect of type `EffectTy` on `value`.
template <typename... EffectTys>
bool hasEffect(Operation *op, Value value);

/// Returns true if `op` has an effect of type `EffectTy` on `value` of type
/// `ValueTy`.
template <typename ValueTy, typename... EffectTys>
bool hasEffect(Operation *op, ValueTy value);

/// Return true if the given operation is unused, and has no side effects on
/// memory that prevent erasing.
bool isOpTriviallyDead(Operation *op);

/// Return true if the given operation would be dead if unused, and has no side
/// effects on memory that would prevent erasing. This is equivalent to checking
/// `isOpTriviallyDead` if `op` was unused.
///
/// Note: Terminators and symbols are never considered to be trivially dead.
bool wouldOpBeTriviallyDead(Operation *op);

/// Returns true if the given operation is free of memory effects.
///
/// An operation is free of memory effects if its implementation of
/// `MemoryEffectOpInterface` indicates that it has no memory effects. For
/// example, it may implement `NoMemoryEffect` in ODS. Alternatively, if the
/// operation has the `HasRecursiveMemoryEffects` trait, then it is free of
/// memory effects if all of its nested operations are free of memory effects.
///
/// If the operation has both, then it is free of memory effects if both
/// conditions are satisfied.
bool isMemoryEffectFree(Operation *op);

/// Returns the side effects of an operation. If the operation has
/// RecursiveMemoryEffects, include all side effects of child operations.
///
/// std::nullopt indicates that an option did not have a memory effect interface
/// and so no result could be obtained. An empty vector indicates that there
/// were no memory effects found (but every operation implemented the memory
/// effect interface or has RecursiveMemoryEffects). If the vector contains
/// multiple effects, these effects may be duplicates.
std::optional<llvm::SmallVector<MemoryEffects::EffectInstance>>
getEffectsRecursively(Operation *rootOp);

/// Returns true if the given operation is speculatable, i.e. has no undefined
/// behavior or other side effects.
///
/// An operation can indicate that it is speculatable by implementing the
/// getSpeculatability hook in the ConditionallySpeculatable op interface.
bool isSpeculatable(Operation *op);

/// Returns true if the given operation is pure, i.e., is speculatable that does
/// not touch memory.
///
/// This function is the C++ equivalent of the `Pure` trait.
bool isPure(Operation *op);

} // namespace mlir

//===----------------------------------------------------------------------===//
// SideEffect Interfaces
//===----------------------------------------------------------------------===//

/// Include the definitions of the side effect interfaces.
#include "mlir/Interfaces/SideEffectInterfaces.h.inc"

#endif // MLIR_INTERFACES_SIDEEFFECTINTERFACES_H
