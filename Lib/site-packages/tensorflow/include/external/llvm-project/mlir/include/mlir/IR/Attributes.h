//===- Attributes.h - MLIR Attribute Classes --------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_IR_ATTRIBUTES_H
#define MLIR_IR_ATTRIBUTES_H

#include "mlir/IR/AttributeSupport.h"
#include "llvm/Support/PointerLikeTypeTraits.h"

namespace mlir {
class AsmState;
class StringAttr;

/// Attributes are known-constant values of operations.
///
/// Instances of the Attribute class are references to immortal key-value pairs
/// with immutable, uniqued keys owned by MLIRContext. As such, an Attribute is
/// a thin wrapper around an underlying storage pointer. Attributes are usually
/// passed by value.
class Attribute {
public:
  /// Utility class for implementing attributes.
  template <typename ConcreteType, typename BaseType, typename StorageType,
            template <typename T> class... Traits>
  using AttrBase = detail::StorageUserBase<ConcreteType, BaseType, StorageType,
                                           detail::AttributeUniquer, Traits...>;

  using ImplType = AttributeStorage;
  using ValueType = void;
  using AbstractTy = AbstractAttribute;

  constexpr Attribute() = default;
  /* implicit */ Attribute(const ImplType *impl)
      : impl(const_cast<ImplType *>(impl)) {}

  Attribute(const Attribute &other) = default;
  Attribute &operator=(const Attribute &other) = default;

  bool operator==(Attribute other) const { return impl == other.impl; }
  bool operator!=(Attribute other) const { return !(*this == other); }
  explicit operator bool() const { return impl; }

  bool operator!() const { return impl == nullptr; }

  /// Casting utility functions. These are deprecated and will be removed,
  /// please prefer using the `llvm` namespace variants instead.
  template <typename... Tys>
  [[deprecated("Use mlir::isa<U>() instead")]]
  bool isa() const;
  template <typename... Tys>
  [[deprecated("Use mlir::isa_and_nonnull<U>() instead")]]
  bool isa_and_nonnull() const;
  template <typename U>
  [[deprecated("Use mlir::dyn_cast<U>() instead")]]
  U dyn_cast() const;
  template <typename U>
  [[deprecated("Use mlir::dyn_cast_or_null<U>() instead")]]
  U dyn_cast_or_null() const;
  template <typename U>
  [[deprecated("Use mlir::cast<U>() instead")]]
  U cast() const;

  /// Return a unique identifier for the concrete attribute type. This is used
  /// to support dynamic type casting.
  TypeID getTypeID() { return impl->getAbstractAttribute().getTypeID(); }

  /// Return the context this attribute belongs to.
  MLIRContext *getContext() const;

  /// Get the dialect this attribute is registered to.
  Dialect &getDialect() const {
    return impl->getAbstractAttribute().getDialect();
  }

  /// Print the attribute. If `elideType` is set, the attribute is printed
  /// without a trailing colon type if it has one.
  void print(raw_ostream &os, bool elideType = false) const;
  void print(raw_ostream &os, AsmState &state, bool elideType = false) const;
  void dump() const;

  /// Print the attribute without dialect wrapping.
  void printStripped(raw_ostream &os) const;
  void printStripped(raw_ostream &os, AsmState &state) const;

  /// Get an opaque pointer to the attribute.
  const void *getAsOpaquePointer() const { return impl; }
  /// Construct an attribute from the opaque pointer representation.
  static Attribute getFromOpaquePointer(const void *ptr) {
    return Attribute(reinterpret_cast<const ImplType *>(ptr));
  }

  friend ::llvm::hash_code hash_value(Attribute arg);

  /// Returns true if `InterfaceT` has been promised by the dialect or
  /// implemented.
  template <typename InterfaceT>
  bool hasPromiseOrImplementsInterface() {
    return dialect_extension_detail::hasPromisedInterface(
               getDialect(), getTypeID(), InterfaceT::getInterfaceID()) ||
           mlir::isa<InterfaceT>(*this);
  }

  /// Returns true if the type was registered with a particular trait.
  template <template <typename T> class Trait>
  bool hasTrait() {
    return getAbstractAttribute().hasTrait<Trait>();
  }

  /// Return the abstract descriptor for this attribute.
  const AbstractTy &getAbstractAttribute() const {
    return impl->getAbstractAttribute();
  }

  /// Walk all of the immediately nested sub-attributes and sub-types. This
  /// method does not recurse into sub elements.
  void walkImmediateSubElements(function_ref<void(Attribute)> walkAttrsFn,
                                function_ref<void(Type)> walkTypesFn) const {
    getAbstractAttribute().walkImmediateSubElements(*this, walkAttrsFn,
                                                    walkTypesFn);
  }

  /// Replace the immediately nested sub-attributes and sub-types with those
  /// provided. The order of the provided elements is derived from the order of
  /// the elements returned by the callbacks of `walkImmediateSubElements`. The
  /// element at index 0 would replace the very first attribute given by
  /// `walkImmediateSubElements`. On success, the new instance with the values
  /// replaced is returned. If replacement fails, nullptr is returned.
  auto replaceImmediateSubElements(ArrayRef<Attribute> replAttrs,
                                   ArrayRef<Type> replTypes) const {
    return getAbstractAttribute().replaceImmediateSubElements(*this, replAttrs,
                                                              replTypes);
  }

  /// Walk this attribute and all attibutes/types nested within using the
  /// provided walk functions. See `AttrTypeWalker` for information on the
  /// supported walk function types.
  template <WalkOrder Order = WalkOrder::PostOrder, typename... WalkFns>
  auto walk(WalkFns &&...walkFns) {
    AttrTypeWalker walker;
    (walker.addWalk(std::forward<WalkFns>(walkFns)), ...);
    return walker.walk<Order>(*this);
  }

  /// Recursively replace all of the nested sub-attributes and sub-types using
  /// the provided map functions. Returns nullptr in the case of failure. See
  /// `AttrTypeReplacer` for information on the support replacement function
  /// types.
  template <typename... ReplacementFns>
  auto replace(ReplacementFns &&...replacementFns) {
    AttrTypeReplacer replacer;
    (replacer.addReplacement(std::forward<ReplacementFns>(replacementFns)),
     ...);
    return replacer.replace(*this);
  }

  /// Return the internal Attribute implementation.
  ImplType *getImpl() const { return impl; }

protected:
  ImplType *impl{nullptr};
};

inline raw_ostream &operator<<(raw_ostream &os, Attribute attr) {
  attr.print(os);
  return os;
}

template <typename... Tys>
bool Attribute::isa() const {
  return llvm::isa<Tys...>(*this);
}

template <typename... Tys>
bool Attribute::isa_and_nonnull() const {
  return llvm::isa_and_present<Tys...>(*this);
}

template <typename U>
U Attribute::dyn_cast() const {
  return llvm::dyn_cast<U>(*this);
}

template <typename U>
U Attribute::dyn_cast_or_null() const {
  return llvm::dyn_cast_if_present<U>(*this);
}

template <typename U>
U Attribute::cast() const {
  return llvm::cast<U>(*this);
}

inline ::llvm::hash_code hash_value(Attribute arg) {
  return DenseMapInfo<const Attribute::ImplType *>::getHashValue(arg.impl);
}

//===----------------------------------------------------------------------===//
// NamedAttribute
//===----------------------------------------------------------------------===//

/// NamedAttribute represents a combination of a name and an Attribute value.
class NamedAttribute {
public:
  NamedAttribute(StringAttr name, Attribute value);

  /// Return the name of the attribute.
  StringAttr getName() const;

  /// Return the dialect of the name of this attribute, if the name is prefixed
  /// by a dialect namespace. For example, `llvm.fast_math` would return the
  /// LLVM dialect (if it is loaded). Returns nullptr if the dialect isn't
  /// loaded, or if the name is not prefixed by a dialect namespace.
  Dialect *getNameDialect() const;

  /// Return the value of the attribute.
  Attribute getValue() const { return value; }

  /// Set the name of this attribute.
  void setName(StringAttr newName);

  /// Set the value of this attribute.
  void setValue(Attribute newValue) {
    assert(value && "expected valid attribute value");
    value = newValue;
  }

  /// Compare this attribute to the provided attribute, ordering by name.
  bool operator<(const NamedAttribute &rhs) const;
  /// Compare this attribute to the provided string, ordering by name.
  bool operator<(StringRef rhs) const;

  bool operator==(const NamedAttribute &rhs) const {
    return name == rhs.name && value == rhs.value;
  }
  bool operator!=(const NamedAttribute &rhs) const { return !(*this == rhs); }

private:
  NamedAttribute(Attribute name, Attribute value) : name(name), value(value) {}

  /// Allow access to internals to enable hashing.
  friend ::llvm::hash_code hash_value(const NamedAttribute &arg);
  friend DenseMapInfo<NamedAttribute>;

  /// The name of the attribute. This is represented as a StringAttr, but
  /// type-erased to Attribute in the field.
  Attribute name;
  /// The value of the attribute.
  Attribute value;
};

inline ::llvm::hash_code hash_value(const NamedAttribute &arg) {
  using AttrPairT = std::pair<Attribute, Attribute>;
  return DenseMapInfo<AttrPairT>::getHashValue(AttrPairT(arg.name, arg.value));
}

/// Allow walking and replacing the subelements of a NamedAttribute.
template <>
struct AttrTypeSubElementHandler<NamedAttribute> {
  template <typename T>
  static void walk(T param, AttrTypeImmediateSubElementWalker &walker) {
    walker.walk(param.getName());
    walker.walk(param.getValue());
  }
  template <typename T>
  static T replace(T param, AttrSubElementReplacements &attrRepls,
                   TypeSubElementReplacements &typeRepls) {
    ArrayRef<Attribute> paramRepls = attrRepls.take_front(2);
    return T(cast<decltype(param.getName())>(paramRepls[0]), paramRepls[1]);
  }
};

//===----------------------------------------------------------------------===//
// AttributeTraitBase
//===----------------------------------------------------------------------===//

namespace AttributeTrait {
/// This class represents the base of an attribute trait.
template <typename ConcreteType, template <typename> class TraitType>
using TraitBase = detail::StorageUserTraitBase<ConcreteType, TraitType>;
} // namespace AttributeTrait

//===----------------------------------------------------------------------===//
// AttributeInterface
//===----------------------------------------------------------------------===//

/// This class represents the base of an attribute interface. See the definition
/// of `detail::Interface` for requirements on the `Traits` type.
template <typename ConcreteType, typename Traits>
class AttributeInterface
    : public detail::Interface<ConcreteType, Attribute, Traits, Attribute,
                               AttributeTrait::TraitBase> {
public:
  using Base = AttributeInterface<ConcreteType, Traits>;
  using InterfaceBase = detail::Interface<ConcreteType, Attribute, Traits,
                                          Attribute, AttributeTrait::TraitBase>;
  using InterfaceBase::InterfaceBase;

protected:
  /// Returns the impl interface instance for the given type.
  static typename InterfaceBase::Concept *getInterfaceFor(Attribute attr) {
#ifndef NDEBUG
    // Check that the current interface isn't an unresolved promise for the
    // given attribute.
    dialect_extension_detail::handleUseOfUndefinedPromisedInterface(
        attr.getDialect(), attr.getTypeID(), ConcreteType::getInterfaceID(),
        llvm::getTypeName<ConcreteType>());
#endif

    return attr.getAbstractAttribute().getInterface<ConcreteType>();
  }

  /// Allow access to 'getInterfaceFor'.
  friend InterfaceBase;
};

//===----------------------------------------------------------------------===//
// Core AttributeTrait
//===----------------------------------------------------------------------===//

/// This trait is used to determine if an attribute is mutable or not. It is
/// attached on an attribute if the corresponding ImplType defines a `mutate`
/// function with proper signature.
namespace AttributeTrait {
template <typename ConcreteType>
using IsMutable = detail::StorageUserTrait::IsMutable<ConcreteType>;
} // namespace AttributeTrait

} // namespace mlir.

namespace llvm {

// Attribute hash just like pointers.
template <>
struct DenseMapInfo<mlir::Attribute> {
  static mlir::Attribute getEmptyKey() {
    auto *pointer = llvm::DenseMapInfo<void *>::getEmptyKey();
    return mlir::Attribute(static_cast<mlir::Attribute::ImplType *>(pointer));
  }
  static mlir::Attribute getTombstoneKey() {
    auto *pointer = llvm::DenseMapInfo<void *>::getTombstoneKey();
    return mlir::Attribute(static_cast<mlir::Attribute::ImplType *>(pointer));
  }
  static unsigned getHashValue(mlir::Attribute val) {
    return mlir::hash_value(val);
  }
  static bool isEqual(mlir::Attribute LHS, mlir::Attribute RHS) {
    return LHS == RHS;
  }
};
template <typename T>
struct DenseMapInfo<
    T, std::enable_if_t<std::is_base_of<mlir::Attribute, T>::value &&
                        !mlir::detail::IsInterface<T>::value>>
    : public DenseMapInfo<mlir::Attribute> {
  static T getEmptyKey() {
    const void *pointer = llvm::DenseMapInfo<const void *>::getEmptyKey();
    return T::getFromOpaquePointer(pointer);
  }
  static T getTombstoneKey() {
    const void *pointer = llvm::DenseMapInfo<const void *>::getTombstoneKey();
    return T::getFromOpaquePointer(pointer);
  }
};

/// Allow LLVM to steal the low bits of Attributes.
template <>
struct PointerLikeTypeTraits<mlir::Attribute> {
  static inline void *getAsVoidPointer(mlir::Attribute attr) {
    return const_cast<void *>(attr.getAsOpaquePointer());
  }
  static inline mlir::Attribute getFromVoidPointer(void *ptr) {
    return mlir::Attribute::getFromOpaquePointer(ptr);
  }
  static constexpr int NumLowBitsAvailable = llvm::PointerLikeTypeTraits<
      mlir::AttributeStorage *>::NumLowBitsAvailable;
};

template <>
struct DenseMapInfo<mlir::NamedAttribute> {
  static mlir::NamedAttribute getEmptyKey() {
    auto emptyAttr = llvm::DenseMapInfo<mlir::Attribute>::getEmptyKey();
    return mlir::NamedAttribute(emptyAttr, emptyAttr);
  }
  static mlir::NamedAttribute getTombstoneKey() {
    auto tombAttr = llvm::DenseMapInfo<mlir::Attribute>::getTombstoneKey();
    return mlir::NamedAttribute(tombAttr, tombAttr);
  }
  static unsigned getHashValue(mlir::NamedAttribute val) {
    return mlir::hash_value(val);
  }
  static bool isEqual(mlir::NamedAttribute lhs, mlir::NamedAttribute rhs) {
    return lhs == rhs;
  }
};

/// Add support for llvm style casts. We provide a cast between To and From if
/// From is mlir::Attribute or derives from it.
template <typename To, typename From>
struct CastInfo<To, From,
                std::enable_if_t<std::is_same_v<mlir::Attribute,
                                                std::remove_const_t<From>> ||
                                 std::is_base_of_v<mlir::Attribute, From>>>
    : NullableValueCastFailed<To>,
      DefaultDoCastIfPossible<To, From, CastInfo<To, From>> {
  /// Arguments are taken as mlir::Attribute here and not as `From`, because
  /// when casting from an intermediate type of the hierarchy to one of its
  /// children, the val.getTypeID() inside T::classof will use the static
  /// getTypeID of the parent instead of the non-static Type::getTypeID that
  /// returns the dynamic ID. This means that T::classof would end up comparing
  /// the static TypeID of the children to the static TypeID of its parent,
  /// making it impossible to downcast from the parent to the child.
  static inline bool isPossible(mlir::Attribute ty) {
    /// Return a constant true instead of a dynamic true when casting to self or
    /// up the hierarchy.
    if constexpr (std::is_base_of_v<To, From>) {
      return true;
    } else {
      return To::classof(ty);
    }
  }
  static inline To doCast(mlir::Attribute attr) { return To(attr.getImpl()); }
};

} // namespace llvm

#endif
