//===- InterfaceSupport.h - MLIR Interface Support Classes ------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines several support classes for defining interfaces.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_SUPPORT_INTERFACESUPPORT_H
#define MLIR_SUPPORT_INTERFACESUPPORT_H

#include "mlir/Support/TypeID.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/Support/TypeName.h"

namespace mlir {
namespace detail {
//===----------------------------------------------------------------------===//
// Interface
//===----------------------------------------------------------------------===//

/// This class represents an abstract interface. An interface is a simplified
/// mechanism for attaching concept based polymorphism to a class hierarchy. An
/// interface is comprised of two components:
/// * The derived interface class: This is what users interact with, and invoke
///   methods on.
/// * An interface `Trait` class: This is the class that is attached to the
///   object implementing the interface. It is the mechanism with which models
///   are specialized.
///
/// Derived interfaces types must provide the following template types:
/// * ConcreteType: The CRTP derived type.
/// * ValueT: The opaque type the derived interface operates on. For example
///           `Operation*` for operation interfaces, or `Attribute` for
///           attribute interfaces.
/// * Traits: A class that contains definitions for a 'Concept' and a 'Model'
///           class. The 'Concept' class defines an abstract virtual interface,
///           where as the 'Model' class implements this interface for a
///           specific derived T type. Both of these classes *must* not contain
///           non-static data. A simple example is shown below:
///
/// ```c++
///    struct ExampleInterfaceTraits {
///      struct Concept {
///        virtual unsigned getNumInputs(T t) const = 0;
///      };
///      template <typename DerivedT> class Model {
///        unsigned getNumInputs(T t) const final {
///          return cast<DerivedT>(t).getNumInputs();
///        }
///      };
///    };
/// ```
///
/// * BaseType: A desired base type for the interface. This is a class
///             that provides specific functionality for the `ValueT`
///             value. For instance the specific `Op` that will wrap the
///             `Operation*` for an `OpInterface`.
/// * BaseTrait: The base type for the interface trait. This is the base class
///              to use for the interface trait that will be attached to each
///              instance of `ValueT` that implements this interface.
///
template <typename ConcreteType, typename ValueT, typename Traits,
          typename BaseType,
          template <typename, template <typename> class> class BaseTrait>
class Interface : public BaseType {
public:
  using Concept = typename Traits::Concept;
  template <typename T>
  using Model = typename Traits::template Model<T>;
  template <typename T>
  using FallbackModel = typename Traits::template FallbackModel<T>;
  using InterfaceBase =
      Interface<ConcreteType, ValueT, Traits, BaseType, BaseTrait>;
  template <typename T, typename U>
  using ExternalModel = typename Traits::template ExternalModel<T, U>;
  using ValueType = ValueT;

  /// This is a special trait that registers a given interface with an object.
  template <typename ConcreteT>
  struct Trait : public BaseTrait<ConcreteT, Trait> {
    using ModelT = Model<ConcreteT>;

    /// Define an accessor for the ID of this interface.
    static TypeID getInterfaceID() { return TypeID::get<ConcreteType>(); }
  };

  /// Construct an interface from an instance of the value type.
  explicit Interface(ValueT t = ValueT())
      : BaseType(t),
        conceptImpl(t ? ConcreteType::getInterfaceFor(t) : nullptr) {
    assert((!t || conceptImpl) &&
           "expected value to provide interface instance");
  }
  Interface(std::nullptr_t) : BaseType(ValueT()), conceptImpl(nullptr) {}

  /// Construct an interface instance from a type that implements this
  /// interface's trait.
  template <typename T,
            std::enable_if_t<std::is_base_of<Trait<T>, T>::value> * = nullptr>
  Interface(T t)
      : BaseType(t),
        conceptImpl(t ? ConcreteType::getInterfaceFor(t) : nullptr) {
    assert((!t || conceptImpl) &&
           "expected value to provide interface instance");
  }

  /// Constructor for a known concept.
  Interface(ValueT t, const Concept *conceptImpl)
      : BaseType(t), conceptImpl(const_cast<Concept *>(conceptImpl)) {
    assert(!t || ConcreteType::getInterfaceFor(t) == conceptImpl);
  }

  /// Constructor for DenseMapInfo's empty key and tombstone key.
  Interface(ValueT t, std::nullptr_t) : BaseType(t), conceptImpl(nullptr) {}

  /// Support 'classof' by checking if the given object defines the concrete
  /// interface.
  static bool classof(ValueT t) { return ConcreteType::getInterfaceFor(t); }

  /// Define an accessor for the ID of this interface.
  static TypeID getInterfaceID() { return TypeID::get<ConcreteType>(); }

protected:
  /// Get the raw concept in the correct derived concept type.
  const Concept *getImpl() const { return conceptImpl; }
  Concept *getImpl() { return conceptImpl; }

private:
  /// A pointer to the impl concept object.
  Concept *conceptImpl;
};

//===----------------------------------------------------------------------===//
// InterfaceMap
//===----------------------------------------------------------------------===//

/// Template utility that computes the number of elements within `T` that
/// satisfy the given predicate.
template <template <class> class Pred, size_t N, typename... Ts>
struct count_if_t_impl : public std::integral_constant<size_t, N> {};
template <template <class> class Pred, size_t N, typename T, typename... Us>
struct count_if_t_impl<Pred, N, T, Us...>
    : public std::integral_constant<
          size_t,
          count_if_t_impl<Pred, N + (Pred<T>::value ? 1 : 0), Us...>::value> {};
template <template <class> class Pred, typename... Ts>
using count_if_t = count_if_t_impl<Pred, 0, Ts...>;

/// This class provides an efficient mapping between a given `Interface` type,
/// and a particular implementation of its concept.
class InterfaceMap {
  /// Trait to check if T provides a static 'getInterfaceID' method.
  template <typename T, typename... Args>
  using has_get_interface_id = decltype(T::getInterfaceID());
  template <typename T>
  using detect_get_interface_id = llvm::is_detected<has_get_interface_id, T>;
  template <typename... Types>
  using num_interface_types_t = count_if_t<detect_get_interface_id, Types...>;

  /// Trait to check if T provides a 'initializeInterfaceConcept' method.
  template <typename T, typename... Args>
  using has_initialize_method =
      decltype(std::declval<T>().initializeInterfaceConcept(
          std::declval<InterfaceMap &>()));
  template <typename T>
  using detect_initialize_method = llvm::is_detected<has_initialize_method, T>;

public:
  InterfaceMap() = default;
  InterfaceMap(InterfaceMap &&) = default;
  InterfaceMap &operator=(InterfaceMap &&rhs) {
    for (auto &it : interfaces)
      free(it.second);
    interfaces = std::move(rhs.interfaces);
    return *this;
  }
  ~InterfaceMap() {
    for (auto &it : interfaces)
      free(it.second);
  }

  /// Construct an InterfaceMap with the given set of template types. For
  /// convenience given that object trait lists may contain other non-interface
  /// types, not all of the types need to be interfaces. The provided types that
  /// do not represent interfaces are not added to the interface map.
  template <typename... Types>
  static InterfaceMap get() {
    constexpr size_t numInterfaces = num_interface_types_t<Types...>::value;
    if constexpr (numInterfaces == 0)
      return InterfaceMap();

    InterfaceMap map;
    (map.insertPotentialInterface<Types>(), ...);
    return map;
  }

  /// Returns an instance of the concept object for the given interface if it
  /// was registered to this map, null otherwise.
  template <typename T>
  typename T::Concept *lookup() const {
    return reinterpret_cast<typename T::Concept *>(lookup(T::getInterfaceID()));
  }

  /// Returns true if the interface map contains an interface for the given id.
  bool contains(TypeID interfaceID) const { return lookup(interfaceID); }

  /// Insert the given interface models.
  template <typename... IfaceModels>
  void insertModels() {
    (insertModel<IfaceModels>(), ...);
  }

private:
  /// Insert the given interface type into the map, ignoring it if it doesn't
  /// actually represent an interface.
  template <typename T>
  inline void insertPotentialInterface() {
    if constexpr (detect_get_interface_id<T>::value)
      insertModel<typename T::ModelT>();
  }

  /// Insert the given interface model into the map.
  template <typename InterfaceModel>
  void insertModel() {
    // FIXME(#59975): Uncomment this when SPIRV no longer awkwardly reimplements
    // interfaces in a way that isn't clean/compatible.
    // static_assert(std::is_trivially_destructible_v<InterfaceModel>,
    //               "interface models must be trivially destructible");

    // Build the interface model, optionally initializing if necessary.
    InterfaceModel *model =
        new (malloc(sizeof(InterfaceModel))) InterfaceModel();
    if constexpr (detect_initialize_method<InterfaceModel>::value)
      model->initializeInterfaceConcept(*this);

    insert(InterfaceModel::Interface::getInterfaceID(), model);
  }
  /// Insert the given set of interface id and concept implementation into the
  /// interface map.
  void insert(TypeID interfaceId, void *conceptImpl);

  /// Compare two TypeID instances by comparing the underlying pointer.
  static bool compare(TypeID lhs, TypeID rhs) {
    return lhs.getAsOpaquePointer() < rhs.getAsOpaquePointer();
  }

  /// Returns an instance of the concept object for the given interface id if it
  /// was registered to this map, null otherwise.
  void *lookup(TypeID id) const {
    const auto *it =
        llvm::lower_bound(interfaces, id, [](const auto &it, TypeID id) {
          return compare(it.first, id);
        });
    return (it != interfaces.end() && it->first == id) ? it->second : nullptr;
  }

  /// A list of interface instances, sorted by TypeID.
  SmallVector<std::pair<TypeID, void *>> interfaces;
};

template <typename ConcreteType, typename ValueT, typename Traits,
          typename BaseType,
          template <typename, template <typename> class> class BaseTrait>
void isInterfaceImpl(
    Interface<ConcreteType, ValueT, Traits, BaseType, BaseTrait> &);

template <typename T>
using is_interface_t = decltype(isInterfaceImpl(std::declval<T &>()));

template <typename T>
using IsInterface = llvm::is_detected<is_interface_t, T>;

} // namespace detail
} // namespace mlir

namespace llvm {

template <typename T>
struct DenseMapInfo<T, std::enable_if_t<mlir::detail::IsInterface<T>::value>> {
  using ValueTypeInfo = llvm::DenseMapInfo<typename T::ValueType>;

  static T getEmptyKey() { return T(ValueTypeInfo::getEmptyKey(), nullptr); }

  static T getTombstoneKey() {
    return T(ValueTypeInfo::getTombstoneKey(), nullptr);
  }

  static unsigned getHashValue(T val) {
    return ValueTypeInfo::getHashValue(val);
  }

  static bool isEqual(T lhs, T rhs) { return ValueTypeInfo::isEqual(lhs, rhs); }
};

} // namespace llvm

#endif
