//===- StorageUniquerSupport.h - MLIR Storage Uniquer Utilities -*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines utility classes for interfacing with StorageUniquer.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_IR_STORAGEUNIQUERSUPPORT_H
#define MLIR_IR_STORAGEUNIQUERSUPPORT_H

#include "mlir/IR/AttrTypeSubElements.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/Support/InterfaceSupport.h"
#include "mlir/Support/StorageUniquer.h"
#include "mlir/Support/TypeID.h"
#include "llvm/ADT/FunctionExtras.h"

namespace mlir {
class InFlightDiagnostic;
class Location;
class MLIRContext;

namespace detail {
/// Utility method to generate a callback that can be used to generate a
/// diagnostic when checking the construction invariants of a storage object.
/// This is defined out-of-line to avoid the need to include Location.h.
llvm::unique_function<InFlightDiagnostic()>
getDefaultDiagnosticEmitFn(MLIRContext *ctx);
llvm::unique_function<InFlightDiagnostic()>
getDefaultDiagnosticEmitFn(const Location &loc);

//===----------------------------------------------------------------------===//
// StorageUserTraitBase
//===----------------------------------------------------------------------===//

/// Helper class for implementing traits for storage classes. Clients are not
/// expected to interact with this directly, so its members are all protected.
template <typename ConcreteType, template <typename> class TraitType>
class StorageUserTraitBase {
protected:
  /// Return the derived instance.
  ConcreteType getInstance() const {
    // We have to cast up to the trait type, then to the concrete type because
    // the concrete type will multiply derive from the (content free) TraitBase
    // class, and we need to be able to disambiguate the path for the C++
    // compiler.
    auto *trait = static_cast<const TraitType<ConcreteType> *>(this);
    return *static_cast<const ConcreteType *>(trait);
  }
};

namespace StorageUserTrait {
/// This trait is used to determine if a storage user, like Type, is mutable
/// or not. A storage user is mutable if ImplType of the derived class defines
/// a `mutate` function with a proper signature. Note that this trait is not
/// supposed to be used publicly. Users should use alias names like
/// `TypeTrait::IsMutable` instead.
template <typename ConcreteType>
struct IsMutable : public StorageUserTraitBase<ConcreteType, IsMutable> {};
} // namespace StorageUserTrait

//===----------------------------------------------------------------------===//
// StorageUserBase
//===----------------------------------------------------------------------===//

namespace storage_user_base_impl {
/// Returns true if this given Trait ID matches the IDs of any of the provided
/// trait types `Traits`.
template <template <typename T> class... Traits>
bool hasTrait(TypeID traitID) {
  TypeID traitIDs[] = {TypeID::get<Traits>()...};
  for (unsigned i = 0, e = sizeof...(Traits); i != e; ++i)
    if (traitIDs[i] == traitID)
      return true;
  return false;
}

// We specialize for the empty case to not define an empty array.
template <>
inline bool hasTrait(TypeID traitID) {
  return false;
}
} // namespace storage_user_base_impl

/// Utility class for implementing users of storage classes uniqued by a
/// StorageUniquer. Clients are not expected to interact with this class
/// directly.
template <typename ConcreteT, typename BaseT, typename StorageT,
          typename UniquerT, template <typename T> class... Traits>
class StorageUserBase : public BaseT, public Traits<ConcreteT>... {
public:
  using BaseT::BaseT;

  /// Utility declarations for the concrete attribute class.
  using Base = StorageUserBase<ConcreteT, BaseT, StorageT, UniquerT, Traits...>;
  using ImplType = StorageT;
  using HasTraitFn = bool (*)(TypeID);

  /// Return a unique identifier for the concrete type.
  static TypeID getTypeID() { return TypeID::get<ConcreteT>(); }

  /// Provide an implementation of 'classof' that compares the type id of the
  /// provided value with that of the concrete type.
  template <typename T>
  static bool classof(T val) {
    static_assert(std::is_convertible<ConcreteT, T>::value,
                  "casting from a non-convertible type");
    return val.getTypeID() == getTypeID();
  }

  /// Returns an interface map for the interfaces registered to this storage
  /// user. This should not be used directly.
  static detail::InterfaceMap getInterfaceMap() {
    return detail::InterfaceMap::template get<Traits<ConcreteT>...>();
  }

  /// Returns the function that returns true if the given Trait ID matches the
  /// IDs of any of the traits defined by the storage user.
  static HasTraitFn getHasTraitFn() {
    return [](TypeID id) {
      return storage_user_base_impl::hasTrait<Traits...>(id);
    };
  }

  /// Returns a function that walks immediate sub elements of a given instance
  /// of the storage user.
  static auto getWalkImmediateSubElementsFn() {
    return [](auto instance, function_ref<void(Attribute)> walkAttrsFn,
              function_ref<void(Type)> walkTypesFn) {
      ::mlir::detail::walkImmediateSubElementsImpl(
          llvm::cast<ConcreteT>(instance), walkAttrsFn, walkTypesFn);
    };
  }

  /// Returns a function that replaces immediate sub elements of a given
  /// instance of the storage user.
  static auto getReplaceImmediateSubElementsFn() {
    return [](auto instance, ArrayRef<Attribute> replAttrs,
              ArrayRef<Type> replTypes) {
      return ::mlir::detail::replaceImmediateSubElementsImpl(
          llvm::cast<ConcreteT>(instance), replAttrs, replTypes);
    };
  }

  /// Attach the given models as implementations of the corresponding interfaces
  /// for the concrete storage user class. The type must be registered with the
  /// context, i.e. the dialect to which the type belongs must be loaded. The
  /// call will abort otherwise.
  template <typename... IfaceModels>
  static void attachInterface(MLIRContext &context) {
    typename ConcreteT::AbstractTy *abstract =
        ConcreteT::AbstractTy::lookupMutable(TypeID::get<ConcreteT>(),
                                             &context);
    if (!abstract)
      llvm::report_fatal_error("Registering an interface for an attribute/type "
                               "that is not itself registered.");

    // Handle the case where the models resolve a promised interface.
    (dialect_extension_detail::handleAdditionOfUndefinedPromisedInterface(
         abstract->getDialect(), abstract->getTypeID(),
         IfaceModels::Interface::getInterfaceID()),
     ...);

    (checkInterfaceTarget<IfaceModels>(), ...);
    abstract->interfaceMap.template insertModels<IfaceModels...>();
  }

  /// Get or create a new ConcreteT instance within the ctx. This
  /// function is guaranteed to return a non null object and will assert if
  /// the arguments provided are invalid.
  template <typename... Args>
  static ConcreteT get(MLIRContext *ctx, Args &&...args) {
    // Ensure that the invariants are correct for construction.
    assert(succeeded(
        ConcreteT::verifyInvariants(getDefaultDiagnosticEmitFn(ctx), args...)));
    return UniquerT::template get<ConcreteT>(ctx, std::forward<Args>(args)...);
  }

  /// Get or create a new ConcreteT instance within the ctx, defined at
  /// the given, potentially unknown, location. If the arguments provided are
  /// invalid, errors are emitted using the provided location and a null object
  /// is returned.
  template <typename... Args>
  static ConcreteT getChecked(const Location &loc, Args &&...args) {
    return ConcreteT::getChecked(getDefaultDiagnosticEmitFn(loc),
                                 std::forward<Args>(args)...);
  }

  /// Get or create a new ConcreteT instance within the ctx. If the arguments
  /// provided are invalid, errors are emitted using the provided `emitError`
  /// and a null object is returned.
  template <typename... Args>
  static ConcreteT getChecked(function_ref<InFlightDiagnostic()> emitErrorFn,
                              MLIRContext *ctx, Args... args) {
    // If the construction invariants fail then we return a null attribute.
    if (failed(ConcreteT::verifyInvariants(emitErrorFn, args...)))
      return ConcreteT();
    return UniquerT::template get<ConcreteT>(ctx, args...);
  }

  /// Get an instance of the concrete type from a void pointer.
  static ConcreteT getFromOpaquePointer(const void *ptr) {
    return ConcreteT((const typename BaseT::ImplType *)ptr);
  }

  /// Utility for easy access to the storage instance.
  ImplType *getImpl() const { return static_cast<ImplType *>(this->impl); }

protected:
  /// Mutate the current storage instance. This will not change the unique key.
  /// The arguments are forwarded to 'ConcreteT::mutate'.
  template <typename... Args>
  LogicalResult mutate(Args &&...args) {
    static_assert(std::is_base_of<StorageUserTrait::IsMutable<ConcreteT>,
                                  ConcreteT>::value,
                  "The `mutate` function expects mutable trait "
                  "(e.g. TypeTrait::IsMutable) to be attached on parent.");
    return UniquerT::template mutate<ConcreteT>(this->getContext(), getImpl(),
                                                std::forward<Args>(args)...);
  }

  /// Default implementation that just returns success.
  template <typename... Args>
  static LogicalResult verifyInvariants(Args... args) {
    return success();
  }

private:
  /// Trait to check if T provides a 'ConcreteEntity' type alias.
  template <typename T>
  using has_concrete_entity_t = typename T::ConcreteEntity;

  /// A struct-wrapped type alias to T::ConcreteEntity if provided and to
  /// ConcreteT otherwise. This is akin to std::conditional but doesn't fail on
  /// the missing typedef. Useful for checking if the interface is targeting the
  /// right class.
  template <typename T,
            bool = llvm::is_detected<has_concrete_entity_t, T>::value>
  struct IfaceTargetOrConcreteT {
    using type = typename T::ConcreteEntity;
  };
  template <typename T>
  struct IfaceTargetOrConcreteT<T, false> {
    using type = ConcreteT;
  };

  /// A hook for static assertion that the external interface model T is
  /// targeting a base class of the concrete attribute/type. The model can also
  /// be a fallback model that works for every attribute/type.
  template <typename T>
  static void checkInterfaceTarget() {
    static_assert(std::is_base_of<typename IfaceTargetOrConcreteT<T>::type,
                                  ConcreteT>::value,
                  "attaching an interface to the wrong attribute/type kind");
  }
};
} // namespace detail
} // namespace mlir

#endif
