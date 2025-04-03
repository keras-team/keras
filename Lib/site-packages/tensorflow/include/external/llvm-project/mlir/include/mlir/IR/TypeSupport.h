//===- TypeSupport.h --------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines support types for registering dialect extended types.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_IR_TYPESUPPORT_H
#define MLIR_IR_TYPESUPPORT_H

#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/StorageUniquerSupport.h"
#include "llvm/ADT/Twine.h"

namespace mlir {
class Dialect;
class MLIRContext;

//===----------------------------------------------------------------------===//
// AbstractType
//===----------------------------------------------------------------------===//

/// This class contains all of the static information common to all instances of
/// a registered Type.
class AbstractType {
public:
  using HasTraitFn = llvm::unique_function<bool(TypeID) const>;
  using WalkImmediateSubElementsFn = function_ref<void(
      Type, function_ref<void(Attribute)>, function_ref<void(Type)>)>;
  using ReplaceImmediateSubElementsFn =
      function_ref<Type(Type, ArrayRef<Attribute>, ArrayRef<Type>)>;

  /// Look up the specified abstract type in the MLIRContext and return a
  /// reference to it.
  static const AbstractType &lookup(TypeID typeID, MLIRContext *context);

  /// Look up the specified abstract type in the MLIRContext and return a
  /// reference to it if it exists.
  static std::optional<std::reference_wrapper<const AbstractType>>
  lookup(StringRef name, MLIRContext *context);

  /// This method is used by Dialect objects when they register the list of
  /// types they contain.
  template <typename T>
  static AbstractType get(Dialect &dialect) {
    return AbstractType(dialect, T::getInterfaceMap(), T::getHasTraitFn(),
                        T::getWalkImmediateSubElementsFn(),
                        T::getReplaceImmediateSubElementsFn(), T::getTypeID(),
                        T::name);
  }

  /// This method is used by Dialect objects to register types with
  /// custom TypeIDs.
  /// The use of this method is in general discouraged in favor of
  /// 'get<CustomType>(dialect)';
  static AbstractType
  get(Dialect &dialect, detail::InterfaceMap &&interfaceMap,
      HasTraitFn &&hasTrait,
      WalkImmediateSubElementsFn walkImmediateSubElementsFn,
      ReplaceImmediateSubElementsFn replaceImmediateSubElementsFn,
      TypeID typeID, StringRef name) {
    return AbstractType(dialect, std::move(interfaceMap), std::move(hasTrait),
                        walkImmediateSubElementsFn,
                        replaceImmediateSubElementsFn, typeID, name);
  }

  /// Return the dialect this type was registered to.
  Dialect &getDialect() const { return const_cast<Dialect &>(dialect); }

  /// Returns an instance of the concept object for the given interface if it
  /// was registered to this type, null otherwise. This should not be used
  /// directly.
  template <typename T>
  typename T::Concept *getInterface() const {
    return interfaceMap.lookup<T>();
  }

  /// Returns true if the type has the interface with the given ID.
  bool hasInterface(TypeID interfaceID) const {
    return interfaceMap.contains(interfaceID);
  }

  /// Returns true if the type has a particular trait.
  template <template <typename T> class Trait>
  bool hasTrait() const {
    return hasTraitFn(TypeID::get<Trait>());
  }

  /// Returns true if the type has a particular trait.
  bool hasTrait(TypeID traitID) const { return hasTraitFn(traitID); }

  /// Walk the immediate sub-elements of the given type.
  void walkImmediateSubElements(Type type,
                                function_ref<void(Attribute)> walkAttrsFn,
                                function_ref<void(Type)> walkTypesFn) const;

  /// Replace the immediate sub-elements of the given type.
  Type replaceImmediateSubElements(Type type, ArrayRef<Attribute> replAttrs,
                                   ArrayRef<Type> replTypes) const;

  /// Return the unique identifier representing the concrete type class.
  TypeID getTypeID() const { return typeID; }

  /// Return the unique name representing the type.
  StringRef getName() const { return name; }

private:
  AbstractType(Dialect &dialect, detail::InterfaceMap &&interfaceMap,
               HasTraitFn &&hasTrait,
               WalkImmediateSubElementsFn walkImmediateSubElementsFn,
               ReplaceImmediateSubElementsFn replaceImmediateSubElementsFn,
               TypeID typeID, StringRef name)
      : dialect(dialect), interfaceMap(std::move(interfaceMap)),
        hasTraitFn(std::move(hasTrait)),
        walkImmediateSubElementsFn(walkImmediateSubElementsFn),
        replaceImmediateSubElementsFn(replaceImmediateSubElementsFn),
        typeID(typeID), name(name) {}

  /// Give StorageUserBase access to the mutable lookup.
  template <typename ConcreteT, typename BaseT, typename StorageT,
            typename UniquerT, template <typename T> class... Traits>
  friend class detail::StorageUserBase;

  /// Look up the specified abstract type in the MLIRContext and return a
  /// (mutable) pointer to it. Return a null pointer if the type could not
  /// be found in the context.
  static AbstractType *lookupMutable(TypeID typeID, MLIRContext *context);

  /// This is the dialect that this type was registered to.
  const Dialect &dialect;

  /// This is a collection of the interfaces registered to this type.
  detail::InterfaceMap interfaceMap;

  /// Function to check if the type has a particular trait.
  HasTraitFn hasTraitFn;

  /// Function to walk the immediate sub-elements of this type.
  WalkImmediateSubElementsFn walkImmediateSubElementsFn;

  /// Function to replace the immediate sub-elements of this type.
  ReplaceImmediateSubElementsFn replaceImmediateSubElementsFn;

  /// The unique identifier of the derived Type class.
  const TypeID typeID;

  /// The unique name of this type. The string is not owned by the context, so
  /// The lifetime of this string should outlive the MLIR context.
  const StringRef name;
};

//===----------------------------------------------------------------------===//
// TypeStorage
//===----------------------------------------------------------------------===//

namespace detail {
struct TypeUniquer;
} // namespace detail

/// Base storage class appearing in a Type.
class TypeStorage : public StorageUniquer::BaseStorage {
  friend detail::TypeUniquer;
  friend StorageUniquer;

public:
  /// Return the abstract type descriptor for this type.
  const AbstractType &getAbstractType() {
    assert(abstractType && "Malformed type storage object.");
    return *abstractType;
  }

protected:
  /// This constructor is used by derived classes as part of the TypeUniquer.
  TypeStorage() {}

private:
  /// Set the abstract type for this storage instance. This is used by the
  /// TypeUniquer when initializing a newly constructed type storage object.
  void initialize(const AbstractType &abstractTy) {
    abstractType = const_cast<AbstractType *>(&abstractTy);
  }

  /// The abstract description for this type.
  AbstractType *abstractType{nullptr};
};

/// Default storage type for types that require no additional initialization or
/// storage.
using DefaultTypeStorage = TypeStorage;

//===----------------------------------------------------------------------===//
// TypeStorageAllocator
//===----------------------------------------------------------------------===//

/// This is a utility allocator used to allocate memory for instances of derived
/// Types.
using TypeStorageAllocator = StorageUniquer::StorageAllocator;

//===----------------------------------------------------------------------===//
// TypeUniquer
//===----------------------------------------------------------------------===//
namespace detail {
/// A utility class to get, or create, unique instances of types within an
/// MLIRContext. This class manages all creation and uniquing of types.
struct TypeUniquer {
  /// Get an uniqued instance of a type T.
  template <typename T, typename... Args>
  static T get(MLIRContext *ctx, Args &&...args) {
    return getWithTypeID<T, Args...>(ctx, T::getTypeID(),
                                     std::forward<Args>(args)...);
  }

  /// Get an uniqued instance of a parametric type T.
  /// The use of this method is in general discouraged in favor of
  /// 'get<T, Args>(ctx, args)'.
  template <typename T, typename... Args>
  static std::enable_if_t<
      !std::is_same<typename T::ImplType, TypeStorage>::value, T>
  getWithTypeID(MLIRContext *ctx, TypeID typeID, Args &&...args) {
#ifndef NDEBUG
    if (!ctx->getTypeUniquer().isParametricStorageInitialized(typeID))
      llvm::report_fatal_error(
          llvm::Twine("can't create type '") + llvm::getTypeName<T>() +
          "' because storage uniquer isn't initialized: the dialect was likely "
          "not loaded, or the type wasn't added with addTypes<...>() "
          "in the Dialect::initialize() method.");
#endif
    return ctx->getTypeUniquer().get<typename T::ImplType>(
        [&, typeID](TypeStorage *storage) {
          storage->initialize(AbstractType::lookup(typeID, ctx));
        },
        typeID, std::forward<Args>(args)...);
  }
  /// Get an uniqued instance of a singleton type T.
  /// The use of this method is in general discouraged in favor of
  /// 'get<T, Args>(ctx, args)'.
  template <typename T>
  static std::enable_if_t<
      std::is_same<typename T::ImplType, TypeStorage>::value, T>
  getWithTypeID(MLIRContext *ctx, TypeID typeID) {
#ifndef NDEBUG
    if (!ctx->getTypeUniquer().isSingletonStorageInitialized(typeID))
      llvm::report_fatal_error(
          llvm::Twine("can't create type '") + llvm::getTypeName<T>() +
          "' because storage uniquer isn't initialized: the dialect was likely "
          "not loaded, or the type wasn't added with addTypes<...>() "
          "in the Dialect::initialize() method.");
#endif
    return ctx->getTypeUniquer().get<typename T::ImplType>(typeID);
  }

  /// Change the mutable component of the given type instance in the provided
  /// context.
  template <typename T, typename... Args>
  static LogicalResult mutate(MLIRContext *ctx, typename T::ImplType *impl,
                              Args &&...args) {
    assert(impl && "cannot mutate null type");
    return ctx->getTypeUniquer().mutate(T::getTypeID(), impl,
                                        std::forward<Args>(args)...);
  }

  /// Register a type instance T with the uniquer.
  template <typename T>
  static void registerType(MLIRContext *ctx) {
    registerType<T>(ctx, T::getTypeID());
  }

  /// Register a parametric type instance T with the uniquer.
  /// The use of this method is in general discouraged in favor of
  /// 'registerType<T>(ctx)'.
  template <typename T>
  static std::enable_if_t<
      !std::is_same<typename T::ImplType, TypeStorage>::value>
  registerType(MLIRContext *ctx, TypeID typeID) {
    ctx->getTypeUniquer().registerParametricStorageType<typename T::ImplType>(
        typeID);
  }
  /// Register a singleton type instance T with the uniquer.
  /// The use of this method is in general discouraged in favor of
  /// 'registerType<T>(ctx)'.
  template <typename T>
  static std::enable_if_t<
      std::is_same<typename T::ImplType, TypeStorage>::value>
  registerType(MLIRContext *ctx, TypeID typeID) {
    ctx->getTypeUniquer().registerSingletonStorageType<TypeStorage>(
        typeID, [&ctx, typeID](TypeStorage *storage) {
          storage->initialize(AbstractType::lookup(typeID, ctx));
        });
  }
};
} // namespace detail

} // namespace mlir

#endif
