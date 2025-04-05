//===- StorageUniquer.h - Common Storage Class Uniquer ----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_SUPPORT_STORAGEUNIQUER_H
#define MLIR_SUPPORT_STORAGEUNIQUER_H

#include "mlir/Support/LLVM.h"
#include "mlir/Support/TypeID.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Allocator.h"
#include <utility>

namespace mlir {
namespace detail {
struct StorageUniquerImpl;

/// Trait to check if ImplTy provides a 'getKey' method with types 'Args'.
template <typename ImplTy, typename... Args>
using has_impltype_getkey_t = decltype(ImplTy::getKey(std::declval<Args>()...));

/// Trait to check if ImplTy provides a 'hashKey' method for 'T'.
template <typename ImplTy, typename T>
using has_impltype_hash_t = decltype(ImplTy::hashKey(std::declval<T>()));
} // namespace detail

/// A utility class to get or create instances of "storage classes". These
/// storage classes must derive from 'StorageUniquer::BaseStorage'.
///
/// For non-parametric storage classes, i.e. singleton classes, nothing else is
/// needed. Instances of these classes can be created by calling `get` without
/// trailing arguments.
///
/// Otherwise, the parametric storage classes may be created with `get`,
/// and must respect the following:
///    - Define a type alias, KeyTy, to a type that uniquely identifies the
///      instance of the storage class.
///      * The key type must be constructible from the values passed into the
///        getComplex call.
///      * If the KeyTy does not have an llvm::DenseMapInfo specialization, the
///        storage class must define a hashing method:
///         'static unsigned hashKey(const KeyTy &)'
///
///    - Provide a method, 'bool operator==(const KeyTy &) const', to
///      compare the storage instance against an instance of the key type.
///
///    - Provide a static construction method:
///        'DerivedStorage *construct(StorageAllocator &, const KeyTy &key)'
///      that builds a unique instance of the derived storage. The arguments to
///      this function are an allocator to store any uniqued data and the key
///      type for this storage.
///
///    - Provide a cleanup method:
///        'void cleanup()'
///      that is called when erasing a storage instance. This should cleanup any
///      fields of the storage as necessary and not attempt to free the memory
///      of the storage itself.
///
/// Storage classes may have an optional mutable component, which must not take
/// part in the unique immutable key. In this case, storage classes may be
/// mutated with `mutate` and must additionally respect the following:
///    - Provide a mutation method:
///        'LogicalResult mutate(StorageAllocator &, <...>)'
///      that is called when mutating a storage instance. The first argument is
///      an allocator to store any mutable data, and the remaining arguments are
///      forwarded from the call site. The storage can be mutated at any time
///      after creation. Care must be taken to avoid excessive mutation since
///      the allocated storage can keep containing previous states. The return
///      value of the function is used to indicate whether the mutation was
///      successful, e.g., to limit the number of mutations or enable deferred
///      one-time assignment of the mutable component.
///
/// All storage classes must be registered with the uniquer via
/// `registerParametricStorageType` or `registerSingletonStorageType`
/// using an appropriate unique `TypeID` for the storage class.
class StorageUniquer {
public:
  /// This class acts as the base storage that all storage classes must derived
  /// from.
  class alignas(8) BaseStorage {
  protected:
    BaseStorage() = default;
  };

  /// This is a utility allocator used to allocate memory for instances of
  /// derived types.
  class StorageAllocator {
  public:
    /// Copy the specified array of elements into memory managed by our bump
    /// pointer allocator.  This assumes the elements are all PODs.
    template <typename T>
    ArrayRef<T> copyInto(ArrayRef<T> elements) {
      if (elements.empty())
        return std::nullopt;
      auto result = allocator.Allocate<T>(elements.size());
      std::uninitialized_copy(elements.begin(), elements.end(), result);
      return ArrayRef<T>(result, elements.size());
    }

    /// Copy the provided string into memory managed by our bump pointer
    /// allocator.
    StringRef copyInto(StringRef str) {
      if (str.empty())
        return StringRef();

      char *result = allocator.Allocate<char>(str.size() + 1);
      std::uninitialized_copy(str.begin(), str.end(), result);
      result[str.size()] = 0;
      return StringRef(result, str.size());
    }

    /// Allocate an instance of the provided type.
    template <typename T>
    T *allocate() {
      return allocator.Allocate<T>();
    }

    /// Allocate 'size' bytes of 'alignment' aligned memory.
    void *allocate(size_t size, size_t alignment) {
      return allocator.Allocate(size, alignment);
    }

    /// Returns true if this allocator allocated the provided object pointer.
    bool allocated(const void *ptr) {
      return allocator.identifyObject(ptr).has_value();
    }

  private:
    /// The raw allocator for type storage objects.
    llvm::BumpPtrAllocator allocator;
  };

  StorageUniquer();
  ~StorageUniquer();

  /// Set the flag specifying if multi-threading is disabled within the uniquer.
  void disableMultithreading(bool disable = true);

  /// Register a new parametric storage class, this is necessary to create
  /// instances of this class type. `id` is the type identifier that will be
  /// used to identify this type when creating instances of it via 'get'.
  template <typename Storage>
  void registerParametricStorageType(TypeID id) {
    // If the storage is trivially destructible, we don't need a destructor
    // function.
    if constexpr (std::is_trivially_destructible_v<Storage>)
      return registerParametricStorageTypeImpl(id, nullptr);
    registerParametricStorageTypeImpl(id, [](BaseStorage *storage) {
      static_cast<Storage *>(storage)->~Storage();
    });
  }
  /// Utility override when the storage type represents the type id.
  template <typename Storage>
  void registerParametricStorageType() {
    registerParametricStorageType<Storage>(TypeID::get<Storage>());
  }
  /// Register a new singleton storage class, this is necessary to get the
  /// singletone instance. `id` is the type identifier that will be used to
  /// access the singleton instance via 'get'. An optional initialization
  /// function may also be provided to initialize the newly created storage
  /// instance, and used when the singleton instance is created.
  template <typename Storage>
  void registerSingletonStorageType(TypeID id,
                                    function_ref<void(Storage *)> initFn) {
    auto ctorFn = [&](StorageAllocator &allocator) {
      auto *storage = new (allocator.allocate<Storage>()) Storage();
      if (initFn)
        initFn(storage);
      return storage;
    };
    registerSingletonImpl(id, ctorFn);
  }
  template <typename Storage>
  void registerSingletonStorageType(TypeID id) {
    registerSingletonStorageType<Storage>(id, std::nullopt);
  }
  /// Utility override when the storage type represents the type id.
  template <typename Storage>
  void registerSingletonStorageType(function_ref<void(Storage *)> initFn = {}) {
    registerSingletonStorageType<Storage>(TypeID::get<Storage>(), initFn);
  }

  /// Gets a uniqued instance of 'Storage'. 'id' is the type id used when
  /// registering the storage instance. 'initFn' is an optional parameter that
  /// can be used to initialize a newly inserted storage instance. This function
  /// is used for derived types that have complex storage or uniquing
  /// constraints.
  template <typename Storage, typename... Args>
  Storage *get(function_ref<void(Storage *)> initFn, TypeID id,
               Args &&...args) {
    // Construct a value of the derived key type.
    auto derivedKey = getKey<Storage>(std::forward<Args>(args)...);

    // Create a hash of the derived key.
    unsigned hashValue = getHash<Storage>(derivedKey);

    // Generate an equality function for the derived storage.
    auto isEqual = [&derivedKey](const BaseStorage *existing) {
      return static_cast<const Storage &>(*existing) == derivedKey;
    };

    // Generate a constructor function for the derived storage.
    auto ctorFn = [&](StorageAllocator &allocator) {
      auto *storage = Storage::construct(allocator, std::move(derivedKey));
      if (initFn)
        initFn(storage);
      return storage;
    };

    // Get an instance for the derived storage.
    return static_cast<Storage *>(
        getParametricStorageTypeImpl(id, hashValue, isEqual, ctorFn));
  }
  /// Utility override when the storage type represents the type id.
  template <typename Storage, typename... Args>
  Storage *get(function_ref<void(Storage *)> initFn, Args &&...args) {
    return get<Storage>(initFn, TypeID::get<Storage>(),
                        std::forward<Args>(args)...);
  }

  /// Gets a uniqued instance of 'Storage' which is a singleton storage type.
  /// 'id' is the type id used when registering the storage instance.
  template <typename Storage>
  Storage *get(TypeID id) {
    return static_cast<Storage *>(getSingletonImpl(id));
  }
  /// Utility override when the storage type represents the type id.
  template <typename Storage>
  Storage *get() {
    return get<Storage>(TypeID::get<Storage>());
  }

  /// Test if there is a singleton storage uniquer initialized for the provided
  /// TypeID. This is only useful for debugging/diagnostic purpose: the uniquer
  /// is initialized when a dialect is loaded.
  bool isSingletonStorageInitialized(TypeID id);

  /// Test if there is a parametric storage uniquer initialized for the provided
  /// TypeID. This is only useful for debugging/diagnostic purpose: the uniquer
  /// is initialized when a dialect is loaded.
  bool isParametricStorageInitialized(TypeID id);

  /// Changes the mutable component of 'storage' by forwarding the trailing
  /// arguments to the 'mutate' function of the derived class.
  template <typename Storage, typename... Args>
  LogicalResult mutate(TypeID id, Storage *storage, Args &&...args) {
    auto mutationFn = [&](StorageAllocator &allocator) -> LogicalResult {
      return static_cast<Storage &>(*storage).mutate(
          allocator, std::forward<Args>(args)...);
    };
    return mutateImpl(id, storage, mutationFn);
  }

private:
  /// Implementation for getting/creating an instance of a derived type with
  /// parametric storage.
  BaseStorage *getParametricStorageTypeImpl(
      TypeID id, unsigned hashValue,
      function_ref<bool(const BaseStorage *)> isEqual,
      function_ref<BaseStorage *(StorageAllocator &)> ctorFn);

  /// Implementation for registering an instance of a derived type with
  /// parametric storage. This method takes an optional destructor function that
  /// destructs storage instances when necessary.
  void registerParametricStorageTypeImpl(
      TypeID id, function_ref<void(BaseStorage *)> destructorFn);

  /// Implementation for getting an instance of a derived type with default
  /// storage.
  BaseStorage *getSingletonImpl(TypeID id);

  /// Implementation for registering an instance of a derived type with default
  /// storage.
  void
  registerSingletonImpl(TypeID id,
                        function_ref<BaseStorage *(StorageAllocator &)> ctorFn);

  /// Implementation for mutating an instance of a derived storage.
  LogicalResult
  mutateImpl(TypeID id, BaseStorage *storage,
             function_ref<LogicalResult(StorageAllocator &)> mutationFn);

  /// The internal implementation class.
  std::unique_ptr<detail::StorageUniquerImpl> impl;

  //===--------------------------------------------------------------------===//
  // Key Construction
  //===--------------------------------------------------------------------===//

  /// Used to construct an instance of 'ImplTy::KeyTy' if there is an
  /// 'ImplTy::getKey' function for the provided arguments.  Otherwise, then we
  /// try to directly construct the 'ImplTy::KeyTy' with the provided arguments.
  template <typename ImplTy, typename... Args>
  static typename ImplTy::KeyTy getKey(Args &&...args) {
    if constexpr (llvm::is_detected<detail::has_impltype_getkey_t, ImplTy,
                                    Args...>::value)
      return ImplTy::getKey(std::forward<Args>(args)...);
    else
      return typename ImplTy::KeyTy(std::forward<Args>(args)...);
  }

  //===--------------------------------------------------------------------===//
  // Key Hashing
  //===--------------------------------------------------------------------===//

  /// Used to generate a hash for the `ImplTy` of a storage instance if
  /// there is a `ImplTy::hashKey.  Otherwise, if there is no `ImplTy::hashKey`
  /// then default to using the 'llvm::DenseMapInfo' definition for
  /// 'DerivedKey' for generating a hash.
  template <typename ImplTy, typename DerivedKey>
  static ::llvm::hash_code getHash(const DerivedKey &derivedKey) {
    if constexpr (llvm::is_detected<detail::has_impltype_hash_t, ImplTy,
                                    DerivedKey>::value)
      return ImplTy::hashKey(derivedKey);
    else
      return DenseMapInfo<DerivedKey>::getHashValue(derivedKey);
  }
};
} // namespace mlir

#endif
