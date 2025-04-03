//===- TypeID.h - TypeID RTTI class -----------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains a definition of the TypeID class. This provides a non
// RTTI mechanism for producing unique type IDs in LLVM.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_SUPPORT_TYPEID_H
#define MLIR_SUPPORT_TYPEID_H

#include "mlir/Support/LLVM.h"
#include "llvm/ADT/DenseMapInfo.h"
#include "llvm/ADT/Hashing.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Allocator.h"
#include "llvm/Support/PointerLikeTypeTraits.h"
#include "llvm/Support/TypeName.h"

namespace mlir {
//===----------------------------------------------------------------------===//
// TypeID
//===----------------------------------------------------------------------===//

/// This class provides an efficient unique identifier for a specific C++ type.
/// This allows for a C++ type to be compared, hashed, and stored in an opaque
/// context. This class is similar in some ways to std::type_index, but can be
/// used for any type. For example, this class could be used to implement LLVM
/// style isa/dyn_cast functionality for a type hierarchy:
///
///  struct Base {
///    Base(TypeID typeID) : typeID(typeID) {}
///    TypeID typeID;
///  };
///
///  struct DerivedA : public Base {
///    DerivedA() : Base(TypeID::get<DerivedA>()) {}
///
///    static bool classof(const Base *base) {
///      return base->typeID == TypeID::get<DerivedA>();
///    }
///  };
///
///  void foo(Base *base) {
///    if (DerivedA *a = llvm::dyn_cast<DerivedA>(base))
///       ...
///  }
///
/// C++ RTTI is a notoriously difficult topic; given the nature of shared
/// libraries many different approaches fundamentally break down in either the
/// area of support (i.e. only certain types of classes are supported), or in
/// terms of performance (e.g. by using string comparison). This class intends
/// to strike a balance between performance and the setup required to enable its
/// use.
///
/// Assume we are adding support for some class Foo, below are the set of ways
/// in which a given c++ type may be supported:
///
///  * Explicitly via `MLIR_DECLARE_EXPLICIT_TYPE_ID` and
///    `MLIR_DEFINE_EXPLICIT_TYPE_ID`
///
///    - This method explicitly defines the type ID for a given type using the
///      given macros. These should be placed at the top-level of the file (i.e.
///      not within any namespace or class). This is the most effective and
///      efficient method, but requires explicit annotations for each type.
///
///      Example:
///
///        // Foo.h
///        MLIR_DECLARE_EXPLICIT_TYPE_ID(Foo);
///
///        // Foo.cpp
///        MLIR_DEFINE_EXPLICIT_TYPE_ID(Foo);
///
///  * Explicitly via `MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID`
///   - This method explicitly defines the type ID for a given type by
///     annotating the class directly. This has similar effectiveness and
///     efficiency to the above method, but should only be used on internal
///     classes; i.e. those with definitions constrained to a specific library
///     (generally classes in anonymous namespaces).
///
///     Example:
///
///       namespace {
///       class Foo {
///       public:
///         MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(Foo)
///       };
///       } // namespace
///
///  * Implicitly via a fallback using the type name
///   - This method implicitly defines a type ID for a given type by using the
///     type name. This method requires nothing explicitly from the user, but
///     pays additional access and initialization cost. Given that this method
///     uses the name of the type, it may not be used for types defined in
///     anonymous namespaces (which is asserted when it can be detected). String
///     names do not provide any guarantees on uniqueness in these contexts.
///
class TypeID {
  /// This class represents the storage of a type info object.
  /// Note: We specify an explicit alignment here to allow use with
  /// PointerIntPair and other utilities/data structures that require a known
  /// pointer alignment.
  struct alignas(8) Storage {};

public:
  TypeID() : TypeID(get<void>()) {}

  /// Comparison operations.
  inline bool operator==(const TypeID &other) const {
    return storage == other.storage;
  }
  inline bool operator!=(const TypeID &other) const {
    return !(*this == other);
  }

  /// Construct a type info object for the given type T.
  template <typename T>
  static TypeID get();
  template <template <typename> class Trait>
  static TypeID get();

  /// Methods for supporting PointerLikeTypeTraits.
  const void *getAsOpaquePointer() const {
    return static_cast<const void *>(storage);
  }
  static TypeID getFromOpaquePointer(const void *pointer) {
    return TypeID(reinterpret_cast<const Storage *>(pointer));
  }

  /// Enable hashing TypeID.
  friend ::llvm::hash_code hash_value(TypeID id);

private:
  TypeID(const Storage *storage) : storage(storage) {}

  /// The storage of this type info object.
  const Storage *storage;

  friend class TypeIDAllocator;
};

/// Enable hashing TypeID.
inline ::llvm::hash_code hash_value(TypeID id) {
  return DenseMapInfo<const TypeID::Storage *>::getHashValue(id.storage);
}

//===----------------------------------------------------------------------===//
// TypeIDResolver
//===----------------------------------------------------------------------===//

namespace detail {
/// This class provides a fallback for resolving TypeIDs. It uses the string
/// name of the type to perform the resolution, and as such does not allow the
/// use of classes defined in "anonymous" contexts.
class FallbackTypeIDResolver {
protected:
  /// Register an implicit type ID for the given type name.
  static TypeID registerImplicitTypeID(StringRef name);
};

/// This class provides a resolver for getting the ID for a given class T. This
/// allows for the derived type to specialize its resolution behavior. The
/// default implementation uses the string name of the type to resolve the ID.
/// This provides a strong definition, but at the cost of performance (we need
/// to do an initial lookup) and is not usable by classes defined in anonymous
/// contexts.
///
/// TODO: The use of the type name is only necessary when building in the
/// presence of shared libraries. We could add a build flag that guarantees
/// "static"-like environments and switch this to a more optimal implementation
/// when that is enabled.
template <typename T, typename Enable = void>
class TypeIDResolver : public FallbackTypeIDResolver {
public:
  /// Trait to check if `U` is fully resolved. We use this to verify that `T` is
  /// fully resolved when trying to resolve a TypeID. We don't technically need
  /// to have the full definition of `T` for the fallback, but it does help
  /// prevent situations where a forward declared type uses this fallback even
  /// though there is a strong definition for the TypeID in the location where
  /// `T` is defined.
  template <typename U>
  using is_fully_resolved_trait = decltype(sizeof(U));
  template <typename U>
  using is_fully_resolved = llvm::is_detected<is_fully_resolved_trait, U>;

  static TypeID resolveTypeID() {
    static_assert(is_fully_resolved<T>::value,
                  "TypeID::get<> requires the complete definition of `T`");
    static TypeID id = registerImplicitTypeID(llvm::getTypeName<T>());
    return id;
  }
};

/// This class provides utilities for resolving the TypeID of a class that
/// provides a `static TypeID resolveTypeID()` method. This allows for
/// simplifying situations when the class can resolve the ID itself. This
/// functionality is separated from the corresponding `TypeIDResolver`
/// specialization below to enable referencing it more easily in different
/// contexts.
struct InlineTypeIDResolver {
  /// Trait to check if `T` provides a static `resolveTypeID` method.
  template <typename T>
  using has_resolve_typeid_trait = decltype(T::resolveTypeID());
  template <typename T>
  using has_resolve_typeid = llvm::is_detected<has_resolve_typeid_trait, T>;

  template <typename T>
  static TypeID resolveTypeID() {
    return T::resolveTypeID();
  }
};
/// This class provides a resolver for getting the ID for a given class T, when
/// the class provides a `static TypeID resolveTypeID()` method. This allows for
/// simplifying situations when the class can resolve the ID itself.
template <typename T>
class TypeIDResolver<
    T, std::enable_if_t<InlineTypeIDResolver::has_resolve_typeid<T>::value>> {
public:
  static TypeID resolveTypeID() {
    return InlineTypeIDResolver::resolveTypeID<T>();
  }
};
} // namespace detail

template <typename T>
TypeID TypeID::get() {
  return detail::TypeIDResolver<T>::resolveTypeID();
}
template <template <typename> class Trait>
TypeID TypeID::get() {
  // An empty class used to simplify the use of Trait types.
  struct Empty {};
  return TypeID::get<Trait<Empty>>();
}

// Declare/define an explicit specialization for TypeID: this forces the
// compiler to emit a strong definition for a class and controls which
// translation unit and shared object will actually have it.
// This can be useful to turn to a link-time failure what would be in other
// circumstances a hard-to-catch runtime bug when a TypeID is hidden in two
// different shared libraries and instances of the same class only gets the same
// TypeID inside a given DSO.
#define MLIR_DECLARE_EXPLICIT_TYPE_ID(CLASS_NAME)                              \
  namespace mlir {                                                             \
  namespace detail {                                                           \
  template <>                                                                  \
  class TypeIDResolver<CLASS_NAME> {                                           \
  public:                                                                      \
    static TypeID resolveTypeID() { return id; }                               \
                                                                               \
  private:                                                                     \
    static SelfOwningTypeID id;                                                \
  };                                                                           \
  } /* namespace detail */                                                     \
  } /* namespace mlir */

#define MLIR_DEFINE_EXPLICIT_TYPE_ID(CLASS_NAME)                               \
  namespace mlir {                                                             \
  namespace detail {                                                           \
  SelfOwningTypeID TypeIDResolver<CLASS_NAME>::id = {};                        \
  } /* namespace detail */                                                     \
  } /* namespace mlir */

// Declare/define an explicit, **internal**, specialization of TypeID for the
// given class. This is useful for providing an explicit specialization of
// TypeID for a class that is known to be internal to a specific library. It
// should be placed within a public section of the declaration of the class.
#define MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(CLASS_NAME)               \
  static ::mlir::TypeID resolveTypeID() {                                      \
    static ::mlir::SelfOwningTypeID id;                                        \
    return id;                                                                 \
  }                                                                            \
  static_assert(                                                               \
      ::mlir::detail::InlineTypeIDResolver::has_resolve_typeid<                \
          CLASS_NAME>::value,                                                  \
      "`MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID` must be placed in a "    \
      "public section of `" #CLASS_NAME "`");

//===----------------------------------------------------------------------===//
// TypeIDAllocator
//===----------------------------------------------------------------------===//

/// This class provides a way to define new TypeIDs at runtime.
/// When the allocator is destructed, all allocated TypeIDs become invalid and
/// therefore should not be used.
class TypeIDAllocator {
public:
  /// Allocate a new TypeID, that is ensured to be unique for the lifetime
  /// of the TypeIDAllocator.
  TypeID allocate() { return TypeID(ids.Allocate()); }

private:
  /// The TypeIDs allocated are the addresses of the different storages.
  /// Keeping those in memory ensure uniqueness of the TypeIDs.
  llvm::SpecificBumpPtrAllocator<TypeID::Storage> ids;
};

//===----------------------------------------------------------------------===//
// SelfOwningTypeID
//===----------------------------------------------------------------------===//

/// Defines a TypeID for each instance of this class by using a pointer to the
/// instance. Thus, the copy and move constructor are deleted.
/// Note: We align by 8 to match the alignment of TypeID::Storage, as we treat
/// an instance of this class similarly to TypeID::Storage.
class alignas(8) SelfOwningTypeID {
public:
  SelfOwningTypeID() = default;
  SelfOwningTypeID(const SelfOwningTypeID &) = delete;
  SelfOwningTypeID &operator=(const SelfOwningTypeID &) = delete;
  SelfOwningTypeID(SelfOwningTypeID &&) = delete;
  SelfOwningTypeID &operator=(SelfOwningTypeID &&) = delete;

  /// Implicitly converts to the owned TypeID.
  operator TypeID() const { return getTypeID(); }

  /// Return the TypeID owned by this object.
  TypeID getTypeID() const { return TypeID::getFromOpaquePointer(this); }
};

} // namespace mlir

//===----------------------------------------------------------------------===//
// Builtin TypeIDs
//===----------------------------------------------------------------------===//

/// Explicitly register a set of "builtin" types.
MLIR_DECLARE_EXPLICIT_TYPE_ID(void)

namespace llvm {
template <>
struct DenseMapInfo<mlir::TypeID> {
  static inline mlir::TypeID getEmptyKey() {
    void *pointer = llvm::DenseMapInfo<void *>::getEmptyKey();
    return mlir::TypeID::getFromOpaquePointer(pointer);
  }
  static inline mlir::TypeID getTombstoneKey() {
    void *pointer = llvm::DenseMapInfo<void *>::getTombstoneKey();
    return mlir::TypeID::getFromOpaquePointer(pointer);
  }
  static unsigned getHashValue(mlir::TypeID val) {
    return mlir::hash_value(val);
  }
  static bool isEqual(mlir::TypeID lhs, mlir::TypeID rhs) { return lhs == rhs; }
};

/// We align TypeID::Storage by 8, so allow LLVM to steal the low bits.
template <>
struct PointerLikeTypeTraits<mlir::TypeID> {
  static inline void *getAsVoidPointer(mlir::TypeID info) {
    return const_cast<void *>(info.getAsOpaquePointer());
  }
  static inline mlir::TypeID getFromVoidPointer(void *ptr) {
    return mlir::TypeID::getFromOpaquePointer(ptr);
  }
  static constexpr int NumLowBitsAvailable = 3;
};

} // namespace llvm

#endif // MLIR_SUPPORT_TYPEID_H
