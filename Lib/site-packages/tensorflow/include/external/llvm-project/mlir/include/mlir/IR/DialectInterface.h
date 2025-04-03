//===- DialectInterface.h - IR Dialect Interfaces ---------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_IR_DIALECTINTERFACE_H
#define MLIR_IR_DIALECTINTERFACE_H

#include "mlir/Support/TypeID.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include <vector>

namespace mlir {
class Dialect;
class MLIRContext;
class Operation;

//===----------------------------------------------------------------------===//
// DialectInterface
//===----------------------------------------------------------------------===//
namespace detail {
/// The base class used for all derived interface types. This class provides
/// utilities necessary for registration.
template <typename ConcreteType, typename BaseT>
class DialectInterfaceBase : public BaseT {
public:
  using Base = DialectInterfaceBase<ConcreteType, BaseT>;

  /// Get a unique id for the derived interface type.
  static TypeID getInterfaceID() { return TypeID::get<ConcreteType>(); }

protected:
  DialectInterfaceBase(Dialect *dialect) : BaseT(dialect, getInterfaceID()) {}
};
} // namespace detail

/// This class represents an interface overridden for a single dialect.
class DialectInterface {
public:
  virtual ~DialectInterface();

  /// The base class used for all derived interface types. This class provides
  /// utilities necessary for registration.
  template <typename ConcreteType>
  using Base = detail::DialectInterfaceBase<ConcreteType, DialectInterface>;

  /// Return the dialect that this interface represents.
  Dialect *getDialect() const { return dialect; }

  /// Return the context that holds the parent dialect of this interface.
  MLIRContext *getContext() const;

  /// Return the derived interface id.
  TypeID getID() const { return interfaceID; }

protected:
  DialectInterface(Dialect *dialect, TypeID id)
      : dialect(dialect), interfaceID(id) {}

private:
  /// The dialect that represents this interface.
  Dialect *dialect;

  /// The unique identifier for the derived interface type.
  TypeID interfaceID;
};

//===----------------------------------------------------------------------===//
// DialectInterfaceCollection
//===----------------------------------------------------------------------===//

namespace detail {
/// This class is the base class for a collection of instances for a specific
/// interface kind.
class DialectInterfaceCollectionBase {
  /// DenseMap info for dialect interfaces that allows lookup by the dialect.
  struct InterfaceKeyInfo : public DenseMapInfo<const DialectInterface *> {
    using DenseMapInfo<const DialectInterface *>::isEqual;

    static unsigned getHashValue(Dialect *key) { return llvm::hash_value(key); }
    static unsigned getHashValue(const DialectInterface *key) {
      return getHashValue(key->getDialect());
    }

    static bool isEqual(Dialect *lhs, const DialectInterface *rhs) {
      if (rhs == getEmptyKey() || rhs == getTombstoneKey())
        return false;
      return lhs == rhs->getDialect();
    }
  };

  /// A set of registered dialect interface instances.
  using InterfaceSetT = DenseSet<const DialectInterface *, InterfaceKeyInfo>;
  using InterfaceVectorT = std::vector<const DialectInterface *>;

public:
  DialectInterfaceCollectionBase(MLIRContext *ctx, TypeID interfaceKind,
                                 StringRef interfaceName);
  virtual ~DialectInterfaceCollectionBase();

protected:
  /// Get the interface for the dialect of given operation, or null if one
  /// is not registered.
  const DialectInterface *getInterfaceFor(Operation *op) const;

  /// Get the interface for the given dialect.
  const DialectInterface *getInterfaceFor(Dialect *dialect) const {
    auto it = interfaces.find_as(dialect);
    return it == interfaces.end() ? nullptr : *it;
  }

  /// An iterator class that iterates the held interface objects of the given
  /// derived interface type.
  template <typename InterfaceT>
  struct iterator
      : public llvm::mapped_iterator_base<iterator<InterfaceT>,
                                          InterfaceVectorT::const_iterator,
                                          const InterfaceT &> {
    using llvm::mapped_iterator_base<iterator<InterfaceT>,
                                     InterfaceVectorT::const_iterator,
                                     const InterfaceT &>::mapped_iterator_base;

    /// Map the element to the iterator result type.
    const InterfaceT &mapElement(const DialectInterface *interface) const {
      return *static_cast<const InterfaceT *>(interface);
    }
  };

  /// Iterator access to the held interfaces.
  template <typename InterfaceT>
  iterator<InterfaceT> interface_begin() const {
    return iterator<InterfaceT>(orderedInterfaces.begin());
  }
  template <typename InterfaceT>
  iterator<InterfaceT> interface_end() const {
    return iterator<InterfaceT>(orderedInterfaces.end());
  }

private:
  /// A set of registered dialect interface instances.
  InterfaceSetT interfaces;
  /// An ordered list of the registered interface instances, necessary for
  /// deterministic iteration.
  // NOTE: SetVector does not provide find access, so it can't be used here.
  InterfaceVectorT orderedInterfaces;
};
} // namespace detail

/// A collection of dialect interfaces within a context, for a given concrete
/// interface type.
template <typename InterfaceType>
class DialectInterfaceCollection
    : public detail::DialectInterfaceCollectionBase {
public:
  using Base = DialectInterfaceCollection<InterfaceType>;

  /// Collect the registered dialect interfaces within the provided context.
  DialectInterfaceCollection(MLIRContext *ctx)
      : detail::DialectInterfaceCollectionBase(
            ctx, InterfaceType::getInterfaceID(),
            llvm::getTypeName<InterfaceType>()) {}

  /// Get the interface for a given object, or null if one is not registered.
  /// The object may be a dialect or an operation instance.
  template <typename Object>
  const InterfaceType *getInterfaceFor(Object *obj) const {
    return static_cast<const InterfaceType *>(
        detail::DialectInterfaceCollectionBase::getInterfaceFor(obj));
  }

  /// Iterator access to the held interfaces.
  using iterator =
      detail::DialectInterfaceCollectionBase::iterator<InterfaceType>;
  iterator begin() const { return interface_begin<InterfaceType>(); }
  iterator end() const { return interface_end<InterfaceType>(); }

private:
  using detail::DialectInterfaceCollectionBase::interface_begin;
  using detail::DialectInterfaceCollectionBase::interface_end;
};

} // namespace mlir

#endif
