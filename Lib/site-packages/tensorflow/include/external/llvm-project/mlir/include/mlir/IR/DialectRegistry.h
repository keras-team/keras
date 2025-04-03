//===- DialectRegistry.h - Dialect Registration and Extension ---*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines functionality for registring and extending dialects.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_IR_DIALECTREGISTRY_H
#define MLIR_IR_DIALECTREGISTRY_H

#include "mlir/IR/MLIRContext.h"
#include "mlir/Support/TypeID.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/MapVector.h"

#include <map>
#include <tuple>

namespace mlir {
class Dialect;

using DialectAllocatorFunction = std::function<Dialect *(MLIRContext *)>;
using DialectAllocatorFunctionRef = function_ref<Dialect *(MLIRContext *)>;
using DynamicDialectPopulationFunction =
    std::function<void(MLIRContext *, DynamicDialect *)>;

//===----------------------------------------------------------------------===//
// DialectExtension
//===----------------------------------------------------------------------===//

/// This class represents an opaque dialect extension. It contains a set of
/// required dialects and an application function. The required dialects control
/// when the extension is applied, i.e. the extension is applied when all
/// required dialects are loaded. The application function can be used to attach
/// additional functionality to attributes, dialects, operations, types, etc.,
/// and may also load additional necessary dialects.
class DialectExtensionBase {
public:
  virtual ~DialectExtensionBase();

  /// Return the dialects that our required by this extension to be loaded
  /// before applying. If empty then the extension is invoked for every loaded
  /// dialect indepently.
  ArrayRef<StringRef> getRequiredDialects() const { return dialectNames; }

  /// Apply this extension to the given context and the required dialects.
  virtual void apply(MLIRContext *context,
                     MutableArrayRef<Dialect *> dialects) const = 0;

  /// Return a copy of this extension.
  virtual std::unique_ptr<DialectExtensionBase> clone() const = 0;

protected:
  /// Initialize the extension with a set of required dialects.
  /// If the list is empty, the extension is invoked for every loaded dialect
  /// independently.
  DialectExtensionBase(ArrayRef<StringRef> dialectNames)
      : dialectNames(dialectNames) {}

private:
  /// The names of the dialects affected by this extension.
  SmallVector<StringRef> dialectNames;
};

/// This class represents a dialect extension anchored on the given set of
/// dialects. When all of the specified dialects have been loaded, the
/// application function of this extension will be executed.
template <typename DerivedT, typename... DialectsT>
class DialectExtension : public DialectExtensionBase {
public:
  /// Applies this extension to the given context and set of required dialects.
  virtual void apply(MLIRContext *context, DialectsT *...dialects) const = 0;

  /// Return a copy of this extension.
  std::unique_ptr<DialectExtensionBase> clone() const final {
    return std::make_unique<DerivedT>(static_cast<const DerivedT &>(*this));
  }

protected:
  DialectExtension()
      : DialectExtensionBase(
            ArrayRef<StringRef>({DialectsT::getDialectNamespace()...})) {}

  /// Override the base apply method to allow providing the exact dialect types.
  void apply(MLIRContext *context,
             MutableArrayRef<Dialect *> dialects) const final {
    unsigned dialectIdx = 0;
    auto derivedDialects = std::tuple<DialectsT *...>{
        static_cast<DialectsT *>(dialects[dialectIdx++])...};
    std::apply([&](DialectsT *...dialect) { apply(context, dialect...); },
               derivedDialects);
  }
};

namespace dialect_extension_detail {

/// Checks if the given interface, which is attempting to be used, is a
/// promised interface of this dialect that has yet to be implemented. If so,
/// emits a fatal error.
void handleUseOfUndefinedPromisedInterface(Dialect &dialect,
                                           TypeID interfaceRequestorID,
                                           TypeID interfaceID,
                                           StringRef interfaceName);

/// Checks if the given interface, which is attempting to be attached, is a
/// promised interface of this dialect that has yet to be implemented. If so,
/// the promised interface is marked as resolved.
void handleAdditionOfUndefinedPromisedInterface(Dialect &dialect,
                                                TypeID interfaceRequestorID,
                                                TypeID interfaceID);

/// Checks if a promise has been made for the interface/requestor pair.
bool hasPromisedInterface(Dialect &dialect, TypeID interfaceRequestorID,
                          TypeID interfaceID);

/// Checks if a promise has been made for the interface/requestor pair.
template <typename ConcreteT, typename InterfaceT>
bool hasPromisedInterface(Dialect &dialect) {
  return hasPromisedInterface(dialect, TypeID::get<ConcreteT>(),
                              InterfaceT::getInterfaceID());
}

} // namespace dialect_extension_detail

//===----------------------------------------------------------------------===//
// DialectRegistry
//===----------------------------------------------------------------------===//

/// The DialectRegistry maps a dialect namespace to a constructor for the
/// matching dialect. This allows for decoupling the list of dialects
/// "available" from the dialects loaded in the Context. The parser in
/// particular will lazily load dialects in the Context as operations are
/// encountered.
class DialectRegistry {
  using MapTy =
      std::map<std::string, std::pair<TypeID, DialectAllocatorFunction>>;

public:
  explicit DialectRegistry();

  template <typename ConcreteDialect>
  void insert() {
    insert(TypeID::get<ConcreteDialect>(),
           ConcreteDialect::getDialectNamespace(),
           static_cast<DialectAllocatorFunction>(([](MLIRContext *ctx) {
             // Just allocate the dialect, the context
             // takes ownership of it.
             return ctx->getOrLoadDialect<ConcreteDialect>();
           })));
  }

  template <typename ConcreteDialect, typename OtherDialect,
            typename... MoreDialects>
  void insert() {
    insert<ConcreteDialect>();
    insert<OtherDialect, MoreDialects...>();
  }

  /// Add a new dialect constructor to the registry. The constructor must be
  /// calling MLIRContext::getOrLoadDialect in order for the context to take
  /// ownership of the dialect and for delayed interface registration to happen.
  void insert(TypeID typeID, StringRef name,
              const DialectAllocatorFunction &ctor);

  /// Add a new dynamic dialect constructor in the registry. The constructor
  /// provides as argument the created dynamic dialect, and is expected to
  /// register the dialect types, attributes, and ops, using the
  /// methods defined in ExtensibleDialect such as registerDynamicOperation.
  void insertDynamic(StringRef name,
                     const DynamicDialectPopulationFunction &ctor);

  /// Return an allocation function for constructing the dialect identified
  /// by its namespace, or nullptr if the namespace is not in this registry.
  DialectAllocatorFunctionRef getDialectAllocator(StringRef name) const;

  // Register all dialects available in the current registry with the registry
  // in the provided context.
  void appendTo(DialectRegistry &destination) const {
    for (const auto &nameAndRegistrationIt : registry)
      destination.insert(nameAndRegistrationIt.second.first,
                         nameAndRegistrationIt.first,
                         nameAndRegistrationIt.second.second);
    // Merge the extensions.
    for (const auto &extension : extensions)
      destination.extensions.try_emplace(extension.first,
                                         extension.second->clone());
  }

  /// Return the names of dialects known to this registry.
  auto getDialectNames() const {
    return llvm::map_range(
        registry,
        [](const MapTy::value_type &item) -> StringRef { return item.first; });
  }

  /// Apply any held extensions that require the given dialect. Users are not
  /// expected to call this directly.
  void applyExtensions(Dialect *dialect) const;

  /// Apply any applicable extensions to the given context. Users are not
  /// expected to call this directly.
  void applyExtensions(MLIRContext *ctx) const;

  /// Add the given extension to the registry.
  bool addExtension(TypeID extensionID,
                    std::unique_ptr<DialectExtensionBase> extension) {
    return extensions.try_emplace(extensionID, std::move(extension)).second;
  }

  /// Add the given extensions to the registry.
  template <typename... ExtensionsT>
  void addExtensions() {
    (addExtension(TypeID::get<ExtensionsT>(), std::make_unique<ExtensionsT>()),
     ...);
  }

  /// Add an extension function that requires the given dialects.
  /// Note: This bare functor overload is provided in addition to the
  /// std::function variant to enable dialect type deduction, e.g.:
  ///  registry.addExtension(+[](MLIRContext *ctx, MyDialect *dialect) {
  ///  ... })
  ///
  /// is equivalent to:
  ///  registry.addExtension<MyDialect>(
  ///     [](MLIRContext *ctx, MyDialect *dialect){ ... }
  ///  )
  template <typename... DialectsT>
  bool addExtension(void (*extensionFn)(MLIRContext *, DialectsT *...)) {
    using ExtensionFnT = void (*)(MLIRContext *, DialectsT *...);

    struct Extension : public DialectExtension<Extension, DialectsT...> {
      Extension(const Extension &) = default;
      Extension(ExtensionFnT extensionFn)
          : DialectExtension<Extension, DialectsT...>(),
            extensionFn(extensionFn) {}
      ~Extension() override = default;

      void apply(MLIRContext *context, DialectsT *...dialects) const final {
        extensionFn(context, dialects...);
      }
      ExtensionFnT extensionFn;
    };
    return addExtension(TypeID::getFromOpaquePointer(
                            reinterpret_cast<const void *>(extensionFn)),
                        std::make_unique<Extension>(extensionFn));
  }

  /// Returns true if the current registry is a subset of 'rhs', i.e. if 'rhs'
  /// contains all of the components of this registry.
  bool isSubsetOf(const DialectRegistry &rhs) const;

private:
  MapTy registry;
  llvm::MapVector<TypeID, std::unique_ptr<DialectExtensionBase>> extensions;
};

} // namespace mlir

#endif // MLIR_IR_DIALECTREGISTRY_H
