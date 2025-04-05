//===- DialectResourceBlobManager.h - Dialect Blob Management ---*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines utility classes for referencing and managing asm resource
// blobs. These classes are intended to more easily facilitate the sharing of
// large blobs, and their definition.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_IR_DIALECTRESOURCEBLOBMANAGER_H
#define MLIR_IR_DIALECTRESOURCEBLOBMANAGER_H

#include "mlir/IR/AsmState.h"
#include "mlir/IR/OpImplementation.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/RWMutex.h"
#include "llvm/Support/SMLoc.h"
#include <optional>

namespace mlir {
//===----------------------------------------------------------------------===//
// DialectResourceBlobManager
//===---------------------------------------------------------------------===//

/// This class defines a manager for dialect resource blobs. Blobs are uniqued
/// by a given key, and represented using AsmResourceBlobs.
class DialectResourceBlobManager {
public:
  /// The class represents an individual entry of a blob.
  class BlobEntry {
  public:
    /// Return the key used to reference this blob.
    StringRef getKey() const { return key; }

    /// Return the blob owned by this entry if one has been initialized. Returns
    /// nullptr otherwise.
    const AsmResourceBlob *getBlob() const { return blob ? &*blob : nullptr; }
    AsmResourceBlob *getBlob() { return blob ? &*blob : nullptr; }

    /// Set the blob owned by this entry.
    void setBlob(AsmResourceBlob &&newBlob) { blob = std::move(newBlob); }

  private:
    BlobEntry() = default;
    BlobEntry(BlobEntry &&) = default;
    BlobEntry &operator=(const BlobEntry &) = delete;
    BlobEntry &operator=(BlobEntry &&) = delete;

    /// Initialize this entry with the given key and blob.
    void initialize(StringRef newKey, std::optional<AsmResourceBlob> newBlob) {
      key = newKey;
      blob = std::move(newBlob);
    }

    /// The key used for this blob.
    StringRef key;

    /// The blob that is referenced by this entry if it is valid.
    std::optional<AsmResourceBlob> blob;

    /// Allow access to the constructors.
    friend DialectResourceBlobManager;
    friend class llvm::StringMapEntryStorage<BlobEntry>;
  };

  /// Return the blob registered for the given name, or nullptr if no blob
  /// is registered.
  BlobEntry *lookup(StringRef name);
  const BlobEntry *lookup(StringRef name) const {
    return const_cast<DialectResourceBlobManager *>(this)->lookup(name);
  }

  /// Update the blob for the entry defined by the provided name. This method
  /// asserts that an entry for the given name exists in the manager.
  void update(StringRef name, AsmResourceBlob &&newBlob);

  /// Insert a new entry with the provided name and optional blob data. The name
  /// may be modified during insertion if another entry already exists with that
  /// name. Returns the inserted entry.
  BlobEntry &insert(StringRef name, std::optional<AsmResourceBlob> blob = {});
  /// Insertion method that returns a dialect specific handle to the inserted
  /// entry.
  template <typename HandleT>
  HandleT insert(typename HandleT::Dialect *dialect, StringRef name,
                 std::optional<AsmResourceBlob> blob = {}) {
    BlobEntry &entry = insert(name, std::move(blob));
    return HandleT(&entry, dialect);
  }

private:
  /// A mutex to protect access to the blob map.
  llvm::sys::SmartRWMutex<true> blobMapLock;

  /// The internal map of tracked blobs. StringMap stores entries in distinct
  /// allocations, so we can freely take references to the data without fear of
  /// invalidation during additional insertion/deletion.
  llvm::StringMap<BlobEntry> blobMap;
};

//===----------------------------------------------------------------------===//
// ResourceBlobManagerDialectInterface
//===---------------------------------------------------------------------===//

/// This class implements a dialect interface that provides common functionality
/// for interacting with a resource blob manager.
class ResourceBlobManagerDialectInterface
    : public DialectInterface::Base<ResourceBlobManagerDialectInterface> {
public:
  ResourceBlobManagerDialectInterface(Dialect *dialect)
      : Base(dialect),
        blobManager(std::make_shared<DialectResourceBlobManager>()) {}

  /// Return the blob manager held by this interface.
  DialectResourceBlobManager &getBlobManager() { return *blobManager; }
  const DialectResourceBlobManager &getBlobManager() const {
    return *blobManager;
  }

  /// Set the blob manager held by this interface.
  void
  setBlobManager(std::shared_ptr<DialectResourceBlobManager> newBlobManager) {
    blobManager = std::move(newBlobManager);
  }

private:
  /// The blob manager owned by the dialect implementing this interface.
  std::shared_ptr<DialectResourceBlobManager> blobManager;
};

/// This class provides a base class for dialects implementing the resource blob
/// interface. It provides several additional dialect specific utilities on top
/// of the generic interface. `HandleT` is the type of the handle used to
/// reference a resource blob.
template <typename HandleT>
class ResourceBlobManagerDialectInterfaceBase
    : public ResourceBlobManagerDialectInterface {
public:
  using ResourceBlobManagerDialectInterface::
      ResourceBlobManagerDialectInterface;

  /// Update the blob for the entry defined by the provided name. This method
  /// asserts that an entry for the given name exists in the manager.
  void update(StringRef name, AsmResourceBlob &&newBlob) {
    getBlobManager().update(name, std::move(newBlob));
  }

  /// Insert a new resource blob entry with the provided name and optional blob
  /// data. The name may be modified during insertion if another entry already
  /// exists with that name. Returns a dialect specific handle to the inserted
  /// entry.
  HandleT insert(StringRef name, std::optional<AsmResourceBlob> blob = {}) {
    return getBlobManager().template insert<HandleT>(
        cast<typename HandleT::Dialect>(getDialect()), name, std::move(blob));
  }

  /// Build resources for each of the referenced blobs within this manager.
  void buildResources(AsmResourceBuilder &provider,
                      ArrayRef<AsmDialectResourceHandle> referencedResources) {
    for (const AsmDialectResourceHandle &handle : referencedResources) {
      if (const auto *dialectHandle = dyn_cast<HandleT>(&handle)) {
        if (auto *blob = dialectHandle->getBlob())
          provider.buildBlob(dialectHandle->getKey(), *blob);
      }
    }
  }
};

//===----------------------------------------------------------------------===//
// DialectResourceBlobHandle
//===----------------------------------------------------------------------===//

/// This class defines a dialect specific handle to a resource blob. These
/// handles utilize a StringRef for the internal key, and an AsmResourceBlob as
/// the underlying data.
template <typename DialectT>
struct DialectResourceBlobHandle
    : public AsmDialectResourceHandleBase<DialectResourceBlobHandle<DialectT>,
                                          DialectResourceBlobManager::BlobEntry,
                                          DialectT> {
  using AsmDialectResourceHandleBase<DialectResourceBlobHandle<DialectT>,
                                     DialectResourceBlobManager::BlobEntry,
                                     DialectT>::AsmDialectResourceHandleBase;
  using ManagerInterface = ResourceBlobManagerDialectInterfaceBase<
      DialectResourceBlobHandle<DialectT>>;

  /// Return the human readable string key for this handle.
  StringRef getKey() const { return this->getResource()->getKey(); }

  /// Return the blob referenced by this handle if the underlying resource has
  /// been initialized. Returns nullptr otherwise.
  AsmResourceBlob *getBlob() { return this->getResource()->getBlob(); }
  const AsmResourceBlob *getBlob() const {
    return this->getResource()->getBlob();
  }

  /// Get the interface for the dialect that owns handles of this type. Asserts
  /// that the dialect is registered.
  static ManagerInterface &getManagerInterface(MLIRContext *ctx) {
    auto *dialect = ctx->getOrLoadDialect<DialectT>();
    assert(dialect && "dialect not registered");

    auto *iface = dialect->template getRegisteredInterface<ManagerInterface>();
    assert(iface && "dialect doesn't provide the blob manager interface?");
    return *iface;
  }
};

} // namespace mlir

#endif // MLIR_IR_DIALECTRESOURCEBLOBMANAGER_H
