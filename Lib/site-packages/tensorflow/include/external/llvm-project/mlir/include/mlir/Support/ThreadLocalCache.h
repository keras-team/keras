//===- ThreadLocalCache.h - ThreadLocalCache class --------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains a definition of the ThreadLocalCache class. This class
// provides support for defining thread local objects with non-static duration.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_SUPPORT_THREADLOCALCACHE_H
#define MLIR_SUPPORT_THREADLOCALCACHE_H

#include "mlir/Support/LLVM.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/Support/ManagedStatic.h"
#include "llvm/Support/Mutex.h"

namespace mlir {
/// This class provides support for defining a thread local object with non
/// static storage duration. This is very useful for situations in which a data
/// cache has very large lock contention.
template <typename ValueT>
class ThreadLocalCache {
  struct PerInstanceState;

  using PointerAndFlag = std::pair<ValueT *, std::atomic<bool>>;

  /// The "observer" is owned by a thread-local cache instance. It is
  /// constructed the first time a `ThreadLocalCache` instance is accessed by a
  /// thread, unless `perInstanceState` happens to get re-allocated to the same
  /// address as a previous one. A `thread_local` instance of this class is
  /// destructed when the thread in which it lives is destroyed.
  ///
  /// This class is called the "observer" because while values cached in
  /// thread-local caches are owned by `PerInstanceState`, a reference is stored
  /// via this class in the TLC. With a double pointer, it knows when the
  /// referenced value has been destroyed.
  struct Observer {
    /// This is the double pointer, explicitly allocated because we need to keep
    /// the address stable if the TLC map re-allocates. It is owned by the
    /// observer and shared with the value owner.
    std::shared_ptr<PointerAndFlag> ptr =
        std::make_shared<PointerAndFlag>(std::make_pair(nullptr, false));
    /// Because the `Owner` instance that lives inside `PerInstanceState`
    /// contains a reference to the double pointer, and likewise this class
    /// contains a reference to the value, we need to synchronize destruction of
    /// the TLC and the `PerInstanceState` to avoid racing. This weak pointer is
    /// acquired during TLC destruction if the `PerInstanceState` hasn't entered
    /// its destructor yet, and prevents it from happening.
    std::weak_ptr<PerInstanceState> keepalive;
  };

  /// This struct owns the cache entries. It contains a reference back to the
  /// reference inside the cache so that it can be written to null to indicate
  /// that the cache entry is invalidated. It needs to do this because
  /// `perInstanceState` could get re-allocated to the same pointer and we don't
  /// remove entries from the TLC when it is deallocated. Thus, we have to reset
  /// the TLC entries to a starting state in case the `ThreadLocalCache` lives
  /// shorter than the threads.
  struct Owner {
    /// Save a pointer to the reference and write it to the newly created entry.
    Owner(Observer &observer)
        : value(std::make_unique<ValueT>()), ptrRef(observer.ptr) {
      observer.ptr->second = true;
      observer.ptr->first = value.get();
    }
    ~Owner() {
      if (std::shared_ptr<PointerAndFlag> ptr = ptrRef.lock()) {
        ptr->first = nullptr;
        ptr->second = false;
      }
    }

    Owner(Owner &&) = default;
    Owner &operator=(Owner &&) = default;

    std::unique_ptr<ValueT> value;
    std::weak_ptr<PointerAndFlag> ptrRef;
  };

  // Keep a separate shared_ptr protected state that can be acquired atomically
  // instead of using shared_ptr's for each value. This avoids a problem
  // where the instance shared_ptr is locked() successfully, and then the
  // ThreadLocalCache gets destroyed before remove() can be called successfully.
  struct PerInstanceState {
    /// Remove the given value entry. This is called when a thread local cache
    /// is destructing but still contains references to values owned by the
    /// `PerInstanceState`. Removal is required because it prevents writeback to
    /// a pointer that was deallocated.
    void remove(ValueT *value) {
      // Erase the found value directly, because it is guaranteed to be in the
      // list.
      llvm::sys::SmartScopedLock<true> threadInstanceLock(instanceMutex);
      auto it = llvm::find_if(instances, [&](Owner &instance) {
        return instance.value.get() == value;
      });
      assert(it != instances.end() && "expected value to exist in cache");
      instances.erase(it);
    }

    /// Owning pointers to all of the values that have been constructed for this
    /// object in the static cache.
    SmallVector<Owner, 1> instances;

    /// A mutex used when a new thread instance has been added to the cache for
    /// this object.
    llvm::sys::SmartMutex<true> instanceMutex;
  };

  /// The type used for the static thread_local cache. This is a map between an
  /// instance of the non-static cache and a weak reference to an instance of
  /// ValueT. We use a weak reference here so that the object can be destroyed
  /// without needing to lock access to the cache itself.
  struct CacheType : public llvm::SmallDenseMap<PerInstanceState *, Observer> {
    ~CacheType() {
      // Remove the values of this cache that haven't already expired. This is
      // required because if we don't remove them, they will contain a reference
      // back to the data here that is being destroyed.
      for (auto &[instance, observer] : *this)
        if (std::shared_ptr<PerInstanceState> state = observer.keepalive.lock())
          state->remove(observer.ptr->first);
    }

    /// Clear out any unused entries within the map. This method is not
    /// thread-safe, and should only be called by the same thread as the cache.
    void clearExpiredEntries() {
      for (auto it = this->begin(), e = this->end(); it != e;) {
        auto curIt = it++;
        if (!curIt->second.ptr->second)
          this->erase(curIt);
      }
    }
  };

public:
  ThreadLocalCache() = default;
  ~ThreadLocalCache() {
    // No cleanup is necessary here as the shared_pointer memory will go out of
    // scope and invalidate the weak pointers held by the thread_local caches.
  }

  /// Return an instance of the value type for the current thread.
  ValueT &get() {
    // Check for an already existing instance for this thread.
    CacheType &staticCache = getStaticCache();
    Observer &threadInstance = staticCache[perInstanceState.get()];
    if (ValueT *value = threadInstance.ptr->first)
      return *value;

    // Otherwise, create a new instance for this thread.
    {
      llvm::sys::SmartScopedLock<true> threadInstanceLock(
          perInstanceState->instanceMutex);
      perInstanceState->instances.emplace_back(threadInstance);
    }
    threadInstance.keepalive = perInstanceState;

    // Before returning the new instance, take the chance to clear out any used
    // entries in the static map. The cache is only cleared within the same
    // thread to remove the need to lock the cache itself.
    staticCache.clearExpiredEntries();
    return *threadInstance.ptr->first;
  }
  ValueT &operator*() { return get(); }
  ValueT *operator->() { return &get(); }

private:
  ThreadLocalCache(ThreadLocalCache &&) = delete;
  ThreadLocalCache(const ThreadLocalCache &) = delete;
  ThreadLocalCache &operator=(const ThreadLocalCache &) = delete;

  /// Return the static thread local instance of the cache type.
  static CacheType &getStaticCache() {
    static LLVM_THREAD_LOCAL CacheType cache;
    return cache;
  }

  std::shared_ptr<PerInstanceState> perInstanceState =
      std::make_shared<PerInstanceState>();
};
} // namespace mlir

#endif // MLIR_SUPPORT_THREADLOCALCACHE_H
