/*
 *
 * Copyright 2018 gRPC authors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 */

#ifndef GRPC_CORE_TSI_SSL_SESSION_CACHE_SSL_SESSION_CACHE_H
#define GRPC_CORE_TSI_SSL_SESSION_CACHE_SSL_SESSION_CACHE_H

#include <grpc/support/port_platform.h>

#include "src/core/tsi/grpc_shadow_boringssl.h"

#include <grpc/slice.h>
#include <grpc/support/sync.h>

extern "C" {
#include <openssl/ssl.h>
}

#include "src/core/lib/avl/avl.h"
#include "src/core/lib/gprpp/memory.h"
#include "src/core/lib/gprpp/ref_counted.h"
#include "src/core/tsi/ssl/session_cache/ssl_session.h"

/// Cache for SSL sessions for sessions resumption.
///
/// Older sessions may be evicted from the cache using LRU policy if capacity
/// limit is hit. All sessions are associated with some key, usually server
/// name. Note that servers are required to share session ticket encryption keys
/// in order for cache to be effective.
///
/// This class is thread safe.

namespace tsi {

class SslSessionLRUCache : public grpc_core::RefCounted<SslSessionLRUCache> {
 public:
  /// Create new LRU cache with the given capacity.
  static grpc_core::RefCountedPtr<SslSessionLRUCache> Create(size_t capacity) {
    return grpc_core::MakeRefCounted<SslSessionLRUCache>(capacity);
  }

  // Use Create function instead of using this directly.
  explicit SslSessionLRUCache(size_t capacity);
  ~SslSessionLRUCache();

  // Not copyable nor movable.
  SslSessionLRUCache(const SslSessionLRUCache&) = delete;
  SslSessionLRUCache& operator=(const SslSessionLRUCache&) = delete;

  /// Returns current number of sessions in the cache.
  size_t Size();
  /// Add \a session in the cache using \a key. This operation may discard older
  /// sessions.
  void Put(const char* key, SslSessionPtr session);
  /// Returns the session from the cache associated with \a key or null if not
  /// found.
  SslSessionPtr Get(const char* key);

 private:
  class Node;

  Node* FindLocked(const grpc_slice& key);
  void Remove(Node* node);
  void PushFront(Node* node);
  void AssertInvariants();

  gpr_mu lock_;
  size_t capacity_;

  Node* use_order_list_head_ = nullptr;
  Node* use_order_list_tail_ = nullptr;
  size_t use_order_list_size_ = 0;
  grpc_avl entry_by_key_;
};

}  // namespace tsi

#endif /* GRPC_CORE_TSI_SSL_SESSION_CACHE_SSL_SESSION_CACHE_H */
