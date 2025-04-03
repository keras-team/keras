/*
 * Copyright 2020 The TensorFlow Runtime Authors
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

// Compact pointer to HostContext
//
// This file declares HostContextPtr, a compact pointer representation for
// HostContext.

#ifndef TFRT_HOST_CONTEXT_HOST_CONTEXT_PTR_H_
#define TFRT_HOST_CONTEXT_HOST_CONTEXT_PTR_H_

#include <array>
#include <cassert>
#include <cstdint>

#include "tfrt/support/mutex.h"

namespace tfrt {

class HostContext;
class HostContextPtr;

// HostContextPool manages all the live HostContext instances. It limits the
// number of live HostContext instances to 256 to allow referecing a HostContext
// with a 1-byte int. This is used to keep sizeof(HostContextPtr) to 1 byte.
class HostContextPool {
 public:
  static constexpr int kCompacity = 256;

  static HostContextPool& instance() {
    static HostContextPool* pool = new HostContextPool();
    return *pool;
  }

  HostContextPtr AllocateForHostContext(HostContext* host);
  void FreeHostContext(HostContext* host);

  HostContext* GetHostContextByIndex(int index) const;

 private:
  HostContextPool() = default;

  mutable mutex mutex_;
  std::array<HostContext*, kCompacity> all_host_contexts_;
};

// HostContextPtr implements a compact pointer for a HostContext by storing the
// instance index of the HostContext object. It is intended to be used in places
// where saving the memory space is important, otherwise, HostContext* should be
// used.
class HostContextPtr {
 public:
  // Implicitly convert HostContext* to HostContextPtr.
  HostContextPtr(HostContext* host);  // NOLINT

  HostContext* operator->() const { return get(); }

  HostContext& operator*() const { return *get(); }

  HostContext* get() const;

 private:
  friend class HostContextPool;
  friend class ReadyChain;

  explicit HostContextPtr(int index) : index_{static_cast<uint8_t>(index)} {
    assert(index < HostContextPool::kCompacity);
  }
  uint8_t index() const { return index_; }

  const uint8_t index_ = 0;
};

}  // namespace tfrt

#endif  // TFRT_HOST_CONTEXT_HOST_CONTEXT_PTR_H_
