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

// CPU thread and memory abstraction
//
// This file declares HostContext.

#ifndef TFRT_HOST_CONTEXT_HOST_CONTEXT_H_
#define TFRT_HOST_CONTEXT_HOST_CONTEXT_H_

#include <type_traits>

#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/Compiler.h"
#include "tfrt/host_context/async_value_ref.h"
#include "tfrt/host_context/device.h"
#include "tfrt/host_context/diagnostic.h"
#include "tfrt/host_context/host_context_ptr.h"
#include "tfrt/host_context/kernel_registry.h"
#include "tfrt/host_context/timer_queue.h"

namespace tfrt {

struct DecodedDiagnostic;

class ConcurrentWorkQueue;
class HostAllocator;
class TypeDescriptor;
class SharedContext;

// This represents one instance of a CPU device, which can have multiple
// threads, a private heap for tensor data, and a way of reporting errors.  We
// limit the maximum number of HostContext objects that can be created in a
// process to kDummyIndex in order to allow encoding a HostContext
// pointer using only one byte (See HostContextPtr). A HostContext instance is
// expected to be re-used through the life-time of a process, so the limited
// instance numbers are not expected to be a problem in practice.
class HostContext {
 public:
  // The host device name that we will use if the caller does not specify the
  // device name.
  static const char* const kDefaultHostDeviceName;

  HostContext(std::function<void(const DecodedDiagnostic&)> diag_handler,
              std::unique_ptr<HostAllocator> allocator,
              std::unique_ptr<ConcurrentWorkQueue> work_queue,
              string_view host_device_name);

  // This constructor uses "CPU:0" as the default host device name.
  HostContext(std::function<void(const DecodedDiagnostic&)> diag_handler,
              std::unique_ptr<HostAllocator> allocator,
              std::unique_ptr<ConcurrentWorkQueue> work_queue);

  HostContext(const HostContext&) = delete;
  HostContext& operator=(const HostContext&) = delete;
  ~HostContext();

  //===--------------------------------------------------------------------===//
  // Error Reporting
  //===--------------------------------------------------------------------===//

  // Emit an error for a specified decoded diagnostic, which gets funneled
  // through a location handler.
  void EmitError(const DecodedDiagnostic& diagnostic);

  std::function<void(const DecodedDiagnostic&)> diag_handler() {
    return diag_handler_;
  }

  //===--------------------------------------------------------------------===//
  // Memory Management
  //===--------------------------------------------------------------------===//

  HostAllocator* allocator() { return allocator_.get(); }

  // Allocate the specified number of bytes with the specified alignment.
  void* AllocateBytes(size_t size, size_t alignment);

  // Deallocate the specified pointer that had the specified size.
  void DeallocateBytes(void* ptr, size_t size);

  // Allocate memory for one or more entries of type T.
  template <typename T>
  T* Allocate(size_t num_elements = 1) {
    return static_cast<T*>(AllocateBytes(sizeof(T) * num_elements, alignof(T)));
  }

  // Deallocate the memory for one or more entries of type T.
  template <typename T>
  void Deallocate(T* ptr, size_t num_elements = 1) {
    DeallocateBytes(ptr, sizeof(T) * num_elements);
  }

  // Allocate and initialize an object of type T.
  template <typename T, typename... Args>
  T* Construct(Args&&... args) {
    T* buf = Allocate<T>();
    return new (buf) T(std::forward<Args>(args)...);
  }

  // Destruct and deallocate space for an object of type T.
  template <typename T>
  void Destruct(T* t) {
    t->~T();
    Deallocate(t);
  }

  //===--------------------------------------------------------------------===//
  // Concurrency
  //===--------------------------------------------------------------------===//

  // Block until the specified values are available (either with a value or an
  // error result).
  //
  // This should not be called by a thread managed by the work queue.
  void Await(ArrayRef<RCReference<AsyncValue>> values);

  // Block until the system is quiescent (no pending work and no inflight work).
  //
  // This should not be called by a thread managed by the work queue.
  void Quiesce();

  // Returns the number of worker threads in the work_queue managed by this CPU
  // device. This does not include any additional threads that might have been
  // created to handle blocking work (enqueued by EnqueueBlockingWork).
  int GetNumWorkerThreads() const;

  // Returns true if the caller thread is one of the work queue threads managed
  // by this context. Returns true only for threads executing non-blocking work.
  bool IsInWorkerThread() const;

  //===--------------------------------------------------------------------===//
  // Shared context
  //===--------------------------------------------------------------------===//
  // Get the shared context instance managed by the host context. Create one, if
  // the shared context instance does not exist yet.
  template <typename SharedContextType>
  SharedContextType& GetOrCreateSharedContext();

  //===--------------------------------------------------------------------===//
  // Kernel Registry
  //===--------------------------------------------------------------------===//
  const KernelRegistry& GetKernelRegistry() { return registry_; }

  KernelRegistry* GetMutableRegistry() { return &registry_; }

  //===--------------------------------------------------------------------===//
  // Device Manager
  //===--------------------------------------------------------------------===//
  DeviceManager* GetDeviceManager() { return &device_mgr_; }

  RCReference<Device> GetHostDeviceRef();
  const Device& GetHostDevice();

  void ResetHostDevice(CpuDevice* device);

  RCReference<Device> GetDeviceRef(string_view device);

  ConcurrentWorkQueue& work_queue() const { return *work_queue_; }

  //===--------------------------------------------------------------------===//
  // TimerQueue
  //===--------------------------------------------------------------------===//
  TimerQueue* GetTimerQueue() { return &timer_queue_; }

 private:
  friend class HostContextPtr;
  friend class HostContextPool;

  // Factory function for creating a SharedContext.
  using SharedContextFactory = std::unique_ptr<SharedContext> (*)(HostContext*);

  class SharedContextManager;

  // Dense ID for different shared context types.
  template <typename SharedContextType>
  static int DenseIdForSharedContext();

  static std::atomic<int> num_shared_context_types_;

  HostContextPtr instance_ptr() const { return instance_ptr_; }

  SharedContext& GetOrCreateSharedContext(int shared_context_id,
                                          SharedContextFactory factory);

  KernelRegistry registry_;
  DeviceManager device_mgr_;
  RCReference<Device> host_device_;
  std::function<void(const DecodedDiagnostic&)> diag_handler_;
  std::unique_ptr<HostAllocator> allocator_;
  std::unique_ptr<ConcurrentWorkQueue> work_queue_;

  std::unique_ptr<SharedContextManager> shared_context_mgr_;
  TimerQueue timer_queue_;
  const HostContextPtr instance_ptr_;
};

template <typename SharedContextType>
SharedContextType& HostContext::GetOrCreateSharedContext() {
  int shared_context_id = DenseIdForSharedContext<SharedContextType>();
  auto factory = [](HostContext* host) -> std::unique_ptr<SharedContext> {
    return std::make_unique<SharedContextType>(host);
  };
  return static_cast<SharedContextType&>(
      GetOrCreateSharedContext(shared_context_id, factory));
}

template <typename SharedContextType>
int HostContext::DenseIdForSharedContext() {
  static int id = num_shared_context_types_++;
  return id;
}

// HostContext free function to Allocate and initialize an object of type T.
template <typename T, typename... Args>
T* HostContextConstruct(HostContext* host, Args&&... args) {
  return host->Construct<T>(std::forward<Args>(args)...);
}

}  // namespace tfrt

#endif  // TFRT_HOST_CONTEXT_HOST_CONTEXT_H_
