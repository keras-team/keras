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

// This file declares KernelRegistry.

#ifndef TFRT_HOST_CONTEXT_KERNEL_REGISTRY_H_
#define TFRT_HOST_CONTEXT_KERNEL_REGISTRY_H_

#include <memory>

#include "llvm/ADT/PointerUnion.h"
#include "llvm/ADT/StringRef.h"
#include "tfrt/support/forward_decls.h"
#include "tfrt/support/variant.h"

namespace tfrt {

class TypeName;
class AsyncKernelFrame;
class SyncKernelFrame;
class HostContext;

// Kernel implementations use this signature, synchronously or asynchronously
// performing some computation and updating results.
using AsyncKernelImplementation = void (*)(AsyncKernelFrame* frame);
using SyncKernelImplementation = void (*)(SyncKernelFrame* frame);

using KernelImplementation =
    Variant<Monostate, AsyncKernelImplementation, SyncKernelImplementation>;

namespace internal {

template <typename TraitT>
AsyncKernelImplementation AsBEFKernel();

}  // namespace internal

// This represents a mapping between the names of the MLIR opcodes to the
// implementations of those functions, along with type mappings.
class KernelRegistry {
 public:
  ~KernelRegistry();
  KernelRegistry(const KernelRegistry&) = delete;
  KernelRegistry& operator=(const KernelRegistry&) = delete;

  // TODO: Rename AddKernel to AddAsyncKernel. This will involve touching a
  // large number of files, so it is better to do this in a separate CL.
  void AddKernel(string_view name, AsyncKernelImplementation fn);
  void AddSyncKernel(string_view name, SyncKernelImplementation fn);

  template <typename KernelTraitT>
  void AddKernel(string_view name) {
    AddKernel(name, internal::AsBEFKernel<KernelTraitT>());
  }

  KernelImplementation GetKernel(string_view name) const;

  TypeName GetType(string_view type) const;

 private:
  KernelRegistry();
  class Impl;
  std::unique_ptr<Impl> impl_;
  friend class HostContext;
};

// Use this macro to add a function that will register kernels that are
// statically linked in the binary. FUNC should be a function pointer with the
// prototype given by the tfrt::KernelRegistration alias.
#define TFRT_STATIC_KERNEL_REGISTRATION(FUNC) \
  TFRT_STATIC_KERNEL_REGISTRATION_(FUNC, __COUNTER__)
#define TFRT_STATIC_KERNEL_REGISTRATION_(FUNC, N) \
  TFRT_STATIC_KERNEL_REGISTRATION__(FUNC, N)
#define TFRT_STATIC_KERNEL_REGISTRATION__(FUNC, N)          \
  static bool tfrt_static_kernel_##N##_registered_ = []() { \
    ::tfrt::AddStaticKernelRegistration(FUNC);              \
    return true;                                            \
  }()

// The type for kernel registration functions. This is the same as the
// prototype for the entry point function for dynamic plugins.
using KernelRegistration = void (*)(KernelRegistry*);

// This is called to register all the statically linked kernels in the given
// registry.
void RegisterStaticKernels(KernelRegistry* kernel_reg);

// Adds a kernel to the registry. This should not be used directly; use
// TFRT_STATIC_KERNEL_REGISTRATION instead.
void AddStaticKernelRegistration(KernelRegistration func);

}  // namespace tfrt

#endif  // TFRT_HOST_CONTEXT_KERNEL_REGISTRY_H_
