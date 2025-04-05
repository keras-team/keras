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

// RCReference<AsyncValue> wrapper
//
// AsyncValueRef<T> is an alias for RCReference<AsyncValue> that carries payload
// type information. The user does not need to pass the payload data type to
// get() or emplace().
//
// Like RCReference<AsyncValue>, it represents one reference on the underlying
// AsyncValue. When a callee returns an AsyncValueRef to a caller, the callee
// also transfers their ownership of a reference on the underlying AsyncValue.

#ifndef TFRT_HOST_CONTEXT_ASYNC_VALUE_REF_H_
#define TFRT_HOST_CONTEXT_ASYNC_VALUE_REF_H_

#include "tfrt/concurrency/async_value_ref.h"
#include "tfrt/host_context/async_value.h"

namespace tfrt {

using ::tsl::AsyncValuePtr;           // NOLINT
using ::tsl::AsyncValueRef;           // NOLINT
using ::tsl::MakeErrorAsyncValueRef;  // NOLINT
using ::tsl::MakeIndirectAsyncValue;  // NOLINT

namespace internal {
using ::tsl::internal::AllocateAndConstruct;  // NOLINT
using ::tsl::internal::PlacementConstruct;    // NOLINT
}  // namespace internal

using ::tsl::MakeAvailableAsyncValueRef;      // NOLINT
using ::tsl::MakeConstructedAsyncValueRef;    // NOLINT
using ::tsl::MakeUnconstructedAsyncValueRef;  // NOLINT

namespace internal {
using ::tsl::internal::AsyncValueStorage;  // NOLINT
}  // namespace internal

using ::tsl::AsyncValueOwningRef;  // NOLINT

}  // namespace tfrt

#endif  // TFRT_HOST_CONTEXT_ASYNC_VALUE_REF_H_
