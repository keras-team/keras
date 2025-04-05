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

// Generic future type used by HostContext
//
// This file declares AsyncValue, a generic "future" type that can be fulfilled
// by an asynchronously provided value or an error.  This is called AsyncValue
// instead of 'future' because there are many different types of futures
// (including std::future) which do not share the same characteristics of this
// type.

#ifndef TFRT_HOST_CONTEXT_ASYNC_VALUE_H_
#define TFRT_HOST_CONTEXT_ASYNC_VALUE_H_

#include "tfrt/concurrency/async_value.h"

namespace tfrt {

using ::tsl::AsyncValue;  // NOLINT

namespace internal {
using ::tsl::internal::ConcreteAsyncValue;  // NOLINT
}  // namespace internal

using ::tsl::DummyValueForErrorAsyncValue;  // NOLINT
using ::tsl::ErrorAsyncValue;               // NOLINT
using ::tsl::IndirectAsyncValue;            // NOLINT

}  // namespace tfrt

#endif  // TFRT_HOST_CONTEXT_ASYNC_VALUE_H_
