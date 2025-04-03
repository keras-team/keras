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

// Helpers and utilities for working with reference counted types.

#ifndef TFRT_SUPPORT_REF_COUNT_H_
#define TFRT_SUPPORT_REF_COUNT_H_

#include "tfrt/concurrency/ref_count.h"

namespace tfrt {

#ifndef NDEBUG
using ::tsl::GetNumReferenceCountedObjects;    // NOLINT
using ::tsl::total_reference_counted_objects;  // NOLINT
#endif
using ::tsl::AddNumReferenceCountedObjects;   // NOLINT
using ::tsl::DropNumReferenceCountedObjects;  // NOLINT

using ::tsl::FormRef;           // NOLINT
using ::tsl::MakeRef;           // NOLINT
using ::tsl::RCReference;       // NOLINT
using ::tsl::ReferenceCounted;  // NOLINT
using ::tsl::swap;              // NOLINT
using ::tsl::TakeRef;           // NOLINT

}  // namespace tfrt

#endif  // TFRT_SUPPORT_REF_COUNT_H_
