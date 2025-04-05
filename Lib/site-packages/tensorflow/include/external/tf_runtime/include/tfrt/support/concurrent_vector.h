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

// A concurent sequential container optimized for read access.

#ifndef TFRT_SUPPORT_CONCURRENT_VECTOR_H_
#define TFRT_SUPPORT_CONCURRENT_VECTOR_H_

#include "tfrt/concurrency/concurrent_vector.h"

namespace tfrt {

using ::tsl::internal::ConcurrentVector;  // NOLINT

}  // namespace tfrt
#endif  // TFRT_SUPPORT_CONCURRENT_VECTOR_H_
