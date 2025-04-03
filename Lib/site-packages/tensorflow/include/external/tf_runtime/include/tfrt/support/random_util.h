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

// Declares a random 64-bit number generator.

#ifndef TFRT_SUPPORT_RANDOM_H_
#define TFRT_SUPPORT_RANDOM_H_

#include <cstdint>

namespace tfrt {
namespace random {

// Return a 64-bit random value.  Different sequences are generated
// in different processes.
uint64_t New64();

}  // namespace random
}  // namespace tfrt

#endif  // TFRT_SUPPORT_RANDOM_H_
