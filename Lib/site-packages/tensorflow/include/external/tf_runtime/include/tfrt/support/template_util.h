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

// This file provides templates to define typenames.

#ifndef TFRT_SUPPORT_TEMPLATE_UTIL_H_
#define TFRT_SUPPORT_TEMPLATE_UTIL_H_

#include <cstddef>

namespace tfrt {

// Return typename T<ElementType, ...> where ElementType is repeated N times in
// the template parameter pack.
template <template <typename...> class T, std::size_t N, typename ElementType,
          typename... ElementTypes>
struct RepeatTypeHelper
    : RepeatTypeHelper<T, N - 1U, ElementType, ElementType, ElementTypes...> {};

// Specialization for the base case
template <template <typename...> class T, typename ElementType,
          typename... ElementTypes>
struct RepeatTypeHelper<T, 0U, ElementType, ElementTypes...> {
  using type = T<ElementTypes...>;
};

template <template <typename...> class T, std::size_t N, typename ElementType>
using RepeatTypeHelperT = typename RepeatTypeHelper<T, N, ElementType>::type;

}  // namespace tfrt

#endif  // TFRT_SUPPORT_TEMPLATE_UTIL_H_
