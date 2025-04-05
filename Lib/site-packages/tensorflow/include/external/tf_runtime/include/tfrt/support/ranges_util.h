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

// This file introduces utilities for working with ranges.

#ifndef TFRT_SUPPORT_RANGES_UTIL_H_
#define TFRT_SUPPORT_RANGES_UTIL_H_

#include <vector>

#include "llvm/ADT/SmallVector.h"
#include "tfrt/support/ranges.h"

namespace tfrt {

namespace detail {
template <typename VectorT, typename RangeT>
VectorT CopyRefToVector(const RangeT& range) {
  VectorT copy;
  copy.reserve(range.size());
  for (auto& v : range) {
    copy.emplace_back(v.CopyRef());
  }

  return copy;
}
}  // namespace detail

// Copy the values in the range into a SmallVector.
template <size_t n, typename RangeT>
llvm::SmallVector<typename RangeT::value_type, n> AsSmallVector(
    const RangeT& range) {
  return {range.begin(), range.end()};
}

template <size_t n, typename RangeT>
llvm::SmallVector<typename RangeT::value_type, n> CopyRefToSmallVector(
    const RangeT& range) {
  return detail::CopyRefToVector<
      llvm::SmallVector<typename RangeT::value_type, n>>(range);
}

// Copy the values in the range into a std::vector.
template <typename RangeT>
std::vector<typename RangeT::value_type> AsVector(const RangeT& range) {
  return {range.begin(), range.end()};
}

// Copy the values in the range into a std::vector.
template <typename RangeT>
std::vector<typename RangeT::value_type> CopyRefToVector(const RangeT& range) {
  return detail::CopyRefToVector<std::vector<typename RangeT::value_type>>(
      range);
}

}  // namespace tfrt

#endif  // TFRT_SUPPORT_RANGES_UTIL_H_
