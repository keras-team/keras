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

// This file implements the ranges concept similar to the one available in
// C++20.

#ifndef TFRT_SUPPORT_RANGES_H_
#define TFRT_SUPPORT_RANGES_H_

#include <cassert>
#include <cstddef>
#include <iterator>

#include "llvm/ADT/STLExtras.h"

namespace tfrt {
namespace views {

// A simple implementation for std::views::counted in C++ 20.
//
// Usage example:
//
// int a[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
// for(int i : views::Counted(a, 3))
//   std::cout << i << ' ';
//
// const auto il = {1, 2, 3, 4, 5};
// for (int i : views::Counted(il.begin() + 1, 3))
//   std::cout << i << ' ';
template <typename IteratorT>
class CountedView {
 public:
  using value_type = typename std::iterator_traits<IteratorT>::value_type;

  CountedView(IteratorT it, size_t count) : it_{it}, count_{count} {}

  IteratorT begin() const { return it_; }
  IteratorT end() const { return it_ + count_; }
  size_t size() const { return count_; }

  decltype(auto) operator[](size_t index) const {
    assert(index < count_);
    return *(it_ + index);
  }

 private:
  IteratorT it_;
  size_t count_;
};

template <typename IteratorT>
CountedView<IteratorT> Counted(IteratorT it, size_t count) {
  return CountedView<IteratorT>(it, count);
}

}  // namespace views

// Base class to implement range that is based on indexing into a BaseT.
// See `llvm::detail::indexed_accessor_range_base` for details.
template <typename DerivedT, typename BaseT, typename T,
          typename PointerT = T *, typename ReferenceT = T &>
using IndexedAccessorRangeBase =
    llvm::detail::indexed_accessor_range_base<DerivedT, BaseT, T, PointerT,
                                              ReferenceT>;

}  // namespace tfrt

#endif  // TFRT_SUPPORT_RANGES_H_
