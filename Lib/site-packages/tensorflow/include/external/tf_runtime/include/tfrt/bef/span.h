/*
 * Copyright 2022 The TensorFlow Runtime Authors
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
#ifndef TFRT_BEF_SPAN_H_
#define TFRT_BEF_SPAN_H_

#include <cstdint>

#include "tfrt/bef/bef.h"

namespace tfrt {
namespace bef {

// Span is a range view of contiguous bef region like bef::Vector. It reads the
// array size and start pointer eagerly, so that the range can be adapted.
template <typename T>
class Span {
 public:
  using value_type = T;
  using iterator = ReadIterator<T>;
  using const_iterator = iterator;

  Span(const char* data, size_t size) : data_(data), size_(size) {}
  Span(const Vector<T>& vec)  // NOLINT(google-explicit-constructor)
      : Span(vec.data(), vec.size()) {}

  const char* data() const { return data_; }

  iterator begin() const { return iterator(data_); }
  iterator end() const { return iterator(data_ + size_ * sizeof(T)); }

  T operator[](size_t index) const {
    assert(index < size());
    auto iter = begin();
    iter += index;
    return *iter;
  }

  size_t size() const { return size_; }
  bool empty() const { return size_ == 0; }

  Span drop_front(int num = 1) const {
    auto beg = begin();
    beg += num;
    assert(size() >= num);
    return Span(beg.data(), size() - num);
  }

 private:
  const char* data_ = nullptr;
  size_t size_ = 0;
};

}  // namespace bef
}  // namespace tfrt

#endif  // TFRT_BEF_SPAN_H_
