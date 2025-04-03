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

// A helper class for working with an array of reference counted types.

#ifndef TFRT_SUPPORT_RC_ARRAY_H_
#define TFRT_SUPPORT_RC_ARRAY_H_

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "tfrt/support/ref_count.h"

namespace tfrt {

template <typename T>
class RCArray {
 public:
  explicit RCArray(llvm::ArrayRef<T*> values) {
    values_.reserve(values.size());
    for (auto* v : values) {
      v->AddRef();
      values_.push_back(v);
    }
  }

  explicit RCArray(llvm::ArrayRef<RCReference<T>> references) {
    values_.reserve(references.size());
    for (auto& ref : references) {
      auto* v = ref.get();
      v->AddRef();
      values_.push_back(v);
    }
  }

  RCArray(RCArray&& other) : values_(std::move(other.values_)) {}

  RCArray& operator=(RCArray&& other) {
    for (auto* v : values_) v->DropRef();
    values_ = std::move(other.values_);
    return *this;
  }

  ~RCArray() {
    for (auto* v : values_) {
      v->DropRef();
    }
  }

  T* operator[](size_t i) const {
    assert(i < values_.size());
    return values_[i];
  }

  // Make an explicit copy of this RCArray, increasing the refcount of every
  // element in the array by one.
  RCArray CopyRef() const { return RCArray(values()); }

  llvm::ArrayRef<T*> values() const { return values_; }

  size_t size() const { return values_.size(); }

  // Not copyable.
  RCArray(const RCArray&) = delete;
  RCArray& operator=(const RCArray&) = delete;

 private:
  llvm::SmallVector<T*, 4> values_;
};

}  // namespace tfrt

#endif  // TFRT_SUPPORT_RC_ARRAY_H_
