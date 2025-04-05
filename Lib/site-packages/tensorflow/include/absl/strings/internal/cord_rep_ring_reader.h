// Copyright 2021 The Abseil Authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef ABSL_STRINGS_INTERNAL_CORD_REP_RING_READER_H_
#define ABSL_STRINGS_INTERNAL_CORD_REP_RING_READER_H_

#include <cassert>
#include <cstddef>
#include <cstdint>

#include "absl/strings/internal/cord_internal.h"
#include "absl/strings/internal/cord_rep_ring.h"
#include "absl/strings/string_view.h"

namespace absl {
ABSL_NAMESPACE_BEGIN
namespace cord_internal {

// CordRepRingReader provides basic navigation over CordRepRing data.
class CordRepRingReader {
 public:
  // Returns true if this instance is not empty.
  explicit operator bool() const { return ring_ != nullptr; }

  // Returns the ring buffer reference for this instance, or nullptr if empty.
  CordRepRing* ring() const { return ring_; }

  // Returns the current node index inside the ring buffer for this instance.
  // The returned value is undefined if this instance is empty.
  CordRepRing::index_type index() const { return index_; }

  // Returns the current node inside the ring buffer for this instance.
  // The returned value is undefined if this instance is empty.
  CordRep* node() const { return ring_->entry_child(index_); }

  // Returns the length of the referenced ring buffer.
  // Requires the current instance to be non empty.
  size_t length() const {
    assert(ring_);
    return ring_->length;
  }

  // Returns the end offset of the last navigated-to chunk, which represents the
  // total bytes 'consumed' relative to the start of the ring. The returned
  // value is never zero. For example, initializing a reader with a ring buffer
  // with a first chunk of 19 bytes will return consumed() = 19.
  // Requires the current instance to be non empty.
  size_t consumed() const {
    assert(ring_);
    return ring_->entry_end_offset(index_);
  }

  // Returns the number of bytes remaining beyond the last navigated-to chunk.
  // Requires the current instance to be non empty.
  size_t remaining() const {
    assert(ring_);
    return length() - consumed();
  }

  // Resets this instance to an empty value
  void Reset() { ring_ = nullptr; }

  // Resets this instance to the start of `ring`. `ring` must not be null.
  // Returns a reference into the first chunk of the provided ring.
  absl::string_view Reset(CordRepRing* ring) {
    assert(ring);
    ring_ = ring;
    index_ = ring_->head();
    return ring_->entry_data(index_);
  }

  // Navigates to the next chunk inside the reference ring buffer.
  // Returns a reference into the navigated-to chunk.
  // Requires remaining() to be non zero.
  absl::string_view Next() {
    assert(remaining());
    index_ = ring_->advance(index_);
    return ring_->entry_data(index_);
  }

  // Navigates to the chunk at offset `offset`.
  // Returns a reference into the navigated-to chunk, adjusted for the relative
  // position of `offset` into that chunk. For example, calling Seek(13) on a
  // ring buffer containing 2 chunks of 10 and 20 bytes respectively will return
  // a string view into the second chunk starting at offset 3 with a size of 17.
  // Requires `offset` to be less than `length()`
  absl::string_view Seek(size_t offset) {
    assert(offset < length());
    size_t current = ring_->entry_end_offset(index_);
    CordRepRing::index_type hint = (offset >= current) ? index_ : ring_->head();
    const CordRepRing::Position head = ring_->Find(hint, offset);
    index_ = head.index;
    auto data = ring_->entry_data(head.index);
    data.remove_prefix(head.offset);
    return data;
  }

 private:
  CordRepRing* ring_ = nullptr;
  CordRepRing::index_type index_;
};

}  // namespace cord_internal
ABSL_NAMESPACE_END
}  // namespace absl

#endif  // ABSL_STRINGS_INTERNAL_CORD_REP_RING_READER_H_
