// Copyright 2020 The Abseil Authors
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

#ifndef ABSL_STRINGS_INTERNAL_CORD_REP_RING_H_
#define ABSL_STRINGS_INTERNAL_CORD_REP_RING_H_

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <iosfwd>
#include <limits>
#include <memory>

#include "absl/container/internal/layout.h"
#include "absl/strings/internal/cord_internal.h"
#include "absl/strings/internal/cord_rep_flat.h"

namespace absl {
ABSL_NAMESPACE_BEGIN
namespace cord_internal {

// All operations modifying a ring buffer are implemented as static methods
// requiring a CordRepRing instance with a reference adopted by the method.
//
// The methods return the modified ring buffer, which may be equal to the input
// if the input was not shared, and having large enough capacity to accommodate
// any newly added node(s). Otherwise, a copy of the input rep with the new
// node(s) added is returned.
//
// Any modification on non shared ring buffers with enough capacity will then
// require minimum atomic operations. Caller should where possible provide
// reasonable `extra` hints for both anticipated extra `flat` byte space, as
// well as anticipated extra nodes required for complex operations.
//
// Example of code creating a ring buffer, adding some data to it,
// and discarding the buffer when done:
//
//   void FunWithRings() {
//     // Create ring with 3 flats
//     CordRep* flat = CreateFlat("Hello");
//     CordRepRing* ring = CordRepRing::Create(flat, 2);
//     ring = CordRepRing::Append(ring, CreateFlat(" "));
//     ring = CordRepRing::Append(ring, CreateFlat("world"));
//     DoSomethingWithRing(ring);
//     CordRep::Unref(ring);
//   }
//
// Example of code Copying an existing ring buffer and modifying it:
//
//   void MoreFunWithRings(CordRepRing* src) {
//     CordRepRing* ring = CordRep::Ref(src)->ring();
//     ring = CordRepRing::Append(ring, CreateFlat("Hello"));
//     ring = CordRepRing::Append(ring, CreateFlat(" "));
//     ring = CordRepRing::Append(ring, CreateFlat("world"));
//     DoSomethingWithRing(ring);
//     CordRep::Unref(ring);
//   }
//
class CordRepRing : public CordRep {
 public:
  // `pos_type` represents a 'logical position'. A CordRepRing instance has a
  // `begin_pos` (default 0), and each node inside the buffer will have an
  // `end_pos` which is the `end_pos` of the previous node (or `begin_pos`) plus
  // this node's length. The purpose is to allow for a binary search on this
  // position, while allowing O(1) prepend and append operations.
  using pos_type = size_t;

  // `index_type` is the type for the `head`, `tail` and `capacity` indexes.
  // Ring buffers are limited to having no more than four billion entries.
  using index_type = uint32_t;

  // `offset_type` is the type for the data offset inside a child rep's data.
  using offset_type = uint32_t;

  // Position holds the node index and relative offset into the node for
  // some physical offset in the contained data as returned by the Find()
  // and FindTail() methods.
  struct Position {
    index_type index;
    size_t offset;
  };

  // The maximum # of child nodes that can be hosted inside a CordRepRing.
  static constexpr size_t kMaxCapacity = (std::numeric_limits<uint32_t>::max)();

  // CordRepring can not be default constructed, moved, copied or assigned.
  CordRepRing() = delete;
  CordRepRing(const CordRepRing&) = delete;
  CordRepRing& operator=(const CordRepRing&) = delete;

  // Returns true if this instance is valid, false if some or all of the
  // invariants are broken. Intended for debug purposes only.
  // `output` receives an explanation of the broken invariants.
  bool IsValid(std::ostream& output) const;

  // Returns the size in bytes for a CordRepRing with `capacity' entries.
  static constexpr size_t AllocSize(size_t capacity);

  // Returns the distance in bytes from `pos` to `end_pos`.
  static constexpr size_t Distance(pos_type pos, pos_type end_pos);

  // Creates a new ring buffer from the provided `rep`. Adopts a reference
  // on `rep`. The returned ring buffer has a capacity of at least `extra + 1`
  static CordRepRing* Create(CordRep* child, size_t extra = 0);

  // `head`, `tail` and `capacity` indexes defining the ring buffer boundaries.
  index_type head() const { return head_; }
  index_type tail() const { return tail_; }
  index_type capacity() const { return capacity_; }

  // Returns the number of entries in this instance.
  index_type entries() const { return entries(head_, tail_); }

  // Returns the logical begin position of this instance.
  pos_type begin_pos() const { return begin_pos_; }

  // Returns the number of entries for a given head-tail range.
  // Requires `head` and `tail` values to be less than `capacity()`.
  index_type entries(index_type head, index_type tail) const {
    assert(head < capacity_ && tail < capacity_);
    return tail - head + ((tail > head) ? 0 : capacity_);
  }

  // Returns the logical end position of entry `index`.
  pos_type const& entry_end_pos(index_type index) const {
    assert(IsValidIndex(index));
    return Layout::Partial().Pointer<0>(data_)[index];
  }

  // Returns the child pointer of entry `index`.
  CordRep* const& entry_child(index_type index) const {
    assert(IsValidIndex(index));
    return Layout::Partial(capacity()).Pointer<1>(data_)[index];
  }

  // Returns the data offset of entry `index`
  offset_type const& entry_data_offset(index_type index) const {
    assert(IsValidIndex(index));
    return Layout::Partial(capacity(), capacity()).Pointer<2>(data_)[index];
  }

  // Appends the provided child node to the `rep` instance.
  // Adopts a reference from `rep` and `child` which may not be null.
  // If the provided child is a FLAT or EXTERNAL node, or a SUBSTRING node
  // containing a FLAT or EXTERNAL node, then flat or external the node is added
  // 'as is', with an offset added for the SUBSTRING case.
  // If the provided child is a RING or CONCAT tree, or a SUBSTRING of a RING or
  // CONCAT tree, then all child nodes not excluded by any start offset or
  // length values are added recursively.
  static CordRepRing* Append(CordRepRing* rep, CordRep* child);

  // Appends the provided string data to the `rep` instance.
  // This function will attempt to utilize any remaining capacity in the last
  // node of the input if that node is not shared (directly or indirectly), and
  // of type FLAT. Remaining data will be added as one or more FLAT nodes.
  // Any last node added to the ring buffer will be allocated with up to
  // `extra` bytes of capacity for (anticipated) subsequent append actions.
  static CordRepRing* Append(CordRepRing* rep, string_view data,
                             size_t extra = 0);

  // Prepends the provided child node to the `rep` instance.
  // Adopts a reference from `rep` and `child` which may not be null.
  // If the provided child is a FLAT or EXTERNAL node, or a SUBSTRING node
  // containing a FLAT or EXTERNAL node, then flat or external the node is
  // prepended 'as is', with an optional offset added for the SUBSTRING case.
  // If the provided child is a RING or CONCAT tree, or a SUBSTRING of a RING
  // or CONCAT tree, then all child nodes not excluded by any start offset or
  // length values are added recursively.
  static CordRepRing* Prepend(CordRepRing* rep, CordRep* child);

  // Prepends the provided string data to the `rep` instance.
  // This function will attempt to utilize any remaining capacity in the first
  // node of the input if that node is not shared (directly or indirectly), and
  // of type FLAT. Remaining data will be added as one or more FLAT nodes.
  // Any first node prepnded to the ring buffer will be allocated with up to
  // `extra` bytes of capacity for (anticipated) subsequent prepend actions.
  static CordRepRing* Prepend(CordRepRing* rep, string_view data,
                              size_t extra = 0);

  // Returns a span referencing potentially unused capacity in the last node.
  // The returned span may be empty if no such capacity is available, or if the
  // current instance is shared. Else, a span of size `n <= size` is returned.
  // If non empty, the ring buffer is adjusted to the new length, with the newly
  // added capacity left uninitialized. Callers should assign a value to the
  // entire span before any other operations on this instance.
  Span<char> GetAppendBuffer(size_t size);

  // Returns a span referencing potentially unused capacity in the first node.
  // This function is identical to GetAppendBuffer except that it returns a span
  // referencing up to `size` capacity directly before the existing data.
  Span<char> GetPrependBuffer(size_t size);

  // Returns a cord ring buffer containing `len` bytes of data starting at
  // `offset`. If the input is not shared, this function will remove all head
  // and tail child nodes outside of the requested range, and adjust the new
  // head and tail nodes as required. If the input is shared, this function
  // returns a new instance sharing some or all of the nodes from the input.
  static CordRepRing* SubRing(CordRepRing* r, size_t offset, size_t len,
                              size_t extra = 0);

  // Returns a cord ring buffer with the first `len` bytes removed.
  // If the input is not shared, this function will remove all head child nodes
  // fully inside the first `length` bytes, and adjust the new head as required.
  // If the input is shared, this function returns a new instance sharing some
  // or all of the nodes from the input.
  static CordRepRing* RemoveSuffix(CordRepRing* r, size_t len,
                                   size_t extra = 0);

  // Returns a cord ring buffer with the last `len` bytes removed.
  // If the input is not shared, this function will remove all head child nodes
  // fully inside the first `length` bytes, and adjust the new head as required.
  // If the input is shared, this function returns a new instance sharing some
  // or all of the nodes from the input.
  static CordRepRing* RemovePrefix(CordRepRing* r, size_t len,
                                   size_t extra = 0);

  // Returns the character at `offset`. Requires that `offset < length`.
  char GetCharacter(size_t offset) const;

  // Returns true if this instance manages a single contiguous buffer, in which
  // case the (optional) output parameter `fragment` is set. Otherwise, the
  // function returns false, and `fragment` is left unchanged.
  bool IsFlat(absl::string_view* fragment) const;

  // Returns true if the data starting at `offset` with length `len` is
  // managed by this instance inside a single contiguous buffer, in which case
  // the (optional) output parameter `fragment` is set to the contiguous memory
  // starting at offset `offset` with length `length`. Otherwise, the function
  // returns false, and `fragment` is left unchanged.
  bool IsFlat(size_t offset, size_t len, absl::string_view* fragment) const;

  // Testing only: set capacity to requested capacity.
  void SetCapacityForTesting(size_t capacity);

  // Returns the CordRep data pointer for the provided CordRep.
  // Requires that the provided `rep` is either a FLAT or EXTERNAL CordRep.
  static const char* GetLeafData(const CordRep* rep);

  // Returns the CordRep data pointer for the provided CordRep.
  // Requires that `rep` is either a FLAT, EXTERNAL, or SUBSTRING CordRep.
  static const char* GetRepData(const CordRep* rep);

  // Advances the provided position, wrapping around capacity as needed.
  // Requires `index` < capacity()
  inline index_type advance(index_type index) const;

  // Advances the provided position by 'n`, wrapping around capacity as needed.
  // Requires `index` < capacity() and `n` <= capacity.
  inline index_type advance(index_type index, index_type n) const;

  // Retreats the provided position, wrapping around 0 as needed.
  // Requires `index` < capacity()
  inline index_type retreat(index_type index) const;

  // Retreats the provided position by 'n', wrapping around 0 as needed.
  // Requires `index` < capacity()
  inline index_type retreat(index_type index, index_type n) const;

  // Returns the logical begin position of entry `index`
  pos_type const& entry_begin_pos(index_type index) const {
    return (index == head_) ? begin_pos_ : entry_end_pos(retreat(index));
  }

  // Returns the physical start offset of entry `index`
  size_t entry_start_offset(index_type index) const {
    return Distance(begin_pos_, entry_begin_pos(index));
  }

  // Returns the physical end offset of entry `index`
  size_t entry_end_offset(index_type index) const {
    return Distance(begin_pos_, entry_end_pos(index));
  }

  // Returns the data length for entry `index`
  size_t entry_length(index_type index) const {
    return Distance(entry_begin_pos(index), entry_end_pos(index));
  }

  // Returns the data for entry `index`
  absl::string_view entry_data(index_type index) const;

  // Returns the position for `offset` as {index, prefix}. `index` holds the
  // index of the entry at the specified offset and `prefix` holds the relative
  // offset inside that entry.
  // Requires `offset` < length.
  //
  // For example we can implement GetCharacter(offset) as:
  //   char GetCharacter(size_t offset) {
  //     Position pos = this->Find(offset);
  //     return this->entry_data(pos.pos)[pos.offset];
  //   }
  inline Position Find(size_t offset) const;

  // Find starting at `head`
  inline Position Find(index_type head, size_t offset) const;

  // Returns the tail position for `offset` as {tail index, suffix}.
  // `tail index` holds holds the index of the entry holding the offset directly
  // before 'offset` advanced by one. 'suffix` holds the relative offset from
  // that relative offset in the entry to the end of the entry.
  // For example, FindTail(length) will return {tail(), 0}, FindTail(length - 5)
  // will return {retreat(tail), 5)} provided the preceding entry contains at
  // least 5 bytes of data.
  // Requires offset >= 1 && offset <= length.
  //
  // This function is very useful in functions that need to clip the end of some
  // ring buffer such as 'RemovePrefix'.
  // For example, we could implement RemovePrefix for non shared instances as:
  //   void RemoveSuffix(size_t n) {
  //     Position pos = FindTail(length - n);
  //     UnrefEntries(pos.pos, this->tail_);
  //     this->tail_ = pos.pos;
  //     entry(retreat(pos.pos)).end_pos -= pos.offset;
  //   }
  inline Position FindTail(size_t offset) const;

  // Find tail starting at `head`
  inline Position FindTail(index_type head, size_t offset) const;

  // Invokes f(index_type index) for each entry inside the range [head, tail>
  template <typename F>
  void ForEach(index_type head, index_type tail, F&& f) const {
    index_type n1 = (tail > head) ? tail : capacity_;
    for (index_type i = head; i < n1; ++i) f(i);
    if (tail <= head) {
      for (index_type i = 0; i < tail; ++i) f(i);
    }
  }

  // Invokes f(index_type index) for each entry inside this instance.
  template <typename F>
  void ForEach(F&& f) const {
    ForEach(head_, tail_, std::forward<F>(f));
  }

  // Dump this instance's data tp stream `s` in human readable format, excluding
  // the actual data content itself. Intended for debug purposes only.
  friend std::ostream& operator<<(std::ostream& s, const CordRepRing& rep);

 private:
  enum class AddMode { kAppend, kPrepend };

  using Layout = container_internal::Layout<pos_type, CordRep*, offset_type>;

  class Filler;
  class Transaction;
  class CreateTransaction;

  static constexpr size_t kLayoutAlignment = Layout::Partial().Alignment();

  // Creates a new CordRepRing.
  explicit CordRepRing(index_type capacity) : capacity_(capacity) {}

  // Returns true if `index` is a valid index into this instance.
  bool IsValidIndex(index_type index) const;

  // Debug use only: validates the provided CordRepRing invariants.
  // Verification of all CordRepRing methods can be enabled by defining
  // EXTRA_CORD_RING_VALIDATION, i.e.: `--copts=-DEXTRA_CORD_RING_VALIDATION`
  // Verification is VERY expensive, so only do it for debugging purposes.
  static CordRepRing* Validate(CordRepRing* rep, const char* file = nullptr,
                               int line = 0);

  // Allocates a CordRepRing large enough to hold `capacity + extra' entries.
  // The returned capacity may be larger if the allocated memory allows for it.
  // The maximum capacity of a CordRepRing is capped at kMaxCapacity.
  // Throws `std::length_error` if `capacity + extra' exceeds kMaxCapacity.
  static CordRepRing* New(size_t capacity, size_t extra);

  // Deallocates (but does not destroy) the provided ring buffer.
  static void Delete(CordRepRing* rep);

  // Destroys the provided ring buffer, decrementing the reference count of all
  // contained child CordReps. The provided 1\`rep` should have a ref count of
  // one (pre decrement destroy call observing `refcount.IsOne()`) or zero
  // (post decrement destroy call observing `!refcount.Decrement()`).
  static void Destroy(CordRepRing* rep);

  // Returns a mutable reference to the logical end position array.
  pos_type* entry_end_pos() {
    return Layout::Partial().Pointer<0>(data_);
  }

  // Returns a mutable reference to the child pointer array.
  CordRep** entry_child() {
    return Layout::Partial(capacity()).Pointer<1>(data_);
  }

  // Returns a mutable reference to the data offset array.
  offset_type* entry_data_offset() {
    return Layout::Partial(capacity(), capacity()).Pointer<2>(data_);
  }

  // Find implementations for the non fast path 0 / length cases.
  Position FindSlow(index_type head, size_t offset) const;
  Position FindTailSlow(index_type head, size_t offset) const;

  // Finds the index of the first node that is inside a reasonable distance
  // of the node at `offset` from which we can continue with a linear search.
  template <bool wrap>
  index_type FindBinary(index_type head, index_type tail, size_t offset) const;

  // Fills the current (initialized) instance from the provided source, copying
  // entries [head, tail). Adds a reference to copied entries if `ref` is true.
  template <bool ref>
  void Fill(const CordRepRing* src, index_type head, index_type tail);

  // Create a copy of 'rep', copying all entries [head, tail), allocating room
  // for `extra` entries. Adds a reference on all copied entries.
  static CordRepRing* Copy(CordRepRing* rep, index_type head, index_type tail,
                           size_t extra = 0);

  // Returns a Mutable CordRepRing reference from `rep` with room for at least
  // `extra` additional nodes. Adopts a reference count from `rep`.
  // This function will return `rep` if, and only if:
  // - rep.entries + extra <= rep.capacity
  // - rep.refcount == 1
  // Otherwise, this function will create a new copy of `rep` with additional
  // capacity to satisfy `extra` extra nodes, and unref the old `rep` instance.
  //
  // If a new CordRepRing can not be allocated, or the new capacity would exceed
  // the maximum capacity, then the input is consumed only, and an exception is
  // thrown.
  static CordRepRing* Mutable(CordRepRing* rep, size_t extra);

  // Slow path for Append(CordRepRing* rep, CordRep* child). This function is
  // exercised if the provided `child` in Append() is not a leaf node, i.e., a
  // ring buffer or old (concat) cord tree.
  static CordRepRing* AppendSlow(CordRepRing* rep, CordRep* child);

  // Appends the provided leaf node. Requires `child` to be FLAT or EXTERNAL.
  static CordRepRing* AppendLeaf(CordRepRing* rep, CordRep* child,
                                 size_t offset, size_t length);

  // Prepends the provided leaf node. Requires `child` to be FLAT or EXTERNAL.
  static CordRepRing* PrependLeaf(CordRepRing* rep, CordRep* child,
                                  size_t offset, size_t length);

  // Slow path for Prepend(CordRepRing* rep, CordRep* child). This function is
  // exercised if the provided `child` in Prepend() is not a leaf node, i.e., a
  // ring buffer or old (concat) cord tree.
  static CordRepRing* PrependSlow(CordRepRing* rep, CordRep* child);

  // Slow path for Create(CordRep* child, size_t extra). This function is
  // exercised if the provided `child` in Prepend() is not a leaf node, i.e., a
  // ring buffer or old (concat) cord tree.
  static CordRepRing* CreateSlow(CordRep* child, size_t extra);

  // Creates a new ring buffer from the provided `child` leaf node. Requires
  // `child` to be FLAT or EXTERNAL. on `rep`.
  // The returned ring buffer has a capacity of at least `1 + extra`
  static CordRepRing* CreateFromLeaf(CordRep* child, size_t offset,
                                     size_t length, size_t extra);

  // Appends or prepends (depending on AddMode) the ring buffer in `ring' to
  // `rep` starting at `offset` with length `len`.
  template <AddMode mode>
  static CordRepRing* AddRing(CordRepRing* rep, CordRepRing* ring,
                              size_t offset, size_t len);

  // Increases the data offset for entry `index` by `n`.
  void AddDataOffset(index_type index, size_t n);

  // Decreases the length for entry `index` by `n`.
  void SubLength(index_type index, size_t n);

  index_type head_;
  index_type tail_;
  index_type capacity_;
  pos_type begin_pos_;

  alignas(kLayoutAlignment) char data_[kLayoutAlignment];

  friend struct CordRep;
};

constexpr size_t CordRepRing::AllocSize(size_t capacity) {
  return sizeof(CordRepRing) - sizeof(data_) +
         Layout(capacity, capacity, capacity).AllocSize();
}

inline constexpr size_t CordRepRing::Distance(pos_type pos, pos_type end_pos) {
  return (end_pos - pos);
}

inline const char* CordRepRing::GetLeafData(const CordRep* rep) {
  return rep->tag != EXTERNAL ? rep->flat()->Data() : rep->external()->base;
}

inline const char* CordRepRing::GetRepData(const CordRep* rep) {
  if (rep->tag >= FLAT) return rep->flat()->Data();
  if (rep->tag == EXTERNAL) return rep->external()->base;
  return GetLeafData(rep->substring()->child) + rep->substring()->start;
}

inline CordRepRing::index_type CordRepRing::advance(index_type index) const {
  assert(index < capacity_);
  return ++index == capacity_ ? 0 : index;
}

inline CordRepRing::index_type CordRepRing::advance(index_type index,
                                                    index_type n) const {
  assert(index < capacity_ && n <= capacity_);
  return (index += n) >= capacity_ ? index - capacity_ : index;
}

inline CordRepRing::index_type CordRepRing::retreat(index_type index) const {
  assert(index < capacity_);
  return (index > 0 ? index : capacity_) - 1;
}

inline CordRepRing::index_type CordRepRing::retreat(index_type index,
                                                    index_type n) const {
  assert(index < capacity_ && n <= capacity_);
  return index >= n ? index - n : capacity_ - n + index;
}

inline absl::string_view CordRepRing::entry_data(index_type index) const {
  size_t data_offset = entry_data_offset(index);
  return {GetRepData(entry_child(index)) + data_offset, entry_length(index)};
}

inline bool CordRepRing::IsValidIndex(index_type index) const {
  if (index >= capacity_) return false;
  return (tail_ > head_) ? (index >= head_ && index < tail_)
                         : (index >= head_ || index < tail_);
}

#ifndef EXTRA_CORD_RING_VALIDATION
inline CordRepRing* CordRepRing::Validate(CordRepRing* rep,
                                          const char* /*file*/, int /*line*/) {
  return rep;
}
#endif

inline CordRepRing::Position CordRepRing::Find(size_t offset) const {
  assert(offset < length);
  return (offset == 0) ? Position{head_, 0} : FindSlow(head_, offset);
}

inline CordRepRing::Position CordRepRing::Find(index_type head,
                                               size_t offset) const {
  assert(offset < length);
  assert(IsValidIndex(head) && offset >= entry_start_offset(head));
  return (offset == 0) ? Position{head_, 0} : FindSlow(head, offset);
}

inline CordRepRing::Position CordRepRing::FindTail(size_t offset) const {
  assert(offset > 0 && offset <= length);
  return (offset == length) ? Position{tail_, 0} : FindTailSlow(head_, offset);
}

inline CordRepRing::Position CordRepRing::FindTail(index_type head,
                                                   size_t offset) const {
  assert(offset > 0 && offset <= length);
  assert(IsValidIndex(head) && offset >= entry_start_offset(head) + 1);
  return (offset == length) ? Position{tail_, 0} : FindTailSlow(head, offset);
}

// Now that CordRepRing is defined, we can define CordRep's helper casts:
inline CordRepRing* CordRep::ring() {
  assert(IsRing());
  return static_cast<CordRepRing*>(this);
}

inline const CordRepRing* CordRep::ring() const {
  assert(IsRing());
  return static_cast<const CordRepRing*>(this);
}

inline bool CordRepRing::IsFlat(absl::string_view* fragment) const {
  if (entries() == 1) {
    if (fragment) *fragment = entry_data(head());
    return true;
  }
  return false;
}

inline bool CordRepRing::IsFlat(size_t offset, size_t len,
                                absl::string_view* fragment) const {
  const Position pos = Find(offset);
  const absl::string_view data = entry_data(pos.index);
  if (data.length() >= len && data.length() - len >= pos.offset) {
    if (fragment) *fragment = data.substr(pos.offset, len);
    return true;
  }
  return false;
}

std::ostream& operator<<(std::ostream& s, const CordRepRing& rep);

}  // namespace cord_internal
ABSL_NAMESPACE_END
}  // namespace absl

#endif  // ABSL_STRINGS_INTERNAL_CORD_REP_RING_H_
