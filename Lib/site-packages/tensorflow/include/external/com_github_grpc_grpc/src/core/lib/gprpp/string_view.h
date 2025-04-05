/*
 *
 * Copyright 2019 gRPC authors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 */
#ifndef GRPC_CORE_LIB_GPRPP_STRING_VIEW_H
#define GRPC_CORE_LIB_GPRPP_STRING_VIEW_H

#include <grpc/support/port_platform.h>

#include <grpc/impl/codegen/slice.h>
#include <grpc/support/alloc.h>
#include <grpc/support/log.h>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <limits>
#include <string>

#include "src/core/lib/gpr/string.h"
#include "src/core/lib/gpr/useful.h"
#include "src/core/lib/gprpp/memory.h"

#if GRPC_USE_ABSL
#include "absl/strings/string_view.h"
#endif

namespace grpc_core {

#if GRPC_USE_ABSL

using StringView = absl::string_view;

#else

// Provides a light-weight view over a char array or a slice, similar but not
// identical to absl::string_view.
//
// Any method that has the same name as absl::string_view MUST HAVE identical
// semantics to what absl::string_view provides.
//
// Methods that are not part of absl::string_view API, must be clearly
// annotated.
//
// StringView does not own the buffers that back the view. Callers must ensure
// the buffer stays around while the StringView is accessible.
//
// Pass StringView by value in functions, since it is exactly two pointers in
// size.
//
// The interface used here is not identical to absl::string_view. Notably, we
// need to support slices while we cannot support std::string, and gpr string
// style functions such as strdup() and cmp(). Once we switch to
// absl::string_view this class will inherit from absl::string_view and add the
// gRPC-specific APIs.
class StringView final {
 public:
  static constexpr size_t npos = std::numeric_limits<size_t>::max();

  constexpr StringView(const char* ptr, size_t size) : ptr_(ptr), size_(size) {}
  constexpr StringView(const char* ptr)
      : StringView(ptr, ptr == nullptr ? 0 : strlen(ptr)) {}
  constexpr StringView() : StringView(nullptr, 0) {}

  constexpr const char* data() const { return ptr_; }
  constexpr size_t size() const { return size_; }
  constexpr bool empty() const { return size_ == 0; }

  StringView substr(size_t start, size_t size = npos) {
    GPR_DEBUG_ASSERT(start + size <= size_);
    return StringView(ptr_ + start, std::min(size, size_ - start));
  }

  constexpr const char& operator[](size_t i) const { return ptr_[i]; }

  const char& front() const { return ptr_[0]; }
  const char& back() const { return ptr_[size_ - 1]; }

  void remove_prefix(size_t n) {
    GPR_DEBUG_ASSERT(n <= size_);
    ptr_ += n;
    size_ -= n;
  }

  void remove_suffix(size_t n) {
    GPR_DEBUG_ASSERT(n <= size_);
    size_ -= n;
  }

  size_t find(char c, size_t pos = 0) const {
    if (empty() || pos >= size_) return npos;
    const char* result =
        static_cast<const char*>(memchr(ptr_ + pos, c, size_ - pos));
    return result != nullptr ? result - ptr_ : npos;
  }

  void clear() {
    ptr_ = nullptr;
    size_ = 0;
  }

  // Converts to `std::basic_string`.
  template <typename Allocator>
  explicit operator std::basic_string<char, std::char_traits<char>, Allocator>()
      const {
    if (data() == nullptr) return {};
    return std::basic_string<char, std::char_traits<char>, Allocator>(data(),
                                                                      size());
  }

  // Compares with other.
  inline int compare(StringView other) {
    const size_t len = GPR_MIN(size(), other.size());
    const int ret = strncmp(data(), other.data(), len);
    if (ret != 0) return ret;
    if (size() == other.size()) return 0;
    if (size() < other.size()) return -1;
    return 1;
  }

 private:
  const char* ptr_;
  size_t size_;
};

inline bool operator==(StringView lhs, StringView rhs) {
  return lhs.size() == rhs.size() &&
         strncmp(lhs.data(), rhs.data(), lhs.size()) == 0;
}

inline bool operator!=(StringView lhs, StringView rhs) { return !(lhs == rhs); }

inline bool operator<(StringView lhs, StringView rhs) {
  return lhs.compare(rhs) < 0;
}

#endif  // GRPC_USE_ABSL

// Converts grpc_slice to StringView.
inline StringView StringViewFromSlice(const grpc_slice& slice) {
  return StringView(reinterpret_cast<const char*>(GRPC_SLICE_START_PTR(slice)),
                    GRPC_SLICE_LENGTH(slice));
}

// Creates a dup of the string viewed by this class.
// Return value is null-terminated and never nullptr.
inline grpc_core::UniquePtr<char> StringViewToCString(const StringView sv) {
  char* str = static_cast<char*>(gpr_malloc(sv.size() + 1));
  if (sv.size() > 0) memcpy(str, sv.data(), sv.size());
  str[sv.size()] = '\0';
  return grpc_core::UniquePtr<char>(str);
}

}  // namespace grpc_core

#endif /* GRPC_CORE_LIB_GPRPP_STRING_VIEW_H */
