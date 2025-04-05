// Copyright 2017 The Abseil Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// Character Map Class
//
// A fast, bit-vector map for 8-bit unsigned characters.
// This class is useful for non-character purposes as well.

#ifndef ABSL_STRINGS_INTERNAL_CHAR_MAP_H_
#define ABSL_STRINGS_INTERNAL_CHAR_MAP_H_

#include <cstddef>
#include <cstdint>
#include <cstring>

#include "absl/base/macros.h"
#include "absl/base/port.h"

namespace absl {
ABSL_NAMESPACE_BEGIN
namespace strings_internal {

class Charmap {
 public:
  constexpr Charmap() : m_() {}

  // Initializes with a given char*.  Note that NUL is not treated as
  // a terminator, but rather a char to be flicked.
  Charmap(const char* str, int len) : m_() {
    while (len--) SetChar(*str++);
  }

  // Initializes with a given char*.  NUL is treated as a terminator
  // and will not be in the charmap.
  explicit Charmap(const char* str) : m_() {
    while (*str) SetChar(*str++);
  }

  constexpr bool contains(unsigned char c) const {
    return (m_[c / 64] >> (c % 64)) & 0x1;
  }

  // Returns true if and only if a character exists in both maps.
  bool IntersectsWith(const Charmap& c) const {
    for (size_t i = 0; i < ABSL_ARRAYSIZE(m_); ++i) {
      if ((m_[i] & c.m_[i]) != 0) return true;
    }
    return false;
  }

  bool IsZero() const {
    for (uint64_t c : m_) {
      if (c != 0) return false;
    }
    return true;
  }

  // Containing only a single specified char.
  static constexpr Charmap Char(char x) {
    return Charmap(CharMaskForWord(x, 0), CharMaskForWord(x, 1),
                   CharMaskForWord(x, 2), CharMaskForWord(x, 3));
  }

  // Containing all the chars in the C-string 's'.
  static constexpr Charmap FromString(const char* s) {
    Charmap ret;
    while (*s) ret = ret | Char(*s++);
    return ret;
  }

  // Containing all the chars in the closed interval [lo,hi].
  static constexpr Charmap Range(char lo, char hi) {
    return Charmap(RangeForWord(lo, hi, 0), RangeForWord(lo, hi, 1),
                   RangeForWord(lo, hi, 2), RangeForWord(lo, hi, 3));
  }

  friend constexpr Charmap operator&(const Charmap& a, const Charmap& b) {
    return Charmap(a.m_[0] & b.m_[0], a.m_[1] & b.m_[1], a.m_[2] & b.m_[2],
                   a.m_[3] & b.m_[3]);
  }

  friend constexpr Charmap operator|(const Charmap& a, const Charmap& b) {
    return Charmap(a.m_[0] | b.m_[0], a.m_[1] | b.m_[1], a.m_[2] | b.m_[2],
                   a.m_[3] | b.m_[3]);
  }

  friend constexpr Charmap operator~(const Charmap& a) {
    return Charmap(~a.m_[0], ~a.m_[1], ~a.m_[2], ~a.m_[3]);
  }

 private:
  constexpr Charmap(uint64_t b0, uint64_t b1, uint64_t b2, uint64_t b3)
      : m_{b0, b1, b2, b3} {}

  static constexpr uint64_t RangeForWord(char lo, char hi, uint64_t word) {
    return OpenRangeFromZeroForWord(static_cast<unsigned char>(hi) + 1, word) &
           ~OpenRangeFromZeroForWord(static_cast<unsigned char>(lo), word);
  }

  // All the chars in the specified word of the range [0, upper).
  static constexpr uint64_t OpenRangeFromZeroForWord(uint64_t upper,
                                                     uint64_t word) {
    return (upper <= 64 * word)
               ? 0
               : (upper >= 64 * (word + 1))
                     ? ~static_cast<uint64_t>(0)
                     : (~static_cast<uint64_t>(0) >> (64 - upper % 64));
  }

  static constexpr uint64_t CharMaskForWord(char x, uint64_t word) {
    const auto unsigned_x = static_cast<unsigned char>(x);
    return (unsigned_x / 64 == word)
               ? (static_cast<uint64_t>(1) << (unsigned_x % 64))
               : 0;
  }

  void SetChar(char c) {
    const auto unsigned_c = static_cast<unsigned char>(c);
    m_[unsigned_c / 64] |= static_cast<uint64_t>(1) << (unsigned_c % 64);
  }

  uint64_t m_[4];
};

// Mirror the char-classifying predicates in <cctype>
constexpr Charmap UpperCharmap() { return Charmap::Range('A', 'Z'); }
constexpr Charmap LowerCharmap() { return Charmap::Range('a', 'z'); }
constexpr Charmap DigitCharmap() { return Charmap::Range('0', '9'); }
constexpr Charmap AlphaCharmap() { return LowerCharmap() | UpperCharmap(); }
constexpr Charmap AlnumCharmap() { return DigitCharmap() | AlphaCharmap(); }
constexpr Charmap XDigitCharmap() {
  return DigitCharmap() | Charmap::Range('A', 'F') | Charmap::Range('a', 'f');
}
constexpr Charmap PrintCharmap() { return Charmap::Range(0x20, 0x7e); }
constexpr Charmap SpaceCharmap() { return Charmap::FromString("\t\n\v\f\r "); }
constexpr Charmap CntrlCharmap() {
  return Charmap::Range(0, 0x7f) & ~PrintCharmap();
}
constexpr Charmap BlankCharmap() { return Charmap::FromString("\t "); }
constexpr Charmap GraphCharmap() { return PrintCharmap() & ~SpaceCharmap(); }
constexpr Charmap PunctCharmap() { return GraphCharmap() & ~AlnumCharmap(); }

}  // namespace strings_internal
ABSL_NAMESPACE_END
}  // namespace absl

#endif  // ABSL_STRINGS_INTERNAL_CHAR_MAP_H_
