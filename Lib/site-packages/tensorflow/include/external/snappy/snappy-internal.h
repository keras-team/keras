// Copyright 2008 Google Inc. All Rights Reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
//     * Redistributions of source code must retain the above copyright
// notice, this list of conditions and the following disclaimer.
//     * Redistributions in binary form must reproduce the above
// copyright notice, this list of conditions and the following disclaimer
// in the documentation and/or other materials provided with the
// distribution.
//     * Neither the name of Google Inc. nor the names of its
// contributors may be used to endorse or promote products derived from
// this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
// "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
// LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
// A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
// OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
// SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
// LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
// DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
// THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
// Internals shared between the Snappy implementation and its unittest.

#ifndef THIRD_PARTY_SNAPPY_SNAPPY_INTERNAL_H_
#define THIRD_PARTY_SNAPPY_SNAPPY_INTERNAL_H_

#include <utility>

#include "snappy-stubs-internal.h"

#if SNAPPY_HAVE_SSSE3
// Please do not replace with <x86intrin.h> or with headers that assume more
// advanced SSE versions without checking with all the OWNERS.
#include <emmintrin.h>
#include <tmmintrin.h>
#endif

#if SNAPPY_HAVE_NEON
#include <arm_neon.h>
#endif

#if SNAPPY_HAVE_SSSE3 || SNAPPY_HAVE_NEON
#define SNAPPY_HAVE_VECTOR_BYTE_SHUFFLE 1
#else
#define SNAPPY_HAVE_VECTOR_BYTE_SHUFFLE 0
#endif

namespace snappy {
namespace internal {

#if SNAPPY_HAVE_VECTOR_BYTE_SHUFFLE
#if SNAPPY_HAVE_SSSE3
using V128 = __m128i;
#elif SNAPPY_HAVE_NEON
using V128 = uint8x16_t;
#endif

// Load 128 bits of integer data. `src` must be 16-byte aligned.
inline V128 V128_Load(const V128* src);

// Load 128 bits of integer data. `src` does not need to be aligned.
inline V128 V128_LoadU(const V128* src);

// Store 128 bits of integer data. `dst` does not need to be aligned.
inline void V128_StoreU(V128* dst, V128 val);

// Shuffle packed 8-bit integers using a shuffle mask.
// Each packed integer in the shuffle mask must be in [0,16).
inline V128 V128_Shuffle(V128 input, V128 shuffle_mask);

// Constructs V128 with 16 chars |c|.
inline V128 V128_DupChar(char c);

#if SNAPPY_HAVE_SSSE3
inline V128 V128_Load(const V128* src) { return _mm_load_si128(src); }

inline V128 V128_LoadU(const V128* src) { return _mm_loadu_si128(src); }

inline void V128_StoreU(V128* dst, V128 val) { _mm_storeu_si128(dst, val); }

inline V128 V128_Shuffle(V128 input, V128 shuffle_mask) {
  return _mm_shuffle_epi8(input, shuffle_mask);
}

inline V128 V128_DupChar(char c) { return _mm_set1_epi8(c); }

#elif SNAPPY_HAVE_NEON
inline V128 V128_Load(const V128* src) {
  return vld1q_u8(reinterpret_cast<const uint8_t*>(src));
}

inline V128 V128_LoadU(const V128* src) {
  return vld1q_u8(reinterpret_cast<const uint8_t*>(src));
}

inline void V128_StoreU(V128* dst, V128 val) {
  vst1q_u8(reinterpret_cast<uint8_t*>(dst), val);
}

inline V128 V128_Shuffle(V128 input, V128 shuffle_mask) {
  assert(vminvq_u8(shuffle_mask) >= 0 && vmaxvq_u8(shuffle_mask) <= 15);
  return vqtbl1q_u8(input, shuffle_mask);
}

inline V128 V128_DupChar(char c) { return vdupq_n_u8(c); }
#endif
#endif  // SNAPPY_HAVE_VECTOR_BYTE_SHUFFLE

// Working memory performs a single allocation to hold all scratch space
// required for compression.
class WorkingMemory {
 public:
  explicit WorkingMemory(size_t input_size);
  ~WorkingMemory();

  // Allocates and clears a hash table using memory in "*this",
  // stores the number of buckets in "*table_size" and returns a pointer to
  // the base of the hash table.
  uint16_t* GetHashTable(size_t fragment_size, int* table_size) const;
  char* GetScratchInput() const { return input_; }
  char* GetScratchOutput() const { return output_; }

 private:
  char* mem_;        // the allocated memory, never nullptr
  size_t size_;      // the size of the allocated memory, never 0
  uint16_t* table_;  // the pointer to the hashtable
  char* input_;      // the pointer to the input scratch buffer
  char* output_;     // the pointer to the output scratch buffer

  // No copying
  WorkingMemory(const WorkingMemory&);
  void operator=(const WorkingMemory&);
};

// Flat array compression that does not emit the "uncompressed length"
// prefix. Compresses "input" string to the "*op" buffer.
//
// REQUIRES: "input_length <= kBlockSize"
// REQUIRES: "op" points to an array of memory that is at least
// "MaxCompressedLength(input_length)" in size.
// REQUIRES: All elements in "table[0..table_size-1]" are initialized to zero.
// REQUIRES: "table_size" is a power of two
//
// Returns an "end" pointer into "op" buffer.
// "end - op" is the compressed size of "input".
char* CompressFragment(const char* input,
                       size_t input_length,
                       char* op,
                       uint16_t* table,
                       const int table_size);

// Find the largest n such that
//
//   s1[0,n-1] == s2[0,n-1]
//   and n <= (s2_limit - s2).
//
// Return make_pair(n, n < 8).
// Does not read *s2_limit or beyond.
// Does not read *(s1 + (s2_limit - s2)) or beyond.
// Requires that s2_limit >= s2.
//
// In addition populate *data with the next 5 bytes from the end of the match.
// This is only done if 8 bytes are available (s2_limit - s2 >= 8). The point is
// that on some arch's this can be done faster in this routine than subsequent
// loading from s2 + n.
//
// Separate implementation for 64-bit, little-endian cpus.
#if !SNAPPY_IS_BIG_ENDIAN && \
    (defined(__x86_64__) || defined(_M_X64) || defined(ARCH_PPC) || \
     defined(ARCH_ARM))
static inline std::pair<size_t, bool> FindMatchLength(const char* s1,
                                                      const char* s2,
                                                      const char* s2_limit,
                                                      uint64_t* data) {
  assert(s2_limit >= s2);
  size_t matched = 0;

  // This block isn't necessary for correctness; we could just start looping
  // immediately.  As an optimization though, it is useful.  It creates some not
  // uncommon code paths that determine, without extra effort, whether the match
  // length is less than 8.  In short, we are hoping to avoid a conditional
  // branch, and perhaps get better code layout from the C++ compiler.
  if (SNAPPY_PREDICT_TRUE(s2 <= s2_limit - 16)) {
    uint64_t a1 = UNALIGNED_LOAD64(s1);
    uint64_t a2 = UNALIGNED_LOAD64(s2);
    if (SNAPPY_PREDICT_TRUE(a1 != a2)) {
      // This code is critical for performance. The reason is that it determines
      // how much to advance `ip` (s2). This obviously depends on both the loads
      // from the `candidate` (s1) and `ip`. Furthermore the next `candidate`
      // depends on the advanced `ip` calculated here through a load, hash and
      // new candidate hash lookup (a lot of cycles). This makes s1 (ie.
      // `candidate`) the variable that limits throughput. This is the reason we
      // go through hoops to have this function update `data` for the next iter.
      // The straightforward code would use *data, given by
      //
      // *data = UNALIGNED_LOAD64(s2 + matched_bytes) (Latency of 5 cycles),
      //
      // as input for the hash table lookup to find next candidate. However
      // this forces the load on the data dependency chain of s1, because
      // matched_bytes directly depends on s1. However matched_bytes is 0..7, so
      // we can also calculate *data by
      //
      // *data = AlignRight(UNALIGNED_LOAD64(s2), UNALIGNED_LOAD64(s2 + 8),
      //                    matched_bytes);
      //
      // The loads do not depend on s1 anymore and are thus off the bottleneck.
      // The straightforward implementation on x86_64 would be to use
      //
      // shrd rax, rdx, cl  (cl being matched_bytes * 8)
      //
      // unfortunately shrd with a variable shift has a 4 cycle latency. So this
      // only wins 1 cycle. The BMI2 shrx instruction is a 1 cycle variable
      // shift instruction but can only shift 64 bits. If we focus on just
      // obtaining the least significant 4 bytes, we can obtain this by
      //
      // *data = ConditionalMove(matched_bytes < 4, UNALIGNED_LOAD64(s2),
      //     UNALIGNED_LOAD64(s2 + 4) >> ((matched_bytes & 3) * 8);
      //
      // Writen like above this is not a big win, the conditional move would be
      // a cmp followed by a cmov (2 cycles) followed by a shift (1 cycle).
      // However matched_bytes < 4 is equal to
      // static_cast<uint32_t>(xorval) != 0. Writen that way, the conditional
      // move (2 cycles) can execute in parallel with FindLSBSetNonZero64
      // (tzcnt), which takes 3 cycles.
      uint64_t xorval = a1 ^ a2;
      int shift = Bits::FindLSBSetNonZero64(xorval);
      size_t matched_bytes = shift >> 3;
      uint64_t a3 = UNALIGNED_LOAD64(s2 + 4);
#ifndef __x86_64__
      a2 = static_cast<uint32_t>(xorval) == 0 ? a3 : a2;
#else
      // Ideally this would just be
      //
      // a2 = static_cast<uint32_t>(xorval) == 0 ? a3 : a2;
      //
      // However clang correctly infers that the above statement participates on
      // a critical data dependency chain and thus, unfortunately, refuses to
      // use a conditional move (it's tuned to cut data dependencies). In this
      // case there is a longer parallel chain anyway AND this will be fairly
      // unpredictable.
      asm("testl %k2, %k2\n\t"
          "cmovzq %1, %0\n\t"
          : "+r"(a2)
          : "r"(a3), "r"(xorval)
          : "cc");
#endif
      *data = a2 >> (shift & (3 * 8));
      return std::pair<size_t, bool>(matched_bytes, true);
    } else {
      matched = 8;
      s2 += 8;
    }
  }
  SNAPPY_PREFETCH(s1 + 64);
  SNAPPY_PREFETCH(s2 + 64);

  // Find out how long the match is. We loop over the data 64 bits at a
  // time until we find a 64-bit block that doesn't match; then we find
  // the first non-matching bit and use that to calculate the total
  // length of the match.
  while (SNAPPY_PREDICT_TRUE(s2 <= s2_limit - 16)) {
    uint64_t a1 = UNALIGNED_LOAD64(s1 + matched);
    uint64_t a2 = UNALIGNED_LOAD64(s2);
    if (a1 == a2) {
      s2 += 8;
      matched += 8;
    } else {
      uint64_t xorval = a1 ^ a2;
      int shift = Bits::FindLSBSetNonZero64(xorval);
      size_t matched_bytes = shift >> 3;
      uint64_t a3 = UNALIGNED_LOAD64(s2 + 4);
#ifndef __x86_64__
      a2 = static_cast<uint32_t>(xorval) == 0 ? a3 : a2;
#else
      asm("testl %k2, %k2\n\t"
          "cmovzq %1, %0\n\t"
          : "+r"(a2)
          : "r"(a3), "r"(xorval)
          : "cc");
#endif
      *data = a2 >> (shift & (3 * 8));
      matched += matched_bytes;
      assert(matched >= 8);
      return std::pair<size_t, bool>(matched, false);
    }
  }
  while (SNAPPY_PREDICT_TRUE(s2 < s2_limit)) {
    if (s1[matched] == *s2) {
      ++s2;
      ++matched;
    } else {
      if (s2 <= s2_limit - 8) {
        *data = UNALIGNED_LOAD64(s2);
      }
      return std::pair<size_t, bool>(matched, matched < 8);
    }
  }
  return std::pair<size_t, bool>(matched, matched < 8);
}
#else
static inline std::pair<size_t, bool> FindMatchLength(const char* s1,
                                                      const char* s2,
                                                      const char* s2_limit,
                                                      uint64_t* data) {
  // Implementation based on the x86-64 version, above.
  assert(s2_limit >= s2);
  int matched = 0;

  while (s2 <= s2_limit - 4 &&
         UNALIGNED_LOAD32(s2) == UNALIGNED_LOAD32(s1 + matched)) {
    s2 += 4;
    matched += 4;
  }
  if (LittleEndian::IsLittleEndian() && s2 <= s2_limit - 4) {
    uint32_t x = UNALIGNED_LOAD32(s2) ^ UNALIGNED_LOAD32(s1 + matched);
    int matching_bits = Bits::FindLSBSetNonZero(x);
    matched += matching_bits >> 3;
    s2 += matching_bits >> 3;
  } else {
    while ((s2 < s2_limit) && (s1[matched] == *s2)) {
      ++s2;
      ++matched;
    }
  }
  if (s2 <= s2_limit - 8) *data = LittleEndian::Load64(s2);
  return std::pair<size_t, bool>(matched, matched < 8);
}
#endif

static inline size_t FindMatchLengthPlain(const char* s1, const char* s2,
                                          const char* s2_limit) {
  // Implementation based on the x86-64 version, above.
  assert(s2_limit >= s2);
  int matched = 0;

  while (s2 <= s2_limit - 8 &&
         UNALIGNED_LOAD64(s2) == UNALIGNED_LOAD64(s1 + matched)) {
    s2 += 8;
    matched += 8;
  }
  if (LittleEndian::IsLittleEndian() && s2 <= s2_limit - 8) {
    uint64_t x = UNALIGNED_LOAD64(s2) ^ UNALIGNED_LOAD64(s1 + matched);
    int matching_bits = Bits::FindLSBSetNonZero64(x);
    matched += matching_bits >> 3;
    s2 += matching_bits >> 3;
  } else {
    while ((s2 < s2_limit) && (s1[matched] == *s2)) {
      ++s2;
      ++matched;
    }
  }
  return matched;
}

// Lookup tables for decompression code.  Give --snappy_dump_decompression_table
// to the unit test to recompute char_table.

enum {
  LITERAL = 0,
  COPY_1_BYTE_OFFSET = 1,  // 3 bit length + 3 bits of offset in opcode
  COPY_2_BYTE_OFFSET = 2,
  COPY_4_BYTE_OFFSET = 3
};
static const int kMaximumTagLength = 5;  // COPY_4_BYTE_OFFSET plus the actual offset.

// Data stored per entry in lookup table:
//      Range   Bits-used       Description
//      ------------------------------------
//      1..64   0..7            Literal/copy length encoded in opcode byte
//      0..7    8..10           Copy offset encoded in opcode byte / 256
//      0..4    11..13          Extra bytes after opcode
//
// We use eight bits for the length even though 7 would have sufficed
// because of efficiency reasons:
//      (1) Extracting a byte is faster than a bit-field
//      (2) It properly aligns copy offset so we do not need a <<8
static constexpr uint16_t char_table[256] = {
    // clang-format off
  0x0001, 0x0804, 0x1001, 0x2001, 0x0002, 0x0805, 0x1002, 0x2002,
  0x0003, 0x0806, 0x1003, 0x2003, 0x0004, 0x0807, 0x1004, 0x2004,
  0x0005, 0x0808, 0x1005, 0x2005, 0x0006, 0x0809, 0x1006, 0x2006,
  0x0007, 0x080a, 0x1007, 0x2007, 0x0008, 0x080b, 0x1008, 0x2008,
  0x0009, 0x0904, 0x1009, 0x2009, 0x000a, 0x0905, 0x100a, 0x200a,
  0x000b, 0x0906, 0x100b, 0x200b, 0x000c, 0x0907, 0x100c, 0x200c,
  0x000d, 0x0908, 0x100d, 0x200d, 0x000e, 0x0909, 0x100e, 0x200e,
  0x000f, 0x090a, 0x100f, 0x200f, 0x0010, 0x090b, 0x1010, 0x2010,
  0x0011, 0x0a04, 0x1011, 0x2011, 0x0012, 0x0a05, 0x1012, 0x2012,
  0x0013, 0x0a06, 0x1013, 0x2013, 0x0014, 0x0a07, 0x1014, 0x2014,
  0x0015, 0x0a08, 0x1015, 0x2015, 0x0016, 0x0a09, 0x1016, 0x2016,
  0x0017, 0x0a0a, 0x1017, 0x2017, 0x0018, 0x0a0b, 0x1018, 0x2018,
  0x0019, 0x0b04, 0x1019, 0x2019, 0x001a, 0x0b05, 0x101a, 0x201a,
  0x001b, 0x0b06, 0x101b, 0x201b, 0x001c, 0x0b07, 0x101c, 0x201c,
  0x001d, 0x0b08, 0x101d, 0x201d, 0x001e, 0x0b09, 0x101e, 0x201e,
  0x001f, 0x0b0a, 0x101f, 0x201f, 0x0020, 0x0b0b, 0x1020, 0x2020,
  0x0021, 0x0c04, 0x1021, 0x2021, 0x0022, 0x0c05, 0x1022, 0x2022,
  0x0023, 0x0c06, 0x1023, 0x2023, 0x0024, 0x0c07, 0x1024, 0x2024,
  0x0025, 0x0c08, 0x1025, 0x2025, 0x0026, 0x0c09, 0x1026, 0x2026,
  0x0027, 0x0c0a, 0x1027, 0x2027, 0x0028, 0x0c0b, 0x1028, 0x2028,
  0x0029, 0x0d04, 0x1029, 0x2029, 0x002a, 0x0d05, 0x102a, 0x202a,
  0x002b, 0x0d06, 0x102b, 0x202b, 0x002c, 0x0d07, 0x102c, 0x202c,
  0x002d, 0x0d08, 0x102d, 0x202d, 0x002e, 0x0d09, 0x102e, 0x202e,
  0x002f, 0x0d0a, 0x102f, 0x202f, 0x0030, 0x0d0b, 0x1030, 0x2030,
  0x0031, 0x0e04, 0x1031, 0x2031, 0x0032, 0x0e05, 0x1032, 0x2032,
  0x0033, 0x0e06, 0x1033, 0x2033, 0x0034, 0x0e07, 0x1034, 0x2034,
  0x0035, 0x0e08, 0x1035, 0x2035, 0x0036, 0x0e09, 0x1036, 0x2036,
  0x0037, 0x0e0a, 0x1037, 0x2037, 0x0038, 0x0e0b, 0x1038, 0x2038,
  0x0039, 0x0f04, 0x1039, 0x2039, 0x003a, 0x0f05, 0x103a, 0x203a,
  0x003b, 0x0f06, 0x103b, 0x203b, 0x003c, 0x0f07, 0x103c, 0x203c,
  0x0801, 0x0f08, 0x103d, 0x203d, 0x1001, 0x0f09, 0x103e, 0x203e,
  0x1801, 0x0f0a, 0x103f, 0x203f, 0x2001, 0x0f0b, 0x1040, 0x2040,
    // clang-format on
};

}  // end namespace internal
}  // end namespace snappy

#endif  // THIRD_PARTY_SNAPPY_SNAPPY_INTERNAL_H_
