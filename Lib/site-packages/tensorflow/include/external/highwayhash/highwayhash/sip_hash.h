// Copyright 2016 Google Inc. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef HIGHWAYHASH_SIP_HASH_H_
#define HIGHWAYHASH_SIP_HASH_H_

// Portable but fast SipHash implementation.

#include <cstddef>
#include <cstring>  // memcpy

#include "highwayhash/arch_specific.h"
#include "highwayhash/compiler_specific.h"
#include "highwayhash/endianess.h"
#include "highwayhash/state_helpers.h"

namespace highwayhash {

// Paper: https://www.131002.net/siphash/siphash.pdf
template <int kUpdateIters, int kFinalizeIters>
class SipHashStateT {
 public:
  using Key = HH_U64[2];
  static const size_t kPacketSize = sizeof(HH_U64);

  explicit HH_INLINE SipHashStateT(const Key& key) {
    v0 = 0x736f6d6570736575ull ^ key[0];
    v1 = 0x646f72616e646f6dull ^ key[1];
    v2 = 0x6c7967656e657261ull ^ key[0];
    v3 = 0x7465646279746573ull ^ key[1];
  }

  HH_INLINE void Update(const char* bytes) {
    HH_U64 packet;
    memcpy(&packet, bytes, sizeof(packet));
    packet = host_from_le64(packet);

    v3 ^= packet;

    Compress<kUpdateIters>();

    v0 ^= packet;
  }

  HH_INLINE HH_U64 Finalize() {
    // Mix in bits to avoid leaking the key if all packets were zero.
    v2 ^= 0xFF;

    Compress<kFinalizeIters>();

    return (v0 ^ v1) ^ (v2 ^ v3);
  }
 private:
  // Rotate a 64-bit value "v" left by N bits.
  template <HH_U64 bits>
  static HH_INLINE HH_U64 RotateLeft(const HH_U64 v) {
    const HH_U64 left = v << bits;
    const HH_U64 right = v >> (64 - bits);
    return left | right;
  }

  template <size_t rounds>
  HH_INLINE void Compress() {
    for (size_t i = 0; i < rounds; ++i) {
      // ARX network: add, rotate, exclusive-or.
      v0 += v1;
      v2 += v3;
      v1 = RotateLeft<13>(v1);
      v3 = RotateLeft<16>(v3);
      v1 ^= v0;
      v3 ^= v2;

      v0 = RotateLeft<32>(v0);

      v2 += v1;
      v0 += v3;
      v1 = RotateLeft<17>(v1);
      v3 = RotateLeft<21>(v3);
      v1 ^= v2;
      v3 ^= v0;

      v2 = RotateLeft<32>(v2);
    }
  }

  HH_U64 v0;
  HH_U64 v1;
  HH_U64 v2;
  HH_U64 v3;
};

using SipHashState = SipHashStateT<2, 4>;
using SipHash13State = SipHashStateT<1, 3>;

// Override the HighwayTreeHash padding scheme with that of SipHash so that
// the hash output matches the known-good values in sip_hash_test.
template <>
HH_INLINE void PaddedUpdate<SipHashState>(const HH_U64 size,
                                          const char* remaining_bytes,
                                          const HH_U64 remaining_size,
                                          SipHashState* state) {
  // Copy to avoid overrunning the input buffer.
  char final_packet[SipHashState::kPacketSize] = {0};
  memcpy(final_packet, remaining_bytes, remaining_size);
  final_packet[SipHashState::kPacketSize - 1] = static_cast<char>(size & 0xFF);
  state->Update(final_packet);
}

template <>
HH_INLINE void PaddedUpdate<SipHash13State>(const HH_U64 size,
                                            const char* remaining_bytes,
                                            const HH_U64 remaining_size,
                                            SipHash13State* state) {
  // Copy to avoid overrunning the input buffer.
  char final_packet[SipHash13State::kPacketSize] = {0};
  memcpy(final_packet, remaining_bytes, remaining_size);
  final_packet[SipHash13State::kPacketSize - 1] =
      static_cast<char>(size & 0xFF);
  state->Update(final_packet);
}

// Fast, cryptographically strong pseudo-random function, e.g. for
// deterministic/idempotent 'random' number generation. See also
// README.md for information on resisting hash flooding attacks.
//
// Robust versus timing attacks because memory accesses are sequential
// and the algorithm is branch-free. Compute time is proportional to the
// number of 8-byte packets and about twice as fast as an sse41 implementation.
//
// "key" is a secret 128-bit key unknown to attackers.
// "bytes" is the data to hash; ceil(size / 8) * 8 bytes are read.
// Returns a 64-bit hash of the given data bytes, which are swapped on
// big-endian CPUs so the return value is the same as on little-endian CPUs.
static HH_INLINE HH_U64 SipHash(const SipHashState::Key& key, const char* bytes,
                                const HH_U64 size) {
  return ComputeHash<SipHashState>(key, bytes, size);
}

// Round-reduced SipHash version (1 update and 3 finalization rounds).
static HH_INLINE HH_U64 SipHash13(const SipHash13State::Key& key,
                                  const char* bytes, const HH_U64 size) {
  return ComputeHash<SipHash13State>(key, bytes, size);
}

template <int kNumLanes, int kUpdateIters, int kFinalizeIters>
static HH_INLINE HH_U64 ReduceSipTreeHash(
    const typename SipHashStateT<kUpdateIters, kFinalizeIters>::Key& key,
    const uint64_t (&hashes)[kNumLanes]) {
  SipHashStateT<kUpdateIters, kFinalizeIters> state(key);

  for (int i = 0; i < kNumLanes; ++i) {
    state.Update(reinterpret_cast<const char*>(&hashes[i]));
  }

  return state.Finalize();
}

}  // namespace highwayhash

#endif  // HIGHWAYHASH_SIP_HASH_H_
