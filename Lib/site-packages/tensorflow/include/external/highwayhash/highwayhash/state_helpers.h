// Copyright 2017 Google Inc. All Rights Reserved.
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

#ifndef HIGHWAYHASH_STATE_H_
#define HIGHWAYHASH_STATE_H_

// Helper functions to split inputs into packets and call State::Update on each.

#include <stdint.h>
#include <cstddef>
#include <cstring>
#include <memory>

#include "highwayhash/compiler_specific.h"

namespace highwayhash {

// uint64_t is unsigned long on Linux; we need 'unsigned long long'
// for interoperability with TensorFlow.
typedef unsigned long long HH_U64;  // NOLINT

// Copies the remaining bytes to a zero-padded buffer, sets the upper byte to
// size % 256 (always possible because this should only be called if the
// total size is not a multiple of the packet size) and updates hash state.
//
// The padding scheme is essentially from SipHash, but permuted for the
// convenience of AVX-2 masked loads. This function must use the same layout so
// that the vector and scalar HighwayTreeHash have the same result.
//
// "remaining_size" is the number of accessible/remaining bytes
// (size % kPacketSize).
//
// Primary template; the specialization for AVX-2 is faster. Intended as an
// implementation detail, do not call directly.
template <class State>
HH_INLINE void PaddedUpdate(const HH_U64 size, const char* remaining_bytes,
                            const HH_U64 remaining_size, State* state) {
  char final_packet[State::kPacketSize] HH_ALIGNAS(32) = {0};

  // This layout matches the AVX-2 specialization in highway_tree_hash.h.
  uint32_t packet4 = static_cast<uint32_t>(size) << 24;

  const size_t remainder_mod4 = remaining_size & 3;
  if (remainder_mod4 != 0) {
    const char* final_bytes = remaining_bytes + remaining_size - remainder_mod4;
    packet4 += static_cast<uint32_t>(final_bytes[0]);
    const int idx1 = remainder_mod4 >> 1;
    const int idx2 = remainder_mod4 - 1;
    packet4 += static_cast<uint32_t>(final_bytes[idx1]) << 8;
    packet4 += static_cast<uint32_t>(final_bytes[idx2]) << 16;
  }

  memcpy(final_packet, remaining_bytes, remaining_size - remainder_mod4);
  memcpy(final_packet + State::kPacketSize - 4, &packet4, sizeof(packet4));

  state->Update(final_packet);
}

// Updates hash state for every whole packet, and once more for the final
// padded packet.
template <class State>
HH_INLINE void UpdateState(const char* bytes, const HH_U64 size, State* state) {
  // Feed entire packets.
  const int kPacketSize = State::kPacketSize;
  static_assert((kPacketSize & (kPacketSize - 1)) == 0, "Size must be 2^i.");
  const size_t remainder = size & (kPacketSize - 1);
  const size_t truncated_size = size - remainder;
  for (size_t i = 0; i < truncated_size; i += kPacketSize) {
    state->Update(bytes + i);
  }

  PaddedUpdate(size, bytes + truncated_size, remainder, state);
}

// Convenience function for updating with the bytes of a string.
template <class String, class State>
HH_INLINE void UpdateState(const String& s, State* state) {
  const char* bytes = reinterpret_cast<const char*>(s.data());
  const size_t size = s.length() * sizeof(typename String::value_type);
  UpdateState(bytes, size, state);
}

// Computes a hash of a byte array using the given hash State class.
//
// Example: const SipHashState::Key key = { 1, 2 }; char data[4];
// ComputeHash<SipHashState>(key, data, sizeof(data));
//
// This function avoids duplicating Update/Finalize in every call site.
// Callers wanting to combine multiple hashes should repeatedly UpdateState()
// and only call State::Finalize once.
template <class State>
HH_U64 ComputeHash(const typename State::Key& key, const char* bytes,
                   const HH_U64 size) {
  State state(key);
  UpdateState(bytes, size, &state);
  return state.Finalize();
}

// Computes a hash of a string's bytes using the given hash State class.
//
// Example: const SipHashState::Key key = { 1, 2 };
// StringHasher<SipHashState>()(key, std::u16string(u"abc"));
//
// A struct with nested function template enables deduction of the String type.
template <class State>
struct StringHasher {
  template <class String>
  HH_U64 operator()(const typename State::Key& key, const String& s) {
    State state(key);
    UpdateState(s, &state);
    return state.Finalize();
  }
};

}  // namespace highwayhash

#endif  // HIGHWAYHASH_STATE_H_
