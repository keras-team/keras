/*******************************************************************************
* Copyright 2023-2024 Intel Corporation
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
*******************************************************************************/

#ifndef COMMON_INT4_HPP
#define COMMON_INT4_HPP

#include <cmath>
#include <cstdint>
#include <limits>
#include <type_traits>

namespace dnnl {
namespace impl {

enum class int4_extract_t : uint8_t { low_half = 0, high_half = 4 };

inline uint8_t extract_half_byte(uint8_t val, int4_extract_t half) {
    uint8_t shift = static_cast<uint8_t>(half);
    return (val >> shift) & 0xF;
}

inline uint8_t insert_half_byte(uint8_t src, uint8_t val, int4_extract_t half) {
    uint8_t shift = static_cast<uint8_t>(half);
    uint8_t mask = half == int4_extract_t::high_half ? 0x0F : 0xF0;
    return (src & mask) | (uint8_t)(val << shift);
}

struct uint4_t {
    template <typename IntegerType,
            typename SFINAE = typename std::enable_if<
                    std::is_integral<IntegerType>::value>::type>
    constexpr uint4_t(IntegerType raw) : raw_(raw) {}
    uint4_t(float val_f32) {
        uint8_t val_uint8 = static_cast<uint8_t>(val_f32);
        raw_ = val_uint8 & 0xF;
    }

    operator float() const { return (float)raw_; }

    uint8_t insert(uint8_t src, int4_extract_t half) const {
        return insert_half_byte(src, raw_, half);
    }

    static uint4_t extract(uint8_t val, int4_extract_t half) {
        return uint4_t(extract_half_byte(val, half));
    }

private:
    uint8_t raw_;
};

static_assert(sizeof(uint4_t) == 1, "uint4_t must be 1 byte");

struct int4_t {
    template <typename IntegerType,
            typename SFINAE = typename std::enable_if<
                    std::is_integral<IntegerType>::value>::type>
    constexpr int4_t(IntegerType i) : raw_(static_cast<uint8_t>(i)) {}
    int4_t(float val_f32) {
        int8_t val_int8 = static_cast<int8_t>(val_f32);
        bool negative = val_f32 < 0;
        // positive numbers have the most significant bit set to 0
        // negative numbers have the most significant bit set to 1
        raw_ = negative ? (val_int8 & 0xF) | 0x8 : val_int8 & 0x7;
    }

    operator float() const {
        float sign = (raw_ & (1 << 3)) ? -1.f : 1.f;
        return sign * (float)(sign == -1 ? (~raw_ & 0xF) + 1 : raw_);
    }

    uint8_t insert(uint8_t src, int4_extract_t half) const {
        return insert_half_byte(src, raw_, half);
    }

    static int4_t extract(uint8_t val, int4_extract_t half) {
        return int4_t(extract_half_byte(val, half));
    }

private:
    uint8_t raw_;
};

static_assert(sizeof(int4_t) == 1, "int4_t must be 1 byte");

} // namespace impl
} // namespace dnnl

#endif
