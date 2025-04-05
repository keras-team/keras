/*******************************************************************************
* Copyright 2019-2024 Intel Corporation
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

#ifndef COMMON_FLOAT16_HPP
#define COMMON_FLOAT16_HPP

#include <cmath>
#include <cstdint>
#include <limits>
#include <type_traits>

#include "bit_cast.hpp"
#include "oneapi/dnnl/dnnl.h"

namespace dnnl {
namespace impl {

struct float16_t {
    uint16_t raw;

    constexpr float16_t(uint16_t raw, bool) : raw(raw) {}

    float16_t() = default;
    float16_t(float f) { (*this) = f; }

    float16_t &operator=(float f);

    operator float() const;
    float f() { return (float)(*this); }

    float16_t &operator+=(float16_t a) {
        (*this) = float(f() + a.f());
        return *this;
    }
};

static_assert(sizeof(float16_t) == 2, "float16_t must be 2 bytes");

inline float16_t &float16_t::operator=(float f) {
    uint32_t i = utils::bit_cast<uint32_t>(f);
    uint32_t s = i >> 31;
    uint32_t e = (i >> 23) & 0xFF;
    uint32_t m = i & 0x7FFFFF;

    uint32_t ss = s;
    uint32_t mm = m >> 13;
    uint32_t r = m & 0x1FFF;
    uint32_t ee = 0;
    int32_t eee = static_cast<int32_t>((e - 127) + 15);

    if (e == 0) {
        // Denormal/zero floats all become zero.
        ee = 0;
        mm = 0;
    } else if (e == 0xFF) {
        // Preserve inf/nan, but set quiet bit for nan (snan->qnan).
        ee = 0x1F;
        if (m != 0) mm |= 0x200;
    } else if (eee > 0 && eee < 0x1F) {
        // Normal range. Perform round to even on mantissa.
        ee = static_cast<uint32_t>(eee);
        if (r > (0x1000 - (mm & 1))) {
            // Round up.
            mm++;
            if (mm == 0x400) {
                // Rounds up to next dyad (or inf).
                mm = 0;
                ee++;
            }
        }
    } else if (eee >= 0x1F) {
        // Overflow.
        ee = 0x1F;
        mm = 0;
    } else {
        // Underflow.
        float ff = fabsf(f) + 0.5f;
        uint32_t ii = utils::bit_cast<uint32_t>(ff);
        ee = 0;
        mm = ii & 0x7FF;
    }

    this->raw = static_cast<uint16_t>((ss << 15) | (ee << 10) | mm);
    return *this;
}

inline float16_t::operator float() const {
    uint32_t ss = raw >> 15;
    uint32_t ee = (raw >> 10) & 0x1F;
    uint32_t mm = raw & 0x3FF;

    uint32_t s = ss;
    uint32_t eee = ee - 15 + 127;
    uint32_t m = mm << 13;
    uint32_t e;

    if (ee == 0) {
        if (mm == 0)
            e = 0;
        else {
            // Half denormal -> float normal
            return (ss ? -1 : 1) * std::scalbn((float)mm, -24);
        }
    } else if (ee == 0x1F) {
        // inf/nan
        e = 0xFF;
        // set quiet bit for nan (snan->qnan)
        if (m != 0) m |= 0x400000;
    } else
        e = eee;

    uint32_t f = (s << 31) | (e << 23) | m;

    return utils::bit_cast<float>(f);
}

void cvt_float_to_float16(float16_t *out, const float *inp, size_t nelems);
void cvt_float16_to_float(float *out, const float16_t *inp, size_t nelems);

// performs element-by-element sum of inp and add float arrays and stores
// result to float16 out array with downconversion
// out[:] = (float16_t)(inp0[:] + inp1[:])
void add_floats_and_cvt_to_float16(
        float16_t *out, const float *inp0, const float *inp1, size_t nelems);

#if DNNL_X64
namespace cpu {
namespace x64 {
bool DNNL_API try_cvt_f16_to_f32(float *, const float16_t *);
bool DNNL_API try_cvt_f32_to_f16(float16_t *, const float *);
} // namespace x64
} // namespace cpu
#endif

} // namespace impl
} // namespace dnnl

#endif
