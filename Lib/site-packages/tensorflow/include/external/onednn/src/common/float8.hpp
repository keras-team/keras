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

#ifndef COMMON_FLOAT8_HPP
#define COMMON_FLOAT8_HPP

#include <cstdint>

#include "common/float16.hpp"

namespace dnnl {
namespace impl {

struct float8_e5m2_t {
    uint8_t raw_bits_;
    float8_e5m2_t() = default;
    constexpr float8_e5m2_t(uint8_t r, bool) : raw_bits_(r) {}
    float8_e5m2_t(float f) { (*this) = f; }
    float8_e5m2_t(float16_t f) { (*this) = f; }

    float8_e5m2_t DNNL_API &operator=(float f);
    float8_e5m2_t DNNL_API &operator=(float16_t f);

    DNNL_API operator float() const;
    DNNL_API operator float16_t() const;

    float8_e5m2_t &operator+=(const float a) {
        (*this) = float {*this} + a;
        return *this;
    }
};
static_assert(sizeof(float8_e5m2_t) == 1, "float8_e5m2_t must be 1 byte");

struct float8_e4m3_t {
    uint8_t raw_bits_;
    float8_e4m3_t() = default;
    constexpr float8_e4m3_t(uint8_t r, bool) : raw_bits_(r) {}
    float8_e4m3_t(float f) { (*this) = f; }
    float8_e4m3_t(float16_t f) { (*this) = f; }

    float8_e4m3_t DNNL_API &operator=(float f);
    float8_e4m3_t DNNL_API &operator=(float16_t f);

    DNNL_API operator float() const;
    DNNL_API operator float16_t() const;

    float8_e4m3_t &operator+=(const float a) {
        (*this) = float {*this} + a;
        return *this;
    }
};
static_assert(sizeof(float8_e5m2_t) == 1, "float8_e4m3_t must be 1 byte");

#if DNNL_X64
namespace cpu {
namespace x64 {
bool DNNL_API try_cvt_f8_e5m2_to_f32(float *, const float8_e5m2_t *);
bool DNNL_API try_cvt_f8_e4m3_to_f32(float *, const float8_e4m3_t *);
bool DNNL_API try_cvt_f8_e5m2_to_f16(float16_t *, const float8_e5m2_t *);
bool DNNL_API try_cvt_f8_e4m3_to_f16(float16_t *, const float8_e4m3_t *);
bool DNNL_API try_cvt_f16_to_f8_e5m2(float8_e5m2_t *, const float16_t *);
bool DNNL_API try_cvt_f16_to_f8_e4m3(float8_e4m3_t *, const float16_t *);
bool DNNL_API try_cvt_f32_to_f8_e5m2(float8_e5m2_t *, const float *);
bool DNNL_API try_cvt_f32_to_f8_e4m3(float8_e4m3_t *, const float *);
} // namespace x64
} // namespace cpu
#endif

} // namespace impl
} // namespace dnnl

#endif
