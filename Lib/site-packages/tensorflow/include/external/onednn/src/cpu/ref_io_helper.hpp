/*******************************************************************************
* Copyright 2021-2024 Intel Corporation
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

#ifndef REF_IO_HELPER_HPP
#define REF_IO_HELPER_HPP

#include <cassert>

#include "common/c_types_map.hpp"
#include "common/dnnl_traits.hpp"
#include "common/type_helpers.hpp"

#include "cpu/simple_q10n.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace io {

inline int load_int_value(data_type_t dt, const void *ptr, dim_t idx) {
    assert(ptr);
#define CASE(dt) \
    case dt: \
        return static_cast<int>( \
                reinterpret_cast<const typename prec_traits<dt>::type *>( \
                        ptr)[idx]);

    using namespace data_type;
    switch (dt) {
        CASE(s32);
        CASE(s8);
        CASE(u8);
        default: assert(!"bad data_type");
    }

#undef CASE
    return INT_MAX;
}

inline float load_float_value(data_type_t dt, const void *ptr, dim_t idx) {
    assert(ptr);
#define CASE(dt) \
    case dt: \
        return static_cast<float>( \
                reinterpret_cast<const typename prec_traits<dt>::type *>( \
                        ptr)[idx]);

    using namespace data_type;
    switch (dt) {
        CASE(f8_e5m2);
        CASE(f8_e4m3);
        CASE(bf16);
        CASE(f16);
        CASE(f32);
        CASE(s32);
        CASE(s8);
        CASE(u8);
        case s4: {
            const auto shift = idx % 2 ? int4_extract_t::high_half
                                       : int4_extract_t::low_half;
            auto val = int4_t::extract(
                    reinterpret_cast<const uint8_t *>(ptr)[idx / 2], shift);
            return static_cast<float>(val);
        }
        case u4: {
            const auto shift = idx % 2 ? int4_extract_t::high_half
                                       : int4_extract_t::low_half;
            auto val = uint4_t::extract(
                    reinterpret_cast<const uint8_t *>(ptr)[idx / 2], shift);
            return static_cast<float>(val);
        }
        default: assert(!"bad data_type");
    }

#undef CASE
    return NAN;
}

inline void store_float_value(data_type_t dt, float val, void *ptr, dim_t idx) {
    assert(ptr);
#define CASE(dt) \
    case dt: { \
        using type_ = typename prec_traits<dt>::type; \
        *(reinterpret_cast<type_ *>(ptr) + idx) \
                = cpu::q10n::saturate_and_round<type_>(val); \
    } break;

    using namespace data_type;
    switch (dt) {
        CASE(f8_e5m2);
        CASE(f8_e4m3);
        CASE(bf16);
        CASE(f16);
        CASE(f32);
        CASE(s32);
        CASE(s8);
        CASE(u8);
        default: assert(!"bad data_type");
    }

#undef CASE
}

} // namespace io
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif

// vim: et ts=4 sw=4 cindent cino+=l0,\:4,N-s
