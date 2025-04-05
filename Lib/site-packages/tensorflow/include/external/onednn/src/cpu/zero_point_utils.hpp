/*******************************************************************************
* Copyright 2020-2021 Intel Corporation
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

#ifndef CPU_ZERO_POINT_UTILS_HPP
#define CPU_ZERO_POINT_UTILS_HPP

#include "common/c_types_map.hpp"
#include "common/primitive_attr.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
/*
 * Structure describing the size zero point padding compensation buffer.
 * Size of buffer is h * w * d * output channels. Size h * w * d represents
 * number of unique application of filter over input spatial where filter
 * overlapped the padding need for calculating unique zero point padding
 * compensation of given case. Pad variables represents number of unique
 * zp padding compensation values resulting from participation of a given type
 * of padding (example top padding) over a given axis (example h). Mid points
 * over given axis represents compensation resulting in the absence of padding
 * over given axis, but where padding over other axis exists. Example: 2D: conv:
 * mid_point_w = true, where filter overlaps top padding, but not right and left
 * padding. Spatial filter w size fits in w range of input image.
 */
struct zero_point_pad_comp_config_t {
    zero_point_pad_comp_config_t() = default;
    zero_point_pad_comp_config_t(const dim_t front_pad, const dim_t back_pad,
            const dim_t top_pad, const dim_t bottom_pad, const dim_t left_pad,
            const dim_t right_pad, const dim_t stride_d, const dim_t stride_h,
            const dim_t stride_w, const dim_t od, const dim_t oh,
            const dim_t ow);

    dim_t top_pad = 0;
    dim_t bottom_pad = 0;
    dim_t left_pad = 0;
    dim_t right_pad = 0;
    dim_t front_pad = 0;
    dim_t back_pad = 0;

    dim_t mid_h = 0;
    dim_t mid_w = 0;
    dim_t mid_d = 0;

    dim_t h = 0;
    dim_t w = 0;
    dim_t d = 0;
};

struct zero_point_config_t {
    zero_point_config_t() = default;
    zero_point_config_t(const primitive_attr_t &attr);

    bool src_exists = false;
    bool dst_exists = false;
    bool src_is_common = false;
    zero_point_pad_comp_config_t src_pad_comp;

    bool zp_exists() const noexcept;
};

struct zero_point_call_params_t {
    zero_point_call_params_t() = default;
    zero_point_call_params_t(const int32_t *src, const int32_t *dst,
            const int32_t *src_comp, const int32_t *src_pad_comp);

    const int32_t *src = nullptr;
    const int32_t *dst = nullptr;
    const int32_t *src_comp = nullptr;
    const int32_t *src_pad_comp = nullptr;
};

bool zero_points_valid(const primitive_attr_t *attr,
        bool per_oc_bcast_accepted = false) noexcept;

void set_zp_src_comp_flags(memory_desc_t &weights_md, bool with_groups);
const int32_t *get_src_zp_comp_from_wei(const int8_t *weights,
        const memory_desc_wrapper &weights_md, bool signed_input, dim_t ngroups,
        dim_t oc);

} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif
