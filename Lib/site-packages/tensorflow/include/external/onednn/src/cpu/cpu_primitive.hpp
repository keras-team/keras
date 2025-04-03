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

#ifndef CPU_CPU_PRIMITIVE_HPP
#define CPU_CPU_PRIMITIVE_HPP

#include <assert.h>

#include "oneapi/dnnl/dnnl_types.h"

#include "common/c_types_map.hpp"
#include "common/primitive_attr.hpp"
#include "common/primitive_exec_types.hpp"
#include "common/utils.hpp"
#include "common/z_magic.hpp"

#include "cpu/ref_io_helper.hpp"

// Use `...` for `msg` and additional variables used in msg
#define VCHECK_ATTR(cond, ...) \
    VCONDCHECK(primitive, exec, check, primitive, (cond), \
            status::invalid_arguments, __VA_ARGS__)

#define DEFINE_SCALES_BUFFER_ATTR_ARG(attr, scales, arg) \
    alignas(16) float CONCAT2(scales, _buf16)[16] = {0}; \
    const float *scales {nullptr}; \
    if ((attr)) { \
        if ((attr)->output_scales_.has_default_values()) { \
            utils::array_set(CONCAT2(scales, _buf16), 1.0f, 16); \
            scales = CONCAT2(scales, _buf16); \
        } else { \
            scales = CTX_IN_MEM(const float *, arg); \
            VCHECK_ATTR(scales != nullptr, \
                    "Scales buffer for arg %d is missing", arg); \
            const auto scales_d = ctx.memory_mdw(arg); \
            VCHECK_ATTR(scales_d.data_type() == data_type::f32, \
                    "Scales data type is not f32"); \
            VCHECK_ATTR(scales_d.ndims() == 1, "Scales ndims is not 1"); \
            if (scales_d.dims()[0] == 1) { \
                utils::array_set(CONCAT2(scales, _buf16), scales[0], 16); \
                scales = CONCAT2(scales, _buf16); \
            } \
        } \
    } \
    MAYBE_UNUSED(scales);

#define DEFINE_SCALES_BUFFER_ATTR(attr, scales) \
    DEFINE_SCALES_BUFFER_ATTR_ARG(attr, scales, DNNL_ARG_ATTR_OUTPUT_SCALES);

#define DEFINE_SCALES_BUFFER(scales) \
    DEFINE_SCALES_BUFFER_ATTR(pd()->attr(), scales)

#define DEFINE_ARG_SCALES_BUFFER_ATTR(attr, scales, arg) \
    alignas(16) float CONCAT2(scales, _buf16)[16] = {0}; \
    const float *scales {nullptr}; \
    if ((attr)) { \
        if ((attr)->scales_.get(arg).has_default_values()) { \
            utils::array_set(CONCAT2(scales, _buf16), 1.0f, 16); \
            scales = CONCAT2(scales, _buf16); \
        } else { \
            scales = CTX_IN_MEM(const float *, DNNL_ARG_ATTR_SCALES | arg); \
            VCHECK_ATTR(scales != nullptr, \
                    "Scales buffer for arg %d is missing", arg); \
            const auto scales_d = ctx.memory_mdw(DNNL_ARG_ATTR_SCALES | arg); \
            VCHECK_ATTR(utils::one_of(scales_d.data_type(), data_type::f32, \
                                data_type::f16, data_type::bf16), \
                    "Unsupported scales data type"); \
            if (scales_d.nelems() == 1) { \
                const float s = cpu::io::load_float_value( \
                        scales_d.data_type(), scales, 0); \
                if (utils::one_of(arg, DNNL_ARG_DST, \
                            DNNL_ARG_ATTR_POST_OP_DW | DNNL_ARG_DST)) { \
                    utils::array_set(CONCAT2(scales, _buf16), 1.f / s, 16); \
                } else { \
                    utils::array_set(CONCAT2(scales, _buf16), s, 16); \
                } \
                scales = CONCAT2(scales, _buf16); \
            } \
        } \
    } \
    MAYBE_UNUSED(scales);

#define DEFINE_ARG_SCALES_BUFFER(scales, arg) \
    DEFINE_ARG_SCALES_BUFFER_ATTR(pd()->attr(), scales, arg)

#define DEFINE_ZERO_POINTS_BUFFER(zero_points_ptr, mem_arg) \
    int32_t CONCAT2(default_zero_point_, mem_arg) = 0; \
    const int32_t *zero_points_ptr \
            = pd()->attr()->zero_points_.defined(mem_arg) \
            ? &CONCAT2(default_zero_point_, mem_arg) \
            : CTX_IN_MEM( \
                    const int32_t *, DNNL_ARG_ATTR_ZERO_POINTS | mem_arg); \
    VCHECK_ATTR(zero_points_ptr != nullptr, \
            "Zero points buffer for arg %d is missing", mem_arg); \
    MAYBE_UNUSED(zero_points_ptr);

#define ASSIGN_ARG_SCALE_VALUE(scale, mem_arg) \
    alignas(16) float CONCAT2(CONCAT2(scales, _buf16), mem_arg)[16] = {0}; \
    if (pd()->attr()->scales_.get(mem_arg).has_default_values()) { \
        utils::array_set(CONCAT2(CONCAT2(scales, _buf16), mem_arg), 1.0f, 16); \
        scale = CONCAT2(CONCAT2(scales, _buf16), mem_arg); \
    } else { \
        const auto scale_d = ctx.memory_mdw(DNNL_ARG_ATTR_SCALES | mem_arg); \
        VCHECK_ATTR(scale_d.data_type() == data_type::f32, \
                "Scales data type is not f32"); \
        VCHECK_ATTR(scale_d.ndims() == 1, "Scales ndims is not 1"); \
        VCHECK_ATTR( \
                scale_d.dims()[0] == 1, "Not a single scale was provided"); \
        const float *scale_p \
                = CTX_IN_MEM(const float *, DNNL_ARG_ATTR_SCALES | mem_arg); \
        VCHECK_ATTR(scale_p != nullptr, "Scales buffer for arg %d is missing", \
                mem_arg); \
        scale = scale_p; \
    }

#define DEFINE_ZERO_POINT_VALUE_ATTR(attr, zero_point, mem_arg) \
    int32_t zero_point = 0; \
    if (!attr->zero_points_.has_default_values(mem_arg)) { \
        const auto zero_points_d \
                = ctx.memory_mdw(DNNL_ARG_ATTR_ZERO_POINTS | mem_arg); \
        VCHECK_ATTR(utils::one_of(zero_points_d.data_type(), data_type::s32, \
                            data_type::s8, data_type::u8, data_type::s4, \
                            data_type::u4), \
                "Unsupported zero points type"); \
        VCHECK_ATTR(zero_points_d.dims()[0] == 1, \
                "Not a single zero points was provided"); \
        const int32_t *zero_points_ptr = CTX_IN_MEM( \
                const int32_t *, DNNL_ARG_ATTR_ZERO_POINTS | mem_arg); \
        VCHECK_ATTR(zero_points_ptr != nullptr, \
                "Zero points buffer for arg %d is missing", mem_arg); \
        zero_point = cpu::io::load_int_value( \
                zero_points_d.data_type(), zero_points_ptr, 0); \
    } \
    MAYBE_UNUSED(zero_point);

#define DEFINE_ZERO_POINT_VALUE(zero_point, mem_arg) \
    DEFINE_ZERO_POINT_VALUE_ATTR(pd()->attr(), zero_point, mem_arg)

#endif // CPU_CPU_PRIMITIVE_HPP
