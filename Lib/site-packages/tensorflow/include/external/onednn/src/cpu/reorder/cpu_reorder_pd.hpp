/*******************************************************************************
* Copyright 2016-2024 Intel Corporation
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

#ifndef CPU_REORDER_CPU_REORDER_PD_HPP
#define CPU_REORDER_CPU_REORDER_PD_HPP

#include <assert.h>

#include "common/c_types_map.hpp"
#include "common/reorder_pd.hpp"
#include "common/utils.hpp"
#include "cpu/cpu_engine.hpp"

namespace dnnl {
namespace impl {
namespace cpu {

struct cpu_reorder_pd_t : public reorder_pd_t {
    using reorder_pd_t::reorder_pd_t;

    status_t init(
            engine_t *engine, engine_t *src_engine, engine_t *dst_engine) {
        const auto &post_ops = attr()->post_ops_;
        bool args_ok = IMPLICATION(post_ops.len() != 0,
                post_ops.len() == 1
                        && post_ops.entry_[0].kind == primitive_kind::sum);
        VDISPATCH_REORDER(args_ok, VERBOSE_UNSUPPORTED_POSTOP);
        return status::success;
    }

    // The function splits dimension products based on input mask and returns
    //   them as `D_start`, `D_mask` and `D_rest`.
    // Its used to estimate amount of memory for scratchpad and precomputed
    //   destination scales.
    void get_D_values(const memory_desc_wrapper &input_d, int mask,
            dim_t *D_start, dim_t *D_mask, dim_t *D_rest) const {
        int ndims = input_d.ndims();
        int ndims_start = 0, ndims_mask = 0;
        // XXX: Currently user can pass a mask that has non-zero values in
        // dimensions that do not exist in a md. Since attributes are created
        // separately mask can't be validated.
        // This line truncates a given mask in range [0, 1 << ndims - 1]
        // TODO: Such masks can be either prohibited at pd creation step at
        // API level or checked by each implementation that relies on it.
        mask &= (1 << ndims) - 1;

        for (; mask > 0 && !(mask & 0x1); mask >>= 1)
            ++ndims_start;
        for (; mask > 0 && mask & 0x1; mask >>= 1)
            ++ndims_mask;
        assert(mask == 0);

        if (D_start)
            *D_start = utils::array_product(input_d.dims(), ndims_start);
        if (D_mask)
            *D_mask = utils::array_product(
                    input_d.dims() + ndims_start, ndims_mask);
        assert(*D_mask >= 1);
        if (D_rest) *D_rest = input_d.nelems() / (*D_start * *D_mask);
    }

    // The function serves same purpose as `dnnl::impl::cpu::precompute_scales`.
    // The reason it's dedicated to reorder is it's the only primitive so far
    //   that utilizes `mask > 0` for destination scales.
    const float *precompute_scales(const memory_tracking::grantor_t &scratchpad,
            const primitive_attr_t *attr, size_t count,
            const float *dst_scales) const {
        using namespace dnnl::impl::memory_tracking::names;

        int mask = -1;
        bool is_set = false;
        auto status = attr->scales_.get(DNNL_ARG_DST, &mask, &is_set);
        if (status != status::success) return nullptr;

        // It's possible that mask > 0 but `count` is still `1`. This case is
        //   covered by `DEFINE_ARG_SCALES_BUFFER` macro and no need to inverse
        //   in such case.
        if (is_set && mask > 0 && count > 1) {
            auto loc_scales = scratchpad.template get<float>(
                    key_reorder_precomputed_dst_scales);
            if (!loc_scales) return nullptr;

            PRAGMA_OMP_SIMD()
            for (size_t c = 0; c < count; c++)
                loc_scales[c] = 1.f / dst_scales[c];

            return loc_scales;
        } else {
            return dst_scales;
        }
    }
};

} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif

// vim: et ts=4 sw=4 cindent cino+=l0,\:4,N-s
