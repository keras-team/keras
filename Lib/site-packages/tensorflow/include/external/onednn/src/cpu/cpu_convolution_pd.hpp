/*******************************************************************************
* Copyright 2016-2021 Intel Corporation
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

#ifndef CPU_CPU_CONVOLUTION_PD_HPP
#define CPU_CPU_CONVOLUTION_PD_HPP

#include <assert.h>

#include "common/c_types_map.hpp"
#include "common/convolution_pd.hpp"
#include "common/type_helpers.hpp"
#include "common/utils.hpp"
#include "cpu/cpu_eltwise_pd.hpp"
#include "cpu/cpu_engine.hpp"

namespace dnnl {
namespace impl {
namespace cpu {

struct cpu_convolution_fwd_pd_t : public convolution_fwd_pd_t {
    using convolution_fwd_pd_t::convolution_fwd_pd_t;

    bool has_padded_dst() const {
        memory_desc_wrapper dst_d(&dst_md_);
        return OC() != dst_d.padded_dims()[1];
    }

    bool wants_padded_bias() const {
        if (!with_bias()) return false;
        return has_padded_dst();
    }

    bool wants_zero_pad_dst() const {
        if (!has_padded_dst()) return false;
        bool is_zero_preserved = true;
        const auto &po = attr()->post_ops_;
        for (int i = 0; i < po.len(); i++) {
            const auto &entry = po.entry_[i];
            if (entry.is_eltwise()) {
                const auto &ee = entry.eltwise;
                is_zero_preserved = is_zero_preserved
                        && cpu_eltwise_fwd_pd_t::eltwise_preserves_zero(
                                ee.alg, ee.alpha, ee.beta);
            }
        }
        return !is_zero_preserved;
    }
};

struct cpu_convolution_bwd_data_pd_t : public convolution_bwd_data_pd_t {
    using convolution_bwd_data_pd_t::convolution_bwd_data_pd_t;
};

struct cpu_convolution_bwd_weights_pd_t : public convolution_bwd_weights_pd_t {
    using convolution_bwd_weights_pd_t::convolution_bwd_weights_pd_t;

    bool wants_padded_bias() const {
        if (!with_bias()) return false;
        memory_desc_wrapper diff_dst_d(&diff_dst_md_);
        return OC() != diff_dst_d.padded_dims()[1];
    }
};

} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif

// vim: et ts=4 sw=4 cindent cino+=l0,\:4,N-s
