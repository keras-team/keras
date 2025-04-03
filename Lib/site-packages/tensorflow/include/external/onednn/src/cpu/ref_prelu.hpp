/*******************************************************************************
* Copyright 2020-2024 Intel Corporation
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

#ifndef CPU_REF_PRELU_HPP
#define CPU_REF_PRELU_HPP

#include <assert.h>

#include "common/c_types_map.hpp"
#include "common/dnnl_thread.hpp"
#include "common/memory_tracking.hpp"
#include "common/primitive.hpp"
#include "common/type_helpers.hpp"
#include "common/utils.hpp"

#include "cpu/platform.hpp"

#include "common/broadcast_strategy.hpp"
#include "cpu/cpu_prelu_pd.hpp"

namespace dnnl {
namespace impl {
namespace cpu {

namespace prelu {
void set_reduction_buffers(
        const dim_t work_amount, dim_t &group_size, dim_t &buf_size);
dim_t get_scalar_scratchpad_offset(const std::size_t ithr,
        const std::size_t nthr, const dim_t work_amount);
} // namespace prelu

using byte = unsigned char;

struct ref_prelu_fwd_t : public primitive_t {
    struct pd_t : public cpu_prelu_fwd_pd_t {
        using cpu_prelu_fwd_pd_t::cpu_prelu_fwd_pd_t;

        DECLARE_COMMON_PD_T("ref:any", ref_prelu_fwd_t);

        status_t init(engine_t *engine) {
            using namespace data_type;
            VDISPATCH_PRELU(is_fwd(), VERBOSE_BAD_PROPKIND);
            VDISPATCH_PRELU(src_md(0)->data_type == dst_md(0)->data_type,
                    VERBOSE_INCONSISTENT_DT, "src", "dst");
            VDISPATCH_PRELU(
                    platform::has_data_type_support(src_md(0)->data_type),
                    VERBOSE_UNSUPPORTED_DT);
            VDISPATCH_PRELU(
                    platform::has_data_type_support(weights_md(0)->data_type),
                    VERBOSE_UNSUPPORTED_DT);
            VDISPATCH_PRELU(
                    attr()->has_default_values(), VERBOSE_UNSUPPORTED_ATTR);
            VDISPATCH_PRELU(set_default_formats(), VERBOSE_UNSUPPORTED_TAG);
            VDISPATCH_PRELU(memory_desc_wrapper(src_md())
                            == memory_desc_wrapper(dst_md()),
                    VERBOSE_INCONSISTENT_MDS, "src", "dst");

            return status::success;
        }
    };

    ref_prelu_fwd_t(const pd_t *apd) : primitive_t(apd) {}

    status_t execute(const exec_ctx_t &ctx) const override {
        return execute_forward(ctx);
    }

private:
    status_t execute_forward(const exec_ctx_t &ctx) const;
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }
};

struct ref_prelu_bwd_t : public primitive_t {
    struct pd_t : public cpu_prelu_bwd_pd_t {
        using cpu_prelu_bwd_pd_t::cpu_prelu_bwd_pd_t;

        DECLARE_COMMON_PD_T("ref:any", ref_prelu_bwd_t);

        status_t init(engine_t *engine) {
            using namespace data_type;
            VDISPATCH_PRELU(!is_fwd(), VERBOSE_BAD_PROPKIND);
            VDISPATCH_PRELU(diff_src_md(0)->data_type == src_md(0)->data_type,
                    VERBOSE_INCONSISTENT_DT, "diff_src", "src");
            VDISPATCH_PRELU(
                    diff_weights_md(0)->data_type == weights_md(0)->data_type,
                    VERBOSE_INCONSISTENT_DT, "diff_weights", "weights");
            VDISPATCH_PRELU(
                    diff_dst_md(0)->data_type == diff_src_md(0)->data_type,
                    VERBOSE_INCONSISTENT_DT, "diff_src", "diff_dst");
            VDISPATCH_PRELU(
                    platform::has_data_type_support(src_md(0)->data_type),
                    VERBOSE_UNSUPPORTED_DT);
            VDISPATCH_PRELU(
                    platform::has_data_type_support(weights_md(0)->data_type),
                    VERBOSE_UNSUPPORTED_DT);
            VDISPATCH_PRELU(
                    attr()->has_default_values(), VERBOSE_UNSUPPORTED_ATTR);
            VDISPATCH_PRELU(set_default_formats(), VERBOSE_UNSUPPORTED_TAG);
            VDISPATCH_PRELU(memory_desc_wrapper(diff_dst_md())
                            == memory_desc_wrapper(diff_src_md()),
                    VERBOSE_INCONSISTENT_MDS, "diff_src", "diff_dst");

            init_scratchpad();

            return status::success;
        }

        int nthr_; // To not exceed the limit in execute used for set up.

    private:
        void init_scratchpad() {
            auto scratchpad = this->scratchpad_registry().registrar();
            dim_t scratchpad_size;
            const memory_desc_wrapper src_d(src_md());
            const memory_desc_wrapper weights_d(weights_md());
            auto broadcast_strategy
                    = get_rhs_arg_broadcasting_strategy(*weights_md(), src_d);
            // Assign `nthr_` here since the amount needed maybe reduced.
            nthr_ = dnnl_get_max_threads();
            // Scratchpad is needed to correctly reduce calculated diff_weights
            // in cases where broadcast is used.
            //
            // example: if data tensor size is NxCxW and weight tensor is 1xCx1,
            // diff_weight tensor would also be of size 1xCx1 and thus each value
            // along C axis would equal: results summed up over N and W for given C.
            //
            // In current implementation reduction is 2 step:
            // results are first copied to buffer and reduced, result is then
            // stored in group buffer. Values in group buffer are then reduced
            // to obtain final value.
            if (broadcast_strategy == broadcasting_strategy_t::no_broadcast) {
                return;
            } else if (broadcast_strategy == broadcasting_strategy_t::scalar) {
                int work_amount = static_cast<int>(src_d.nelems());
                nthr_ = nstl::min(nthr_, work_amount);
                scratchpad_size = prelu::get_scalar_scratchpad_offset(
                        nthr_, nthr_, src_d.nelems());
            } else {
                dim_t group_size, buf_size;
                nthr_ = nstl::min(nthr_, static_cast<int>(weights_d.nelems()));
                dim_t work_amount = src_d.nelems() / weights_d.nelems();
                prelu::set_reduction_buffers(work_amount, group_size, buf_size);
                scratchpad_size = nthr_ * (group_size + buf_size);
            }
            scratchpad.book(memory_tracking::names::key_prelu_reduction,
                    scratchpad_size, types::data_type_size(dnnl_f32));
        }
    };

    ref_prelu_bwd_t(const pd_t *apd) : primitive_t(apd) {}

    status_t execute(const exec_ctx_t &ctx) const override {
        return execute_backward(ctx);
    }

private:
    status_t execute_backward(const exec_ctx_t &ctx) const;
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }

    float ker(const byte *src, const byte *weights, const byte *diff_dst,
            byte *diff_src, dim_t data_off, dim_t weight_off) const;
    void calculate_scalar(const byte *src, const byte *weights,
            byte *diff_weights, const byte *diff_dst, byte *diff_src,
            float *scratchpad_buf) const;
    void calculate_no_broadcast(const byte *src, const byte *weights,
            byte *diff_weights, const byte *diff_dst, byte *diff_src,
            float *scratchpad_buf) const;
    void calculate_shared_axes(const byte *src, const byte *weights,
            byte *diff_weights, const byte *diff_dst, byte *diff_src,
            float *scratchpad_buf) const;
};

} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif

// vim: et ts=4 sw=4 cindent cino+=l0,\:4,N-s
