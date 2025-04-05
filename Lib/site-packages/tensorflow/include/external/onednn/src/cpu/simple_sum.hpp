/*******************************************************************************
* Copyright 2017-2024 Intel Corporation
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

#ifndef CPU_SIMPLE_SUM_HPP
#define CPU_SIMPLE_SUM_HPP

#include "common/dnnl_thread.hpp"
#include "common/primitive.hpp"
#include "common/type_helpers.hpp"

#include "cpu/cpu_sum_pd.hpp"
#include "cpu/platform.hpp"

namespace dnnl {
namespace impl {
namespace cpu {

struct sum_xf16_params_t {
    dim_t ws_cvt_elements_per_thread_;
    dim_t ws_acc_elements_per_thread_;
    dim_t ws_elements_per_thread_;
    dim_t acc_loop_step_;
};

template <data_type_t src_data_type, data_type_t dst_data_type = src_data_type>
struct simple_sum_t : public primitive_t {
    struct pd_t : public cpu_sum_pd_t {
        using cpu_sum_pd_t::cpu_sum_pd_t;

        DECLARE_SUM_PD_T("simple:any", simple_sum_t);

        status_t init(engine_t *engine) {
            const int n = n_inputs();

            VDISPATCH_SUM(platform::has_data_type_support(src_data_type),
                    VERBOSE_UNSUPPORTED_DT);
            VDISPATCH_SUM(platform::has_data_type_support(dst_data_type),
                    VERBOSE_UNSUPPORTED_DT);
            VDISPATCH_SUM(cpu_sum_pd_t::init(engine) == status::success,
                    VERBOSE_BAD_ENGINE_KIND);
            VDISPATCH_SUM(n <= max_num_arrs,
                    "number of inputs exceed max number of arrays");

            const memory_desc_wrapper o_d(dst_md());
            VDISPATCH_SUM(o_d.data_type() == dst_data_type,
                    VERBOSE_INCONSISTENT_DT, "o_d", "dst");
            VDISPATCH_SUM(o_d.is_dense(), VERBOSE_UNSUPPORTED_SPARSE_CFG);

            for (int i = 0; i < n; ++i) {
                const memory_desc_wrapper i_d(src_md(i));
                VDISPATCH_SUM(
                        utils::everyone_is(src_data_type, i_d.data_type()),
                        VERBOSE_UNSUPPORTED_DT);
                VDISPATCH_SUM(o_d.similar_to(i_d, true, false, 0),
                        VERBOSE_INCONSISTENT_MDS, "o_d", "i_d");
                VDISPATCH_SUM(i_d.is_dense(), VERBOSE_UNSUPPORTED_SPARSE_CFG);
            }
            nthr_ = dnnl_get_max_threads();
            compute_blocking();
            init_scratchpad();
            return status::success;
        }
        int nthr_ = 1;
        sum_xf16_params_t xf16_params_;
        dim_t block_size_ = 0, nelems_ = 0, blocks_number_ = 0, tail_ = 0;

    private:
        void compute_blocking() {
            const int block_size_bytes
                    = utils::one_of(
                              src_data_type, data_type::bf16, data_type::f16)
                    ? 16 * platform::get_cache_line_size()
                    : platform::get_per_core_cache_size(1) / 2;
            block_size_ = block_size_bytes / (int)sizeof(src_data_type);
            const memory_desc_wrapper o_d(dst_md());
            nelems_ = o_d.nelems();
            blocks_number_ = nelems_ / block_size_;
            tail_ = nelems_ % block_size_;
        }

        void init_scratchpad() {
            if (utils::one_of(src_data_type, data_type::bf16, data_type::f16)) {
                const bool is_dst_xf16 = utils::one_of(
                        dst_data_type, data_type::bf16, data_type::f16);
                xf16_params_.ws_cvt_elements_per_thread_
                        = platform::get_cache_line_size()
                        / (int)sizeof(acc_data_t);

                xf16_params_.ws_acc_elements_per_thread_ = is_dst_xf16
                        ? xf16_params_.ws_cvt_elements_per_thread_
                        : 0;

                xf16_params_.acc_loop_step_ = is_dst_xf16
                        ? xf16_params_.ws_cvt_elements_per_thread_
                        : 1;

                xf16_params_.ws_elements_per_thread_
                        = xf16_params_.ws_cvt_elements_per_thread_
                        + xf16_params_.ws_acc_elements_per_thread_;
                const dim_t cvt_buf_sz
                        = xf16_params_.ws_elements_per_thread_ * nthr_;
                auto scratchpad = scratchpad_registry().registrar();
                scratchpad.template book<acc_data_t>(
                        memory_tracking::names::key_sum_srcs_cvt, cvt_buf_sz);
            }
        }
    };

    simple_sum_t(const pd_t *apd) : primitive_t(apd) {}

    status_t execute(const exec_ctx_t &ctx) const override;

    enum { max_num_arrs = 16 };
    typedef typename prec_traits<src_data_type>::type src_data_t;
    typedef typename prec_traits<dst_data_type>::type dst_data_t;
    typedef typename prec_traits<data_type::f32>::type acc_data_t;

private:
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }
};

} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif

// vim: et ts=4 sw=4 cindent cino+=l0,\:4,N-s
