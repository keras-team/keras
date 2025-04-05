/*******************************************************************************
* Copyright 2020-2023 Intel Corporation
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

#ifndef CPU_X64_LRN_JIT_LRN_AVX512_NHWC_EXECUTOR_HPP
#define CPU_X64_LRN_JIT_LRN_AVX512_NHWC_EXECUTOR_HPP

#include "cpu/x64/lrn/jit_avx512_common_lrn_bwd_nhwc.hpp"
#include "cpu/x64/lrn/jit_avx512_common_lrn_fwd_nhwc.hpp"
#include "cpu/x64/lrn/lrn_executor.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {
namespace lrn {

template <::dnnl::impl::data_type_t d_type, typename PD_T>
class lrn_avx512_nhwc_executor_fwd_t : public i_lrn_executor_t {
public:
    lrn_avx512_nhwc_executor_fwd_t(const PD_T *pd)
        : ker_(utils::make_unique<
                lrn::jit_avx512_common_lrn_kernel_fwd_nhwc_t<d_type>>(pd->C(),
                pd->desc()->prop_kind,
                pd->desc()->lrn_alpha / pd->desc()->local_size,
                pd->desc()->lrn_beta, pd->desc()->lrn_k,
                pd->desc()->local_size))
        , N_(pd->MB())
        , C_(pd->C())
        , H_(pd->H())
        , W_(pd->W()) {}

    using data_t = typename prec_traits<d_type>::type;

    status_t create_kernel() override { return ker_->create_kernel(); }

    status_t execute(const exec_ctx_t &ctx) const override {
        status_t status = status::success;
        const auto src = CTX_IN_MEM(const data_t *, DNNL_ARG_SRC);
        const auto dst = CTX_OUT_CLEAN_MEM(data_t *, DNNL_ARG_DST, status);
        CHECK(status);
        const auto ws = CTX_OUT_CLEAN_MEM(data_t *, DNNL_ARG_WORKSPACE, status);
        CHECK(status);

        const auto ker = ker_.get();
        parallel_nd(N_, H_ * W_, [&](dim_t n, dim_t pixel_id) {
            typename lrn::jit_avx512_common_lrn_kernel_fwd_t<
                    d_type>::jit_args_fwd_t args;
            const auto offset = n * C_ * H_ * W_ + pixel_id * C_;
            const auto ws_offset0 = offset * 2;
            const auto ws_offset1 = ws_offset0 + C_;

            args.src = &src[offset];
            args.dst = &dst[offset];
            args.ws0 = ws ? &ws[ws_offset0] : nullptr;
            args.ws1 = ws ? &ws[ws_offset1] : nullptr;

            (*ker)(&args);
        });

        return status::success;
    }

    virtual ~lrn_avx512_nhwc_executor_fwd_t() = default;

private:
    std::unique_ptr<jit_avx512_common_lrn_kernel_fwd_nhwc_t<d_type>> ker_;
    const dim_t N_;
    const dim_t C_;
    const dim_t H_;
    const dim_t W_;
};
template <::dnnl::impl::data_type_t d_type, typename PD_T>
class lrn_avx512_nhwc_executor_bwd_t : public i_lrn_executor_t {
public:
    lrn_avx512_nhwc_executor_bwd_t(const PD_T *pd)
        : ker_ {utils::make_unique<
                lrn::jit_avx512_common_lrn_kernel_bwd_nhwc_t<d_type>>(pd->C(),
                pd->desc()->lrn_alpha / pd->desc()->local_size,
                pd->desc()->lrn_beta, pd->desc()->local_size)}
        , N_(pd->MB())
        , C_(pd->C())
        , H_(pd->H())
        , W_(pd->W()) {}
    using data_t = typename prec_traits<d_type>::type;

    status_t create_kernel() override { return ker_->create_kernel(); }

    status_t execute(const exec_ctx_t &ctx) const override {
        status_t status = status::success;
        auto src = CTX_IN_MEM(data_t *, DNNL_ARG_SRC);
        auto diff_src = CTX_OUT_CLEAN_MEM(data_t *, DNNL_ARG_DIFF_SRC, status);
        CHECK(status);
        auto diff_dst = CTX_IN_MEM(data_t *, DNNL_ARG_DIFF_DST);
        auto ws = CTX_IN_MEM(data_t *, DNNL_ARG_WORKSPACE);

        const auto ker = ker_.get();
        parallel_nd(N_, H_ * W_, [&](dim_t n, dim_t pixel_id) {
            typename lrn::jit_avx512_common_lrn_kernel_bwd_nhwc_t<
                    d_type>::jit_args_bwd_t args;
            const auto offset = n * C_ * H_ * W_ + pixel_id * C_;
            const auto ws_offset0 = offset * 2;
            const auto ws_offset1 = ws_offset0 + C_;

            args.src = &src[offset];
            args.diff_dst = &diff_dst[offset];
            args.ws0 = &ws[ws_offset0];
            args.ws1 = &ws[ws_offset1];
            args.diff_src = &diff_src[offset];

            (*ker)(&args);
        });

        return status::success;
    }

    virtual ~lrn_avx512_nhwc_executor_bwd_t() = default;

private:
    std::unique_ptr<jit_avx512_common_lrn_kernel_bwd_nhwc_t<d_type>> ker_;
    const dim_t N_;
    const dim_t C_;
    const dim_t H_;
    const dim_t W_;
};

} // namespace lrn
} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif
