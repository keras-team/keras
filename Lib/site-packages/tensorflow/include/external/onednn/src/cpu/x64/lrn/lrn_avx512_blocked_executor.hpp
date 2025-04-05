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

#ifndef CPU_X64_LRN_JIT_LRN_AVX512_BLOCKED_EXECUTOR_HPP
#define CPU_X64_LRN_JIT_LRN_AVX512_BLOCKED_EXECUTOR_HPP

#include "cpu/x64/lrn/jit_avx512_common_lrn_bwd_blocked.hpp"
#include "cpu/x64/lrn/jit_avx512_common_lrn_fwd_blocked.hpp"
#include "cpu/x64/lrn/lrn_executor.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {
namespace lrn {

template <::dnnl::impl::data_type_t d_type, typename PD_T>
class lrn_avx512_blocked_executor_fwd_t : public i_lrn_executor_t {
public:
    lrn_avx512_blocked_executor_fwd_t(const PD_T *pd)
        : ker_(nullptr)
        , ker_first_(nullptr)
        , ker_last_(nullptr)
        , N_(pd->MB())
        , C_(pd->C())
        , H_(pd->H())
        , W_(pd->W())
        , use_h_parallelism_(H_ > 28 ? 1 : 0) {

        const int local_size = pd->desc()->local_size;
        const float alpha = pd->desc()->lrn_alpha / local_size;
        const float beta = pd->desc()->lrn_beta;
        const auto pk = pd->desc()->prop_kind;
        const float k = pd->desc()->lrn_k;

        if (C_ / vsize_ == 1) {
            ker_ = utils::make_unique<
                    lrn::jit_avx512_common_lrn_kernel_fwd_blocked_t<d_type>>(
                    lrn::nChw16c_across_t(H_, W_, lrn::across_version::Single),
                    pk, use_h_parallelism_, alpha, beta, k, local_size);
        } else {
            ker_ = utils::make_unique<
                    lrn::jit_avx512_common_lrn_kernel_fwd_blocked_t<d_type>>(
                    lrn::nChw16c_across_t(H_, W_, lrn::across_version::Middle),
                    pk, use_h_parallelism_, alpha, beta, k, local_size);
            ker_first_ = utils::make_unique<
                    lrn::jit_avx512_common_lrn_kernel_fwd_blocked_t<d_type>>(
                    lrn::nChw16c_across_t(H_, W_, lrn::across_version::First),
                    pk, use_h_parallelism_, alpha, beta, k, local_size);
            ker_last_ = utils::make_unique<
                    lrn::jit_avx512_common_lrn_kernel_fwd_blocked_t<d_type>>(
                    lrn::nChw16c_across_t(H_, W_, lrn::across_version::Last),
                    pk, use_h_parallelism_, alpha, beta, k, local_size);
        }
    }

    using data_t = typename prec_traits<d_type>::type;

    status_t create_kernel() override {
        CHECK(ker_->create_kernel());
        if (ker_first_) CHECK(ker_first_->create_kernel());
        if (ker_last_) CHECK(ker_last_->create_kernel());
        return status::success;
    }

    status_t execute(const exec_ctx_t &ctx) const override {
        status_t status = status::success;
        const auto src = CTX_IN_MEM(const data_t *, DNNL_ARG_SRC);
        const auto dst = CTX_OUT_CLEAN_MEM(data_t *, DNNL_ARG_DST, status);
        CHECK(status);
        const auto ws = CTX_OUT_CLEAN_MEM(data_t *, DNNL_ARG_WORKSPACE, status);
        CHECK(status);

        const auto ker = ker_.get();
        const auto ker_first = ker_first_.get();
        const auto ker_last = ker_last_.get();

        parallel(0, [&](const int ithr, const int nthr) {
            size_t start {0}, end {0};
            const int C16 = C_ / vsize_;
            const size_t work_amount
                    = use_h_parallelism_ ? N_ * C16 * H_ : N_ * C16;

            balance211(work_amount, nthr, ithr, start, end);
            if (use_h_parallelism_) {
                int n {0}, c16 {0}, h {0};
                nd_iterator_init(start, n, N_, c16, C16, h, H_);
                for (size_t iwork = start; iwork < end; ++iwork) {
                    const auto offset = n * C_ * H_ * W_
                            + c16 * H_ * W_ * vsize_ + h * W_ * vsize_;
                    const auto ws_offset0 = n * C_ * H_ * 2 * W_
                            + c16 * H_ * 2 * W_ * vsize_ + h * 2 * W_ * vsize_;
                    const auto ws_offset1 = ws_offset0 + W_ * vsize_;

                    typename lrn::jit_avx512_common_lrn_kernel_fwd_t<
                            d_type>::jit_args_fwd_t args;
                    args.src = &src[offset];
                    args.dst = &dst[offset];
                    args.ws0 = ws ? &ws[ws_offset0] : nullptr;
                    args.ws1 = ws ? &ws[ws_offset1] : nullptr;

                    if (C16 == 1)
                        (*ker)(&args);
                    else if (c16 == 0)
                        (*ker_first)(&args);
                    else if (c16 == C16 - 1)
                        (*ker_last)(&args);
                    else
                        (*ker)(&args);
                    nd_iterator_step(n, N_, c16, C16, h, H_);
                }
            } else {
                int n {0}, c16 {0};
                nd_iterator_init(start, n, N_, c16, C16);
                for (size_t iwork = start; iwork < end; ++iwork) {
                    const auto offset
                            = n * C_ * H_ * W_ + c16 * H_ * W_ * vsize_;
                    const auto ws_offset0
                            = n * C_ * H_ * 2 * W_ + c16 * H_ * 2 * W_ * vsize_;
                    const auto ws_offset1 = ws_offset0 + H_ * W_ * vsize_;

                    typename lrn::jit_avx512_common_lrn_kernel_fwd_t<
                            d_type>::jit_args_fwd_t args;
                    args.src = &src[offset];
                    args.dst = &dst[offset];
                    args.ws0 = ws ? &ws[ws_offset0] : nullptr;
                    args.ws1 = ws ? &ws[ws_offset1] : nullptr;

                    if (C16 == 1)
                        (*ker)(&args);
                    else if (c16 == 0)
                        (*ker_first)(&args);
                    else if (c16 == C16 - 1)
                        (*ker_last)(&args);
                    else
                        (*ker)(&args);

                    nd_iterator_step(n, N_, c16, C16);
                }
            }
        });

        return status::success;
    }

private:
    std::unique_ptr<lrn::jit_avx512_common_lrn_kernel_fwd_blocked_t<d_type>>
            ker_, ker_first_, ker_last_;
    static constexpr int vsize_ = 16;
    const int N_;
    const int C_;
    const int H_;
    const int W_;
    const int use_h_parallelism_;
};

template <::dnnl::impl::data_type_t d_type, typename PD_T>
class lrn_avx512_blocked_executor_bwd_t : public i_lrn_executor_t {
public:
    lrn_avx512_blocked_executor_bwd_t(const PD_T *pd)
        : ker_(nullptr)
        , ker_first_(nullptr)
        , ker_last_(nullptr)
        , N_(pd->MB())
        , C_(pd->C())
        , H_(pd->H())
        , W_(pd->W())
        , use_h_parallelism_(H_ > 28 ? 1 : 0) {

        const int local_size = pd->desc()->local_size;
        const float alpha = pd->desc()->lrn_alpha / local_size;
        const float beta = pd->desc()->lrn_beta;

        if (C_ / vsize_ == 1) {
            ker_ = utils::make_unique<
                    lrn::jit_avx512_common_lrn_kernel_bwd_blocked_t<d_type>>(
                    lrn::nChw16c_across_t(H_, W_, lrn::across_version::Single),
                    alpha, beta, local_size, use_h_parallelism_);
        } else {
            ker_ = utils::make_unique<
                    lrn::jit_avx512_common_lrn_kernel_bwd_blocked_t<d_type>>(
                    lrn::nChw16c_across_t(H_, W_, lrn::across_version::Middle),
                    alpha, beta, local_size, use_h_parallelism_);
            ker_first_ = utils::make_unique<
                    lrn::jit_avx512_common_lrn_kernel_bwd_blocked_t<d_type>>(
                    lrn::nChw16c_across_t(H_, W_, lrn::across_version::First),
                    alpha, beta, local_size, use_h_parallelism_);
            ker_last_ = utils::make_unique<
                    lrn::jit_avx512_common_lrn_kernel_bwd_blocked_t<d_type>>(
                    lrn::nChw16c_across_t(H_, W_, lrn::across_version::Last),
                    alpha, beta, local_size, use_h_parallelism_);
        }
    }

    using data_t = typename prec_traits<d_type>::type;

    status_t create_kernel() override {
        CHECK(ker_->create_kernel());
        if (ker_first_) CHECK(ker_first_->create_kernel());
        if (ker_last_) CHECK(ker_last_->create_kernel());
        return status::success;
    }

    status_t execute(const exec_ctx_t &ctx) const override {
        status_t status = status::success;
        const auto src = CTX_IN_MEM(const data_t *, DNNL_ARG_SRC);
        const auto diff_dst = CTX_IN_MEM(const data_t *, DNNL_ARG_DIFF_DST);
        const auto ws = CTX_IN_MEM(const data_t *, DNNL_ARG_WORKSPACE);
        const auto diff_src
                = CTX_OUT_CLEAN_MEM(data_t *, DNNL_ARG_DIFF_SRC, status);
        CHECK(status);

        const auto ker = ker_.get();
        const auto ker_first = ker_first_.get();
        const auto ker_last = ker_last_.get();

        parallel(0, [&](const int ithr, const int nthr) {
            size_t start {0}, end {0};
            const int C16 = C_ / vsize_;
            const size_t work_amount
                    = use_h_parallelism_ ? N_ * C16 * H_ : N_ * C16;

            balance211(work_amount, nthr, ithr, start, end);
            if (use_h_parallelism_) {
                int n {0}, c16 {0}, h {0};
                nd_iterator_init(start, n, N_, h, H_, c16, C16);
                for (size_t iwork = start; iwork < end; ++iwork) {
                    const auto offset = n * C_ * H_ * W_
                            + c16 * H_ * W_ * vsize_ + h * W_ * vsize_;
                    const auto ws_offset0 = n * C_ * H_ * 2 * W_
                            + c16 * H_ * 2 * W_ * vsize_ + h * 2 * W_ * vsize_;
                    const auto ws_offset1 = ws_offset0 + W_ * vsize_;

                    typename lrn::jit_avx512_common_lrn_kernel_bwd_blocked_t<
                            d_type>::jit_args_bwd_t args;
                    args.src = &src[offset];
                    args.diff_dst = &diff_dst[offset];
                    args.ws0 = ws ? &ws[ws_offset0] : nullptr;
                    args.ws1 = ws ? &ws[ws_offset1] : nullptr;
                    args.diff_src = &diff_src[offset];

                    if (C16 == 1)
                        (*ker)(&args);
                    else if (c16 == 0)
                        (*ker_first)(&args);
                    else if (c16 == C16 - 1)
                        (*ker_last)(&args);
                    else
                        (*ker)(&args);
                    nd_iterator_step(n, N_, h, H_, c16, C16);
                }
            } else {
                int n {0}, c16 {0};
                nd_iterator_init(start, n, N_, c16, C16);
                for (size_t iwork = start; iwork < end; ++iwork) {
                    const auto offset
                            = n * C_ * H_ * W_ + c16 * H_ * W_ * vsize_;
                    const auto ws_offset0
                            = n * C_ * H_ * 2 * W_ + c16 * H_ * 2 * W_ * vsize_;
                    const auto ws_offset1 = ws_offset0 + H_ * W_ * vsize_;

                    typename lrn::jit_avx512_common_lrn_kernel_bwd_blocked_t<
                            d_type>::jit_args_bwd_t args;
                    args.src = &src[offset];
                    args.diff_dst = &diff_dst[offset];
                    args.ws0 = ws ? &ws[ws_offset0] : nullptr;
                    args.ws1 = ws ? &ws[ws_offset1] : nullptr;
                    args.diff_src = &diff_src[offset];

                    if (C16 == 1)
                        (*ker)(&args);
                    else if (c16 == 0)
                        (*ker_first)(&args);
                    else if (c16 == C16 - 1)
                        (*ker_last)(&args);
                    else
                        (*ker)(&args);

                    nd_iterator_step(n, N_, c16, C16);
                }
            }
        });

        return status::success;
    }

private:
    std::unique_ptr<lrn::jit_avx512_common_lrn_kernel_bwd_blocked_t<d_type>>
            ker_, ker_first_, ker_last_;
    static constexpr int vsize_ = 16;
    const int N_;
    const int C_;
    const int H_;
    const int W_;
    const int use_h_parallelism_;
};

} // namespace lrn
} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif
