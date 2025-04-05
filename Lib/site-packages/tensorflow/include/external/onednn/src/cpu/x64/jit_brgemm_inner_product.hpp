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

#ifndef CPU_X64_BRGEMM_INNER_PRODUCT_HPP
#define CPU_X64_BRGEMM_INNER_PRODUCT_HPP

#include "common/c_types_map.hpp"
#include "common/dnnl_thread.hpp"
#include "common/memory_tracking.hpp"
#include "common/primitive.hpp"
#include "common/utils.hpp"

#include "cpu/cpu_inner_product_pd.hpp"
#include "cpu/scale_utils.hpp"

#include "cpu/x64/amx_tile_configure.hpp"
#include "cpu/x64/brgemm/brgemm.hpp"
#include "cpu/x64/brgemm/brgemm_containers.hpp"
#include "cpu/x64/cpu_barrier.hpp"
#include "cpu/x64/cpu_reducer.hpp"
#include "cpu/x64/jit_avx512_core_scale_precompute.hpp"
#include "cpu/x64/jit_brgemm_inner_product_utils.hpp"
#include "cpu/x64/jit_brgemm_post_ops.hpp"
#include "cpu/x64/jit_brgemm_transpose_utils.hpp"
#include "cpu/x64/jit_transpose_utils.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {

template <cpu_isa_t isa>
struct brgemm_inner_product_fwd_t : public primitive_t {
    struct pd_t : public cpu_inner_product_fwd_pd_t {
        pd_t(const inner_product_desc_t *adesc, const primitive_attr_t *attr,
                const typename pd_t::base_class *hint_fwd_pd)
            : cpu_inner_product_fwd_pd_t(adesc, attr, hint_fwd_pd) {}

        DECLARE_COMMON_PD_T(JIT_IMPL_NAME_HELPER("brgemm:", isa, ""),
                brgemm_inner_product_fwd_t);

        status_t init(engine_t *engine) {
            using namespace utils;
            using namespace data_type;

            auto src_dt = invariant_src_md()->data_type;
            auto dst_dt = invariant_dst_md()->data_type;
            auto wei_dt = invariant_wei_md()->data_type;
            const bool is_int8 = one_of(src_dt, u8, s8);

            using skip_mask_t = primitive_attr_t::skip_mask_t;
            auto skip_mask = skip_mask_t::post_ops | skip_mask_t::sum_dt
                    | skip_mask_t::fpmath_mode;
            if (is_int8) skip_mask |= skip_mask_t::scales_runtime;

            if (!mayiuse(isa)) return status::unimplemented;

            VDISPATCH_INNER_PRODUCT(is_fwd(), VERBOSE_BAD_PROPKIND);
            VDISPATCH_INNER_PRODUCT(
                    expect_data_types(src_dt, wei_dt, data_type::undef, dst_dt,
                            data_type::undef),
                    VERBOSE_UNSUPPORTED_DT);
            VDISPATCH_INNER_PRODUCT(
                    IMPLICATION(with_bias() && is_int8,
                            one_of(bias_md_.data_type, f32, bf16, s32, s8, u8)),
                    VERBOSE_UNSUPPORTED_DT);
            VDISPATCH_INNER_PRODUCT(
                    IMPLICATION(with_bias() && !is_int8,
                            one_of(bias_md_.data_type, f32, src_dt)),
                    VERBOSE_UNSUPPORTED_DT);
            VDISPATCH_INNER_PRODUCT(
                    attr()->has_default_values(skip_mask, dst_dt),
                    VERBOSE_UNSUPPORTED_ATTR);
            VDISPATCH_INNER_PRODUCT(
                    attr()->post_ops_.check_sum_consistency(dst_dt, is_int8),
                    VERBOSE_UNSUPPORTED_POSTOP);
            VDISPATCH_INNER_PRODUCT(
                    !has_zero_dim_memory(), VERBOSE_EMPTY_TENSOR, "");
            VDISPATCH_INNER_PRODUCT(
                    arg_scales_ok(), VERBOSE_UNSUPPORTED_SCALES_CFG);

            CHECK(jbgp_.init_conf(isa, *desc(), src_md_, weights_md_, dst_md_,
                    bias_md_, attr_, dnnl_get_max_threads()));

            bool are_post_ops_applicable = one_of(true, jbgp_.with_sum,
                    jbgp_.with_bias, jbgp_.with_scales, jbgp_.with_eltwise,
                    jbgp_.with_binary, jbgp_.acc_dt != jbgp_.dst_dt,
                    jbgp_.req_s8s8_compensation, jbgp_.with_dst_scales);

            const float alpha = 1.0;
            const float beta = 1.0;
            const float beta_init = 0.0;

            for_(int i_bs = 0; i_bs < 2; i_bs++)
            for_(int i_init = 0; i_init < 2; i_init++)
            for_(int i_M = 0; i_M < 2; i_M++)
            for_(int i_N = 0; i_N < 2; i_N++)
            for (int i_K = 0; i_K < 2; i_K++) {
                auto vbeta = (i_init) ? beta_init : beta;
                auto vM = (i_M) ? jbgp_.M_tail : jbgp_.M;
                auto vN = (i_N) ? jbgp_.N_tail : jbgp_.N;
                auto vK = (i_K) ? jbgp_.K_tail : jbgp_.K;
                int bs = get_brg_batchsize(i_bs, i_K);
                int idx = get_brg_kernel_idx(i_bs, i_init, i_M, i_N, i_K, bs);
                if (idx < 0) continue;
                brgemm_desc_t &brg = brg_descs_[idx];
                CHECK(brgemm_desc_init(&brg, isa, jbgp_.brg_type, jbgp_.src_dt,
                        jbgp_.wei_dt, false, false, brgemm_row_major, alpha,
                        vbeta, jbgp_.LDA, jbgp_.LDB, jbgp_.LDC, vM, vN, vK));

                CHECK(brgemm_desc_set_postops(
                        &brg, attr(), &dst_md_, jbgp_.LDD, jbgp_.bia_dt));

                brgemm_attr_t brgattr;
                if (jbgp_.is_amx) {
                    brgattr.max_bs = bs;
                    brgattr.wary_tail_read = false;
                    brgattr.hint_expected_A_size = jbgp_.mb * jbgp_.ic;
                    brgattr.hint_expected_B_size = jbgp_.oc * jbgp_.ic;
                    brgattr.hint_expected_C_size = jbgp_.mb * jbgp_.oc;
                    brgattr.hint_innermost_loop = brgemm_innermost_undef;
                    brgattr.use_uker = jbgp_.use_uker;
                    brgattr.use_interleave_stores = jbgp_.use_interleave_stores;
                    brgattr.hint_prefetching = jbgp_.hint_prefetching;
                    brgattr.fpmath_mode = attr()->fpmath_.mode_;
                }
                if (are_post_ops_applicable && jbgp_.nthr_ic_b > 1) {
                    brgattr.generate_skip_accumulation = true;
                }

                CHECK(brgemm_desc_set_attr(&brg, brgattr));

                if (jbgp_.is_amx)
                    jbgp_.amx_buf_size_per_thread
                            = nstl::max(brg.get_wsp_buffer_size(),
                                    jbgp_.amx_buf_size_per_thread);
            }

            auto scratchpad = scratchpad_registry().registrar();
            jbgp_.init_scratchpad(scratchpad);
            if (jbgp_.with_scales)
                book_precomputed_scales(scratchpad, attr()->scales_, OC());

            return status::success;
        }

        bool arg_scales_ok() const {
            std::vector<int> supported_args
                    = {DNNL_ARG_SRC, DNNL_ARG_WEIGHTS, DNNL_ARG_DST};
            return attr_scales_ok(supported_args);
        }

        int get_brg_kernel_idx(bool is_bs_tail, bool do_initialization,
                bool is_M_tail, bool is_N_tail, bool is_K_tail, int bs) const {
            auto vM = (is_M_tail) ? jbgp_.M_tail : jbgp_.M;
            auto vN = (is_N_tail) ? jbgp_.N_tail : jbgp_.N;
            auto vK = (is_K_tail) ? jbgp_.K_tail : jbgp_.K;

            if (vM == 0 || vN == 0 || vK == 0 || bs == 0 || jbgp_.LDA < vK
                    || jbgp_.LDB < vN || jbgp_.LDC < vN)
                return -1;
            return brgemm_inner_product_utils::get_brg_kernel_index(is_bs_tail,
                    do_initialization, is_M_tail, is_N_tail, is_K_tail);
        }

        int get_brg_batchsize(bool is_bs_tail, bool is_K_tail) const {
            auto adj_ic = jbgp_.use_buffer_a
                    ? utils::rnd_up(jbgp_.ic, jbgp_.ic_block)
                    : jbgp_.ic;
            auto bs = (is_K_tail)
                    ? 1
                    : ((is_bs_tail) ? (adj_ic / jbgp_.K) % jbgp_.gemm_batch_size
                                    : jbgp_.gemm_batch_size);
            return bs;
        }

        brgemm_desc_t
                brg_descs_[brgemm_inner_product_utils::max_num_brg_kernels_ip];
        brgemm_inner_product_utils::jit_brgemm_ip_fwd_conf_t jbgp_;
    };

    brgemm_inner_product_fwd_t(const pd_t *apd) : primitive_t(apd) {}

    status_t init(engine_t *engine) override {
        for_(int i_bs = 0; i_bs < 2; i_bs++)
        for_(int i_M = 0; i_M < 2; i_M++)
        for_(int i_N = 0; i_N < 2; i_N++)
        for_(int i_K = 0; i_K < 2; i_K++)
        for (int i_init = 0; i_init < 2; i_init++) {
            int bs = pd()->get_brg_batchsize(i_bs, i_K);
            int idx = pd()->get_brg_kernel_idx(i_bs, i_init, i_M, i_N, i_K, bs);
            if (idx < 0) continue;

            brgemm_kernel_t *ker = nullptr;
            CHECK(brgemm_kernel_create(&ker, pd()->brg_descs_[idx]));
            CHECK(safe_ptr_assign(brg_kernels_[idx], ker));
            if (pd()->jbgp_.is_amx)
                brgemm_palettes_.insert(idx, pd()->brg_descs_[idx]);
        }
        if (pd()->jbgp_.use_buffer_a)
            CHECK(create_brgemm_copy_to_coarse(copy_src_kernel_, &pd()->jbgp_));
        if (pd()->jbgp_.nthr_ic_b > 1) {
            CHECK(safe_ptr_assign(
                    acc_ker_, new cpu_accumulator_1d_t<data_type::f32>()));
            CHECK(acc_ker_->create_kernel());
        }

        // JIT to precompute scales
        const bool is_jit_supported = mayiuse(avx512_core);
        const auto attr = pd()->attr();
        if (is_jit_supported && pd()->OC() > 1 && req_copy_scales(attr)) {
            const auto &attr_scales = attr->scales_;
            int wei_scale_mask = attr_scales.get(DNNL_ARG_WEIGHTS).mask_;
            if (wei_scale_mask != 0) {
                CHECK(safe_ptr_assign(jit_scale_precompute_,
                        new jit_avx512_core_scale_precompute_t(attr)));
                CHECK(jit_scale_precompute_->create_kernel());
            }
        }

        return status::success;
    }

    status_t execute(const exec_ctx_t &ctx) const override {
        return execute_forward(ctx);
    }

private:
    status_t execute_forward(const exec_ctx_t &ctx) const;
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }

    std::unique_ptr<brgemm_kernel_t>
            brg_kernels_[brgemm_inner_product_utils::max_num_brg_kernels_ip];
    std::unique_ptr<jit_brgemm_copy_to_coarse_t> copy_src_kernel_;
    std::unique_ptr<cpu_accumulator_1d_t<data_type::f32>> acc_ker_;
    std::unique_ptr<jit_avx512_core_scale_precompute_t> jit_scale_precompute_;
    brgemm_containers::brgemm_palette_container_t brgemm_palettes_ {
            brgemm_inner_product_utils::max_num_brg_kernels_ip};
};

template <cpu_isa_t isa>
struct brgemm_inner_product_bwd_data_t : public primitive_t {
    struct pd_t : public cpu_inner_product_bwd_data_pd_t {
        pd_t(const inner_product_desc_t *adesc, const primitive_attr_t *attr,
                const inner_product_fwd_pd_t *hint_fwd_pd)
            : cpu_inner_product_bwd_data_pd_t(adesc, attr, hint_fwd_pd) {}

        DECLARE_COMMON_PD_T(JIT_IMPL_NAME_HELPER("brgemm_bwd_d:", isa, ""),
                brgemm_inner_product_bwd_data_t);

        status_t init(engine_t *engine) {
            using skip_mask_t = primitive_attr_t::skip_mask_t;

            auto diff_src_dt = invariant_src_md()->data_type;
            auto diff_dst_dt = invariant_dst_md()->data_type;
            auto wei_dt = invariant_wei_md()->data_type;

            if (!mayiuse(isa)) return status::unimplemented;
            VDISPATCH_INNER_PRODUCT(
                    desc()->prop_kind == prop_kind::backward_data,
                    VERBOSE_BAD_PROPKIND);
            VDISPATCH_INNER_PRODUCT(
                    !has_zero_dim_memory(), VERBOSE_EMPTY_TENSOR, "");
            VDISPATCH_INNER_PRODUCT(utils::one_of(diff_dst_dt, data_type::f32,
                                            data_type::bf16, data_type::f16),
                    VERBOSE_UNSUPPORTED_DT);
            VDISPATCH_INNER_PRODUCT(wei_dt == diff_dst_dt,
                    VERBOSE_INCONSISTENT_DT, "weights", "diff_dst");
            VDISPATCH_INNER_PRODUCT(
                    utils::one_of(diff_src_dt, data_type::f32, diff_dst_dt),
                    VERBOSE_UNSUPPORTED_DT);
            VDISPATCH_INNER_PRODUCT(
                    attr()->has_default_values(skip_mask_t::fpmath_mode),
                    VERBOSE_UNSUPPORTED_ATTR);

            memory_desc_t dummy_bias_md;
            CHECK(jbgp_.init_conf(isa, *desc(), diff_src_md_, weights_md_,
                    diff_dst_md_, dummy_bias_md, attr_,
                    dnnl_get_max_threads()));

            const float alpha = 1.0;
            const float beta = 1.0;
            const float beta_init = 0.0;
            const auto dt_b = isa == avx512_core_fp16 && jbgp_.use_buffer_b
                    ? data_type::f32
                    : wei_dt;

            for_(int i_bs = 0; i_bs < 2; i_bs++)
            for_(int i_init = 0; i_init < 2; i_init++)
            for_(int i_M = 0; i_M < 2; i_M++)
            for_(int i_N = 0; i_N < 2; i_N++)
            for (int i_K = 0; i_K < 2; i_K++) {
                auto vbeta = (i_init) ? beta_init : beta;
                auto vM = (i_M) ? jbgp_.M_tail : jbgp_.M;
                auto vN = (i_N) ? jbgp_.N_tail : jbgp_.N;
                auto vK = (i_K) ? jbgp_.K_tail : jbgp_.K;
                int bs = get_brg_batchsize(i_bs, i_K);
                int idx = get_brg_kernel_idx(i_bs, i_init, i_M, i_N, i_K, bs);
                if (idx < 0) continue;

                brgemm_desc_t &brg = brg_descs_[idx];
                CHECK(brgemm_desc_init(&brg, isa, jbgp_.brg_type, diff_dst_dt,
                        dt_b, false, false, brgemm_row_major, alpha, vbeta,
                        jbgp_.LDA, jbgp_.LDB, jbgp_.LDC, vM, vN, vK));

                CHECK(brgemm_desc_set_postops(
                        &brg, attr(), &diff_src_md_, jbgp_.LDD, jbgp_.bia_dt));
                if (jbgp_.is_amx) {
                    brgemm_attr_t brgattr;
                    brgattr.max_bs = bs;
                    brgattr.wary_tail_read = false;
                    brgattr.hint_expected_A_size = jbgp_.mb * jbgp_.oc;
                    brgattr.hint_expected_B_size = jbgp_.oc * jbgp_.ic;
                    brgattr.hint_expected_C_size = jbgp_.mb * jbgp_.ic;
                    brgattr.hint_innermost_loop = brgemm_innermost_undef;
                    brgattr.use_uker = jbgp_.use_uker;
                    brgattr.use_interleave_stores = jbgp_.use_interleave_stores;
                    brgattr.hint_prefetching = jbgp_.hint_prefetching;
                    brgattr.fpmath_mode = attr()->fpmath_.mode_;

                    CHECK(brgemm_desc_set_attr(&brg, brgattr));
                    jbgp_.amx_buf_size_per_thread
                            = nstl::max(brg.get_wsp_buffer_size(),
                                    jbgp_.amx_buf_size_per_thread);
                }
            }

            auto scratchpad = scratchpad_registry().registrar();
            jbgp_.init_scratchpad(scratchpad);

            return status::success;
        }

        int get_brg_kernel_idx(bool is_bs_tail, bool do_initialization,
                bool is_M_tail, bool is_N_tail, bool is_K_tail, int bs) const {
            auto vM = (is_M_tail) ? jbgp_.M_tail : jbgp_.M;
            auto vN = (is_N_tail) ? jbgp_.N_tail : jbgp_.N;
            auto vK = (is_K_tail) ? jbgp_.K_tail : jbgp_.K;

            if (vM == 0 || vN == 0 || vK == 0 || bs == 0 || jbgp_.LDA < vK
                    || jbgp_.LDB < vN || jbgp_.LDC < vN)
                return -1;
            return brgemm_inner_product_utils::get_brg_kernel_index(is_bs_tail,
                    do_initialization, is_M_tail, is_N_tail, is_K_tail);
        }

        int get_brg_batchsize(bool is_bs_tail, bool is_K_tail) const {
            auto adj_oc = jbgp_.use_buffer_a
                    ? utils::rnd_up(jbgp_.oc, jbgp_.oc_block)
                    : jbgp_.oc;
            auto bs = (is_K_tail) ? 1
                                  : ((is_bs_tail) ? (adj_oc / jbgp_.oc_block)
                                                          % jbgp_.nb_oc_blocking
                                                  : jbgp_.nb_oc_blocking);

            return bs;
        }

        brgemm_desc_t
                brg_descs_[brgemm_inner_product_utils::max_num_brg_kernels_ip];
        brgemm_inner_product_utils::jit_brgemm_ip_bwd_d_conf_t jbgp_;
    };

    brgemm_inner_product_bwd_data_t(const pd_t *apd) : primitive_t(apd) {}

    status_t init(engine_t *engine) override {
        const auto &jbgp = pd()->jbgp_;
        for_(int i_bs = 0; i_bs < 2; i_bs++)
        for_(int i_M = 0; i_M < 2; i_M++)
        for_(int i_N = 0; i_N < 2; i_N++)
        for_(int i_K = 0; i_K < 2; i_K++)
        for (int i_init = 0; i_init < 2; i_init++) {
            int bs = pd()->get_brg_batchsize(i_bs, i_K);
            int idx = pd()->get_brg_kernel_idx(i_bs, i_init, i_M, i_N, i_K, bs);
            if (idx < 0) continue;

            brgemm_kernel_t *ker = nullptr;
            CHECK(brgemm_kernel_create(&ker, pd()->brg_descs_[idx]));
            CHECK(safe_ptr_assign(brg_kernels_[idx], ker));
            if (jbgp.is_amx)
                brgemm_palettes_.insert(idx, pd()->brg_descs_[idx]);
        }

        if (pd()->jbgp_.use_buffer_a)
            CHECK(create_brgemm_copy_to_coarse(
                    copy_diff_dst_kernel_, &pd()->jbgp_));
        if (jbgp.use_buffer_b)
            CHECK(create_brgemm_trans_wei(trans_B_kernel_, &pd()->jbgp_));

        if (jbgp.nthr_oc_b > 1) {
            CHECK(safe_ptr_assign(
                    acc_ker_, new cpu_accumulator_1d_t<data_type::f32>()));
            CHECK(acc_ker_->create_kernel());
        }

        return status::success;
    }

    status_t execute(const exec_ctx_t &ctx) const override {
        execute_backward_data(ctx);
        return status::success;
    }

private:
    void execute_backward_data(const exec_ctx_t &ctx) const;
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }

    std::unique_ptr<brgemm_kernel_t>
            brg_kernels_[brgemm_inner_product_utils::max_num_brg_kernels_ip];
    std::unique_ptr<jit_brgemm_copy_to_coarse_t> copy_diff_dst_kernel_;
    std::unique_ptr<jit_brgemm_trans_wei_t> trans_B_kernel_;
    std::unique_ptr<cpu_accumulator_1d_t<data_type::f32>> acc_ker_;
    brgemm_containers::brgemm_palette_container_t brgemm_palettes_ {
            brgemm_inner_product_utils::max_num_brg_kernels_ip};
};

template <cpu_isa_t isa>
struct brgemm_inner_product_bwd_weights_t : public primitive_t {
    struct pd_t : public cpu_inner_product_bwd_weights_pd_t {
        pd_t(const inner_product_desc_t *adesc, const primitive_attr_t *attr,
                const inner_product_fwd_pd_t *hint_fwd_pd)
            : cpu_inner_product_bwd_weights_pd_t(adesc, attr, hint_fwd_pd) {}

        DECLARE_COMMON_PD_T(JIT_IMPL_NAME_HELPER("brgemm_bwd_w:", isa, ""),
                brgemm_inner_product_bwd_weights_t);

        status_t init(engine_t *engine) {
            using skip_mask_t = primitive_attr_t::skip_mask_t;

            auto src_dt = invariant_src_md()->data_type;
            auto diff_wei_type = invariant_wei_md()->data_type;
            auto diff_dst_type = invariant_dst_md()->data_type;

            if (!mayiuse(isa)) return status::unimplemented;
            VDISPATCH_INNER_PRODUCT(
                    desc()->prop_kind == prop_kind::backward_weights,
                    VERBOSE_BAD_PROPKIND);
            VDISPATCH_INNER_PRODUCT(
                    !has_zero_dim_memory(), VERBOSE_EMPTY_TENSOR, "");
            VDISPATCH_INNER_PRODUCT(utils::one_of(src_dt, data_type::f32,
                                            data_type::bf16, data_type::f16),
                    VERBOSE_UNSUPPORTED_DT);
            VDISPATCH_INNER_PRODUCT(diff_dst_type == src_dt,
                    VERBOSE_INCONSISTENT_DT, "diff_dst", "src");
            VDISPATCH_INNER_PRODUCT(
                    utils::one_of(diff_wei_type, data_type::f32, src_dt),
                    VERBOSE_UNSUPPORTED_DT);
            VDISPATCH_INNER_PRODUCT(
                    attr()->has_default_values(skip_mask_t::fpmath_mode),
                    VERBOSE_UNSUPPORTED_ATTR);

            CHECK(jbgp_.init_conf(isa, *desc(), src_md_, diff_weights_md_,
                    diff_dst_md_, diff_bias_md_, attr_,
                    dnnl_get_max_threads()));

            const float alpha = 1.0;
            const float beta = 1.0;
            const float beta_init = 0.0;
            const auto dt_a = isa == avx512_core_fp16 && jbgp_.use_buffer_a
                    ? data_type::f32
                    : jbgp_.src_dt;
            const auto dt_b = isa == avx512_core_fp16 && jbgp_.use_buffer_b
                    ? data_type::f32
                    : jbgp_.dst_dt;

            for_(int i_bs = 0; i_bs < 2; i_bs++)
            for_(int i_init = 0; i_init < 2; i_init++)
            for_(int i_M = 0; i_M < 2; i_M++)
            for_(int i_N = 0; i_N < 2; i_N++)
            for (int i_K = 0; i_K < 2; i_K++) {
                auto vbeta = (i_init) ? beta_init : beta;
                auto vM = (i_M) ? jbgp_.M_tail : jbgp_.M;
                auto vN = (i_N) ? jbgp_.N_tail : jbgp_.N;
                auto vK = (i_K) ? jbgp_.K_tail : jbgp_.K;
                int bs = get_brg_batchsize(i_bs, i_K);
                int idx = get_brg_kernel_idx(i_bs, i_init, i_M, i_N, i_K, bs);
                if (idx < 0) continue;
                brgemm_desc_t &brg = brg_descs_[idx];
                CHECK(brgemm_desc_init(&brg, isa, jbgp_.brg_type, dt_a, dt_b,
                        false, false, brgemm_row_major, alpha, vbeta, jbgp_.LDA,
                        jbgp_.LDB, jbgp_.LDC, vM, vN, vK));
                if (jbgp_.is_amx) {
                    brgemm_attr_t brgattr;
                    brgattr.max_bs = bs;
                    brgattr.wary_tail_read = false;
                    brgattr.hint_expected_A_size = jbgp_.mb * jbgp_.ic;
                    brgattr.hint_expected_B_size = jbgp_.mb * jbgp_.oc;
                    brgattr.hint_expected_C_size = jbgp_.ic * jbgp_.oc;
                    brgattr.hint_innermost_loop = brgemm_innermost_undef;
                    brgattr.use_uker = jbgp_.use_uker;
                    brgattr.use_interleave_stores = jbgp_.use_interleave_stores;
                    brgattr.hint_prefetching = jbgp_.hint_prefetching;
                    brgattr.fpmath_mode = attr()->fpmath_.mode_;

                    CHECK(brgemm_desc_set_attr(&brg, brgattr));
                    jbgp_.amx_buf_size_per_thread
                            = nstl::max(brg.get_wsp_buffer_size(),
                                    jbgp_.amx_buf_size_per_thread);
                }
            }

            auto scratchpad = scratchpad_registry().registrar();
            jbgp_.init_scratchpad(scratchpad);

            return status::success;
        }

        int get_brg_kernel_idx(bool is_bs_tail, bool do_initialization,
                bool is_M_tail, bool is_N_tail, bool is_K_tail, int bs) const {
            auto vM = (is_M_tail) ? jbgp_.M_tail : jbgp_.M;
            auto vN = (is_N_tail) ? jbgp_.N_tail : jbgp_.N;
            auto vK = (is_K_tail) ? jbgp_.K_tail : jbgp_.K;

            if (vM == 0 || vN == 0 || vK == 0 || bs == 0 || jbgp_.LDA < vK
                    || jbgp_.LDB < vN || jbgp_.LDC < vN)
                return -1;
            return brgemm_inner_product_utils::get_brg_kernel_index(is_bs_tail,
                    do_initialization, is_M_tail, is_N_tail, is_K_tail);
        }

        int get_brg_batchsize(bool is_bs_tail, bool is_K_tail) const {
            auto bs = (is_K_tail) ? 1
                                  : ((is_bs_tail) ? (jbgp_.os / jbgp_.os_block)
                                                          % jbgp_.nb_os_blocking
                                                  : jbgp_.nb_os_blocking);
            return bs;
        }

        brgemm_desc_t
                brg_descs_[brgemm_inner_product_utils::max_num_brg_kernels_ip];
        brgemm_inner_product_utils::jit_brgemm_ip_bwd_w_conf_t jbgp_;
    };

    brgemm_inner_product_bwd_weights_t(const pd_t *apd) : primitive_t(apd) {}

    status_t init(engine_t *engine) override {
        const auto &jbgp = pd()->jbgp_;
        for_(int i_bs = 0; i_bs < 2; i_bs++)
        for_(int i_M = 0; i_M < 2; i_M++)
        for_(int i_N = 0; i_N < 2; i_N++)
        for_(int i_K = 0; i_K < 2; i_K++)
        for (int i_init = 0; i_init < 2; i_init++) {
            int bs = pd()->get_brg_batchsize(i_bs, i_K);
            int idx = pd()->get_brg_kernel_idx(i_bs, i_init, i_M, i_N, i_K, bs);
            if (idx < 0) continue;

            brgemm_kernel_t *ker = nullptr;
            CHECK(brgemm_kernel_create(&ker, pd()->brg_descs_[idx]));
            CHECK(safe_ptr_assign(brg_kernels_[idx], ker));
            if (jbgp.is_amx)
                brgemm_palettes_.insert(idx, pd()->brg_descs_[idx]);

            if (jbgp.with_bias && i_M == 0 && i_init == 0) {
                kernels_db_[i_K][i_N] = nullptr;
                auto db_desc = pd()->brg_descs_[idx];
                db_desc.reduce_dim = (i_K) ? jbgp.K_tail : jbgp.K;
                if (db_desc.reduce_dim > 0 && db_desc.load_dim > 0) {
                    CHECK(safe_ptr_assign(kernels_db_[i_K][i_N],
                            new ker_diff_bias_t(jbgp, db_desc)));
                    CHECK(kernels_db_[i_K][i_N]->create_kernel());
                }
            }
        }
        if (jbgp.is_amx) {
            ext_ic_block_ = jbgp.ic_block_ext;
            ext_oc_block_ = jbgp.oc_block_ext;
        }
        CHECK(create_brgemm_trans_src(trans_A_kernel_, &pd()->jbgp_));

        if (jbgp.use_buffer_b)
            CHECK(create_brgemm_trans_to_vnni(trans_B_kernel_, &pd()->jbgp_,
                    jit_brgemm_trans_to_vnni_t::matrix_to_transform::matrix_B));

        if (!jbgp.is_amx) {
            if (jbgp.wei_dt != jbgp.acc_dt)
                CHECK(create_brgemm_trans_to_vnni(trans_C_kernel_, &pd()->jbgp_,
                        jit_brgemm_trans_to_vnni_t::matrix_to_transform::
                                matrix_C));
        } else if (utils::one_of(
                           jbgp.wei_dt, data_type::bf16, data_type::f16)) {
            CHECK(create_brgemm_amx_ip_trans_wei(diff_wei_trans_kernel_,
                    &pd()->jbgp_, ext_ic_block_, ext_oc_block_));
        }
        if (jbgp.nthr_mb > 1) {
            CHECK(safe_ptr_assign(
                    acc_ker_, new cpu_accumulator_1d_t<data_type::f32>()));
            CHECK(acc_ker_->create_kernel());
        }

        return status::success;
    }

    status_t execute(const exec_ctx_t &ctx) const override {
        execute_backward_weights(ctx);
        return status::success;
    }

private:
    struct thread_info_t;
    using ker_diff_bias_t
            = jit_brgemm_kernel_diff_bias_t<typename cpu_isa_traits<isa>::Vmm>;
    std::unique_ptr<ker_diff_bias_t> kernels_db_[2][2];
    std::unique_ptr<brgemm_kernel_t>
            brg_kernels_[brgemm_inner_product_utils::max_num_brg_kernels_ip];
    std::unique_ptr<jit_brgemm_trans_src_t> trans_A_kernel_;
    std::unique_ptr<jit_brgemm_trans_to_vnni_t> trans_B_kernel_;
    std::unique_ptr<jit_brgemm_trans_to_vnni_t> trans_C_kernel_;
    std::unique_ptr<cpu_accumulator_1d_t<data_type::f32>> acc_ker_;
    std::unique_ptr<jit_amx_ip_trans_diff_wei> diff_wei_trans_kernel_;

    void execute_backward_weights(const exec_ctx_t &ctx) const;
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }
    void compute_diff_weights_and_bias(const thread_info_t *ti) const;
    void reduce_and_convert_diff_weights_and_bias(
            const thread_info_t *ti) const;
    void transform_matrix_a_chunk(char *tr_src, const char *src,
            int trans_batch, int current_m, int current_k) const;
    void transform_matrix_b_chunk(char *tr_diff_dst, const char *diff_dst,
            int trans_batch, int current_col_size, int current_row_size) const;
    void transpose_matrix_c_chunk(const thread_info_t *ti, const dim_t ocb,
            const dim_t icb, int oc_size, int ic_size, dim_t kd, dim_t kh,
            dim_t kw, bool is_reduction = false) const;

    brgemm_containers::brgemm_palette_container_t brgemm_palettes_ {
            brgemm_inner_product_utils::max_num_brg_kernels_ip};
    char *get_wei_acc_ptr(const thread_info_t *ti, dim_t ocb, dim_t icb,
            dim_t kd, dim_t kh, dim_t kw, int reduction_buf_idx = -1) const;

    int ext_ic_block_ = 0;
    int ext_oc_block_ = 0;
};

} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif

// vim: et ts=4 sw=4 cindent cino+=l0,\:4,N-s
