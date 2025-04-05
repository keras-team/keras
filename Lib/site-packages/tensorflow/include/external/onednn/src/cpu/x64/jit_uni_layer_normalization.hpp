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

#ifndef CPU_X64_JIT_UNI_LAYER_NORMALIZATION_HPP
#define CPU_X64_JIT_UNI_LAYER_NORMALIZATION_HPP

#include <memory>

#include "common/c_types_map.hpp"
#include "common/dnnl_thread.hpp"
#include "common/memory_tracking.hpp"
#include "common/primitive.hpp"
#include "common/reorder.hpp"
#include "common/stream.hpp"
#include "common/utils.hpp"

#include "cpu/cpu_layer_normalization_pd.hpp"

#include "cpu/x64/cpu_isa_traits.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {

struct stat_and_data_kernel_t {
    static stat_and_data_kernel_t *create(const layer_normalization_pd_t *pd);
    virtual ~stat_and_data_kernel_t() = default;

    virtual void operator()(const void *src, void *dst, const float *scale,
            const float *shift, float *mean, float *var,
            const float *src_scales, const float *dst_scales,
            const void *post_ops_binary_rhs_arg_vec,
            const size_t block_size) const {};

    virtual status_t create_kernel() { return status::success; }

protected:
    stat_and_data_kernel_t(const layer_normalization_pd_t *pd) : pd_(pd) {}

    const layer_normalization_pd_t *pd_;
};

struct diff_ss_kernel_t {
    static diff_ss_kernel_t *create(const layer_normalization_pd_t *pd);
    virtual ~diff_ss_kernel_t() = default;

    virtual void operator()(const void *src, const void *diff_dst,
            float *diff_gamma, float *diff_beta, const float *mean,
            const float *var, float *const inv_sqrtvar,
            const size_t block_size) const {};

    virtual status_t create_kernel() { return status::success; }

protected:
    diff_ss_kernel_t(const layer_normalization_pd_t *pd) : pd_(pd) {}

    const layer_normalization_pd_t *pd_;
};

struct diff_data_kernel_t {
    static diff_data_kernel_t *create(const layer_normalization_pd_t *pd);
    virtual ~diff_data_kernel_t() = default;

    virtual void operator()(const void *src, const void *diff_dst,
            void *diff_src, const float *ss, const float *mean,
            float *const inv_sqrtvar, const size_t block_size) const {};

    virtual status_t create_kernel() { return status::success; }

protected:
    diff_data_kernel_t(const layer_normalization_pd_t *pd) : pd_(pd) {}

    const layer_normalization_pd_t *pd_;
};

struct jit_uni_layer_normalization_fwd_t : public primitive_t {
    struct pd_t : public cpu_layer_normalization_fwd_pd_t {
        using cpu_layer_normalization_fwd_pd_t::
                cpu_layer_normalization_fwd_pd_t;

        DECLARE_COMMON_PD_T("jit:uni", jit_uni_layer_normalization_fwd_t);

        status_t init(engine_t *engine);

        bool use_tmp_stats() const { return reorder_pd_ || stats_are_tmp(); }

        std::shared_ptr<primitive_desc_t> reorder_pd_;
        memory_desc_t reordered_stat_md_;

    private:
        void init_scratchpad() {
            using namespace memory_tracking::names;
            auto scratchpad = scratchpad_registry().registrar();
            if (use_tmp_stats()) {
                scratchpad.template book<float>(
                        key_lnorm_tmp_mean, across_axis());
                scratchpad.template book<float>(
                        key_lnorm_tmp_var, across_axis());
            }
            if (reordered_stat_md_ != *stat_md() && !stats_are_tmp()) {
                scratchpad.book(key_nested, reorder_pd_->scratchpad_registry());
            }
        }
    };

    status_t init(engine_t *engine) override {
        if (pd()->reorder_pd_)
            pd()->reorder_pd_->create_primitive(reorder_, engine);
        CHECK(safe_ptr_assign(
                stat_and_data_kernel_, stat_and_data_kernel_t::create(pd())));
        if (stat_and_data_kernel_)
            CHECK(stat_and_data_kernel_->create_kernel());
        return status::success;
    }

    jit_uni_layer_normalization_fwd_t(const pd_t *apd) : primitive_t(apd) {}
    virtual ~jit_uni_layer_normalization_fwd_t() = default;

    void reorder_stat(const exec_ctx_t &ctx, engine_t *engine,
            const memory_arg_t &in, const memory_arg_t &out) const {
        using namespace memory_tracking::names;
        exec_args_t r_args;
        r_args[DNNL_ARG_SRC] = in;
        r_args[DNNL_ARG_DST] = out;
        exec_ctx_t r_ctx(ctx, std::move(r_args));

        nested_scratchpad_t ns(ctx, key_nested, reorder_);
        r_ctx.set_scratchpad_grantor(ns.grantor());
        reorder_->execute(r_ctx);
    }

    status_t execute(const exec_ctx_t &ctx) const override {
        /* LN supports arbitrary layout for input/output statistics.
         * For best performance we compute LN with statistics in the same format
         * as data tensor (i.e. data in abcd, stats in abc) and user's
         * input/output statistics are reordered if necessary */
        using namespace memory_tracking::names;
        engine_t *engine = ctx.stream()->engine();
        auto scratchpad = ctx.get_scratchpad_grantor();
        auto mean_mem = scratchpad.get_memory_storage(key_lnorm_tmp_mean);
        auto variance_mem = scratchpad.get_memory_storage(key_lnorm_tmp_var);
        memory_t mean(engine, &(pd()->reordered_stat_md_), std::move(mean_mem));
        memory_t variance(
                engine, &(pd()->reordered_stat_md_), std::move(variance_mem));

        // reorder input stats
        if (pd()->stats_are_src() && reorder_) {
            reorder_stat(
                    ctx, engine, ctx.args().at(DNNL_ARG_MEAN), {&mean, false});
            reorder_stat(ctx, engine, ctx.args().at(DNNL_ARG_VARIANCE),
                    {&variance, false});
        }
        status_t status = execute_forward(ctx);
        if (status != status::success) return status;
        // reorder output stats
        if (!pd()->stats_are_src() && reorder_) {
            reorder_stat(
                    ctx, engine, {&mean, true}, ctx.args().at(DNNL_ARG_MEAN));
            reorder_stat(ctx, engine, {&variance, true},
                    ctx.args().at(DNNL_ARG_VARIANCE));
        }

        return status::success;
    }

private:
    status_t execute_forward(const exec_ctx_t &ctx) const;
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }

    std::unique_ptr<stat_and_data_kernel_t> stat_and_data_kernel_;
    std::shared_ptr<primitive_t> reorder_;
};

struct jit_uni_layer_normalization_bwd_t : public primitive_t {
    struct pd_t : public cpu_layer_normalization_bwd_pd_t {
        using cpu_layer_normalization_bwd_pd_t::
                cpu_layer_normalization_bwd_pd_t;

        DECLARE_COMMON_PD_T("jit:uni", jit_uni_layer_normalization_bwd_t);

        status_t init(engine_t *engine) {
            using namespace data_type;
            const memory_desc_wrapper src_d(src_md());

            VDISPATCH_LNORM(!is_fwd(), VERBOSE_BAD_PROPKIND);
            VDISPATCH_LNORM(!has_zero_dim_memory(), VERBOSE_EMPTY_TENSOR, "");
            // disabling verbose dispatch checks for unsupported isa for better readability
            if (!mayiuse(avx2))
                return status::unimplemented; // sse41 is not supported yet

            VDISPATCH_LNORM(utils::one_of(src_md()->data_type, f32, bf16, f16),
                    VERBOSE_UNSUPPORTED_DT);
            VDISPATCH_LNORM(
                    utils::one_of(diff_dst_md()->data_type, f32, bf16, f16),
                    VERBOSE_UNSUPPORTED_DT);
            VDISPATCH_LNORM(
                    utils::one_of(diff_src_md()->data_type, f32, bf16, f16),
                    VERBOSE_UNSUPPORTED_DT);
            VDISPATCH_LNORM(IMPLICATION(utils::one_of(bf16, src_md()->data_type,
                                                diff_dst_md()->data_type,
                                                diff_src_md()->data_type),
                                    mayiuse(avx512_core)),
                    VERBOSE_ISA_DT_MISMATCH);
            VDISPATCH_LNORM(IMPLICATION(utils::one_of(f16, src_md()->data_type,
                                                diff_dst_md()->data_type,
                                                diff_src_md()->data_type),
                                    mayiuse(avx512_core_fp16)),
                    VERBOSE_ISA_DT_MISMATCH);
            VDISPATCH_LNORM(
                    stat_md()->data_type == f32, VERBOSE_UNSUPPORTED_DT);
            VDISPATCH_LNORM(check_scale_shift_data_type(),
                    VERBOSE_UNSUPPORTED_FEATURE,
                    "unsupported scale or shift data type");
            VDISPATCH_LNORM(
                    attr()->has_default_values(), VERBOSE_UNSUPPORTED_ATTR);
            VDISPATCH_LNORM(
                    set_default_formats_common(), VERBOSE_UNSUPPORTED_TAG);
            VDISPATCH_LNORM(src_d.is_blocking_desc(), VERBOSE_BLOCKING_FAIL,
                    "blocking descriptor fail");
            // plain format, last logical dim is last physical
            VDISPATCH_LNORM(src_d.blocking_desc().strides[ndims() - 1] == 1,
                    VERBOSE_BLOCKING_FAIL, "bad stride value");

            CHECK(fill_compatible_stats_md(*src_md(), reordered_stat_md_));

            if (reordered_stat_md_ != *stat_md()) {
                CHECK(reorder_primitive_desc_create(
                        reorder_pd_, engine, stat_md(), &reordered_stat_md_));
            }

            nthr_ = dnnl_get_max_threads();
            init_scratchpad();
            return status::success;
        }

        bool use_tmp_stats() const { return reorder_pd_.get(); }

        std::shared_ptr<primitive_desc_t> reorder_pd_;
        memory_desc_t reordered_stat_md_;
        int nthr_; // To not exceed the limit in execute used for set up.

    private:
        void init_scratchpad() {
            using namespace memory_tracking::names;
            auto scratchpad = scratchpad_registry().registrar();
            if (use_tmp_stats()) {
                scratchpad.template book<float>(
                        key_lnorm_tmp_mean, across_axis());
                scratchpad.template book<float>(
                        key_lnorm_tmp_var, across_axis());
            }
            scratchpad.template book<float>(
                    key_lnorm_reduction, 2 * norm_axis() * nthr_);
            scratchpad.template book<float>(
                    key_lnorm_tmp_diff_ss, 2 * norm_axis());
            if (reordered_stat_md_ != *stat_md() && !stats_are_tmp()) {
                scratchpad.book(key_nested, reorder_pd_->scratchpad_registry());
            }
            scratchpad.template book<float>(
                    key_lnorm_inv_sqrtvar, across_axis());
        }
    };

    status_t init(engine_t *engine) override {
        if (pd()->reorder_pd_)
            pd()->reorder_pd_->create_primitive(reorder_, engine);
        CHECK(safe_ptr_assign(diff_ss_kernel_, diff_ss_kernel_t::create(pd())));
        CHECK(safe_ptr_assign(
                diff_data_kernel_, diff_data_kernel_t::create(pd())));
        if (diff_ss_kernel_) CHECK(diff_ss_kernel_->create_kernel());
        if (diff_data_kernel_) CHECK(diff_data_kernel_->create_kernel());
        return status::success;
    }

    jit_uni_layer_normalization_bwd_t(const pd_t *apd) : primitive_t(apd) {}
    virtual ~jit_uni_layer_normalization_bwd_t() = default;

    void reorder_stat(const exec_ctx_t &ctx, engine_t *engine,
            const memory_arg_t &in, const memory_arg_t &out) const {
        using namespace memory_tracking::names;
        exec_args_t r_args;
        r_args[DNNL_ARG_SRC] = in;
        r_args[DNNL_ARG_DST] = out;
        exec_ctx_t r_ctx(ctx, std::move(r_args));

        nested_scratchpad_t ns(ctx, key_nested, reorder_);
        r_ctx.set_scratchpad_grantor(ns.grantor());
        reorder_->execute(r_ctx);
    }

    status_t execute(const exec_ctx_t &ctx) const override {
        using namespace memory_tracking::names;
        /* LN supports arbitrary layout for input/output statistics.
         * For best performance we compute LN with statistics in the same format
         * as data tensor (i.e. data in abcd, stats in abc) and user's
         * input/output statistics are reordered if necessary */

        if (reorder_) {
            engine_t *engine = ctx.stream()->engine();
            auto scratchpad = ctx.get_scratchpad_grantor();
            auto mean_mem = scratchpad.get_memory_storage(key_lnorm_tmp_mean);
            auto variance_mem
                    = scratchpad.get_memory_storage(key_lnorm_tmp_var);
            memory_t mean(
                    engine, &(pd()->reordered_stat_md_), std::move(mean_mem));
            memory_t variance(engine, &(pd()->reordered_stat_md_),
                    std::move(variance_mem));
            reorder_stat(
                    ctx, engine, ctx.args().at(DNNL_ARG_MEAN), {&mean, false});
            reorder_stat(ctx, engine, ctx.args().at(DNNL_ARG_VARIANCE),
                    {&variance, false});
        }

        return execute_backward(ctx);
    }

private:
    status_t execute_backward(const exec_ctx_t &ctx) const;
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }

    std::unique_ptr<diff_ss_kernel_t> diff_ss_kernel_;
    std::unique_ptr<diff_data_kernel_t> diff_data_kernel_;
    std::shared_ptr<primitive_t> reorder_;
};

} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif

// vim: et ts=4 sw=4 cindent cino^=l0,\:0,N-s
