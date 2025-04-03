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

#ifndef CPU_SIMPLE_LAYER_NORMALIZATION_HPP
#define CPU_SIMPLE_LAYER_NORMALIZATION_HPP

#include <memory>

#include "common/c_types_map.hpp"
#include "common/dnnl_thread.hpp"
#include "common/memory_tracking.hpp"
#include "common/primitive.hpp"
#include "common/reorder_pd.hpp"
#include "common/stream.hpp"
#include "common/utils.hpp"

#include "cpu/primitive_attr_postops.hpp"

#include "cpu/cpu_layer_normalization_pd.hpp"

namespace dnnl {
namespace impl {
namespace cpu {

struct simple_layer_normalization_fwd_t : public primitive_t {
    struct pd_t : public cpu_layer_normalization_fwd_pd_t {
        using cpu_layer_normalization_fwd_pd_t::
                cpu_layer_normalization_fwd_pd_t;

        DECLARE_COMMON_PD_T("simple:any", simple_layer_normalization_fwd_t);

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
        bool post_ops_ok() const {
            return ref_post_ops_t::primitive_kind_ok(attr()->post_ops_);
        }
    };

    status_t init(engine_t *engine) override {
        if (pd()->reorder_pd_)
            pd()->reorder_pd_->create_primitive(reorder_, engine);

        ref_post_ops
                = utils::make_unique<ref_post_ops_t>(pd()->attr()->post_ops_);
        if (!ref_post_ops) return status::out_of_memory;
        CHECK(ref_post_ops->init(pd()->dst_md()));

        return status::success;
    }

    simple_layer_normalization_fwd_t(const pd_t *apd) : primitive_t(apd) {}

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

    std::shared_ptr<primitive_t> reorder_;
    std::unique_ptr<ref_post_ops_t> ref_post_ops;
};

struct simple_layer_normalization_bwd_t : public primitive_t {
    struct pd_t : public cpu_layer_normalization_bwd_pd_t {
        using cpu_layer_normalization_bwd_pd_t::
                cpu_layer_normalization_bwd_pd_t;

        DECLARE_COMMON_PD_T("simple:any", simple_layer_normalization_bwd_t);

        status_t init(engine_t *engine);

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
        return status::success;
    }

    simple_layer_normalization_bwd_t(const pd_t *apd) : primitive_t(apd) {}

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

    std::shared_ptr<primitive_t> reorder_;
};

} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif

// vim: et ts=4 sw=4 cindent cino^=l0,\:0,N-s
