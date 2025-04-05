/*******************************************************************************
* Copyright 2023 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/
#ifndef CPU_REORDER_SIMPLE_SPARSE_REORDER_HPP
#define CPU_REORDER_SIMPLE_SPARSE_REORDER_HPP

#include <bitset>
#include <iostream>

#include <assert.h>
#include "common/c_types_map.hpp"
#include "common/dnnl_thread.hpp"
#include "common/math_utils.hpp"
#include "common/primitive.hpp"
#include "common/reorder.hpp"

#include "common/primitive_attr.hpp"
#include "common/stream.hpp"
#include "common/tag_traits.hpp"
#include "common/type_helpers.hpp"
#include "common/utils.hpp"
#include "cpu/cpu_primitive.hpp"
#include "cpu/reorder/cpu_reorder_pd.hpp"
#include "cpu/simple_q10n.hpp"

namespace dnnl {
namespace impl {
namespace cpu {

// The following cases can be covered:
//
// Note: `sparse_tag` is a regular format tag describing
//        a regular tensor with sparse data.
//
// - sparse_tag -> sparse_tag
// - encoding -> encoding
//
// - sparse_tag -> dense_tag
// - dense_tag -> sparse_tag
//
// - sparse_tag -> encoding
// - encoding -> sparse_tag
//
// - dense_tag -> encoding
// - encoding -> dense_tag
#define SIMPLE_SPARSE_REORDER_TEMPL_DECL \
    impl::data_type_t type_i, typename fmt_i_t, fmt_i_t fmt_i, \
            impl::data_type_t type_o, typename fmt_o_t, fmt_o_t fmt_o
#define SIMPLE_SPARSE_REORDER_TEMPL_CALL \
    type_i, fmt_i_t, fmt_i, type_o, fmt_o_t, fmt_o

template <SIMPLE_SPARSE_REORDER_TEMPL_DECL, typename spec = void>
struct simple_sparse_reorder_impl {};

namespace {
template <typename T>
constexpr bool is_format_tag(T) {
    return std::is_same<T, format_tag_t>::value;
}
} // namespace

template <SIMPLE_SPARSE_REORDER_TEMPL_DECL>
struct simple_sparse_reorder_impl<SIMPLE_SPARSE_REORDER_TEMPL_CALL,
        typename utils::enable_if<(is_format_tag(fmt_i)
                                          && (fmt_i == format_tag::any))
                && (is_format_tag(fmt_o)
                        && (fmt_o == format_tag::any))>::type> {

    static bool is_applicable(const memory_desc_wrapper &input_d,
            const memory_desc_wrapper &output_d, const primitive_attr_t *attr) {
        // This reorder expects a non-plain format for destination.
        return input_d.is_blocking_desc() && output_d.is_sparse_desc()
                && output_d.sparse_desc().encoding == sparse_encoding::packed
                && output_d.blocking_desc().inner_nblks > 0
                && output_d.blk_size() % 64 == 0;
    }

    static size_t get_scratchpad_size(const memory_desc_wrapper &input_d,
            const memory_desc_wrapper &output_d) {
        const auto nelems = output_d.nelems(true);
        const auto tmp_output_sz = nelems * output_d.data_type_size();
        const auto nnz_per_blocks_sz
                = nelems / output_d.blk_size() * sizeof(dim_t);
        return tmp_output_sz + nnz_per_blocks_sz;
    }

    static status_t execute(const cpu_reorder_pd_t *pd, const exec_ctx_t &ctx,
            const std::shared_ptr<primitive_t> &reorder) {
        auto output_values = CTX_OUT_MEM(data_t<type_o> *, DNNL_ARG_TO, 0);
        auto output_offsets = CTX_OUT_MEM(int64_t *, DNNL_ARG_TO, 1);
        auto output_bitmask = CTX_OUT_MEM(uint64_t *, DNNL_ARG_TO, 2);

        engine_t *engine = ctx.stream()->engine();
        const auto scratchpad = ctx.get_scratchpad_grantor();
        auto wspace_mem_storage = scratchpad.get_memory_storage(
                memory_tracking::names::key_reorder_space);
        memory_t wspace_mem(
                engine, reorder->pd()->dst_md(), std::move(wspace_mem_storage));

        exec_args_t r_args;
        r_args[DNNL_ARG_SRC] = ctx.args().at(DNNL_ARG_FROM);
        r_args[DNNL_ARG_DST] = {&wspace_mem, false};
        exec_ctx_t r_ctx(ctx, std::move(r_args));

        nested_scratchpad_t ns(
                ctx, memory_tracking::names::key_nested, reorder);
        r_ctx.set_scratchpad_grantor(ns.grantor());
        reorder->execute(r_ctx);

        auto *wspace = scratchpad.template get<data_t<type_o>>(
                memory_tracking::names::key_reorder_space);

        const auto output_d = ctx.memory_mdw(DNNL_ARG_TO, pd->dst_md());
        const auto nelems = output_d.nelems(true);
        const auto blk_sz = output_d.blk_size();
        const auto nblks = nelems / blk_sz;

        dim_t *nnz_per_blocks
                = reinterpret_cast<dim_t *>(reinterpret_cast<char *>(wspace)
                        + nelems * output_d.data_type_size());

        static constexpr int bitmask_step = sizeof(uint64_t) * CHAR_BIT;
        // Fill output_bitmask and move non-zero elements to the begining of the
        // blocks. Also, remember number of non-zero elements per-block to
        // calculate output_offsets later.
        parallel_nd(nblks, [&](dim_t b) {
            dim_t nnz_per_blk = 0;
            for (dim_t i = 0; i < blk_sz / bitmask_step; i++) {
                uint64_t &bm = output_bitmask[b * blk_sz / bitmask_step + i];
                bm = 0;
                for (dim_t j = 0; j < bitmask_step; j++) {
                    const auto v = wspace[b * blk_sz + bitmask_step * i + j];
                    if (v != 0) {
                        wspace[b * blk_sz + nnz_per_blk++] = v;
                        bm |= (uint64_t(1) << j);
                    }
                }
            }
            nnz_per_blocks[b] = nnz_per_blk;
        });

        // Calculate output_offsets using previously computed number of non-zero
        // elements in each block.
        parallel_nd(nblks, [&](dim_t b) {
            dim_t off = 0;
            if (b != 0) {
                for (dim_t i = 0; i < b; i++) {
                    off += nnz_per_blocks[i];
                }
            }
            output_offsets[b] = off;
        });

        // Use the calculated output_offsets and number of non-zero elements
        // per block to copy the non-zero elements that we moved to the
        // begining of the blocks to output_values.
        parallel_nd(nblks, [&](dim_t b) {
            const auto nnz_per_blk = nnz_per_blocks[b];
            const auto blk_off = output_offsets[b];
            for (dim_t i = 0; i < nnz_per_blk; i++) {
                output_values[blk_off + i] = wspace[b * blk_sz + i];
            }
        });

        return status::success;
    }
};

template <SIMPLE_SPARSE_REORDER_TEMPL_DECL, typename spec = void>
struct simple_sparse_reorder_t : public primitive_t {
    struct pd_t : public cpu_reorder_pd_t {
        using cpu_reorder_pd_t::cpu_reorder_pd_t;
        DECLARE_COMMON_PD_T("simple::any", simple_sparse_reorder_t);

        std::shared_ptr<primitive_desc_t> reorder_pd_;

    private:
        static status_t create(reorder_pd_t **reorder_pd, engine_t *engine,
                const primitive_attr_t *attr, engine_t *src_engine,
                const memory_desc_t *src_md, engine_t *dst_engine,
                const memory_desc_t *dst_md) {

            const bool args_ok = src_md->data_type == type_i
                    && dst_md->data_type == type_o
                    && simple_sparse_reorder_impl<
                            SIMPLE_SPARSE_REORDER_TEMPL_CALL>::
                            is_applicable(src_md, dst_md, attr);
            if (!args_ok) return status::invalid_arguments;

            auto _pd = make_unique_pd<pd_t>(attr, src_engine->kind(), src_md,
                    dst_engine->kind(), dst_md);
            if (_pd == nullptr) return status::out_of_memory;
            CHECK(_pd->init(engine, src_engine, dst_engine));

            CHECK(_pd->init_scratchpad_md());
            return safe_ptr_assign(*reorder_pd, _pd.release());
        }

        status_t init(
                engine_t *engine, engine_t *src_engine, engine_t *dst_engine) {
            // Convert sparse packed desc to blocking desc.
            auto converted_dst_md = cvt_sparse_packed2blocked(*this->dst_md());

            CHECK(reorder_primitive_desc_create(
                    reorder_pd_, engine, src_md(), &converted_dst_md, attr()));

            const size_t scratchpad_sz_ = simple_sparse_reorder_impl<
                    SIMPLE_SPARSE_REORDER_TEMPL_CALL>::
                    get_scratchpad_size(src_md(), dst_md());
            auto scratchpad = scratchpad_registry().registrar();
            scratchpad.book(memory_tracking::names::key_reorder_space,
                    scratchpad_sz_, 1, 16);
            scratchpad.book(memory_tracking::names::key_nested,
                    reorder_pd_->scratchpad_registry());
            return status::success;
        }

        friend dnnl::impl::impl_list_item_t;
    };

    simple_sparse_reorder_t(const pd_t *apd) : primitive_t(apd) {}

    status_t init(engine_t *engine) override {
        return pd()->reorder_pd_->create_primitive(reorder_, engine);
    }

    status_t execute(const exec_ctx_t &ctx) const override {
        return simple_sparse_reorder_impl<
                SIMPLE_SPARSE_REORDER_TEMPL_CALL>::execute(pd(), ctx, reorder_);
    }

private:
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }
    std::shared_ptr<primitive_t> reorder_;
};

#undef SIMPLE_SPARSE_REORDER_TEMPL_DECL
#undef SIMPLE_SPARSE_REORDER_TEMPL_CALL

} // namespace cpu
} // namespace impl
} // namespace dnnl
#endif
