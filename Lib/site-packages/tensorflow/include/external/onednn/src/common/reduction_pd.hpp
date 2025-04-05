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

#ifndef COMMON_REDUCTION_PD_HPP
#define COMMON_REDUCTION_PD_HPP

#include "c_types_map.hpp"
#include "primitive_desc.hpp"
#include "utils.hpp"

#define VDISPATCH_REDUCTION(cond, msg, ...) \
    VCONDCHECK(primitive, create, dispatch, reduction, (cond), \
            status::unimplemented, "%s," msg, this->info(engine), \
            ##__VA_ARGS__)

#define VDISPATCH_REDUCTION_SC(f, msg, ...) \
    VCHECK(primitive, create, dispatch, reduction, (f), "%s," msg, \
            this->info(engine), ##__VA_ARGS__)

namespace dnnl {
namespace impl {

status_t reduction_desc_init(reduction_desc_t *reduction_desc,
        alg_kind_t alg_kind, const memory_desc_t *src_desc,
        const memory_desc_t *dst_desc, float p, float eps);

struct reduction_pd_t : public primitive_desc_t {
    static constexpr auto base_pkind = primitive_kind::reduction;

    typedef reduction_pd_t hint_class;

    const reduction_desc_t *desc() const { return &desc_; }
    const op_desc_t *op_desc() const override {
        return reinterpret_cast<const op_desc_t *>(this->desc());
    }

    status_t query(query_t what, int idx, void *result) const override {
        switch (what) {
            case query::alg_kind:
                *(alg_kind_t *)result = desc()->alg_kind;
                break;
            case query::p_f32: *(float *)result = desc()->p; break;
            case query::epsilon_f32: *(float *)result = desc()->eps; break;
            default: return primitive_desc_t::query(what, idx, result);
        }
        return status::success;
    }

    arg_usage_t arg_usage(int arg) const override {
        switch (arg) {
            case DNNL_ARG_SRC: return arg_usage_t::input;
            case DNNL_ARG_DST: return arg_usage_t::output;
            default: return primitive_desc_t::arg_usage(arg);
        }
    }

    const memory_desc_t *arg_md(
            int arg, bool user_input = false) const override {
        switch (arg) {
            case DNNL_ARG_SRC: return src_md(0);
            case DNNL_ARG_DST: return dst_md(0, user_input);
            default: return primitive_desc_t::arg_md(arg);
        }
    }

    const memory_desc_t *src_md(
            int index = 0, bool user_input = false) const override {
        if (index == 0) return user_input ? &desc()->src_desc : &src_md_;
        return &glob_zero_md;
    }
    const memory_desc_t *dst_md(
            int index = 0, bool user_input = false) const override {
        if (index == 0) return user_input ? &desc()->dst_desc : &dst_md_;
        return &glob_zero_md;
    }

    int n_inputs() const override { return 1 + n_binary_po_inputs(); }
    int n_outputs() const override { return 1; }

    static void memory_desc_reduce_dim(memory_desc_t &md, int dim) {
        if (md.format_kind != format_kind::blocked) return;

        // Update reduced dim
        md.dims[dim] = 1;

        dims_t blocks = {0};
        memory_desc_wrapper(md).compute_blocks(blocks);

        // Reduced dim should be padded in case of inner blocks to preserve
        // layout
        md.padded_dims[dim] = blocks[dim];

        // Update strides of dimensions which depend on reduced dim
        int perm[DNNL_MAX_NDIMS];
        for (int i = 0; i < md.ndims; ++i)
            perm[i] = i;

        auto &blk_d = md.format_desc.blocking;

        dims_t strides;
        utils::array_copy(strides, blk_d.strides, md.ndims);

        // compute ou_dims. It is required to get correct perm
        dims_t ou_dims;
        for (int i = 0; i < md.ndims; ++i)
            ou_dims[i] = md.padded_dims[i] / blocks[i];

        utils::simultaneous_sort(strides, ou_dims, perm, md.ndims,
                [](stride_t a, stride_t b) { return a - b; });

        auto stride = md.padded_dims[dim] / blocks[dim] * blk_d.strides[dim];
        for (int _d = 0; _d < md.ndims; ++_d) {
            const auto d = perm[_d];
            if (strides[_d] > blk_d.strides[dim]) {
                blk_d.strides[d] = stride;
                stride *= md.padded_dims[d] / blocks[d];
            }
        }
    }

protected:
    reduction_desc_t desc_;

    memory_desc_t src_md_;
    memory_desc_t dst_md_;

    reduction_pd_t(const reduction_desc_t *adesc, const primitive_attr_t *attr,
            const hint_class *hint_fwd)
        : primitive_desc_t(attr, base_pkind)
        , desc_(*adesc)
        , src_md_(desc_.src_desc)
        , dst_md_(desc_.dst_desc) {}

    status_t set_default_params() {
        if (dst_md_.format_kind != format_kind::any) return status::success;

        return set_dst_format();
    }

    status_t set_dst_format() {
        memory_desc_t new_dst_md = src_md_;
        new_dst_md.data_type = dst_md_.data_type;
        for (int d = 0; d < src_md_.ndims; d++)
            if (src_md_.dims[d] != dst_md_.dims[d])
                memory_desc_reduce_dim(new_dst_md, d);
        dst_md_ = new_dst_md;

        return status::success;
    }
};

} // namespace impl
} // namespace dnnl

#endif
