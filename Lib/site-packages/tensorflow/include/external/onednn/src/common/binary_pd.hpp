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

#ifndef COMMON_BINARY_PD_HPP
#define COMMON_BINARY_PD_HPP

#include <assert.h>

#include "oneapi/dnnl/dnnl.h"

#include "c_types_map.hpp"
#include "primitive_desc.hpp"
#include "utils.hpp"

#define VDISPATCH_BINARY(cond, msg, ...) \
    VCONDCHECK(primitive, create, dispatch, binary, (cond), \
            status::unimplemented, "%s," msg, this->info(engine), \
            ##__VA_ARGS__)

#define VDISPATCH_BINARY_SC(f, msg, ...) \
    VCHECK(primitive, create, dispatch, binary, (f), "%s," msg, \
            this->info(engine), ##__VA_ARGS__)

namespace dnnl {
namespace impl {

struct binary_pd_t : public primitive_desc_t {
    static constexpr auto base_pkind = primitive_kind::binary;

    typedef binary_pd_t base_class;
    typedef binary_pd_t hint_class;

    const binary_desc_t *desc() const { return &desc_; }
    const op_desc_t *op_desc() const override {
        return reinterpret_cast<const op_desc_t *>(this->desc());
    }

    status_t query(query_t what, int idx, void *result) const override {
        switch (what) {
            case query::alg_kind:
                *(alg_kind_t *)result = desc()->alg_kind;
                break;
            default: return primitive_desc_t::query(what, idx, result);
        }
        return status::success;
    }

    arg_usage_t arg_usage(int arg) const override {
        if (arg == DNNL_ARG_SRC_0 || arg == DNNL_ARG_SRC_1)
            return arg_usage_t::input;

        if (arg == DNNL_ARG_DST) return arg_usage_t::output;

        return primitive_desc_t::arg_usage(arg);
    }

    const memory_desc_t *arg_md(
            int arg, bool user_input = false) const override {
        switch (arg) {
            case DNNL_ARG_SRC_0: return src_md(0);
            case DNNL_ARG_SRC_1: return src_md(1);
            case DNNL_ARG_DST: return dst_md(0, user_input);
            default: return primitive_desc_t::arg_md(arg);
        }
    }

    const memory_desc_t *src_md(
            int index = 0, bool user_input = false) const override {
        if (index == 0) return user_input ? &desc()->src_desc[0] : &src0_md_;
        if (index == 1) return user_input ? &desc()->src_desc[1] : &src1_md_;
        return &glob_zero_md;
    }
    const memory_desc_t *dst_md(
            int index = 0, bool user_input = false) const override {
        if (index == 0) return user_input ? &desc()->dst_desc : &dst_md_;
        return &glob_zero_md;
    }

    int n_inputs() const override { return 2 + n_binary_po_inputs(); }
    int n_outputs() const override { return 1; }

    const dims_t &broadcast_dims() const { return broadcast_dims_; }

    bool has_zero_dim_memory() const {
        return memory_desc_wrapper(src_md(0)).has_zero_dim();
    }

    int ndims() const { return memory_desc_wrapper(src_md(0)).ndims(); }

    bool is_tensor_op() const {
        const memory_desc_wrapper src0_d(src_md(0));
        const memory_desc_wrapper src1_d(src_md(1));
        return src0_d.consistent_with(src1_d);
    }

protected:
    binary_desc_t desc_;

    memory_desc_t src0_md_;
    memory_desc_t src1_md_;
    memory_desc_t dst_md_;

    dims_t broadcast_dims_;

    binary_pd_t(const binary_desc_t *adesc, const primitive_attr_t *attr,
            const binary_pd_t *hint_fwd_pd)
        : primitive_desc_t(attr, base_pkind)
        , desc_(*adesc)
        , src0_md_(desc_.src_desc[0])
        , src1_md_(desc_.src_desc[1])
        , dst_md_(desc_.dst_desc) {
        init_broadcast_dims();
    }

    status_t set_default_params() {
        if (src1_md_.format_kind == format_kind::any) {
            const memory_desc_wrapper src_d(src_md(0));
            if (src_d.is_blocking_desc()) {
                CHECK(memory_desc_init_by_blocking_desc(
                        src1_md_, src_d.blocking_desc()));
            }
        }

        if (dst_md_.format_kind == format_kind::any) {
            const memory_desc_wrapper src_d(src_md(0));
            if (src_d.is_blocking_desc()) {
                CHECK(memory_desc_init_by_blocking_desc(
                        dst_md_, src_d.blocking_desc()));
            }
        }

        return status::success;
    }

    bool attr_post_ops_ok() const {
        using namespace primitive_kind;
        const auto &p = attr()->post_ops_;
        switch (p.len()) {
            case 0: return true;
            case 1: return p.contain(sum, 0) || p.contain(eltwise, 0);
            case 2: return p.contain(sum, 0) && p.contain(eltwise, 1);
            default: return false;
        }
    }

    bool attr_scales_ok(const std::vector<int> &supported_args
            = {DNNL_ARG_SRC_0, DNNL_ARG_SRC_1, DNNL_ARG_DST}) const {
        bool ok = attr()->scales_.has_default_values(supported_args);
        for (int arg : supported_args) {
            const auto &mask = attr()->scales_.get(arg).mask_;
            ok = ok && (mask == 0);
        }
        return ok;
    }

private:
    void init_broadcast_dims() {
        const dims_t &dims_A = src_md(0)->dims;
        const dims_t &dims_B = src_md(1)->dims;

        for (int d = 0; d < ndims(); ++d)
            broadcast_dims_[d]
                    = (dims_A[d] == dims_B[d] && dims_A[d] != 1) ? 0 : 1;
    }
};

} // namespace impl
} // namespace dnnl

#endif
