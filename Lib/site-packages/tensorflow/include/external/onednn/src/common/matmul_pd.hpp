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

#ifndef COMMON_MATMUL_PD_HPP
#define COMMON_MATMUL_PD_HPP

#include <assert.h>

#include "oneapi/dnnl/dnnl.h"

#include "c_types_map.hpp"
#include "primitive_desc.hpp"
#include "utils.hpp"

#define VDISPATCH_MATMUL(cond, msg, ...) \
    VCONDCHECK(primitive, create, dispatch, matmul, (cond), \
            status::unimplemented, "%s," msg, this->info(engine), \
            ##__VA_ARGS__)

#define VDISPATCH_MATMUL_SC(f, msg, ...) \
    VCHECK(primitive, create, dispatch, matmul, f, msg, ##__VA_ARGS__);

namespace dnnl {
namespace impl {

status_t matmul_desc_init(matmul_desc_t *matmul_desc,
        const memory_desc_t *src_desc, const memory_desc_t *weights_desc,
        const memory_desc_t *bias_desc, const memory_desc_t *dst_desc);

struct matmul_pd_t : public primitive_desc_t {
    static constexpr auto base_pkind = primitive_kind::matmul;

    typedef matmul_pd_t base_class;
    typedef matmul_pd_t hint_class;

    const matmul_desc_t *desc() const { return &desc_; }
    const op_desc_t *op_desc() const override {
        return reinterpret_cast<const op_desc_t *>(this->desc());
    }

    arg_usage_t arg_usage(int arg) const override {
        const bool input = utils::one_of(arg, DNNL_ARG_SRC, DNNL_ARG_WEIGHTS);
        if (input) return arg_usage_t::input;

        if (arg == DNNL_ARG_BIAS && with_bias()) return arg_usage_t::input;

        if (arg == DNNL_ARG_DST) return arg_usage_t::output;

        return primitive_desc_t::arg_usage(arg);
    }

    const memory_desc_t *arg_md(
            int arg, bool user_input = false) const override {
        switch (arg) {
            case DNNL_ARG_SRC: return src_md(0);
            case DNNL_ARG_WEIGHTS: return weights_md(0);
            case DNNL_ARG_BIAS: return weights_md(1);
            case DNNL_ARG_DST: return dst_md(0, user_input);
            default: return primitive_desc_t::arg_md(arg);
        }
    }

    const memory_desc_t *src_md(
            int index = 0, bool user_input = false) const override {
        if (index == 0) return user_input ? &desc()->src_desc : &src_md_;
        return &glob_zero_md;
    }

    const memory_desc_t *weights_md(
            int index = 0, bool user_input = false) const override {
        if (index == 0)
            return user_input ? &desc()->weights_desc : &weights_md_;
        if (index == 1) return user_input ? &desc()->bias_desc : &bias_md_;
        return &glob_zero_md;
    }

    const memory_desc_t *dst_md(
            int index = 0, bool user_input = false) const override {
        if (index == 0) return user_input ? &desc()->dst_desc : &dst_md_;
        return &glob_zero_md;
    }

    int n_inputs() const override {
        return 2 + with_bias() + n_binary_po_inputs() + n_prelu_po_inputs();
    }
    int n_outputs() const override { return 1; }

    bool has_zero_dim_memory() const {
        return memory_desc_wrapper(src_md(0)).has_zero_dim()
                || memory_desc_wrapper(weights_md(0)).has_zero_dim()
                || memory_desc_wrapper(dst_md(0)).has_zero_dim();
    }

    int ndims() const { return dst_md_.ndims; }

    dim_t ldc() const {
        return memory_desc_wrapper(dst_md(0))
                .blocking_desc()
                .strides[ndims() - 2];
    }

    bool with_bias() const { return bias_md_.ndims != 0; }
    bool batched() const { return ndims() > 2; }

    dim_t batch() const {
        return utils::array_product(dst_md_.dims, ndims() - 2);
    }
    dim_t M() const { return dst_md_.dims[ndims() - 2]; }
    dim_t N() const { return dst_md_.dims[ndims() - 1]; }
    dim_t K() const { return src_md_.dims[ndims() - 1]; }

    bool is_bias_1xN() const {
        if (!with_bias()) return false;

        const auto &dims = weights_md(1)->dims;
        const int n_dims = ndims();
        for (int i = 0; i < n_dims - 1; ++i) {
            if (dims[i] != 1) return false;
        }

        return dims[n_dims - 1] == N();
    }

    // Quantization mask frequently used for weights scales and zero points
    int wei_qmask_N() const {
        const int wei_ndims = weights_md(0)->ndims;
        assert(wei_ndims >= 2);
        return 1 << (wei_ndims - 1);
    }

    int wei_qmask_K() const {
        const int wei_ndims = weights_md(0)->ndims;
        assert(wei_ndims >= 2);
        return 1 << (wei_ndims - 2);
    }

    virtual bool attr_scales_ok(const std::vector<int> &supported_args
            = {DNNL_ARG_SRC, DNNL_ARG_WEIGHTS, DNNL_ARG_DST}) const {
        bool ok = attr()->scales_.has_default_values(supported_args);
        for (int arg : supported_args) {
            const auto &mask = attr()->scales_.get(arg).mask_;
            if (arg == DNNL_ARG_WEIGHTS) {
                const auto &sc = attr()->scales_.get(arg);
                ok = ok
                        && utils::one_of(mask, 0, wei_qmask_N(),
                                wei_qmask_K() + wei_qmask_N());
                ok = ok && utils::one_of(sc.ndims_, 0, 2)
                        && IMPLICATION(sc.ndims_ == 2,
                                sc.group_dims_[1] == 1
                                        && K() % sc.group_dims_[0] == 0);
            } else
                ok = ok && (mask == 0);
        }
        return ok;
    }

protected:
    matmul_desc_t desc_;

    memory_desc_t src_md_;
    memory_desc_t weights_md_;
    memory_desc_t bias_md_;
    memory_desc_t dst_md_;

    matmul_pd_t(const matmul_desc_t *adesc, const primitive_attr_t *attr,
            const matmul_pd_t *hint_fwd_pd)
        : primitive_desc_t(attr, base_pkind)
        , desc_(*adesc)
        , src_md_(desc_.src_desc)
        , weights_md_(desc_.weights_desc)
        , bias_md_(desc_.bias_desc)
        , dst_md_(desc_.dst_desc) {}

    // temporary solution to deal with format `any`
    bool set_default_formats() {
        for (auto md : {&src_md_, &weights_md_, &bias_md_, &dst_md_}) {
            memory_desc_wrapper mdw(md);
            if (mdw.format_any()) {
                if (mdw.has_runtime_dims_or_strides()) return false;
                status_t status = memory_desc_init_by_strides(*md, nullptr);
                if (status != status::success) return false;
            }
        }

        return true;
    }

    // All implementations that do not support sparse inputs/outputs should
    // call this function.
    bool is_dense_format_kind() {
        return impl::is_dense_format_kind(
                {&src_md_, &weights_md_, &bias_md_, &dst_md_});
    }
};

} // namespace impl
} // namespace dnnl

#endif
