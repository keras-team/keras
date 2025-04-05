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

#ifndef COMMON_GEMM_PD_HPP
#define COMMON_GEMM_PD_HPP

#include "oneapi/dnnl/dnnl.h"

#include "common/c_types_map.hpp"
#include "common/gemm_utils.hpp"
#include "common/primitive_desc.hpp"
#include "common/utils.hpp"

namespace dnnl {
namespace impl {

#define VDISPATCH_GEMM(cond, msg, ...) \
    VCONDCHECK(primitive, create, dispatch, gemm, (cond), \
            status::unimplemented, "%s," msg, this->info(engine), \
            ##__VA_ARGS__)

#define VDISPATCH_GEMM_SC(f, msg, ...) \
    VCHECK(primitive, create, dispatch, gemm, (f), "%s," msg, \
            this->info(engine), ##__VA_ARGS__)

struct gemm_pd_t : public primitive_desc_t {
    static constexpr auto base_pkind = primitive_kind::gemm;

    typedef gemm_pd_t base_class;
    typedef gemm_pd_t hint_class;

    const gemm_desc_t *desc() const { return &desc_; }
    const op_desc_t *op_desc() const override {
        return reinterpret_cast<const op_desc_t *>(this->desc());
    }

    arg_usage_t arg_usage(int arg) const override {
        if (utils::one_of(arg, DNNL_ARG_SRC_0, DNNL_ARG_SRC_1))
            return arg_usage_t::input;

        if (arg == DNNL_ARG_DST) return arg_usage_t::output;

        return primitive_desc_t::arg_usage(arg);
    }

    const memory_desc_t *arg_md(
            int arg, bool user_input = false) const override {
        switch (arg) {
            case DNNL_ARG_SRC_0: return src_md(0);
            case DNNL_ARG_SRC_1: return src_md(1);
            case DNNL_ARG_BIAS: return src_md(2);
            case DNNL_ARG_DST: return dst_md(0, user_input);
            default: return primitive_desc_t::arg_md(arg);
        }
    }

    const memory_desc_t *src_md(
            int index = 0, bool user_input = false) const override {
        switch (index) {
            case 0: return &desc_.a_desc;
            case 1: return &desc_.b_desc;
            case 2: return &desc_.bias_desc;
            default: return &glob_zero_md;
        }
    }
    const memory_desc_t *dst_md(
            int index = 0, bool user_input = false) const override {
        return index == 0 ? &desc_.c_desc : &glob_zero_md;
    }
    bool with_bias() const { return desc_.bias_desc.ndims != 0; }

    int n_inputs() const override { return 2; }
    int n_outputs() const override { return 1; }

protected:
    // Note: we do not copy memory desc locally to avoid
    // overheads. This means we lose the users memory descs when we
    // resolve the 'any' tags.
    gemm_desc_t desc_;

    gemm_pd_t(const gemm_desc_t *adesc, const primitive_attr_t *attr,
            const hint_class *hint_fwd_pd)
        : primitive_desc_t(attr, base_pkind), desc_(*adesc) {}

    // By default, we just resolve 'any' with blocked layout and trivial strides
    bool set_default_format(memory_desc_t *md) {
        memory_desc_wrapper mdw(md);
        if (mdw.format_any()) {
            if (mdw.has_runtime_dims_or_strides()) return false;
            status_t status = memory_desc_init_by_strides(*md, nullptr);
            if (status != status::success) return false;
        }

        return true;
    }

    bool set_default_formats() {
        bool ok = true;

        for (auto md : {&desc_.a_desc, &desc_.b_desc, &desc_.bias_desc,
                     &desc_.c_desc}) {
            ok = ok && set_default_format(md);
        }

        auto status = attr_.post_ops_.set_default_formats(&desc_.c_desc);
        ok = ok && (status == status::success);

        return ok;
    }
};

} // namespace impl
} // namespace dnnl

#endif
