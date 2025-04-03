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

#ifndef CPU_PRIMITIVE_ATTR_POSTOPS_HPP
#define CPU_PRIMITIVE_ATTR_POSTOPS_HPP

#include <vector>

#include "common/primitive.hpp"
#include "common/primitive_attr.hpp"

namespace dnnl {
namespace impl {
namespace cpu {

float compute_binary_scalar(alg_kind_t alg, float x, float y);
float compute_eltwise_scalar_fwd(
        const alg_kind_t alg, float s, float alpha, float beta);
float compute_eltwise_scalar_bwd(
        const alg_kind_t alg, float dd, float s, float alpha, float beta);

struct ref_binary_scalar_t {
    ref_binary_scalar_t(alg_kind_t alg);
    ref_binary_scalar_t(const post_ops_t::entry_t::binary_t &binary);

    float compute_scalar(float src0, float src1) const;

private:
    const alg_kind_t alg_;
};

struct ref_eltwise_scalar_fwd_t {
    ref_eltwise_scalar_fwd_t(
            alg_kind_t alg, float alpha, float beta, float scale);
    ref_eltwise_scalar_fwd_t(const post_ops_t::entry_t::eltwise_t &eltwise);

    float compute_scalar(float s) const;

    const alg_kind_t alg_;
    const float alpha_;
    const float beta_;
    const float scale_;
};

struct ref_post_ops_t {
    struct args_t {
        args_t() : dst_val(0.f), ctx(nullptr), l_offset(-1), dst_md(nullptr) {}

        float dst_val; // sum arg
        const exec_ctx_t *ctx; // binary arg
        dim_t l_offset; // binary arg
        const memory_desc_t *dst_md; // binary arg
    };

    ref_post_ops_t(const post_ops_t &po, bool skip_sum = false);

    virtual ~ref_post_ops_t() = default;

    status_t init(const memory_desc_t *dst_md);

    void execute(float &res, const args_t &args = args_t()) const;

    static bool primitive_kind_ok(const post_ops_t &po) {
        using namespace primitive_kind;
        return po.has_default_values({binary, eltwise, prelu, sum});
    }

private:
    const post_ops_t &po_;
    // some primitives for example gemm are able to perform sum postop itself,
    // in such cases executing sum should be skipped
    const bool skip_sum_;

    std::vector<ref_eltwise_scalar_fwd_t> eltwise_po_;
    std::vector<ref_binary_scalar_t> binary_po_;
    std::vector<memory_desc_t> prelu_md_;
};

} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif
