/*******************************************************************************
* Copyright 2019-2023 Intel Corporation
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

#ifndef COMMON_GEMM_UTILS_HPP
#define COMMON_GEMM_UTILS_HPP

#include "oneapi/dnnl/dnnl.h"

#include "common/c_types_map.hpp"
#include "common/nstl.hpp"
#include "common/primitive_desc_iterator.hpp"
#include "common/utils.hpp"

namespace dnnl {
namespace impl {

static inline status_t check_gemm_input(char transa, char transb, int m, int n,
        int k, int lda, int ldb, int ldc, float alpha, float beta) {
    using namespace status;
    bool consistency = true && utils::one_of(transa, 'T', 't', 'N', 'n')
            && utils::one_of(transb, 'T', 't', 'N', 'n') && m >= 0 && n >= 0
            && k >= 0;
    if (!consistency) return invalid_arguments;
    bool isTransA = utils::one_of(transa, 'T', 't');
    bool isTransB = utils::one_of(transb, 'T', 't');
    int nrowA = isTransA ? k : m;
    int nrowB = isTransB ? n : k;
    consistency = true && lda >= nstl::max(1, nrowA)
            && ldb >= nstl::max(1, nrowB) && ldc >= nstl::max(1, m);
    if (!consistency) return invalid_arguments;

    return success;
}

static inline status_t check_gemm_x8x8s32_input(char offsetc, char transa,
        char transb, int m, int n, int k, int lda, int ldb, int ldc,
        float alpha, float beta) {
    using namespace status;
    if (!utils::one_of(offsetc, 'F', 'f', 'C', 'c', 'R', 'r'))
        return invalid_arguments;
    return check_gemm_input(
            transa, transb, m, n, k, lda, ldb, ldc, alpha, beta);
}

// This function makes a 2d tensor from an nd tensor.
// the 2d tensor just collapes dims[1...ndims-1] from the nd tensor
// The only reason we do not use reshape here is that we want to allow
// fusing blocked dimensions and padded dimensions.
static inline status_t init_2d_desc(memory_desc_t *md_2d,
        const memory_desc_t *md_nd, bool transpose_dims = false) {
    auto p_dims = md_nd->padded_dims;
    auto blk = md_nd->format_desc.blocking;
    auto strides = blk.strides;

    // we assume that the innermost dimension always has stride 1
    assert(IMPLICATION(blk.inner_nblks == 0,
            utils::array_min(strides, md_nd->ndims) == 1));

    // TODO: add checks to see if the memory descriptor can be 2d-fied
    // TODO: change signature to specifiy at which dimension shall we 2d-fy (currently 1st)
    auto p_dim1 = utils::array_product(p_dims + 1, md_nd->ndims - 1);
    auto stride1 = blk.inner_nblks == 0
            ? utils::array_min(strides + 1, md_nd->ndims - 1)
            : 1;

    if (transpose_dims) {
        dnnl_dims_t dims_2d = {p_dim1, p_dims[0]};
        dnnl_dims_t strides_2d = {stride1, strides[0]};
        return memory_desc_init_by_strides(
                *md_2d, 2, dims_2d, md_nd->data_type, strides_2d);
    } else {
        dnnl_dims_t dims_2d = {p_dims[0], p_dim1};
        dnnl_dims_t strides_2d = {strides[0], stride1};
        return memory_desc_init_by_strides(
                *md_2d, 2, dims_2d, md_nd->data_type, strides_2d);
    }
}

static inline status_t create_2d_desc(memory_desc_t *md_2d, int d0, int d1,
        data_type_t dt, transpose_t trans, int ld) {
    dnnl_dims_t dims_2d = {d0, d1};
    if (trans == transpose::notrans) {
        dnnl_dims_t strides_2d = {ld, 1};
        return memory_desc_init_by_strides(*md_2d, 2, dims_2d, dt, strides_2d);
    } else {
        dnnl_dims_t strides_2d = {1, ld};
        return memory_desc_init_by_strides(*md_2d, 2, dims_2d, dt, strides_2d);
    }
}

static inline gemm_desc_t create_gemm_desc(const memory_desc_t *a_md,
        const memory_desc_t *b_md, const memory_desc_t *c_md,
        const memory_desc_t *bias_md, data_type_t acc_dt, engine_t *engine,
        sum_ab_t sum_ab = sum_ab::sum_none,
        data_type_t sum_ab_dt = data_type::undef) {
    auto gemm_desc = gemm_desc_t();
    gemm_desc.primitive_kind = primitive_kind::gemm;
    gemm_desc.a_desc = *a_md;
    gemm_desc.b_desc = *b_md;
    gemm_desc.c_desc = *c_md;
    gemm_desc.bias_desc = *bias_md;
    gemm_desc.acc_type = acc_dt;
    gemm_desc.sum_ab = sum_ab;
    gemm_desc.sum_ab_type = sum_ab_dt;
    // Downgrade accumulation type for f16 if allowed.
    if (engine->mayiuse_f16_accumulator_with_f16()
            && utils::everyone_is(
                    data_type::f16, a_md->data_type, b_md->data_type)) {
        gemm_desc.acc_type = data_type::f16;
    }
    return gemm_desc;
}

static inline status_t create_gemm_pd(
        std::shared_ptr<primitive_desc_t> &gemm_pd_, engine_t *engine,
        const memory_desc_t *a_md, const memory_desc_t *b_md,
        const memory_desc_t *c_md, const memory_desc_t *bias_md,
        data_type_t acc_dt, const primitive_attr_t *attr, bool skip_ref = false,
        sum_ab_t sum_ab = sum_ab::sum_none,
        data_type_t sum_ab_dt = data_type::undef) {
    auto gemm_desc = create_gemm_desc(
            a_md, b_md, c_md, bias_md, acc_dt, engine, sum_ab, sum_ab_dt);

    primitive_attr_t gemm_attr = *attr;

    primitive_desc_iterator_t it(
            engine, (op_desc_t *)&gemm_desc, &gemm_attr, nullptr);

    gemm_pd_ = *(++it);
    if (!gemm_pd_) return status::unimplemented;
    if (skip_ref && strstr(gemm_pd_.get()->name(), "ref") != NULL)
        return status::unimplemented;

    return status::success;
}

static inline bool is_md_gemm_compatible_plain_format(
        const memory_desc_t *md, bool is_dst = false) {

    if (md->format_kind != format_kind::blocked) return false;

    auto &blk_desc = md->format_desc.blocking;

    if (blk_desc.inner_nblks != 0) return false;

    return (blk_desc.strides[md->ndims - 1] == 1)
            || (!is_dst && blk_desc.strides[md->ndims - 2] == 1);
}

} // namespace impl
} // namespace dnnl

#endif
