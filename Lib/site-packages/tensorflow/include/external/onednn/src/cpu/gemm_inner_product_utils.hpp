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

#ifndef CPU_GEMM_INNER_PRODUCT_UTILS_HPP
#define CPU_GEMM_INNER_PRODUCT_UTILS_HPP

#include "common/broadcast_strategy.hpp"
#include "common/c_types_map.hpp"
#include "common/type_helpers.hpp"
#include "common/utils.hpp"

#include "cpu/cpu_inner_product_pd.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace inner_product_utils {

struct pp_kernel_t {
    static pp_kernel_t *create(size_t OC, size_t MB, dim_t dst_mb_stride,
            const primitive_attr_t *attr, data_type_t bias_dt,
            data_type_t acc_dt, const memory_desc_t *dst_md, bool skip_sum);
    static pp_kernel_t *create(
            const cpu_inner_product_fwd_pd_t *pd, bool skip_sum) {
        return create(pd->OC(), pd->MB(), pd->OC(), pd->attr(),
                pd->desc()->bias_desc.data_type, pd->desc()->accum_data_type,
                pd->dst_md(), skip_sum);
    }

    virtual ~pp_kernel_t() = default;

    // mb kernel only supports single-threaded execution where performance
    // degradation is larger
    bool sequential_kernel() const { return mb_blk_kernel_; }

    virtual void operator()(void *dst, const void *acc, const char *bias,
            const float *scales, float dst_scale, size_t start,
            size_t dst_logical_off, size_t dim1_off, size_t end,
            size_t runtime_oc, dim_t dst_mb_stride,
            const float *dst_zero_points,
            const void *post_ops_binary_rhs_arg_vec, const void *dst_orig,
            size_t first_mb_matrix_addr_off, const exec_ctx_t &ctx,
            const memory_desc_t &dst_md) const = 0;

    virtual status_t create_kernel() { return status::success; }

protected:
    pp_kernel_t(size_t OC, size_t MB, dim_t dst_mb_stride,
            const primitive_attr_t *attr, data_type_t bias_dt,
            data_type_t acc_dt, const memory_desc_t *dst_md, bool skip_sum);

    size_t OC_;
    size_t MB_;
    dim_t dst_mb_stride_;
    data_type_t bias_data_type_;
    data_type_t acc_data_type_;
    data_type_t dst_data_type_;
    size_t bias_data_type_size_ = 0;
    size_t acc_data_type_size_ = 4;
    size_t dst_data_type_size_ = 0;
    bool do_scale_ = false;
    size_t scale_idx_mult_ = 0;
    bool do_eltwise_ = false;
    bool do_binary_ = false;
    bool do_prelu_ = false;
    bool do_sum_ = false;
    bool do_dst_scale_ = false;
    bool do_dst_zero_points_ = false;
    float sum_scale_ = 0.f;
    int32_t sum_zp_ = 0;
    data_type_t sum_data_type_;
    bool mb_blk_kernel_ = false;
    post_ops_t post_ops_;
    int ndims_;

    bool has_trivial_mb_stride() const {
        return (!runtime_oc()) && (OC_ == (size_t)dst_mb_stride_);
    }
    bool do_bias() const { return bias_data_type_ != data_type::undef; }
    bool runtime_oc() const { return OC_ == (size_t)DNNL_RUNTIME_DIM_VAL; }
    bool runtime_mb() const { return MB_ == (size_t)DNNL_RUNTIME_DIM_VAL; }
};

inline const bcast_set_t &gemm_default_strategies() {
    static const bcast_set_t s
            = {broadcasting_strategy_t::scalar, broadcasting_strategy_t::per_oc,
                    broadcasting_strategy_t::per_oc_spatial,
                    broadcasting_strategy_t::no_broadcast};
    return s;
}

bool post_ops_ok(const post_ops_t &post_ops, const memory_desc_wrapper *dst_d,
        const bcast_set_t &enabled_bcast_strategy = gemm_default_strategies());
bool post_ops_ok(const post_ops_t &post_ops, const memory_desc_t *dst_d,
        const bcast_set_t &enabled_bcast_strategy = gemm_default_strategies());

} // namespace inner_product_utils
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif
