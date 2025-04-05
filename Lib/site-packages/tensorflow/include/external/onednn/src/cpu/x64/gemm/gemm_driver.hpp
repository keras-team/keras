/*******************************************************************************
* Copyright 2018-2020 Intel Corporation
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

#ifndef CPU_X64_GEMM_GEMM_DRIVER_HPP
#define CPU_X64_GEMM_GEMM_DRIVER_HPP

#include "oneapi/dnnl/dnnl_types.h"

#include "common/c_types_map.hpp"

#include "cpu/x64/gemm/gemm_info.hpp"
#include "cpu/x64/gemm/gemm_pack_storage.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {

template <typename a_type, typename b_type, typename c_type>
dnnl_status_t gemm_driver(const char *transA, const char *transB,
        const char *offsetC, const dim_t *m, const dim_t *n, const dim_t *k,
        const float *alpha, const a_type *a, const dim_t *lda, const a_type *oa,
        const b_type *b, const dim_t *ldb, const b_type *ob, const float *beta,
        c_type *c, const dim_t *ldc, const c_type *oc,
        const bool force_jit_nocopy_gemm, pack_type packing = pack_type::none,
        gemm_pack_storage_t *pack_dst = NULL, bool measure_only = false);

void prep_ref_gemm_s8u8s32_pack(
        bool do_a, dim_t rows, dim_t cols, gemm_pack_storage_t *pack_dst);

dnnl_status_t ref_gemm_s8u8s32_pack(const void *src, dim_t ld_src, dim_t rows,
        dim_t cols, int trans, gemm_pack_storage_t *dst_pack);

} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif // CPU_X64_GEMM_GEMM_DRIVER_HPP
