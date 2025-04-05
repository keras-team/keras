/*******************************************************************************
* Copyright 2020 Intel Corporation
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

#ifndef CPU_GEMM_GEMM_MSAN_UNPOISON_HPP
#define CPU_GEMM_GEMM_MSAN_UNPOISON_HPP

namespace dnnl {
namespace impl {
namespace cpu {

inline void msan_unpoison_matrix(
        void *C, dim_t M, dim_t N, dim_t LDC, size_t typesize) {
    assert(C != nullptr && LDC >= M && typesize);
    if (msan_enabled && C != nullptr) {
        if ((M <= 0) || (N <= 0)) return;

        size_t col_size = M * typesize;
        size_t col_stride = LDC * typesize;
        uint8_t *col = (uint8_t *)C;
        for (dim_t j = 0; j < N; j++) {
            msan_unpoison(col, col_size);
            col += col_stride;
        }
    }
}

} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif
