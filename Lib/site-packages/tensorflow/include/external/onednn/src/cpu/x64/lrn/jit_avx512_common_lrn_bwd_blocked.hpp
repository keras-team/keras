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

#ifndef CPU_X64_LRN_JIT_AVX512_COMMON_LRN_BWD_BLOCKED_HPP
#define CPU_X64_LRN_JIT_AVX512_COMMON_LRN_BWD_BLOCKED_HPP

#include "cpu/x64/lrn/jit_avx512_common_lrn_bwd_base.hpp"
#include "cpu/x64/lrn/jit_avx512_common_lrn_utils.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {
namespace lrn {

using namespace dnnl::impl::status;
using namespace dnnl::impl::utils;
using namespace data_type;
using namespace Xbyak;
using namespace Xbyak::util;

template <data_type_t d_type>
class jit_avx512_common_lrn_kernel_bwd_blocked_t
    : public jit_avx512_common_lrn_kernel_bwd_t<d_type> {
public:
    using data_t = typename prec_traits<d_type>::type;

    struct jit_args_bwd_t {
        const data_t *src, *diff_dst, *ws0, *ws1;
        data_t *diff_src;
    };

    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_avx512_common_lrn_kernel_bwd_blocked_t)

    jit_avx512_common_lrn_kernel_bwd_blocked_t(const struct nChw16c_across_t &J,
            float alpha, float beta, int local_size, int use_h_parallel);

private:
    void generate() override;
    void compute_loop(int loop_size_param);

    int xmm_size_, zmm_size_, buffer_block_, buffer_nest_offset_,
            src_prev_offset_;
    int HW_, W_;
    across_version version_;

    const Reg64 hw_ = r10;

    const int xws1_prev_ = 3;
    const int xdiffdst_prev_ = 4;
    const int zws1_ = 3;

    const int xws1_next_ = 3;
    const int xdiffdst_next_ = 5;

    int use_h_parallelism_;
};

} // namespace lrn
} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif
