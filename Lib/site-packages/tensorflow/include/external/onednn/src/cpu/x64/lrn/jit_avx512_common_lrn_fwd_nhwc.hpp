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

#ifndef CPU_X64_LRN_JIT_AVX512_COMMON_LRN_FWD_NHWC_HPP
#define CPU_X64_LRN_JIT_AVX512_COMMON_LRN_FWD_NHWC_HPP

#include "cpu/x64/lrn/jit_avx512_common_lrn_fwd_base.hpp"
#include "cpu/x64/lrn/jit_avx512_common_lrn_utils.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {
namespace lrn {

template <data_type_t d_type>
class jit_avx512_common_lrn_kernel_fwd_nhwc_t
    : public jit_avx512_common_lrn_kernel_fwd_t<d_type> {
public:
    jit_avx512_common_lrn_kernel_fwd_nhwc_t(unsigned C, prop_kind_t prop_kind,
            float alpha, float beta, float k, int local_size);

    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_avx512_common_lrn_kernel_fwd_nhwc_t);

private:
    void generate() override;
    void set_up_ker_params();
    void execute_compute_loop(unsigned num_full_16c_blocks, unsigned C_tail);
    void compute_loop(across_version version, tail_mode tail_mode,
            unsigned C_tail = 0, int loop_size_param = 1);
    void compute(int loop_size_param);
    void increment_loop_params(std::size_t offset);
    void load_compute_data(
            across_version version, tail_mode tail_proc, int loop_size_param);
    void store_compute_data(
            int loop_size_param, tail_mode tail_mode, unsigned C_tail);
    void reserve_stack_space(std::size_t space);
    void unreserve_stack_space(std::size_t space);
    void load_data_to_stack(
            unsigned C_tail, across_version version, tail_mode tail_mode);

    const std::vector<int> tmp_mask_prev_;
    const std::vector<int> tmp_mask_next_;
    static constexpr int tmp_load_to_stack_idx_prev_ = 12;
    static constexpr int tmp_load_to_stack_idx_tail_ = 13;
    static constexpr int tmp_store_from_stack_idx_tail_ = 14;

    static constexpr int zmm_size = 64;
    const Reg64 mask_ = r10;
    const Reg64 blockC_ = r9;

    const int half_ls_;
    unsigned C;
};

} // namespace lrn
} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif
