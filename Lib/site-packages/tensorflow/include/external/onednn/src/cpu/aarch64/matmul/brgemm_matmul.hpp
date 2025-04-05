/*******************************************************************************
* Copyright 2021-2023 Intel Corporation
* Copyright 2024 FUJITSU LIMITED
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

#ifndef CPU_AARCH64_MATMUL_BRGEMM_MATMUL_HPP
#define CPU_AARCH64_MATMUL_BRGEMM_MATMUL_HPP

#include "common/c_types_map.hpp"
#include "common/primitive.hpp"
#include "common/type_helpers.hpp"

#include "cpu/matmul/cpu_matmul_pd.hpp"

#include "cpu/aarch64/brgemm/brgemm.hpp"
#include "cpu/aarch64/brgemm/brgemm_containers.hpp"
#include "cpu/aarch64/brgemm/brgemm_utils.hpp"
#include "cpu/aarch64/cpu_reducer.hpp"
#include "cpu/aarch64/matmul/brgemm_matmul_copy_utils.hpp"
#include "cpu/aarch64/matmul/brgemm_matmul_utils.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace aarch64 {
namespace matmul {

namespace {
constexpr int dynamic_m_tails[] = {32, 16, 8, 1};
constexpr int max_num_dynamic_m_tails
        = sizeof(dynamic_m_tails) / sizeof(dynamic_m_tails[0]);
constexpr int max_num_brg_kernels_matmul
        = 2 * 2 * 2 * 2 * (max_num_dynamic_m_tails + 1 /* main kernel size */);

inline int get_brg_kernel_index(const brgemm_matmul_conf_t &bgmmc,
        bool is_bs_tail, bool do_initialization, int m_ker_idx, bool is_N_tail,
        bool is_K_tail, int bs) {
    const int max_m_ker_idx
            = bgmmc.is_runtime_M ? max_num_dynamic_m_tails + 1 : 2;
    if (m_ker_idx >= max_m_ker_idx) return -1;

    auto vM = m_ker_idx > 0
            ? (bgmmc.is_runtime_M ? dynamic_m_tails[m_ker_idx - 1]
                                  : bgmmc.M_tail)
            : bgmmc.M_blk;
    auto vN = (is_N_tail) ? bgmmc.N_tail : bgmmc.N_blk;
    auto vK = (is_K_tail) ? bgmmc.K_tail : bgmmc.K_blk;
    if (vM == 0 || vN == 0 || vK == 0 || bs == 0 || bgmmc.LDA < vK
            || bgmmc.LDB < vN || bgmmc.LDC < vN)
        return -1;

    int idx = 16 * m_ker_idx + 8 * (int)is_bs_tail + 4 * (int)do_initialization
            + 2 * (int)is_N_tail + (int)is_K_tail;
    assert(idx < max_num_brg_kernels_matmul);
    return idx;
}

inline int get_brg_batchsize(
        const brgemm_matmul_conf_t &bgmmc, bool is_bs_tail, bool is_K_tail) {
    auto bs = is_K_tail  ? 1
            : is_bs_tail ? bgmmc.brgemm_batch_tail_size
                         : bgmmc.brgemm_batch_size;
    return bs;
}
} // namespace

template <cpu_isa_t isa>
struct brgemm_matmul_t : public primitive_t {
    struct pd_t : public ::dnnl::impl::cpu::matmul::cpu_matmul_pd_t {
        using ::dnnl::impl::cpu::matmul::cpu_matmul_pd_t::cpu_matmul_pd_t;

        DECLARE_COMMON_PD_T(
                JIT_IMPL_NAME_HELPER("brg:", isa, ""), brgemm_matmul_t);

        status_t init(engine_t *engine);
        int get_brg_kernel_idx(bool is_bs_tail, bool do_initialization,
                int m_ker_idx, bool is_N_tail, bool is_K_tail) const {
            int bs = get_brg_batchsize(bgmmc_, is_bs_tail, is_K_tail);
            return get_brg_kernel_index(bgmmc_, is_bs_tail, do_initialization,
                    m_ker_idx, is_N_tail, is_K_tail, bs);
        }
        const brgemm_t &get_brg_desc(int idx) const { return brg_descs_[idx]; }
        const brgemm_matmul_conf_t &get_brgemm_matmul_conf() const {
            return bgmmc_;
        }

    private:
        brgemm_t brg_descs_[max_num_brg_kernels_matmul];
        brgemm_matmul_conf_t bgmmc_;
    };

    brgemm_matmul_t(const pd_t *apd) : primitive_t(apd) {}

    status_t init(engine_t *engine) override;
    static constexpr data_type_t acc_type = data_type::s32;

    status_t execute(const exec_ctx_t &ctx) const override {
        return execute_body(ctx);
    }

private:
    struct brg_matmul_exec_ctx_t;

    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }
    status_t execute_body(const exec_ctx_t &ctx) const;
    void compute_kernel(const brg_matmul_exec_ctx_t &brgmm_ctx, int ithr,
            int b_idx, int m_blk_idx, int n_blk_idx, int k_blk_idx,
            bool do_init, int &prev_ker_idx) const;
    void copy_a_chunk_in_buffer(const brg_matmul_exec_ctx_t &brgmm_ctx,
            int ithr, int b_idx, int m_blk_idx, int k_blk_idx) const;
    void copy_b_chunk_in_buffer(const brg_matmul_exec_ctx_t &brgmm_ctx,
            int ithr, int b_idx, int n_blk_idx, int k_blk_idx) const;
    void maybe_reduce_partial_results_and_apply_postops(
            const brg_matmul_exec_ctx_t &brgmm_ctx) const;
    void accumulate(
            char *result_ptr, const char *reduce_ptr, size_t size) const;

    std::unique_ptr<brgemm_kernel_t> brg_kernels_[max_num_brg_kernels_matmul];

    std::unique_ptr<jit_brgemm_matmul_copy_b_t> copy_B_kernel_;
    std::unique_ptr<jit_brgemm_matmul_copy_a_t> copy_A_kernel_;
    std::unique_ptr<cpu_accumulator_1d_t<data_type::f32>> acc_ker_f32_;
    std::unique_ptr<cpu_accumulator_1d_t<data_type::s32>> acc_ker_s32_;
};

} // namespace matmul
} // namespace aarch64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif
