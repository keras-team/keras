/*******************************************************************************
* Copyright 2021-2024 Intel Corporation
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

#ifndef CPU_X64_RNN_BRGEMM_UTILS_RNN_HPP
#define CPU_X64_RNN_BRGEMM_UTILS_RNN_HPP

#include <memory>
#include "common/c_types_map.hpp"
#include "common/memory_tracking.hpp"
#include "cpu/x64/brgemm/brgemm.hpp"
#include "cpu/x64/jit_brgemm_transpose_utils.hpp"
#include "cpu/x64/matmul/brgemm_matmul_copy_utils.hpp"
#include "cpu/x64/rnn/jit_brgemm_transpose_single_row.hpp"
#include "cpu/x64/rnn/jit_diff_weights_peephole.hpp"
#include "cpu/x64/rnn/jit_gates_reduction.hpp"

namespace dnnl {
namespace impl {
namespace cpu {

namespace rnn_utils {
struct rnn_conf_t;
}

namespace x64 {

struct jit_brgemm_trans_src_t;

namespace rnn_brgemm_utils {

using brgemm_ker_ptr_t = std::unique_ptr<brgemm_kernel_t>;
using brgemm_pallete_t = char[64];
using srcatch_gates_reorder_ker_ptr_t
        = std::unique_ptr<matmul::jit_brgemm_matmul_copy_b_t>;

struct rnn_brgemm_base_t {
    static void init_scratchpad(const cpu::rnn_utils::rnn_conf_t &rnn,
            memory_tracking::registrar_t &scratchpad, dim_t gemm_acc_type_size,
            dim_t gemm_acc_align);
    static constexpr dim_t num_base_kernels_ = 3;
    static constexpr dim_t num_proj_kernels_ = 4;
    static constexpr dim_t num_vanilla_gru_iter_part2_kernels_ = 4;
};

template <prop_kind_t aprop>
struct rnn_brgemm_t;

template <>
struct rnn_brgemm_t<prop_kind::forward> : public rnn_brgemm_base_t {
    using rnn_brgemm_base_t::init_scratchpad;

    static status_t configure_brgemm(cpu::rnn_utils::rnn_conf_t &rnn,
            alg_kind_t cell_kind, dim_t src_layer_type_size,
            dim_t scratch_type_size);
    status_t init_kernels(const cpu::rnn_utils::rnn_conf_t &rnn,
            data_type_t src_type, data_type_t weights_type);

    brgemm_desc_t desc_layer_b0_[num_base_kernels_];
    brgemm_desc_t desc_iter_b0_[num_base_kernels_];
    brgemm_desc_t desc_iter_b1_[num_base_kernels_];
    brgemm_desc_t desc_layer_N_tail_b0_[num_base_kernels_];
    brgemm_desc_t desc_iter_N_tail_b0_[num_base_kernels_];
    brgemm_desc_t desc_iter_N_tail_b1_[num_base_kernels_];

    brgemm_desc_t desc_layer_K1_tail_b1_[num_base_kernels_];
    brgemm_desc_t desc_layer_NK1_tail_b1_[num_base_kernels_];
    brgemm_desc_t desc_iter_K2_tail_b1_[num_base_kernels_];
    brgemm_desc_t desc_iter_NK2_tail_b1_[num_base_kernels_];

    brgemm_desc_t desc_layermerged_b0_[num_base_kernels_];
    brgemm_desc_t desc_layermerged_N_tail_b0_[num_base_kernels_];
    brgemm_desc_t desc_layermerged_K1_tail_b1_[num_base_kernels_];
    brgemm_desc_t desc_layermerged_NK1_tail_b1_[num_base_kernels_];

    brgemm_desc_t desc_proj_b0_[num_proj_kernels_];
    brgemm_desc_t desc_proj_N_tail_b0_[num_proj_kernels_];
    brgemm_desc_t desc_proj_N_tail_b1_[num_proj_kernels_];
    brgemm_desc_t desc_proj_K_tail_b1_[num_proj_kernels_];
    brgemm_desc_t desc_proj_NK_tail_b1_[num_proj_kernels_];

    // Set of brgemm descriptor for 2nd part of iteration gemm in vanulla GRU
    // cell
    brgemm_desc_t desc_iter_p2_b1_[num_vanilla_gru_iter_part2_kernels_];
    brgemm_desc_t desc_iter_p2_N_tail_b1_[num_vanilla_gru_iter_part2_kernels_];
    brgemm_desc_t desc_iter_p2_K2_tail_b1_[num_vanilla_gru_iter_part2_kernels_];
    brgemm_desc_t
            desc_iter_p2_NK2_tail_b1_[num_vanilla_gru_iter_part2_kernels_];

    brgemm_ker_ptr_t kernel_layer_b0_[num_base_kernels_];
    brgemm_ker_ptr_t kernel_layer_b1_[num_base_kernels_];
    brgemm_ker_ptr_t kernel_iter_b0_[num_base_kernels_];
    brgemm_ker_ptr_t kernel_iter_b1_[num_base_kernels_];
    brgemm_ker_ptr_t kernel_layer_N_tail_b0_[num_base_kernels_];
    brgemm_ker_ptr_t kernel_layer_N_tail_b1_[num_base_kernels_];
    brgemm_ker_ptr_t kernel_iter_N_tail_b0_[num_base_kernels_];
    brgemm_ker_ptr_t kernel_iter_N_tail_b1_[num_base_kernels_];

    brgemm_ker_ptr_t kernel_layer_K1_tail_b1_[num_base_kernels_];
    brgemm_ker_ptr_t kernel_layer_NK1_tail_b1_[num_base_kernels_];
    brgemm_ker_ptr_t kernel_iter_K2_tail_b1_[num_base_kernels_];
    brgemm_ker_ptr_t kernel_iter_NK2_tail_b1_[num_base_kernels_];

    brgemm_ker_ptr_t kernel_layermerged_b0_[num_base_kernels_];
    brgemm_ker_ptr_t kernel_layermerged_b1_[num_base_kernels_];
    brgemm_ker_ptr_t kernel_layermerged_N_tail_b0_[num_base_kernels_];
    brgemm_ker_ptr_t kernel_layermerged_N_tail_b1_[num_base_kernels_];
    brgemm_ker_ptr_t kernel_layermerged_K1_tail_b1_[num_base_kernels_];
    brgemm_ker_ptr_t kernel_layermerged_NK1_tail_b1_[num_base_kernels_];

    brgemm_ker_ptr_t kernel_proj_b0_[num_proj_kernels_];
    brgemm_ker_ptr_t kernel_proj_N_tail_b0_[num_proj_kernels_];
    brgemm_ker_ptr_t kernel_proj_N_tail_b1_[num_proj_kernels_];
    brgemm_ker_ptr_t kernel_proj_K_tail_b1_[num_proj_kernels_];
    brgemm_ker_ptr_t kernel_proj_NK_tail_b1_[num_proj_kernels_];

    // Set of brgemm kernels for 2nd part of iteration gemm in vanulla GRU cell
    brgemm_ker_ptr_t kernel_iter_p2_b1_[num_vanilla_gru_iter_part2_kernels_];
    brgemm_ker_ptr_t
            kernel_iter_p2_N_tail_b1_[num_vanilla_gru_iter_part2_kernels_];
    brgemm_ker_ptr_t
            kernel_iter_p2_K2_tail_b1_[num_vanilla_gru_iter_part2_kernels_];
    brgemm_ker_ptr_t
            kernel_iter_p2_NK2_tail_b1_[num_vanilla_gru_iter_part2_kernels_];

    brgemm_pallete_t pallete_buff_iter_;
    brgemm_pallete_t pallete_buff_iter_n_tail_;
    brgemm_pallete_t pallete_buff_layer_;
    brgemm_pallete_t pallete_buff_layer_n_tail_;

    brgemm_pallete_t pallete_buff_k1_tail_;
    brgemm_pallete_t pallete_buff_k2_tail_;
    brgemm_pallete_t pallete_buff_nk1_tail_;
    brgemm_pallete_t pallete_buff_nk2_tail_;
    brgemm_pallete_t pallete_buff_proj_;
    brgemm_pallete_t pallete_buff_nproj_tail_;
    brgemm_pallete_t pallete_buff_kproj_tail_;
    brgemm_pallete_t pallete_buff_nkproj_tail_;

    brgemm_pallete_t pallete_buff_layermerged_;
    brgemm_pallete_t pallete_buff_layermerged_n_tail_;
    brgemm_pallete_t pallete_buff_layermerged_k1_tail_;
    brgemm_pallete_t pallete_buff_layermerged_nk1_tail_;

private:
    status_t brgemm_rnn_init_tiles(
            brgemm_desc_t *desc, brgemm_pallete_t pallete);
    status_t brgemm_rnn_init_tiles_proj(
            brgemm_desc_t *desc, brgemm_pallete_t pallete);
    status_t brgemm_rnn_init_tiles(
            brgemm_desc_t *desc, dim_t size, brgemm_pallete_t pallete);
};

struct rnn_diff_src_brgemm_t {
    brgemm_desc_t desc_iter_layer_beta0_;
    brgemm_desc_t desc_iter_layer_beta1_;
    brgemm_desc_t desc_layer_N_tail_beta0_;
    brgemm_desc_t desc_layer_N_tail_beta1_;
    brgemm_desc_t desc_iter_N_tail_beta0_;
    brgemm_desc_t desc_iter_N_tail_beta1_;
    brgemm_desc_t desc_iter_layer_K_tail_beta1_;
    brgemm_desc_t desc_layer_NK_tail_beta1_;
    brgemm_desc_t desc_iter_NK_tail_beta1_;

    brgemm_ker_ptr_t kernel_iter_layer_beta0_ = nullptr;
    brgemm_ker_ptr_t kernel_iter_layer_beta1_ = nullptr;
    brgemm_ker_ptr_t kernel_layer_N_tail_beta0_ = nullptr;
    brgemm_ker_ptr_t kernel_layer_N_tail_beta1_ = nullptr;
    brgemm_ker_ptr_t kernel_iter_N_tail_beta0_ = nullptr;
    brgemm_ker_ptr_t kernel_iter_N_tail_beta1_ = nullptr;
    brgemm_ker_ptr_t kernel_iter_layer_K_tail_beta1_ = nullptr;
    brgemm_ker_ptr_t kernel_layer_NK_tail_beta1_ = nullptr;
    brgemm_ker_ptr_t kernel_iter_NK_tail_beta1_ = nullptr;

    brgemm_pallete_t pallete_buff_iter_layer_ = {};
    brgemm_pallete_t pallete_buff_iter_layer_k_tail_ = {};
    brgemm_pallete_t pallete_buff_iter_n_tail_ = {};
    brgemm_pallete_t pallete_buff_layer_n_tail_ = {};
    brgemm_pallete_t pallete_buff_iter_nk_tail_ = {};
    brgemm_pallete_t pallete_buff_layer_nk_tail_ = {};
};

struct rnn_diff_wei_brgemm_t {
    brgemm_desc_t desc_iter_beta1_;
    brgemm_desc_t desc_layer_beta1_;
    brgemm_desc_t desc_iter_N_tail_beta1_;
    brgemm_desc_t desc_layer_N_tail_beta1_;
    brgemm_desc_t desc_iter_NK_tail_beta1_;
    brgemm_desc_t desc_layer_NK_tail_beta1_;
    brgemm_desc_t desc_iter_K_tail_beta1_;
    brgemm_desc_t desc_layer_K_tail_beta1_;

    brgemm_ker_ptr_t kernel_iter_beta1_ = nullptr;
    brgemm_ker_ptr_t kernel_layer_beta1_ = nullptr;
    brgemm_ker_ptr_t kernel_iter_N_tail_beta1_ = nullptr;
    brgemm_ker_ptr_t kernel_layer_N_tail_beta1_ = nullptr;
    brgemm_ker_ptr_t kernel_iter_NK_tail_beta1_ = nullptr;
    brgemm_ker_ptr_t kernel_layer_NK_tail_beta1_ = nullptr;
    brgemm_ker_ptr_t kernel_iter_K_tail_beta1_ = nullptr;
    brgemm_ker_ptr_t kernel_layer_K_tail_beta1_ = nullptr;

    brgemm_pallete_t pallete_buff_iter_ = {};
    brgemm_pallete_t pallete_buff_layer_ = {};
    brgemm_pallete_t pallete_buff_iter_n_tail_ = {};
    brgemm_pallete_t pallete_buff_layer_n_tail_ = {};
    brgemm_pallete_t pallete_buff_iter_nk_tail_ = {};
    brgemm_pallete_t pallete_buff_layer_nk_tail_ = {};
    brgemm_pallete_t pallete_buff_iter_k_tail_ = {};
    brgemm_pallete_t pallete_buff_layer_k_tail_ = {};

    srcatch_gates_reorder_ker_ptr_t srcatch_gates_reorder_kernel_;
};

template <>
struct rnn_brgemm_t<prop_kind::backward> : public rnn_brgemm_base_t {
public:
    static void init_scratchpad(const cpu::rnn_utils::rnn_conf_t &rnn,
            memory_tracking::registrar_t &scratchpad, dim_t gemm_acc_type_size,
            dim_t gemm_acc_align);
    static status_t configure_brgemm(cpu::rnn_utils::rnn_conf_t &rnn,
            alg_kind_t cell_kind, dim_t src_layer_type_size,
            dim_t scratch_type_size);

    status_t init_kernels(const cpu::rnn_utils::rnn_conf_t &rnn,
            data_type_t src_type, data_type_t weights_type);

    rnn_diff_src_brgemm_t diff_src_;
    rnn_diff_wei_brgemm_t diff_wei_;

    std::unique_ptr<jit_gates_reduction_t> kernel_gates_reduction_;
    std::unique_ptr<jit_gates_reduction_t> kernel_gates_reduction_tail_;

    std::unique_ptr<jit_brgemm_transpose_single_row_t>
            kernel_transpose_single_row_iter_;
    std::unique_ptr<jit_brgemm_transpose_single_row_t>
            kernel_transpose_single_row_layer_;

    std::unique_ptr<jit_brgemm_trans_src_t>
            kernel_transpose_iter_[num_base_kernels_];
    std::unique_ptr<jit_brgemm_trans_src_t>
            kernel_transpose_layer_[num_base_kernels_];

    std::unique_ptr<jit_diff_weights_peephole_t> kernel_peephole_;
    std::unique_ptr<jit_diff_weights_peephole_t> kernel_peephole_tail_;

private:
    static void configure_brgemm_peephole(cpu::rnn_utils::rnn_conf_t &rnn);

    status_t init_peephole_kernels(const cpu::rnn_utils::rnn_conf_t &rnn);
};

} // namespace rnn_brgemm_utils
} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif
