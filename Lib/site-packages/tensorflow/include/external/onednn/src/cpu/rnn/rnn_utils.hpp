/*******************************************************************************
* Copyright 2018-2024 Intel Corporation
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

#ifndef CPU_RNN_RNN_UTILS_HPP
#define CPU_RNN_RNN_UTILS_HPP

#include <memory>
#include <type_traits>

#include "common/c_types_map.hpp"
#include "common/memory_desc_wrapper.hpp"
#include "common/primitive.hpp"
#include "common/utils.hpp"

#include "cpu/platform.hpp"

#include "cpu/gemm/gemm_pack.hpp"

#if DNNL_X64
#include "cpu/x64/cpu_isa_traits.hpp"
#endif

#define rnn_postgemm_sig(f) \
    void f(const rnn_utils::rnn_conf_t &rnn, \
            rnn_utils::cell_position_t cell_position, gates_t *ws_gates_, \
            scratch_t *scratch_gates_, const dst_layer_t *augru_attention_, \
            dst_layer_t *dst_layer_, void *dst_iter_c_, \
            const src_iter_t *src_iter_, const void *src_iter_c_, \
            gemm_acc_t *diff_src_layer_, gemm_acc_t *diff_augru_attention_, \
            gemm_acc_t *diff_src_iter_, gemm_acc_t *diff_src_iter_c_, \
            gemm_acc_t *diff_dst_layer_, gemm_acc_t *diff_dst_iter_, \
            gemm_acc_t *diff_dst_iter_c_, const float *weights_peephole_, \
            const void *bias_, gates_t *ws_grid_, scratch_t *scratch_cell_, \
            dst_iter_t *dst_iter_, float *weights_scales_, int block_step) \
            const

#if DNNL_X64
#define rnn_merged_layer_execution_sig(f) \
    dnnl_status_t f(const exec_ctx_t &ctx, const rnn_utils::rnn_conf_t &rnn, \
            rnn_utils::cell_position_t cell_position, weights_t **w_layer_, \
            const src_layer_t *src_layer_, scratch_t *scratch_gates_, \
            gemm_acc_t *diff_src_layer_, gemm_acc_t *diff_w_layer_, \
            gemm_acc_t *amx_scratchpad, \
            x64::brgemm_batch_element_t *addr_batch_global) const

#define rnn_cell_execution_sig(f) \
    dnnl_status_t f(const exec_ctx_t &ctx, const rnn_utils::rnn_conf_t &rnn, \
            rnn_utils::cell_position_t cell_position, dst_layer_t *dst_layer_, \
            void *dst_iter_c_, gemm_acc_t *diff_src_layer_, \
            gemm_acc_t *diff_augru_attention_, gemm_acc_t *diff_src_iter_, \
            gemm_acc_t *diff_src_iter_c_, weights_t **w_layer_, \
            weights_t **w_iter_, weights_t **w_projection_, \
            const float *weights_peephole_, const float *w_proj_comp, \
            void **bias_, const src_layer_t *src_layer_, \
            const src_layer_t *augru_attention_, const src_iter_t *src_iter_, \
            const void *src_iter_c_, gemm_acc_t *diff_dst_layer_, \
            gemm_acc_t *diff_dst_iter_, gemm_acc_t *diff_dst_iter_c_, \
            gemm_acc_t *diff_w_layer_, gemm_acc_t *diff_w_iter_, \
            float *diff_weights_projection_, float *diff_weights_peephole_, \
            float *diff_bias_, gates_t *ws_gates_, scratch_t *scratch_gates_, \
            ht_t *proj_ht_, gemm_acc_t *scratch_diff_ht_, gates_t *ws_grid_, \
            scratch_t *scratch_cell_, scratch_t *scratch_gates_blocked_, \
            scratch_t *scratch_src_layer_, scratch_t *scratch_src_iter_, \
            dst_iter_t *dst_iter_, gemm_acc_t *amx_scratchpad, \
            x64::brgemm_batch_element_t *addr_batch_global) const

#define rnn_grid_execution_sig(f) \
    dnnl_status_t f(const exec_ctx_t &ctx, const rnn_utils::rnn_conf_t &rnn, \
            weights_t **weights_layer_, weights_t **weights_iter_, \
            weights_t **weights_projection_, const float *weights_peephole_, \
            const float *w_proj_comp, void **bias_, \
            const src_layer_t *src_layer_, \
            const src_layer_t *augru_attention_, const src_iter_t *src_iter_, \
            const void *src_iter_c_, dst_layer_t *dst_layer_, \
            dst_iter_t *dst_iter_, void *dst_iter_c_, \
            src_layer_t *ws_states_layer_, src_iter_t *ws_states_iter_, \
            void *ws_states_iter_c_, gemm_acc_t *ws_diff_states_layer_, \
            gemm_acc_t *ws_diff_states_iter_, \
            gemm_acc_t *ws_diff_states_iter_c_, gates_t *ws_gates_, \
            ht_t *ws_ht_, gates_t *ws_grid_, scratch_t *scratch_gates_, \
            ht_t *scratch_ht_, gemm_acc_t *scratch_diff_ht_, \
            scratch_t *scratch_cell_, scratch_t *scratch_gates_blocked_, \
            scratch_t *scratch_src_layer_, scratch_t *scratch_src_iter_, \
            gemm_acc_t *diff_augru_attention_, \
            gemm_acc_t *diff_weights_layer_, gemm_acc_t *diff_weights_iter_, \
            float *diff_weights_projection_, float *diff_weights_peephole_, \
            float *diff_bias_, gemm_acc_t *amx_scratchpad, \
            x64::brgemm_batch_element_t *addr_batch_global) const
#else
#define rnn_merged_layer_execution_sig(f) \
    dnnl_status_t f(const rnn_utils::rnn_conf_t &rnn, \
            rnn_utils::cell_position_t cell_position, weights_t **w_layer_, \
            const src_layer_t *src_layer_, scratch_t *scratch_gates_, \
            gemm_acc_t *diff_src_layer_, gemm_acc_t *diff_w_layer_) const

#define rnn_cell_execution_sig(f) \
    dnnl_status_t f(const exec_ctx_t &ctx, const rnn_utils::rnn_conf_t &rnn, \
            rnn_utils::cell_position_t cell_position, dst_layer_t *dst_layer_, \
            void *dst_iter_c_, gemm_acc_t *diff_src_layer_, \
            gemm_acc_t *diff_augru_attention_, gemm_acc_t *diff_src_iter_, \
            gemm_acc_t *diff_src_iter_c_, weights_t **w_layer_, \
            weights_t **w_iter_, weights_t **w_projection_, \
            const float *weights_peephole_, const float *w_proj_comp, \
            void **bias_, const src_layer_t *src_layer_, \
            const src_layer_t *augru_attention_, const src_iter_t *src_iter_, \
            const void *src_iter_c_, gemm_acc_t *diff_dst_layer_, \
            gemm_acc_t *diff_dst_iter_, gemm_acc_t *diff_dst_iter_c_, \
            gemm_acc_t *diff_w_layer_, gemm_acc_t *diff_w_iter_, \
            float *diff_weights_projection_, float *diff_weights_peephole_, \
            float *diff_bias_, gates_t *ws_gates_, scratch_t *scratch_gates_, \
            ht_t *proj_ht_, gemm_acc_t *scratch_diff_ht_, gates_t *ws_grid_, \
            scratch_t *scratch_cell_, dst_iter_t *dst_iter_, \
            gemm_acc_t *amx_scratchpad) const

#define rnn_grid_execution_sig(f) \
    dnnl_status_t f(const exec_ctx_t &ctx, const rnn_utils::rnn_conf_t &rnn, \
            weights_t **weights_layer_, weights_t **weights_iter_, \
            weights_t **weights_projection_, const float *weights_peephole_, \
            const float *w_proj_comp, void **bias_, \
            const src_layer_t *src_layer_, \
            const src_layer_t *augru_attention_, const src_iter_t *src_iter_, \
            const void *src_iter_c_, dst_layer_t *dst_layer_, \
            dst_iter_t *dst_iter_, void *dst_iter_c_, \
            src_layer_t *ws_states_layer_, src_iter_t *ws_states_iter_, \
            void *ws_states_iter_c_, gemm_acc_t *ws_diff_states_layer_, \
            gemm_acc_t *ws_diff_states_iter_, \
            gemm_acc_t *ws_diff_states_iter_c_, gates_t *ws_gates_, \
            ht_t *ws_ht_, gates_t *ws_grid_, scratch_t *scratch_gates_, \
            ht_t *scratch_ht_, gemm_acc_t *scratch_diff_ht_, \
            scratch_t *scratch_cell_, gemm_acc_t *diff_augru_attention_, \
            gemm_acc_t *diff_weights_layer_, gemm_acc_t *diff_weights_iter_, \
            float *diff_weights_projection_, float *diff_weights_peephole_, \
            float *diff_bias_, gemm_acc_t *amx_scratchpad) const
#endif

#define rnn_matmul_sig(f) \
    dnnl_status_t f(const exec_ctx_t &ctx, \
            const std::shared_ptr<dnnl::impl::primitive_t> &matmul_prim, \
            const weights_t *a_, const gemm_data_t *b_, gemm_acc_t *c_) const

#define rnn_gemm_sig(f) \
    dnnl_status_t f(const char transA, const char transB, dim_t m, dim_t n, \
            dim_t k, const float alpha, const weights_t *a_, const dim_t ldA, \
            const gemm_data_t *b_, const dim_t ldB, const float beta, \
            gemm_acc_t *c_, const dim_t ldC) const

#define rnn_bias_prepare_sig(f) \
    void f(const rnn_utils::rnn_conf_t &rnn, void **bias_, const void *b_, \
            void *scratch_bias_) const

#define rnn_bias_prepare_sig_templ(f) \
    template <typename T> \
    static void f(const rnn_utils::rnn_conf_t &rnn, T **bias_, const T *b_, \
            T *scratch_bias_)

#define rnn_bias_finalize_sig(f) \
    void f(const rnn_utils::rnn_conf_t &rnn, void *scratch_bias_, \
            const float *w_iter_comp, const float *w_layer_comp) const

#define rnn_weights_assign_sig(f) \
    void f(const rnn_utils::rnn_conf_t &rnn, const memory_desc_t *md, \
            int n_parts, const int *gates_per_part, weights_t **weights_, \
            const weights_t *w_) const

namespace dnnl {
namespace impl {
namespace cpu {

namespace rnn_utils {

enum execution_direction_t {
    l2r,
    r2l,
    bi_concat,
    bi_sum,
};

enum cell_position_t {
    middle_cell = 0x0,
    first_layer = 0x1,
    first_iter = 0x2,
    last_layer = 0x4,
    last_iter = 0x8,
    c_state_first_iter = 0x10,
    c_state_last_iter = 0x20,
    merged_iter = 0x40,
    merged_layer = 0x80
};

enum class weights_type_t {
    layer,
    iter,
    projection,
    peephole,
};

inline cell_position_t &operator|=(cell_position_t &lhs, cell_position_t rhs) {
    lhs = static_cast<cell_position_t>(
            static_cast<unsigned>(lhs) | static_cast<unsigned>(rhs));
    return lhs;
}

inline cell_position_t operator|(cell_position_t lhs, cell_position_t rhs) {
    return static_cast<cell_position_t>(
            static_cast<unsigned>(lhs) | static_cast<unsigned>(rhs));
}

enum data_type_conf_t {
    all_f32,
    all_bf16,
    all_f16,
    u8u8u8f32,
    f32u8f32f32,
    u8u8u8u8,
    f32u8f32u8,
    s8s8s8f32,
    f32s8f32f32,
    s8s8s8s8,
    f32s8f32s8
};

enum brgemm_rnn_execute_loop_order_t {
    // default for kernels w/o loop order choice
    undefined = 0x0,
    // m_blocking loop is outermost
    mblk_nblk = 0x1,
    // n_blocking loop is outermost
    nblk_mblk = 0x2
};

struct diff_src_brgemm_conf_t {
    dim_t M = 0, N = 0, K = 0;

    dim_t n_block = 0, N_blocks = 0, n_tail = 0;
    dim_t m_block = 0, M_blocks = 0;

    dim_t K_blocks = 0, k_block = 0, k_tail = 0;
    dim_t Kpadded = 0;

    dim_t N_iter = 0, N_layer = 0;
    dim_t N_layer_blocks = 0, n_layer_tail = 0;
    dim_t N_iter_blocks = 0, n_iter_tail = 0;
    dim_t LDA = 0, LDB = 0, LDC = 0;

#if DNNL_X64
    x64::cpu_isa_t isa = x64::isa_undef;
#endif

    brgemm_rnn_execute_loop_order_t loop_order
            = brgemm_rnn_execute_loop_order_t::undefined;
    int gates_block;
};

struct diff_wei_brgemm_conf_t {
    dim_t M = 0, M_layer = 0, M_iter = 0, N = 0, K = 0;

    dim_t n_block = 0, N_blocks = 0, n_tail = 0;
    dim_t m_block = 0, M_blocks = 0;
    dim_t K_blocks = 0, k_block = 0, k_tail = 0;
    dim_t Kpadded = 0;

    dim_t LDA_layer = 0, LDA_iter = 0, LDB = 0, LDC_iter = 0, LDC_layer = 0;

    bool global_transpose = false;

#if DNNL_X64
    x64::cpu_isa_t isa = x64::isa_undef;
#endif

    brgemm_rnn_execute_loop_order_t loop_order
            = brgemm_rnn_execute_loop_order_t::undefined;
};

struct rnn_conf_t {
    execution_direction_t exec_dir;
    data_type_conf_t dt_conf;
    data_type_t cell_dt = data_type::undef; // The data type used by cell
    data_type_t bias_dt = data_type::undef;
    data_type_t src_iter_c_dt = data_type::undef;
    data_type_t dst_iter_c_dt = data_type::undef;

    int n_layer = 0, n_iter = 0, n_dir = 0, n_gates = 0, n_states = 0;
    int mb = 0;
    int slc = 0, sic = 0, dhc = 0, dic = 0, dlc = 0;
    //int gates_ld, gates_nld, gates_ws_ld;

    int n_parts_weights_layer = 0;
    int parts_weights_layer[DNNL_RNN_MAX_N_PARTS];
    size_t part_weights_layer_pack_size[DNNL_RNN_MAX_N_PARTS];

    int n_parts_weights_iter = 0;
    int parts_weights_iter[DNNL_RNN_MAX_N_PARTS];
    size_t part_weights_iter_pack_size[DNNL_RNN_MAX_N_PARTS];

    int n_parts_weights_projection = 0;
    int parts_weights_projection[DNNL_RNN_MAX_N_PARTS];
    size_t part_weights_projection_pack_size[DNNL_RNN_MAX_N_PARTS];

    int n_bias = 0, n_parts_bias = 0, parts_bias[DNNL_RNN_MAX_N_PARTS];

    /* Size of packed data in bytes */
    size_t weights_layer_comp_offset = 0, weights_layer_pack_size = 0;
    size_t weights_iter_comp_offset = 0, weights_iter_pack_size = 0;
    size_t weights_projection_comp_offset = 0, weights_projection_pack_size = 0;

    bool copy_bias = 0;
    int weights_layer_ld = 0, weights_layer_nld = 0;
    int diff_weights_layer_ld = 0, diff_weights_layer_nld = 0;
    int weights_iter_ld = 0, weights_iter_nld = 0;
    int diff_weights_iter_ld = 0, diff_weights_iter_nld = 0;
    int weights_projection_ld = 0, weights_projection_nld = 0;
    int diff_weights_projection_ld = 0, diff_weights_projection_nld = 0;

    int proj_ht_ld = 0, proj_ht_nld = 0;

    int ws_gates_ld = 0, ws_gates_nld = 0;
    int ws_ht_ld = 0, ws_ht_nld = 0;
    int ws_states_layer_ld = 0, ws_states_layer_nld = 0;
    int ws_states_iter_ld = 0, ws_states_iter_nld = 0;
    int ws_states_iter_c_ld = 0, ws_states_iter_c_nld = 0;
    int ws_diff_states_layer_ld = 0, ws_diff_states_layer_nld = 0;
    int ws_diff_states_iter_ld = 0, ws_diff_states_iter_nld = 0;
    int ws_diff_states_iter_c_ld = 0, ws_diff_states_iter_c_nld = 0;

    int scratch_gates_ld = 0, scratch_gates_nld = 0;
    int scratch_ht_ld = 0, scratch_ht_nld = 0;
    int scratch_diff_ht_ld = 0, scratch_diff_ht_nld = 0;

    int src_layer_ld_ = 0, src_layer_nld_ = 0;
    int src_iter_ld_ = 0, src_iter_nld_ = 0;
    int src_iter_c_ld_ = 0, src_iter_c_nld_ = 0;
    int dst_layer_ld_ = 0, dst_layer_nld_ = 0;
    int dst_iter_ld_ = 0, dst_iter_nld_ = 0;
    int dst_iter_c_ld_ = 0, dst_iter_c_nld_ = 0;

    int weights_iter_compensation_size = 0, weights_layer_compensation_size = 0;
    bool is_fwd = 0, is_training = 0, is_lbr = 0, is_lstm_peephole = 0,
         is_lstm_projection = 0, is_augru = 0, is_orig_gru = 0;
    bool use_workspace = 0;

    // Size of workspace for each tensor in bytes
    // Notes:
    // 1. For non-LSTMP ws_states_iter_size == ws_states_layer_size. The corresponding
    //    pointers should point to the same places.
    size_t ws_gates_size = 0;
    size_t ws_ht_size = 0;
    size_t ws_states_layer_size = 0;
    size_t ws_states_iter_size = 0;
    size_t ws_states_iter_c_size = 0;
    size_t ws_diff_states_layer_size = 0;
    size_t ws_diff_states_iter_size = 0;
    size_t ws_diff_states_iter_c_size = 0;
    size_t scratch_gates_size = 0;

    size_t scratch_gates_blocked_size = 0;
    size_t scratch_gates_blocked_nested_reorder_size = 0;
    size_t scratch_src_layer_size = 0;
    size_t scratch_src_layer_nested_reorder_size = 0;
    size_t scratch_src_iter_size = 0;
    size_t scratch_src_iter_nested_reorder_size = 0;

    size_t scratch_ht_size = 0;
    size_t scratch_diff_ht_size = 0;
    size_t scratch_cell_size = 0;
    size_t ws_grid_comp_size = 0;
    size_t ws_per_cell = 0;
    size_t ws_bias_size = 0;

    bool src_layer_is_trivial_stride = false;
    bool dst_layer_is_trivial_stride = false;
    bool merge_gemm_iter = false, merge_gemm_layer = false,
         force_nocopy = false, use_layer_packed_gemm = false,
         use_iter_packed_gemm = false, use_projection_packed_gemm = false;
    int n_iter_scratch_gates = 0;

    bool diff_weights_overwrite = false;
    bool use_matmul = false;

    inline bool is_int8_conf() const {
        return is_signed_int8_conf() || is_unsigned_int8_conf();
    }
    inline bool is_signed_int8_conf() const {
        return utils::one_of(
                dt_conf, s8s8s8f32, f32s8f32f32, s8s8s8s8, f32s8f32s8);
    }
    inline bool is_unsigned_int8_conf() const {
        return utils::one_of(
                dt_conf, u8u8u8f32, f32u8f32f32, u8u8u8u8, f32u8f32u8);
    }

    inline bool is_cell_dt_int8() const {
        return is_cell_dt_signed_int8() || is_cell_dt_unsigned_int8();
    }
    inline bool is_cell_dt_signed_int8() const {
        return cell_dt == data_type::s8;
    }
    inline bool is_cell_dt_unsigned_int8() const {
        return cell_dt == data_type::u8;
    }

    inline bool is_cell_int8_amx() const {
#if DNNL_X64
        return brgemm_isa == x64::avx512_core_amx && is_cell_dt_int8();
#else
        return false;
#endif
    }

    inline bool is_bf16_conf() const { return dt_conf == all_bf16; }
    inline bool is_f16_conf() const { return dt_conf == all_f16; }
    inline bool is_xf16_conf() const { return is_bf16_conf() || is_f16_conf(); }
    inline bool is_f32_conf() const { return dt_conf == all_f32; }

    inline bool is_cell_dt_f32() const { return cell_dt == data_type::f32; }
    inline bool is_cell_dt_bf16() const { return cell_dt == data_type::bf16; }
    inline bool is_cell_dt_f16() const { return cell_dt == data_type::f16; }
    inline bool is_cell_dt_xf16() const {
        return is_cell_dt_bf16() || is_cell_dt_f16();
    }
    inline bool is_cell_bf16_amx() const {
#if DNNL_X64
        return brgemm_isa == x64::avx512_core_amx && is_cell_dt_bf16();
#else
        return false;
#endif
    }
    inline bool is_cell_f16_amx() const {
#if DNNL_X64
        return brgemm_isa == x64::avx512_core_amx_fp16 && is_cell_dt_f16();
#else
        return false;
#endif
    }

    inline bool is_cell_xf16_amx() const {
        return is_cell_bf16_amx() || is_cell_f16_amx();
    }

    inline bool is_cell_amx() const {
        return is_cell_bf16_amx() || is_cell_int8_amx() || is_cell_f16_amx();
    }

    inline bool is_bf32() const { return is_cell_bf16_amx() && is_f32_conf(); }

    inline bool skip_src_layer_copy() const {
        return (exec_dir == l2r) && !is_bf32()
                && utils::one_of(dt_conf, s8s8s8f32, f32s8f32f32, s8s8s8s8,
                        f32s8f32s8, u8u8u8u8, u8u8u8f32, f32u8f32u8,
                        f32u8f32f32, all_f32, all_bf16, all_f16);
    }
    inline bool skip_src_iter_copy() const {
        return (exec_dir == l2r) && (src_iter_ld_ > 0) && !is_bf32()
                && utils::one_of(dt_conf, s8s8s8s8, s8s8s8f32, u8u8u8u8,
                        u8u8u8f32, all_f32, all_bf16, all_f16);
    }
    inline bool skip_dst_layer_copy() const {
        return (exec_dir == l2r) && !is_bf32()
                && utils::one_of(dt_conf, s8s8s8s8, f32s8f32s8, u8u8u8u8,
                        f32u8f32u8, all_f32, all_bf16, all_f16);
    }
    inline bool skip_dst_iter_copy() const {
        return (exec_dir == l2r) && (dst_iter_ld_ > 0) && !is_bf32()
                && utils::one_of(dt_conf, s8s8s8s8, s8s8s8f32, u8u8u8u8,
                        u8u8u8f32, all_f32, all_bf16, all_f16);
    }

    inline dim_t src_layer_ld(cell_position_t cell_position) const {
        return (cell_position & first_layer) && skip_src_layer_copy()
                ? src_layer_ld_
                : (cell_position & last_iter) && skip_dst_iter_copy()
                ? dst_iter_ld_
                : ws_states_layer_ld;
    }

    inline dim_t src_iter_ld(cell_position_t cell_position) const {
        return (cell_position & first_iter) && skip_src_iter_copy()
                ? src_iter_ld_
                : ((cell_position & last_layer) && skip_dst_layer_copy()
                                        && !(cell_position & first_iter)
                                ? dst_layer_ld_
                                : ws_states_iter_ld);
    }

    inline dim_t layer_brgemm_desc(cell_position_t cell_position) const {
        return ((cell_position & first_layer) && skip_src_layer_copy()) ? 0
                : ((cell_position & last_iter) && skip_dst_iter_copy()) ? 1
                                                                        : 2;
    }

    inline dim_t iter_brgemm_desc(cell_position_t cell_position) const {
        return ((cell_position & first_iter) && skip_src_iter_copy()) ? 0
                : ((cell_position & last_layer) && skip_dst_layer_copy()
                          && !(cell_position & first_iter))
                ? 1
                : 2;
    }

    // Returns index of brgemm kernel for 2nd part of iteration gemm in vanilla
    // GRU cell for the current position.
    // Note: this method must be aligned with dst_iter_part2_ld() and LDA2_2[]
    // values initialization order
    inline dim_t iter_part2_brgemm_desc(cell_position_t cell_position) const {
        if (cell_position & last_layer) {
            return (cell_position & last_layer) && skip_dst_layer_copy()  ? 0
                    : (cell_position & last_iter) && skip_dst_iter_copy() ? 1
                                                                          : 2;
        } else {
            return (cell_position & last_iter) && skip_dst_iter_copy() ? 1 : 3;
        }
    }

    inline dim_t src_iter_c_ld(cell_position_t cell_position) const {
        return (cell_position & c_state_first_iter) ? src_iter_c_ld_
                                                    : ws_states_iter_c_ld;
    }

    inline dim_t dst_layer_ld(
            cell_position_t cell_position, bool after_proj = false) const {
        // We use scratch_ht and not dst_layer for lstmp
        if (is_lstm_projection && !after_proj) return scratch_ht_ld;

        return (cell_position & last_layer) && skip_dst_layer_copy()
                ? dst_layer_ld_
                : (cell_position & last_iter) && skip_dst_iter_copy()
                ? dst_iter_ld_
                : ws_states_layer_ld;
    }

    inline dim_t dst_brgemm_desc(
            cell_position_t cell_position, bool after_proj = false) const {
        // We use scratch_ht and not dst_layer for lstmp
        if (is_lstm_projection && !after_proj) return 0;

        return (cell_position & last_layer) && skip_dst_layer_copy()  ? 1
                : (cell_position & last_iter) && skip_dst_iter_copy() ? 2
                                                                      : 3;
    }

    inline dim_t dst_iter_ld(cell_position_t cell_position) const {
        return (cell_position & last_iter) && skip_dst_iter_copy()
                ? dst_iter_ld_
                : ws_states_iter_ld;
    }

    // Returns dst tensor leading dimension for 2nd part of iteration gemm in
    // vanilla GRU cell for the current position
    inline dim_t dst_iter_part2_ld(cell_position_t cell_position) const {
        return (cell_position & last_layer) ? dst_layer_ld(cell_position)
                                            : dst_iter_ld(cell_position);
    }

    inline dim_t dst_iter_c_ld(cell_position_t cell_position) const {
        return (cell_position & c_state_last_iter) ? dst_iter_c_ld_
                                                   : ws_states_iter_c_ld;
    }

    // // when skipping copy, the output ld can be states_ws_ld,
    // // dst_iter_ld or dst_layer_ld depending on the cell position
    // inline dim_t dst_ld(cell_position_t cell_position) const {
    //     return (cell_position & last_layer) ? dst_layer_ld(cell_position)
    //                                         : dst_iter_ld(cell_position);
    // }
    inline dim_t dst_copy_ld(cell_position_t cell_position) const {
        return dst_iter_ld(cell_position);
    }

    inline bool need_gemm_layer(cell_position_t cell_position) const {
        // In case of merge_gemm_layer we might still need a layer gemm if we store
        // the states of the last iteration in the destination memory. The
        // exception of this rule is the first layer though, in which case all
        // states are kept in user's src_layer, hence making full merged gemm
        // possible.
        return IMPLICATION(merge_gemm_layer,
                skip_dst_iter_copy() && (cell_position & last_iter)
                        && !(cell_position & first_layer));
    }

    // get diff_weights_beta based on cell position
    inline float diff_weights_beta(cell_position_t cell_position) const {
        if (diff_weights_overwrite) {
            // Initialize diff weights if needed
            if (cell_position & merged_iter) return 0.0f;
            if ((cell_position & merged_layer)
                    && !need_gemm_layer(cell_position | last_iter))
                return 0.0f;
            if (cell_position & last_iter) return 0.0f;
        }
        return 1.0f;
    }

    bool is_brgemm;

    diff_src_brgemm_conf_t diff_src_brgemm;
    diff_wei_brgemm_conf_t diff_wei_brgemm;

    dim_t M, N, K1, K2;

    dim_t LDB1, LDB2;
    dim_t LDA1[3];
    dim_t LDA2[3];
    // LDA for iter part2 gemm in vanilla gru cell
    dim_t LDA2_2[4];
    dim_t LDC;

    dim_t m_block, M_blocks;
    dim_t n_block, N_blocks, n_tail;

    dim_t k2_block, k1_block, k1_tail, k2_tail;
    dim_t KB1_blocks, KB2_blocks;
    dim_t K1padded, K2padded;

    dim_t Kproj, Kprojpadded;
    dim_t kproj_block, KBproj_blocks, kproj_tail;

    dim_t Nproj, Nproj_blocks, nproj_tail;
    dim_t LDAproj, LDBproj, LDCproj[4];
    int dhc_block_peephole, dhc_tail_peephole, dhc_blocks_peephole;
    bool brgemm_fwd_iter_layer_fuse_possible = false;

    dim_t nthr;
#if DNNL_X64
    x64::cpu_isa_t brgemm_isa;
#endif
    bool unfused_post_gemm;
    brgemm_rnn_execute_loop_order_t loop_order
            = brgemm_rnn_execute_loop_order_t::undefined;

    // for merged layer computation in brgemm
    dim_t Mlayermerged;
    dim_t mlayermerged_block, Mlayermerged_blocks;
};

bool is_ldigo(const memory_desc_wrapper &md);
bool is_ldgoi(const memory_desc_wrapper &md);
bool is_ldio(const memory_desc_wrapper &md);
bool is_ldoi(const memory_desc_wrapper &md);
bool is_ldigo_blocked(const memory_desc_wrapper &md);
bool is_ldgoi_blocked(const memory_desc_wrapper &md);
bool is_ldio_blocked(const memory_desc_wrapper &md);
bool is_ldoi_blocked(const memory_desc_wrapper &md);

int get_good_ld(int dim, int sizeof_dt);

template <typename T>
bool init_conf(rnn_conf_t &rnn, const rnn_desc_t &rd,
        const primitive_attr_t &attr, const memory_desc_wrapper &src_layer_d,
        const memory_desc_wrapper &src_iter_d,
        const memory_desc_wrapper &src_iter_c_d,
        const memory_desc_wrapper &weights_layer_d,
        const memory_desc_wrapper &weights_iter_d,
        const memory_desc_wrapper &weights_projection_d,
        const memory_desc_wrapper &dst_layer_d,
        const memory_desc_wrapper &dst_iter_d,
        const memory_desc_wrapper &dst_iter_c_d,
        const memory_desc_wrapper &bias_d) {
    rnn.is_fwd = utils::one_of(rd.prop_kind, prop_kind::forward_training,
            prop_kind::forward_inference);
    rnn.is_training = utils::one_of(
            rd.prop_kind, prop_kind::forward_training, prop_kind::backward);
    rnn.is_lbr = utils::one_of(rd.cell_kind, dnnl_lbr_gru, dnnl_lbr_augru);
    rnn.is_lstm_peephole = rd.cell_kind == dnnl_vanilla_lstm
            && !memory_desc_wrapper(rd.weights_peephole_desc).is_zero();
    rnn.is_lstm_projection = rd.cell_kind == dnnl_vanilla_lstm
            && !memory_desc_wrapper(rd.weights_projection_desc).is_zero();
    rnn.is_augru
            = utils::one_of(rd.cell_kind, dnnl_lbr_augru, dnnl_vanilla_augru);
    rnn.bias_dt = bias_d.is_zero() ? data_type::f32 : bias_d.data_type();
    rnn.src_iter_c_dt = src_iter_c_d.is_zero() ? data_type::f32
                                               : src_iter_c_d.data_type();
    rnn.dst_iter_c_dt = dst_iter_c_d.is_zero() ? data_type::f32
                                               : dst_iter_c_d.data_type();

    rnn.cell_dt = data_traits<typename T::src_layer_t>::data_type;
    switch (rd.direction) {
        case dnnl_unidirectional_left2right: rnn.exec_dir = l2r; break;
        case dnnl_unidirectional_right2left: rnn.exec_dir = r2l; break;
        case dnnl_bidirectional_concat: rnn.exec_dir = bi_concat; break;
        case dnnl_bidirectional_sum: rnn.exec_dir = bi_sum; break;
        default: break;
    }

    if (utils::everyone_is(data_type::f32, src_layer_d.data_type(),
                dst_layer_d.data_type(), weights_layer_d.data_type()))
        rnn.dt_conf = all_f32;
    else if (utils::everyone_is(data_type::bf16, src_layer_d.data_type(),
                     dst_layer_d.data_type(), weights_layer_d.data_type())) {
        if (!platform::has_data_type_support(data_type::bf16)) return false;
#if DNNL_X64
        if (!(x64::mayiuse(x64::avx512_core) || x64::mayiuse(x64::avx2_vnni_2)))
            return false;
#endif
        rnn.dt_conf = all_bf16;
    } else if (utils::everyone_is(data_type::f16, src_layer_d.data_type(),
                       dst_layer_d.data_type(), weights_layer_d.data_type())) {
        if (!platform::has_data_type_support(data_type::f16)) return false;
#if DNNL_X64
        if (!(x64::mayiuse(x64::avx512_core_fp16)
                    || x64::mayiuse(x64::avx2_vnni_2)))
            return false;
#endif
        rnn.dt_conf = all_f16;
    } else if (dst_layer_d.data_type() == data_type::u8) {
        if (IMPLICATION(
                    src_iter_d.md_, src_iter_d.data_type() == data_type::u8))
            rnn.dt_conf = u8u8u8u8;
        else
            rnn.dt_conf = f32u8f32u8;
    } else if (dst_layer_d.data_type() == data_type::s8) {
        if (IMPLICATION(
                    src_iter_d.md_, src_iter_d.data_type() == data_type::s8))
            rnn.dt_conf = s8s8s8s8;
        else
            rnn.dt_conf = f32s8f32s8;

    } else if (dst_layer_d.data_type() == data_type::f32) {
        if (IMPLICATION(
                    src_iter_d.md_, src_iter_d.data_type() == data_type::u8))
            rnn.dt_conf = u8u8u8f32;
        else if (IMPLICATION(src_iter_d.md_,
                         src_iter_d.data_type() == data_type::s8))
            rnn.dt_conf = s8s8s8f32;
        else if (IMPLICATION(src_layer_d.md_,
                         src_layer_d.data_type() == data_type::s8))
            rnn.dt_conf = f32s8f32f32;
        else
            rnn.dt_conf = f32u8f32f32;
    }

    if (!rnn.is_fwd && !platform::has_training_support(src_layer_d.data_type()))
        return false;

    // Set problem members defining problem sizes
    rnn.n_layer = weights_layer_d.dims()[0];
    rnn.n_iter = src_layer_d.dims()[0];
    rnn.n_dir = weights_layer_d.dims()[1];
    rnn.n_gates = weights_layer_d.dims()[3];
    rnn.n_states = rd.cell_kind == dnnl_vanilla_lstm ? 2 : 1;
    rnn.n_bias = rnn.n_gates + rnn.is_lbr;
    rnn.mb = src_layer_d.dims()[1];
    rnn.sic = weights_iter_d.dims()[2];
    rnn.slc = weights_layer_d.dims()[2];
    rnn.dhc = weights_layer_d.dims()[4];
    rnn.dlc = rnn.is_lstm_projection ? weights_projection_d.dims()[3] : rnn.dhc;
    // All supported cells have dic == dlc
    rnn.dic = rnn.dlc;

    // set members with user memories leading dimensions
    // Assumption: weights datatype size is the same as state datatype size
    assert(types::data_type_size(weights_layer_d.data_type())
            == types::data_type_size(src_layer_d.data_type()));

    // set workspace leading dimensions (and non leading-dimensions)

    // the ws and scratch proj_ht need to match as we use them interchangeably
    assert(IMPLICATION(rnn.is_lstm_projection,
            sizeof(typename T::ht_t) == sizeof(typename T::dst_iter_t)));
    rnn.proj_ht_nld = rnn.mb;
    rnn.proj_ht_ld = get_good_ld(rnn.dhc, sizeof(typename T::ht_t));

    rnn.ws_gates_nld = rnn.mb;
    rnn.ws_gates_ld
            = get_good_ld(rnn.dhc * rnn.n_gates, sizeof(typename T::gates_t));
    rnn.ws_ht_nld = rnn.proj_ht_nld;
    rnn.ws_ht_ld = rnn.proj_ht_ld;

    rnn.ws_states_layer_nld = rnn.mb;
    static_assert(std::is_same<typename T::src_layer_t,
                          typename T::src_iter_t>::value,
            "src_layer_t and src_iter_t must be the same");
    rnn.ws_states_layer_ld
            = get_good_ld(nstl::max(rnn.sic, nstl::max(rnn.slc, rnn.dlc)),
                    sizeof(typename T::src_layer_t));
    // there is no need for al separate ws_states_iter for now as all
    // supported cell have dst_iter == dst_layer
    rnn.ws_states_iter_nld = rnn.ws_states_layer_nld;
    rnn.ws_states_iter_ld = rnn.ws_states_layer_ld;

    // we do not need a good ld for iter_c as it is not involved in GEMM
    rnn.ws_states_iter_c_nld = rnn.mb;
    rnn.ws_states_iter_c_ld = rnn.dhc;

    // TODO: be more restrictive on the leading dimensions
    rnn.ws_diff_states_layer_nld = rnn.mb;
    rnn.ws_diff_states_layer_ld = get_good_ld(
            nstl::max(nstl::max(rnn.slc, rnn.dic), nstl::max(rnn.sic, rnn.dhc)),
            sizeof(typename T::gemm_acc_t));

    rnn.ws_diff_states_iter_nld = rnn.mb;
    rnn.ws_diff_states_iter_ld = get_good_ld(
            nstl::max(nstl::max(rnn.slc, rnn.dic), nstl::max(rnn.sic, rnn.dhc)),
            sizeof(typename T::gemm_acc_t));

    rnn.ws_diff_states_iter_c_nld = rnn.mb;
    rnn.ws_diff_states_iter_c_ld = rnn.dhc;

    // set scratch (not)leading dimensions
    // scratch gates is used to store intermediate gates before postgemm operation
    // temporary: we also use it in lstmp as temporary scratchpad
    // between projection and downconversion, hence the max with dlc
    rnn.scratch_gates_nld = rnn.mb;
    rnn.scratch_gates_ld
            = get_good_ld(nstl::max(rnn.dlc, rnn.n_gates * rnn.dhc),
                    sizeof(typename T::scratch_t));
    rnn.scratch_ht_nld = rnn.proj_ht_nld;
    rnn.scratch_ht_ld = rnn.proj_ht_ld;

    rnn.scratch_diff_ht_nld = rnn.mb;
    rnn.scratch_diff_ht_ld
            = get_good_ld(rnn.dlc, sizeof(typename T::gemm_acc_t));

    // Assumption: {src,dst}_layer has tnc layout, {src,dst}_iter has ldnc,
    rnn.src_layer_ld_ = src_layer_d.blocking_desc().strides[1];
    rnn.dst_layer_ld_ = dst_layer_d.blocking_desc().strides[1];
    rnn.src_iter_ld_ = types::is_zero_md(src_iter_d.md_)
            ? 0
            : src_iter_d.blocking_desc().strides[2];
    rnn.dst_iter_ld_ = types::is_zero_md(dst_iter_d.md_)
            ? 0
            : dst_iter_d.blocking_desc().strides[2];
    rnn.src_iter_c_ld_ = types::is_zero_md(src_iter_c_d.md_)
            ? 0
            : src_iter_c_d.blocking_desc().strides[2];
    rnn.dst_iter_c_ld_ = types::is_zero_md(dst_iter_c_d.md_)
            ? 0
            : dst_iter_c_d.blocking_desc().strides[2];

    /* Set the correct number of weights parts */
    rnn.is_orig_gru = utils::one_of(
            rd.cell_kind, alg_kind::vanilla_gru, alg_kind::vanilla_augru);
    rnn.n_parts_weights_layer = 1;
    rnn.parts_weights_layer[0] = rnn.n_gates;
    rnn.parts_weights_layer[1] = 0;

    rnn.n_parts_weights_iter = rnn.is_orig_gru ? 2 : 1;
    rnn.parts_weights_iter[0] = rnn.is_orig_gru ? 2 : rnn.n_gates;
    rnn.parts_weights_iter[1] = rnn.is_orig_gru ? 1 : 0;

    rnn.n_parts_weights_projection = 1;
    rnn.parts_weights_projection[0] = 1;

    rnn.n_parts_bias = 1;
    rnn.parts_bias[0] = rnn.n_bias;
    rnn.parts_bias[1] = 0;

    rnn.use_matmul = !rnn.is_brgemm && rnn.is_fwd // TODO: Enable BWD
    // TODO: Below checks are for legacy and a performance study is
    // required to avoid regressions.
#if DNNL_X64
            && IMPLICATION(
                    rnn.is_cell_dt_bf16(), !x64::mayiuse(x64::avx512_core))
#endif
            && !rnn.is_cell_dt_f32() && !rnn.is_cell_dt_int8();

    /* Decide which gemm implementation to use: packed/nonpacked jit/cblas
     * and if to merge gemm across iterations */
    const bool is_f32 = rnn.dt_conf == all_f32,
               is_bf16 = rnn.dt_conf == all_bf16;
    const bool is_gru = utils::one_of(rd.cell_kind, alg_kind::vanilla_gru,
            alg_kind::lbr_gru, alg_kind::vanilla_augru, alg_kind::lbr_augru);
    const bool is_inference = !rnn.is_training;

    // To be able to merge the GEMM on the layer input when not
    // copying, we need to have a trivial stride for the T dimension
    rnn.src_layer_is_trivial_stride = src_layer_d.blocking_desc().strides[0]
            == (rnn.src_layer_ld_ * rnn.mb);
    rnn.dst_layer_is_trivial_stride = dst_layer_d.blocking_desc().strides[0]
            == (rnn.dst_layer_ld_ * rnn.mb);

    rnn.merge_gemm_layer = !(rnn.is_brgemm || rnn.use_matmul)
            ? ((rnn.is_fwd && rnn.src_layer_is_trivial_stride)
                      || ((rd.prop_kind == prop_kind::backward)
                              && rnn.dst_layer_is_trivial_stride))
                    && (((rnn.is_fwd && rnn.mb < 128) || !rnn.is_fwd)
                            || rnn.is_int8_conf())
            : false;
    rnn.merge_gemm_iter = !(rnn.is_brgemm || rnn.use_matmul)
            ? rnn.dst_layer_is_trivial_stride && !(rnn.is_fwd || is_gru)
            : false;
    rnn.force_nocopy = false;
#if DNNL_X64
    rnn.force_nocopy = x64::mayiuse(x64::avx)
            && ((is_inference && (rnn.n_layer > 1 || rnn.mb < 100))
                    || (rnn.is_training && rnn.dhc < 500));
#endif

    /* Decide to copy bias */
    rnn.copy_bias = rnn.is_int8_conf();

    rnn.use_layer_packed_gemm = !(rnn.is_brgemm || rnn.use_matmul)
            ? utils::one_of(weights_layer_d.format_kind(), format_kind::any,
                      format_kind::rnn_packed)
                    && is_inference
                    && ((is_f32 && pack_sgemm_supported() && rnn.n_iter == 1)
                            || rnn.is_int8_conf() || is_bf16)
            : false;
    rnn.use_iter_packed_gemm = !(rnn.is_brgemm || rnn.use_matmul)
            ? utils::one_of(weights_iter_d.format_kind(), format_kind::any,
                      format_kind::rnn_packed)
                    && is_inference
                    && ((is_f32 && pack_sgemm_supported() && rnn.mb >= 16)
                            || rnn.is_int8_conf() || is_bf16)
            : false;
    rnn.use_projection_packed_gemm = !(rnn.is_brgemm || rnn.use_matmul)
            ? utils::one_of(weights_projection_d.format_kind(),
                      format_kind::any, format_kind::rnn_packed)
                    && is_inference
                    && ((is_f32 && pack_sgemm_supported() && rnn.n_iter == 1)
                            || rnn.is_int8_conf() || is_bf16)
            : false;

    rnn.diff_weights_overwrite = rd.flags & rnn_flags::diff_weights_overwrite;

#if DNNL_CPU_RUNTIME == DNNL_RUNTIME_THREADPOOL || BUILD_GEMM_KERNELS_NONE
    // XXX: Threadpool runtime may use different number of threads at execute
    // and create stages. GEMM packed API is not aware of number of threads as
    // of now. In order to synchronize all layers, GEMM pack API should be
    // modified to accept number of threads instead of taking it from
    // `dnnl_get_max_threads()`, and rnn_packed_desc_t should be updated with
    // `nthr` member to pass this information between different parts of packed
    // API, since `get_size` call happens on RNN side, while packing happens
    // on reorder side. Consider enabling later.
    // `test_iface_runtime_attr` was disabled for RNN with threadpool due to
    // this is the only working approach for int8 computations in RNN for now.
    // Consider enabling it once resolved.
    rnn.use_layer_packed_gemm = false;
    rnn.use_iter_packed_gemm = false;
    rnn.use_projection_packed_gemm = false;
#endif

    /* Set packed gemm sizes */
    /* TODO: investigate the benefit of mixing packed and non-packed weights parts */
    const auto set_pack_sizes
            = [&](bool merge, bool &do_pack, size_t &weights_pack_size,
                      int &n_parts, int *parts, size_t *parts_pack_size,
                      size_t &comp_offset, int ic, int oc, int weights_oc,
                      dim_t data_ld) -> bool {
        bool pack = true;
        weights_pack_size = 0;
        for (int p = 0; p < n_parts; p++) {
            const dim_t m_p = rnn.is_fwd ? (parts[p] * oc) : ic;
            const dim_t k_p = rnn.is_fwd ? ic : (parts[p] * oc);
            const dim_t n_p
                    = merge ? static_cast<dim_t>(rnn.mb) * rnn.n_iter : rnn.mb;
            bool pack_part = true;

            dnnl_status_t st = dnnl_success;
            switch (rnn.dt_conf) {
                case all_f32:
                    st = sgemm_pack_get_size("A", "N", "N", &m_p, &n_p, &k_p,
                            &m_p, &data_ld, &parts_pack_size[p], &pack_part);
                    break;
                case s8s8s8f32:
                case f32s8f32f32:
                case s8s8s8s8:
                case f32s8f32s8:
                    st = gemm_s8s8s32_pack_get_size("A", "N", "N", &m_p, &n_p,
                            &k_p, &m_p, &data_ld, &parts_pack_size[p],
                            &pack_part);
                    break;
                case u8u8u8f32:
                case f32u8f32f32:
                case u8u8u8u8:
                case f32u8f32u8:
                    st = gemm_s8u8s32_pack_get_size("A", "N", "N", &m_p, &n_p,
                            &k_p, &m_p, &data_ld, &parts_pack_size[p],
                            &pack_part);
                    break;
                case all_bf16:
                    st = gemm_bf16bf16f32_pack_get_size("A", "N", "N", &m_p,
                            &n_p, &k_p, &m_p, &data_ld, &parts_pack_size[p],
                            &pack_part);
                    break;
                default: assert(!"Unsupported configuration");
            }
            if (st != dnnl_success) return false;

            pack = pack && pack_part;
            weights_pack_size += rnn.n_layer * rnn.n_dir * parts_pack_size[p];
        }

        // NOTE: pack is updated only for f32. We force pack for int8
        do_pack = (rnn.dt_conf == all_f32) ? pack : true;
        comp_offset = weights_pack_size;
        const bool need_compensation = rnn.is_int8_conf();
        weights_pack_size += (need_compensation ? rnn.n_layer * rnn.n_dir : 0)
                * weights_oc * sizeof(float);

        return true;
    };
    // TODO: the activation leading dimension can vary for first layer/iteration
    if (rnn.use_layer_packed_gemm) {
        bool ok = set_pack_sizes(rnn.merge_gemm_layer,
                rnn.use_layer_packed_gemm, rnn.weights_layer_pack_size,
                rnn.n_parts_weights_layer, rnn.parts_weights_layer,
                rnn.part_weights_layer_pack_size, rnn.weights_layer_comp_offset,
                rnn.slc, rnn.dhc, rnn.n_gates * rnn.dhc,
                rnn.ws_states_layer_ld);
        if (!ok) return false;
    }

    if (rnn.use_iter_packed_gemm) {
        bool ok = set_pack_sizes(rnn.merge_gemm_iter, rnn.use_iter_packed_gemm,
                rnn.weights_iter_pack_size, rnn.n_parts_weights_iter,
                rnn.parts_weights_iter, rnn.part_weights_iter_pack_size,
                rnn.weights_iter_comp_offset, rnn.sic, rnn.dhc,
                rnn.n_gates * rnn.dhc, rnn.ws_states_iter_ld);
        if (!ok) return false;
    }

    if (rnn.use_projection_packed_gemm) {
        bool ok = set_pack_sizes(false, rnn.use_projection_packed_gemm,
                rnn.weights_projection_pack_size,
                rnn.n_parts_weights_projection, rnn.parts_weights_projection,
                rnn.part_weights_projection_pack_size,
                rnn.weights_projection_comp_offset, rnn.dhc, rnn.dic, rnn.dic,
                rnn.scratch_ht_ld);
        if (!ok) return false;
    }

    return true;
}

template <typename T>
void set_conf(rnn_conf_t &rnn, const rnn_desc_t &rd,
        const memory_desc_wrapper &weights_layer_d,
        const memory_desc_wrapper &weights_iter_d,
        const memory_desc_wrapper &weights_projection_d,
        const memory_desc_wrapper &diff_weights_layer_d,
        const memory_desc_wrapper &diff_weights_iter_d,
        const memory_desc_wrapper &diff_weights_projection_d) {

    // Set leading dimensions for input weights arrays depending on input format
    const auto set_dims
            = [&](const memory_desc_wrapper &md, int &ld, int &nld) {
                  ld = 0;
                  nld = 0;
                  if (md.is_blocking_desc()) {
                      if (is_ldigo(md)) {
                          ld = (int)md.blocking_desc().strides[2];
                          nld = md.dims()[2];
                      } else if (is_ldgoi(md)) {
                          ld = (int)md.blocking_desc().strides[4];
                          nld = md.dims()[3] * md.dims()[4];
                      } else if (is_ldoi(md)) {
                          ld = (int)md.blocking_desc().strides[3];
                          nld = md.dims()[3];
                      } else if (is_ldio(md)) {
                          ld = (int)md.blocking_desc().strides[2];
                          nld = md.dims()[2];
                      } else
                          assert(!"unsupported weights format");
                  }
              };
    set_dims(weights_layer_d, rnn.weights_layer_ld, rnn.weights_layer_nld);
    set_dims(weights_iter_d, rnn.weights_iter_ld, rnn.weights_iter_nld);
    set_dims(weights_projection_d, rnn.weights_projection_ld,
            rnn.weights_projection_nld);
    if (!rnn.is_fwd) {
        set_dims(diff_weights_layer_d, rnn.diff_weights_layer_ld,
                rnn.diff_weights_layer_nld);
        set_dims(diff_weights_iter_d, rnn.diff_weights_iter_ld,
                rnn.diff_weights_iter_nld);
        set_dims(diff_weights_projection_d, rnn.diff_weights_projection_ld,
                rnn.diff_weights_projection_nld);
    }

    assert(weights_layer_d.data_type() == weights_iter_d.data_type());
    assert(IMPLICATION(diff_weights_layer_d.ndims() != 0,
            (diff_weights_layer_d.data_type()
                    == diff_weights_iter_d.data_type())));

    /* Set workspace sizes to store:
     * states to compute a pass
     * diff states to compute bwd pass (training onl)y
     * intermediate results from the gates
     */

    assert(sizeof(typename T::src_layer_t) == sizeof(typename T::dst_layer_t));
    assert(sizeof(typename T::src_iter_t) == sizeof(typename T::dst_iter_t));
}

template <typename T>
void set_workspace_sizes(rnn_conf_t &rnn, const rnn_desc_t &rd) {
    rnn.use_workspace = rnn.is_training;
    // TODO: for inference, we can make ws_states_* smaller, but
    // dependant of the grid execution though
    rnn.ws_states_layer_size = (size_t)(rnn.n_layer + 1) * rnn.n_dir
            * (rnn.n_iter + 1) * rnn.mb * rnn.ws_states_layer_ld
            * sizeof(typename T::src_layer_t);
    rnn.ws_states_iter_size = (size_t)(rnn.n_layer + 1) * rnn.n_dir
            * (rnn.n_iter + 1) * rnn.mb * rnn.ws_states_iter_ld
            * sizeof(typename T::src_iter_t);
    bool is_lstm = rd.cell_kind == dnnl_vanilla_lstm;
    rnn.ws_states_iter_c_size = is_lstm ? (size_t)(rnn.n_layer + 1) * rnn.n_dir
                    * (rnn.n_iter + 1) * rnn.mb * rnn.ws_states_iter_c_ld
                    * types::data_type_size(rnn.src_iter_c_dt)
                                        : 0;

    rnn.ws_diff_states_layer_size = rnn.is_training
            ? (size_t)(rnn.n_layer + 1) * rnn.n_dir * (rnn.n_iter + 1) * rnn.mb
                    * rnn.ws_diff_states_layer_ld
                    * sizeof(typename T::gemm_acc_t)
            : (size_t)0;
    rnn.ws_diff_states_iter_size = rnn.is_training
            ? (size_t)(rnn.n_layer + 1) * rnn.n_dir * (rnn.n_iter + 1) * rnn.mb
                    * rnn.ws_diff_states_iter_ld
                    * sizeof(typename T::gemm_acc_t)
            : (size_t)0;
    rnn.ws_diff_states_iter_c_size = rnn.is_training && is_lstm
            ? (size_t)(rnn.n_layer + 1) * rnn.n_dir * (rnn.n_iter + 1) * rnn.mb
                    * rnn.ws_diff_states_iter_c_ld
                    * sizeof(typename T::gemm_acc_t)
            : (size_t)0;

    rnn.ws_gates_size = rnn.is_training
            ? (size_t)rnn.n_layer * rnn.n_dir * rnn.n_iter * rnn.ws_gates_nld
                    * rnn.ws_gates_ld * sizeof(typename T::gates_t)
            : (size_t)0;
    rnn.ws_ht_size = rnn.is_training
            ? (size_t)rnn.n_layer * rnn.n_dir * rnn.n_iter * rnn.ws_ht_nld
                    * rnn.ws_ht_ld * sizeof(typename T::dst_iter_t)
            : (size_t)0;
    rnn.n_iter_scratch_gates
            = (rnn.merge_gemm_layer || rnn.merge_gemm_iter) ? rnn.n_iter : 1;
    rnn.scratch_gates_size = sizeof(typename T::scratch_t)
            * rnn.n_iter_scratch_gates * rnn.scratch_gates_nld
            * rnn.scratch_gates_ld;
    rnn.scratch_ht_size
            = sizeof(typename T::ht_t) * rnn.scratch_ht_nld * rnn.scratch_ht_ld;
    rnn.scratch_diff_ht_size = rnn.is_training ? sizeof(typename T::gemm_acc_t)
                    * rnn.scratch_diff_ht_nld * rnn.scratch_diff_ht_ld
                                               : (size_t)0;

    /* set other sizes */
    /// scratchpad buffer for each cell to hold intermediate data in gru/lbr_gru
    rnn.scratch_cell_size = rnn.is_lbr
            ? (size_t)rnn.scratch_gates_nld * rnn.scratch_gates_ld
                    * sizeof(typename T::gemm_acc_t)
            : (utils::one_of(rd.cell_kind, alg_kind::vanilla_gru,
                       alg_kind::vanilla_augru)
                            ? (size_t)rnn.ws_states_layer_nld
                                    * rnn.ws_states_layer_ld
                                    * sizeof(typename T::gemm_acc_t)
                            : 0);
    /// workspace needed for lbr GRU
    rnn.ws_per_cell = (size_t)rnn.is_lbr * rnn.mb * rnn.dhc
            * sizeof(typename T::gemm_acc_t);
    rnn.ws_grid_comp_size = (size_t)rnn.is_lbr * rnn.is_training * rnn.n_layer
            * rnn.n_dir * rnn.n_iter * rnn.ws_per_cell * sizeof(float);
    /// bias ws needed to add compensation in int8
    rnn.ws_bias_size = (size_t)rnn.n_layer * rnn.n_dir * rnn.n_bias * rnn.dhc
            * types::data_type_size(rnn.bias_dt);
}

void set_offsets(const rnn_conf_t &rnn, size_t &ws_gates_offset,
        size_t &ws_ht_offset, size_t &ws_state_layer_offset,
        size_t &ws_states_iter_offset, size_t &ws_states_iter_c_offset,
        size_t &ws_diff_states_layer_offset, size_t &ws_diff_states_iter_offset,
        size_t &ws_diff_states_iter_c_offset, size_t &ws_grid_comp_offset,
        size_t &ws_bias_offset, size_t &scratch_gates_offset,
        size_t &scratch_ht_offset, size_t &scratch_diff_ht_offset,
        size_t &scratch_cell_offset, size_t &scratchpad_size,
        size_t &workspace_size);

void get_scratchpad_and_workspace_sizes(
        const rnn_conf_t &rnn, size_t &scratchpad_size, size_t &workspace_size);
status_t set_expected_desc(rnn_conf_t &rnn, memory_desc_t &weights_md,
        weights_type_t weights_type);
status_t set_good_strides(memory_desc_t &weights_md, format_tag_t tag);

using byte = unsigned char;
template <size_t Tdims>
struct raw_array_offset_calculator_t {
    template <typename... Targs>
    raw_array_offset_calculator_t(
            const byte *base, const dim_t dt_size, Targs... Fargs)
        : base_ptr_(base), dt_size_(dt_size), dims_ {Fargs...} {}

    template <typename... Targs>
    raw_array_offset_calculator_t(std::nullptr_t, Targs... Fargs) = delete;

    template <typename... Targs>
    inline const void *operator()(Targs... Fargs) const {
        assert(static_cast<bool>(base_ptr_));
        return base_ptr_ + (offset(1, Fargs...) * dt_size_);
    }

private:
    template <typename... Targs>
    inline size_t offset(size_t const dimension, size_t element) const {
        return element;
    }
    template <typename... Targs>
    inline size_t offset(
            size_t const dimension, size_t theta, size_t element) const {
        return element + (dims_[dimension] * theta);
    }

    template <typename... Targs>
    inline size_t offset(size_t const dimension, size_t theta, size_t element,
            Targs... Fargs) const {
        const size_t t_prime = element + (dims_[dimension] * theta);
        return offset(dimension + 1, t_prime, Fargs...);
    }

    const byte *const base_ptr_;
    const dim_t dt_size_;
    const int dims_[Tdims];
};

template <typename... Targs>
raw_array_offset_calculator_t<sizeof...(Targs)> make_raw_aoc(
        const void *base, const dim_t dt_size, Targs... Fargs) {
    return raw_array_offset_calculator_t<sizeof...(Targs)>(
            static_cast<const byte *>(base), dt_size,
            std::forward<Targs>(Fargs)...);
}

template <typename T>
struct ws_gates_aoc {
    ws_gates_aoc(const rnn_conf_t &rnn, T *data)
        : gates_(data, rnn.ws_gates_nld, rnn.ws_gates_ld), DHC_(rnn.dhc) {}
    T &operator()(int batch, int gate, int dhc) const {
        return gates_(batch, gate * DHC_ + dhc);
    }

private:
    const dnnl::impl::utils::array_offset_calculator<T, 2> gates_;
    const int DHC_;
};
using ws_gates_aoc_t = ws_gates_aoc<float>;
using ws_gates_aoc_s32_t = ws_gates_aoc<int32_t>;

template <typename T>
struct ws_ht_aoc {
    ws_ht_aoc(const rnn_conf_t &rnn, T *data)
        : ht_(data, rnn.ws_ht_nld, rnn.ws_ht_ld) {}
    T &operator()(int batch, int dhc) const { return ht_(batch, dhc); }

private:
    const dnnl::impl::utils::array_offset_calculator<T, 2> ht_;
};

template <typename T>
struct scratch_gates_aoc {
    scratch_gates_aoc(const rnn_conf_t &rnn, T *data)
        : gates_(data, rnn.scratch_gates_nld, rnn.scratch_gates_ld)
        , DHC_(rnn.dhc) {}
    T &operator()(int batch, int gate, int dhc) const {
        return gates_(batch, gate * DHC_ + dhc);
    }

private:
    const dnnl::impl::utils::array_offset_calculator<T, 2> gates_;
    const int DHC_;
};
using scratch_gates_aoc_t = scratch_gates_aoc<float>;
using scratch_gates_aoc_s32_t = scratch_gates_aoc<int32_t>;

template <typename T>
struct scratch_ht_aoc {
    scratch_ht_aoc(const rnn_conf_t &rnn, T *data)
        : ht_(data, rnn.scratch_ht_nld, rnn.scratch_ht_ld) {}
    T &operator()(int batch, int dhc) const { return ht_(batch, dhc); }

private:
    const dnnl::impl::utils::array_offset_calculator<T, 2> ht_;
};
using scratch_ht_aoc_t = scratch_ht_aoc<float>;
using scratch_ht_aoc_s32_t = scratch_ht_aoc<int32_t>;

template <typename T>
struct weights_peephole_aoc_t {
    weights_peephole_aoc_t(const rnn_conf_t &rnn, T *data)
        : weights_peephole_(data, 3, rnn.dhc) {}
    T &operator()(int g, int dhc) const { return weights_peephole_(g, dhc); }

private:
    const utils::array_offset_calculator<T, 2> weights_peephole_;
};

float to_float(const void *data, const data_type_t dt);

struct bias_linear_exec_aoc_t {
    bias_linear_exec_aoc_t(const rnn_conf_t &rnn, void **bias)
        : bias_dt_(rnn.bias_dt), bias_present_(static_cast<bool>(bias)) {

        if (bias_dt_ == data_type::f32)
            new (std::addressof(bias_f32_aoc_))
                    utils::array_offset_calculator<float *, 3>(
                            reinterpret_cast<float **>(bias), rnn.n_layer,
                            rnn.n_dir, rnn.n_parts_bias);
        else if (bias_dt_ == data_type::bf16)
            new (std::addressof(bias_bf16_aoc_))
                    utils::array_offset_calculator<bfloat16_t *, 3>(
                            reinterpret_cast<bfloat16_t **>(bias), rnn.n_layer,
                            rnn.n_dir, rnn.n_parts_bias);
        else if (bias_dt_ == data_type::f16)
            new (std::addressof(bias_f16_aoc_))
                    utils::array_offset_calculator<float16_t *, 3>(
                            reinterpret_cast<float16_t **>(bias), rnn.n_layer,
                            rnn.n_dir, rnn.n_parts_bias);
        else
            assert("unsupported data type");
    }

    void **operator()(int layer, int dir) const {
        if (bias_present_) {
            if (bias_dt_ == data_type::f32)
                return reinterpret_cast<void **>(
                        &bias_f32_aoc_.operator()(layer, dir, 0));
            else if (bias_dt_ == data_type::bf16)
                return reinterpret_cast<void **>(
                        &bias_bf16_aoc_.operator()(layer, dir, 0));
            else if (bias_dt_ == data_type::f16)
                return reinterpret_cast<void **>(
                        &bias_f16_aoc_.operator()(layer, dir, 0));
            else
                assert("unsupported data type");
        }

        return nullptr;
    }

    ~bias_linear_exec_aoc_t() {
        if (bias_dt_ == data_type::f32)
            bias_f32_aoc_.~array_offset_calculator<float *, 3>();
        else if (bias_dt_ == data_type::bf16)
            bias_bf16_aoc_.~array_offset_calculator<bfloat16_t *, 3>();
        else if (bias_dt_ == data_type::f16)
            bias_f16_aoc_.~array_offset_calculator<float16_t *, 3>();
        else
            assert("unsupported data type");
    }

    DNNL_DISALLOW_COPY_AND_ASSIGN(bias_linear_exec_aoc_t);
    bias_linear_exec_aoc_t(bias_linear_exec_aoc_t &&) = delete;
    bias_linear_exec_aoc_t &operator=(bias_linear_exec_aoc_t &&) = delete;

private:
    data_type_t bias_dt_;
    bool bias_present_;
    union {
        utils::array_offset_calculator<float *, 3> bias_f32_aoc_;
        utils::array_offset_calculator<bfloat16_t *, 3> bias_bf16_aoc_;
        utils::array_offset_calculator<float16_t *, 3> bias_f16_aoc_;
    };
};

template <typename T>
struct ws_states_layer_aoc {
    ws_states_layer_aoc(const rnn_conf_t &rnn, T *data, int leading_dim)
        : state_(data, rnn.ws_states_layer_nld, leading_dim) {}
    ws_states_layer_aoc(const rnn_conf_t &rnn, T *data)
        : state_(data, rnn.ws_states_layer_nld, rnn.ws_states_layer_ld) {}
    T &operator()(int batch, int dhc) const { return state_(batch, dhc); }

private:
    const dnnl::impl::utils::array_offset_calculator<T, 2> state_;
};

template <typename T>
struct ws_states_iter_aoc {
    ws_states_iter_aoc(const rnn_conf_t &rnn, T *data, int leading_dim)
        : state_(data, rnn.ws_states_iter_nld, leading_dim) {}
    ws_states_iter_aoc(const rnn_conf_t &rnn, T *data)
        : state_(data, rnn.ws_states_iter_nld, rnn.ws_states_iter_ld) {}
    T &operator()(int batch, int dhc) const { return state_(batch, dhc); }

private:
    const dnnl::impl::utils::array_offset_calculator<T, 2> state_;
};

template <typename T>
struct augru_attention_aoc {
    augru_attention_aoc(const rnn_conf_t &rnn, T *data)
        : state_(data, rnn.mb) {}
    T &operator()(int batch) const { return state_(batch); }

private:
    const dnnl::impl::utils::array_offset_calculator<T, 1> state_;
};

template <typename T>
struct ws_diff_states_layer_aoc {
    ws_diff_states_layer_aoc(const rnn_conf_t &rnn, T *data)
        : diff_states_layer_(data, rnn.ws_diff_states_layer_nld,
                rnn.ws_diff_states_layer_ld) {}
    T &operator()(int batch, int dhc) const {
        return diff_states_layer_(batch, dhc);
    }

private:
    const dnnl::impl::utils::array_offset_calculator<T, 2> diff_states_layer_;
};

template <typename T>
struct ws_diff_states_iter_aoc {
    ws_diff_states_iter_aoc(const rnn_conf_t &rnn, T *data)
        : diff_states_iter_(data, rnn.ws_diff_states_iter_nld,
                rnn.ws_diff_states_iter_ld) {}
    T &operator()(int batch, int dhc) const {
        return diff_states_iter_(batch, dhc);
    }

private:
    const dnnl::impl::utils::array_offset_calculator<T, 2> diff_states_iter_;
};

template <typename T>
struct ws_diff_states_iter_c_aoc {
    ws_diff_states_iter_c_aoc(const rnn_conf_t &rnn, T *data)
        : diff_states_iter_c_(data, rnn.ws_diff_states_iter_c_nld,
                rnn.ws_diff_states_iter_c_ld) {}
    T &operator()(int batch, int dhc) const {
        return diff_states_iter_c_(batch, dhc);
    }

private:
    const dnnl::impl::utils::array_offset_calculator<T, 2> diff_states_iter_c_;
};

struct ws_diff_w_iter_aoc_t {
    ws_diff_w_iter_aoc_t(const rnn_conf_t &rnn, float *data)
        : diff_weights_iter_(
                data, rnn.diff_weights_iter_nld, rnn.diff_weights_iter_ld)
        , DHC_(rnn.dhc) {}
    float &operator()(int sic, int gate, int dhc) const {
        return diff_weights_iter_(sic, gate * DHC_ + dhc);
    }

private:
    const dnnl::impl::utils::array_offset_calculator<float, 2>
            diff_weights_iter_;
    const int DHC_;
};

const void *inc_ptr(const void *data, data_type_t data_type, int offset);
void *inc_ptr(void *data, data_type_t data_type, int offset);

} // namespace rnn_utils
} // namespace cpu
} // namespace impl
} // namespace dnnl
#endif
