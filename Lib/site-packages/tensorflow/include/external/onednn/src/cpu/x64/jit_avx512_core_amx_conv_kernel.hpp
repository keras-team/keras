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

#ifndef CPU_X64_JIT_AVX512_CORE_AMX_CONV_KERNEL_HPP
#define CPU_X64_JIT_AVX512_CORE_AMX_CONV_KERNEL_HPP

#include <queue>

#include "common/c_types_map.hpp"
#include "common/memory_tracking.hpp"

#include "cpu/x64/injectors/jit_uni_postops_injector.hpp"
#include "cpu/x64/jit_avx512_core_bf16cvt.hpp"
#include "cpu/x64/jit_generator.hpp"
#include "cpu/x64/jit_primitive_conf.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {

/* This struct computes the compensation for src_zero_point related to
 * padding */
struct jit_avx512_core_amx_compute_zp_pbuff_t : public jit_generator {

    using reg64_t = const Xbyak::Reg64;

    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_avx512_core_amx_compute_zp_pbuff_t)

    jit_avx512_core_amx_compute_zp_pbuff_t(const jit_conv_conf_t &ajcp)
        : jit_generator(jit_name(), avx512_core_amx), jcp(ajcp) {}

    static const int max_regs_ur = 30;

private:
    jit_conv_conf_t jcp;

    typedef enum { no_last_block, last_ic_block } ic_block_t;
    const int ic_inner_block = 4;

    Xbyak::Label permb_idx_label;
    Xbyak::Label ic_mask_label;

    const reg64_t reg_zp_pbuff = r8;
    const reg64_t reg_src_zero_point = r9;
    const reg64_t reg_filt = r10;
    const reg64_t aux_reg_filt = r11;
    const reg64_t aux_reg_filt_d = r15;

    const reg64_t reg_oc_blocks = r12;
    const reg64_t reg_icb = r13;
    const reg64_t reg_oi = r14;
    const reg64_t reg_kj = rax;
    const reg64_t reg_ki = rbx;
    const reg64_t reg_overflow = reg_kj;
    const reg64_t reg_scratch = rsi;

    const Xbyak::Zmm zmm_one = Xbyak::Zmm(31);
    const Xbyak::Zmm zmm_permb = Xbyak::Zmm(30);

    const Xbyak::Opmask kmask_ic_block = Xbyak::Opmask(1);
    const Xbyak::Opmask ktail_mask = Xbyak::Opmask(2);

    void prepare_output(int ur_w);
    void store_output(int ur_w, bool last_oc_block_flag);
    void compute_ker(int ur_w, int pad_l, int pad_r,
            ic_block_t last_ic_block_flag, bool padded);
    void kh_loop(int ur_w, int pad_l, int pad_r, ic_block_t last_ic_block_flag,
            bool handle_h_pad);
    void kd_loop(int ur_w, int pad_l, int pad_r, ic_block_t last_ic_block_flag,
            bool handle_h_pad);
    void icb_loop(int ur_w, int pad_l, int pad_r, bool handle_h_pad);
    void unroll_width(const bool h_padding);

    void generate() override;

    Xbyak::Zmm zmm_out(int i_ur, int i_oc) {
        int idx = i_ur * jcp.nb_oc_blocking + i_oc;
        assert(idx < max_regs_ur);
        return Xbyak::Zmm(idx);
    }
    int get_ow_start(int ki, int pad_l) {
        return nstl::max(0,
                utils::div_up(pad_l - ki * (jcp.dilate_w + 1), jcp.stride_w));
    }
    int get_ow_end(int ur_w, int ki, int pad_r) {
        int filter_overlap = pad_r - (jcp.kw - 1 - ki) * (jcp.dilate_w + 1);
        return ur_w - nstl::max(0, utils::div_up(filter_overlap, jcp.stride_w));
    }
};

struct jit_avx512_core_amx_copy_to_wbuffer_t : public jit_generator {
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_avx512_core_amx_copy_to_wbuffer_t)

    using reg64_t = Xbyak::Reg64;

    jit_avx512_core_amx_copy_to_wbuffer_t(const jit_conv_conf_t &ajcp)
        : jit_generator(jit_name(), avx512_core_amx), jcp(ajcp) {}

private:
    jit_conv_conf_t jcp;

    const reg64_t reg_src = rax;
    const reg64_t reg_dst = rbx;
    const reg64_t reg_tmp = rdx;

    const Xbyak::Opmask kmask_load = k2;

    const Xbyak::Zmm zmm_src = zmm0;
    const Xbyak::Zmm zmm_dst = zmm1;
    const Xbyak::Zmm zmm_idx = zmm2;
    const Xbyak::Zmm zmm_zero = zmm3;

    void generate() override;
};

struct jit_avx512_core_amx_copy_to_pbuffer_t : public jit_generator {
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_avx512_core_amx_copy_to_pbuffer_t)

    using reg64_t = Xbyak::Reg64;

    jit_avx512_core_amx_copy_to_pbuffer_t(const jit_conv_conf_t &ajcp)
        : jit_generator(jit_name(), avx512_core_amx), jcp(ajcp) {}

private:
    jit_conv_conf_t jcp;

    const reg64_t reg_inp_ptr = r15;
    const reg64_t reg_out_ptr = r14;

    const reg64_t reg_aux_inp_ptr = r13;
    const reg64_t reg_aux_out_ptr = r12;

    const reg64_t reg_khp = r10;

    /* relow stuff */
    const reg64_t reg_kht = r11;
    const reg64_t reg_tov = r9;
    const reg64_t reg_bov = r8;
    const reg64_t reg_kwp = rax;
    const reg64_t reg_lov = reg_aux_inp_ptr;
    const reg64_t reg_rov = rbx;
    const reg64_t reg_save_out_ptr = rdx;
    const reg64_t reg_cnt = rbp;
    /* relow stuff */

    /* non-relow stuff */
    const reg64_t reg_kdp = abi_not_param1;
    const reg64_t reg_kdc = rbp;
    const reg64_t reg_khc = r11;

    const reg64_t reg_kh_over = r8;
    const reg64_t reg_tover = rax;
    const reg64_t reg_bover = rbx;

    const reg64_t reg_owb = rdx;
    /* non-relow stuff */

    const reg64_t reg_tmp = rsi;

    const Xbyak::Opmask &ktail_mask = k2;

    const Xbyak::Ymm &ymm_tmp = ymm0;
    const Xbyak::Zmm &zmm_tmp = zmm0;
    const Xbyak::Zmm &zmm_zero = zmm1;

    void generate() override;
    void copy_row(int icb);
    void copy_row_body(int lpad, int iw_len, int icb);
    void copy_row_reduced_lowering();
};

struct jit_avx512_core_amx_fwd_kernel_t : public jit_generator {
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_avx512_core_amx_fwd_kernel_t)

    jit_avx512_core_amx_fwd_kernel_t(const jit_conv_conf_t &ajcp,
            const primitive_attr_t &attr, const memory_desc_t &dst_md);

    status_t create_kernel() override;

    static status_t init_conf(jit_conv_conf_t &jcp,
            const convolution_desc_t &cd, memory_desc_t &src_pd,
            memory_desc_t &weights_pd, memory_desc_t &dst_pd,
            memory_desc_t &bias_pd, primitive_attr_t &attr, int nthreads);
    static status_t init_scratchpad(memory_tracking::registrar_t &scratchpad,
            const jit_conv_conf_t &jcp, const primitive_attr_t &attr);

    inline int accum_with_upper_bound(
            int upper_bound, int lower_value, int upper_value) {
        return nstl::min(upper_bound,
                nstl::min(upper_bound, lower_value)
                        + nstl::max(0, upper_bound - upper_value));
    }

    /*  Calculate and store the limits relevant to 'ow_block'. These limits
     * allow the driver code to determine which 'ow_block' is currently being
     * executed. There can be at most 5 different 'ow_block', each corresponding
     * to:
     * - l_pad block
     * - middle block (no padding)
     * - middle & r_pad shift block
     * - r_pad (full) block
     * - r_pad tail
     *  */
    static void set_ow_blk_limits(jit_conv_conf_t &jcp);

    /*  Calculate and store the limits relevant to each 'oh_block'. Each
     * 'oh_block' size is 'nb_oh_blocking * oh_per_tile'. These limits allow
     * the driver code to determine which 'oh_block' is currently being
     * executed, and what is the oh value required to advance the limits index.
     *
     *  There can be at most 6 different 'oh_blk', depending on the sizes of
     * 't_pad_output', 'b_pad_output' and their overlap with
     * 'nb_oh_blocking * oh_per_tile'.
     *
     *  For example, given the following input dimensions of {height_size = 12,
     * oh_blk_size = 2, top_padding = 5 (t_pad), bottom_padding = 2 (b_pad)},
     * the 4 output height blocks and limits are:
     *
     *          H: _           H_blks:_  Limits:
     *          0 | |               0|X|
     *          1 | |                |X|_4
     *          2 | | t_pad
     *          3 | |                 _
     *          4 |_|               1|X|
     *          5 | |                |_|_5
     *          6 | |               2| |
     *          7 | |                |_|_9
     *          8 | |
     *          9 |_|                 _
     *          10| | b_pad         3|X|
     *          11|_|                |X|_11
     *
     *                        -where 'x' represents
     *                        an 'h_blk' with output
     *                        padding.
     *  */
    static void set_oh_blk_limits(jit_conv_conf_t &jcp);

    void tile_configure(char *tcfg_buff);

    jit_conv_conf_t jcp;
    const primitive_attr_t &attr_;

    const jit_avx512_core_amx_copy_to_pbuffer_t &copy_to_pbuffer() const {
        return *copy_to_pbuffer_;
    }
    const jit_avx512_core_amx_copy_to_wbuffer_t &copy_to_wbuffer() const {
        return *copy_to_wbuffer_;
    }
    const jit_avx512_core_amx_compute_zp_pbuff_t &zp_pbuff_kernel() const {
        return *zp_pbuff_kernel_;
    }

private:
    constexpr static int isa_simd_width_
            = cpu_isa_traits<avx512_core>::vlen / sizeof(float);
    std::unique_ptr<injector::jit_uni_postops_injector_t<avx512_core>>
            postops_injector_;
    std::unique_ptr<jit_avx512_core_amx_copy_to_pbuffer_t> copy_to_pbuffer_;
    std::unique_ptr<jit_avx512_core_amx_copy_to_wbuffer_t> copy_to_wbuffer_;
    std::unique_ptr<jit_avx512_core_amx_compute_zp_pbuff_t> zp_pbuff_kernel_;

    enum {
        zmm_idx_limit_bf16 = 29,
        zmm_idx_limit_int8 = 27,
    };

    int prv_width_ = 0;
    int row_count_ = 0;
    bool is_store_done_ = false;
    bool is_buffer_empty_ = true;

    struct w_pad_output {
        int l_pad_output;
        int r_pad_output;
        w_pad_output(int l_, int r_) : l_pad_output(l_), r_pad_output(r_) {}
    };
    std::queue<w_pad_output> w_padding;

    /* data regs */
    const Xbyak::Reg64 reg_inp_ptr = r15;
    const Xbyak::Reg64 reg_wei_ptr = r14;
    const Xbyak::Reg64 reg_out_ptr = r13;
    const Xbyak::Reg64 reg_wsp_ptr = r12;

    const Xbyak::Reg64 reg_kd = r9;

    const Xbyak::Reg64 reg_bias = r11;
    const Xbyak::Reg64 reg_ptr_scales = r10;
    const Xbyak::Reg64 reg_ptr_sum_scale = r9;
    const Xbyak::Reg64 reg_ptr_sum_zp = abi_not_param1;
    const Xbyak::Reg64 reg_aux_saturation = reg_ptr_sum_scale;

    const Xbyak::Reg64 reg_inp_stride = rbx;
    const Xbyak::Reg64 reg_wei_stride = rdx;
    // zero-point computation
    const Xbyak::Reg64 reg_zp_compensation = rax;
    const Xbyak::Reg64 reg_src_zero_point = r8;
    const Xbyak::Reg64 reg_zero_point_pbuff = rsi;
    const Xbyak::Reg64 reg_dst_zero_point = abi_not_param1;
    const Xbyak::Reg64 reg_dst_scale = reg_dst_zero_point;

    // rbp - reserved for EVEX compression
    const Xbyak::Reg64 reg_last_h = abi_not_param1;
    const Xbyak::Reg64 reg_jmp_blk = reg_last_h;

    // temporary, used in generate() function only
    const Xbyak::Reg64 reg_oc_blocks = rax;
    const Xbyak::Reg64 reg_tmp = r8;

    const Xbyak::Opmask &ktail_mask = k2;

    const Xbyak::Zmm &zmm_bias = zmm31;
    const Xbyak::Zmm &zmm_saturation = zmm_bias;
    const Xbyak::Zmm &zmm_zero = zmm30;
    const Xbyak::Zmm &zmm_prev_dst = zmm29;
    const Xbyak::Zmm &zmm_sum_zp = zmm26;
    /* zero-point */
    const Xbyak::Zmm &zmm_zp = zmm29;
    const Xbyak::Zmm &zmm_src_zp = zmm28;
    const Xbyak::Zmm &zmm_dst_zp = zmm27;
    /* dst scale */
    const Xbyak::Zmm &zmm_dst_scale = zmm25;

    const Xbyak::Reg64 bin_injector_helper_reg_1 = r14;
    const Xbyak::Reg64 bin_injector_helper_reg_2 = r15;
    const Xbyak::Reg64 bin_injector_helper_reg_3 = r11;

    // AUX: Steps, shifts and offsets
    size_t get_inp_icb_step() const;
    size_t get_wei_icb_step() const;
    size_t get_inp_d_step() const;
    size_t get_inp_h_step() const;
    size_t get_wei_d_step() const;
    size_t get_wei_h_step() const;
    size_t get_out_ocb_offset(int ohb, int ocb, size_t typesize) const;
    size_t get_out_row_offset(int ohb, int ocb, int j, size_t typesize) const;
    size_t get_out_shift(int width, size_t typesize) const;
    size_t get_wsp_ocb_offset(int ohb, int ocb) const;
    size_t get_wsp_row_offset(int ohb, int ocb, int j) const;
    size_t get_wsp_shift() const;
    size_t get_wei_offset(int ocb, int kw) const;
    size_t get_inp_shift() const;
    size_t get_inp_offset(int ohb, int kw) const;
    size_t get_zp_comp_offset(int ocb, int zp_h, int zp_w) const;
    int get_zp_index_offset(
            int index, int mid, int s_pad_output, int e_pad_output);

    int get_out_tensor(int h, int i, bool is_h_tail = false) const;
    int get_inp_tensor(int h, bool is_h_tail = false) const;
    int get_wei_tensor(int i) const;

    void prepare_output(int tail);
    void init_runtime_counters(bool start_with_last_tile_block);
    size_t reduce_to_block(const int block_size, const int pad_output);
    size_t reduce_to_blocked_dims(const int dim_size, const int block_size,
            const int s_pad_output, const int e_pad_output);
    void cvt2ps(data_type_t type_in, const Xbyak::Zmm &ymm_in,
            const Xbyak::Operand &op, bool mask_flag = false);
    Xbyak::Zmm zmm_out(const int idx) {
        const int upper_limit = jcp.src_dt == data_type::bf16
                ? zmm_idx_limit_bf16
                : zmm_idx_limit_int8;
        assert(upper_limit > idx);
        MAYBE_UNUSED(upper_limit);
        return Xbyak::Zmm(idx);
    }
    Xbyak::Ymm ymm_mask(
            const Xbyak::Ymm &zmm_in, bool mask_flag, bool store = false);
    Xbyak::Zmm zmm_mask(
            const Xbyak::Zmm &zmm_in, bool mask_flag, bool store = false);
    void apply_sum(const Xbyak::Zmm &zmm_out, const float *p_sum_scale,
            const int32_t *p_sum_zp, const Xbyak::Address &addr,
            const bool mask_flag);
    void apply_postops(const Xbyak::Zmm &zmm_out, const float *p_sum_scale,
            const int32_t *p_sum_zp, const Xbyak::Address &addr,
            const size_t off, const bool mask_flag);
    inline void store_output_ymm_bf16(
            const int idx, const Xbyak::Address &addr, const bool mask_flag);
    void store_output_vector_bf16(
            const Xbyak::Zmm &zmm_out, int ocb, int h, int w);
    void store_output_vector_int8(const Xbyak::Zmm &zmm_out, int ocb, int h,
            int w, const bool compute_zp, const int zp_h, const int zp_w);
    void store_output_vector(const Xbyak::Zmm &zmm_out, int ocb, int h, int w,
            const bool compute_zp = false, const int zp_h = 0,
            const int zp_w = 0);
    void store_output(int width, int tail, bool do_store,
            const bool handle_h_block, const int t_pad_output,
            const int b_pad_output, const int l_pad_output,
            const int r_pad_output, const bool is_last_oh_block,
            const bool zp_3d_pad = false);
    void interleave_store(int width, int const t_pad_output,
            int const b_pad_output, const bool zp_3d_pad = false);
    void compute_icb_loop(int width, bool do_store, const bool handle_h_block,
            const int t_pad_output, const int b_pad_output,
            const int l_pad_output, const int r_pad_output,
            const bool zp_3d_pad, const bool is_last_oh_block = false);
    void dispatch_icb_loop(int width, bool do_store, const int l_pad_output,
            const int r_pad_output, const bool zp_3d_pad);
    void dispatch_zp_3d_compute(int width, bool do_store,
            const int l_pad_output, const int r_pad_output);
    void compute_ow_loop();

    void generate() override;
};

struct jit_avx512_core_amx_bwd_data_copy_kernel_t : public jit_generator {
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_avx512_core_amx_bwd_data_copy_kernel_t)

    using reg64_t = Xbyak::Reg64;

    jit_avx512_core_amx_bwd_data_copy_kernel_t(jit_conv_conf_t &ajcp)
        : jit_generator(jit_name(), avx512_core_amx), jcp(ajcp) {}

private:
    jit_conv_conf_t jcp;

    // pointers
    const reg64_t reg_ptr_inp = r15;
    const reg64_t reg_ptr_out = r14;

    // auxiliary pointers
    const reg64_t reg_ptr_aux_inp_h = r13;
    const reg64_t reg_ptr_aux_inp_w = r12;
    const reg64_t reg_ptr_aux_out = r11;

    // variables
    const reg64_t reg_khp = r10; // kh padding
    const reg64_t reg_tov = r9; // top overflow
    const reg64_t reg_bov = reg_tov; // bottom overflow
    const reg64_t reg_kwp = rax; // kw padding
    const reg64_t reg_lov = rbx; // left overflow
    const reg64_t reg_rov = abi_not_param1; // right overflow
    const reg64_t reg_kd = r8; // 3d filter

    // counters
    const reg64_t reg_cnt_khp = rdx;
    const reg64_t reg_cnt_tmp = rbp;
    const reg64_t reg_cnt_ocb = rsi;

    const reg64_t reg_tmp = reg_cnt_tmp;

    const Xbyak::Opmask ktail_mask = k2;

    const Xbyak::Zmm zmm_tmp = zmm1;
    const Xbyak::Zmm zmm_zero = zmm0;

    void generate() override;
    void copy_row(bool is_masked);
    void kd_loop(bool is_masked);
};

struct jit_avx512_core_amx_bwd_data_kernel_t : public jit_generator {
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_avx512_core_amx_bwd_data_kernel_t)

    jit_avx512_core_amx_bwd_data_kernel_t(
            const jit_conv_conf_t &ajcp, const primitive_attr_t &attr)
        : jit_generator(jit_name(), avx512_core_amx)
        , jcp(ajcp)
        , attr_(attr)
        , eltwise_injector_(nullptr) {
        if (jcp.with_eltwise)
            eltwise_injector_ = new jit_uni_eltwise_injector_f32<avx512_core>(
                    this, jcp.eltwise);
        bwd_data_copy_kernel_
                = new jit_avx512_core_amx_bwd_data_copy_kernel_t(jcp);
    }
    status_t create_kernel() override {
        CHECK(jit_generator::create_kernel());
        CHECK(bwd_data_copy_kernel_->create_kernel());
        return status::success;
    }
    ~jit_avx512_core_amx_bwd_data_kernel_t() {
        delete eltwise_injector_;
        delete bwd_data_copy_kernel_;
    }

    static bool post_ops_ok(const jit_conv_conf_t &jcp, primitive_attr_t &attr);

    static status_t init_conf(jit_conv_conf_t &jcp,
            const convolution_desc_t &cd, memory_desc_t &diff_src_pd,
            memory_desc_t &weights_pd, memory_desc_t &diff_dst_pd,
            memory_desc_t *bias_pd, primitive_attr_t &attr, int nthreads);
    static void init_scratchpad(memory_tracking::registrar_t &scratchpad,
            const jit_conv_conf_t &jcp, const primitive_attr_t &attr);

    void tile_configure(char *tcfg_buff);

    jit_conv_conf_t jcp;
    const primitive_attr_t &attr_;

    const jit_avx512_core_amx_bwd_data_copy_kernel_t &
    bwd_data_copy_kernel() const {
        return *bwd_data_copy_kernel_;
    }

private:
    jit_uni_eltwise_injector_f32<avx512_core> *eltwise_injector_;
    jit_avx512_core_amx_bwd_data_copy_kernel_t *bwd_data_copy_kernel_;

    int prv_width_ = 0;
    int row_count_ = 0;
    bool is_store_done_ = false;
    bool is_buffer_empty_ = true;

    bool is_store_done_save_ = false;
    int prv_width_save_ = 0;

    /* data regs */
    const Xbyak::Reg64 reg_inp_ptr = r15;
    const Xbyak::Reg64 reg_wei_ptr = r14;
    const Xbyak::Reg64 reg_out_ptr = r13;
    const Xbyak::Reg64 reg_wsp_ptr = r12;

    const Xbyak::Reg64 reg_bias = r11;
    const Xbyak::Reg64 reg_ptr_scales = r10;
    const Xbyak::Reg64 reg_ptr_dst_scales = r10;
    const Xbyak::Reg64 reg_ptr_sum_scale = r9;
    const Xbyak::Reg64 reg_ptr_sum_zp = abi_not_param1;
    const Xbyak::Reg64 reg_aux_saturation = reg_ptr_sum_scale;

    const Xbyak::Reg64 reg_aux_inp_ptr = r8;
    const Xbyak::Reg64 reg_inp_stride = rbx;
    const Xbyak::Reg64 reg_wei_stride = rdx;

    // rbp - reserved for EVEX compression
    const Xbyak::Reg64 reg_last_h = abi_not_param1;
    const Xbyak::Reg64 reg_kd = rsi;

    // temporary, used in generate() function only
    const Xbyak::Reg64 reg_ic_blocks = rax;
    const Xbyak::Reg64 reg_tmp = reg_aux_inp_ptr;

    const Xbyak::Opmask ktail_mask = k2;

    const Xbyak::Zmm zmm_bias = zmm31;
    const Xbyak::Zmm zmm_saturation = zmm_bias;
    const Xbyak::Zmm zmm_zero = zmm30;
    const Xbyak::Zmm zmm_prev_dst = zmm29;
    const Xbyak::Zmm zmm_sum_zp = zmm28;
    /* dst scale */
    const Xbyak::Zmm &zmm_dst_scale = zmm27;

    // AUX: Steps, shifts and offsets
    size_t get_inp_ocb_step() const;
    size_t get_inp_offset(int ihb, int kh, int kw) const;
    size_t get_inp_shift() const;
    size_t get_inp_d_step() const;
    size_t get_out_icb_offset(int ihb, int icb) const;
    size_t get_out_row_offset(int ihb, int icb, int j) const;
    size_t get_out_shift(int width) const;
    size_t get_wei_kh_step() const;
    size_t get_wei_ocb_step() const;
    size_t get_wei_offset(int icb, int kh, int kw) const;
    size_t get_wei_d_step() const;
    size_t get_wsp_icb_offset(int ihb, int icb) const;
    size_t get_wsp_row_offset(int ihb, int icb, int j) const;
    size_t get_wsp_shift() const;

    int get_out_tensor(int h, int i) const;
    int get_inp_tensor(int h) const;
    int get_wei_tensor(int i) const;

    inline bool gaps_in_store() {
        const int gen_kd = (jcp.kd - 1) * (jcp.dilate_d + 1) + 1;
        return gen_kd < jcp.stride_d || jcp.dilate_d > 0;
    }

    void prepare_output();
    void init_runtime_counters(bool start_with_last_tile_block);

    bool maybe_eltwise(int position);
    void cvt2ps(data_type_t type_in, const Xbyak::Zmm &ymm_in,
            const Xbyak::Operand &op, bool mask_flag = false);
    Xbyak::Ymm ymm_mask(
            const Xbyak::Ymm &zmm_in, bool mask_flag, bool store = false);
    Xbyak::Zmm zmm_mask(
            const Xbyak::Zmm &zmm_in, bool mask_flag, bool store = false);

    void store_output_vector_xf16(
            const Xbyak::Zmm &zmm_out, int icb, int ihb, int iw);
    void store_output_vector_int8(
            const Xbyak::Zmm &zmm_out, int icb, int ihb, int iw);
    void store_output_vector(
            const Xbyak::Zmm &zmm_out, int icb, int ih, int iw);
    void store_output(int width, bool do_store);
    void skipped_interleave_store();
    void interleave_store(int width);
    void compute_ocb_loop(int width, bool do_interleave_store);
    void compute_kd_loop(int width, bool do_store, bool handle_skipped_stores);
    void compute_iw_loop();

    void generate() override;
};

struct jit_avx512_core_amx_bwd_weights_kernel_t : public jit_generator {

    jit_avx512_core_amx_bwd_weights_kernel_t(const jit_conv_conf_t &ajcp)
        : jit_generator(jit_name(), avx512_core_amx), jcp(ajcp) {}

    ~jit_avx512_core_amx_bwd_weights_kernel_t() {}

    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_avx512_core_amx_bwd_weights_kernel_t)

    static status_t init_conf(jit_conv_conf_t &jcp,
            const convolution_desc_t &cd, memory_desc_t &src_md,
            memory_desc_t &diff_weights_md, memory_desc_t &diff_bias_md,
            memory_desc_t &diff_dst_md, int nthreads);
    static status_t init_scratchpad(memory_tracking::registrar_t &scratchpad,
            const jit_conv_conf_t &jcp, memory_desc_t &src_md,
            memory_desc_t &diff_weights_md, memory_desc_t &diff_dst_md);

    void tile_configure(char *tcfg_buff);

    const jit_conv_conf_t &jcp;

private:
    int get_wei_tensor(int ocb, int icb) const;
    int get_src_tensor(int icb) const;
    int get_ddst_tensor(int ocb) const;

    using reg64_t = const Xbyak::Reg64;
    static const int max_ur_w;

    reg64_t param = abi_param1;
    reg64_t reg_src = rax;
    reg64_t reg_kernel = rdx;
    reg64_t reg_ddst = rsi;
    reg64_t b_ic = abi_not_param1;
    reg64_t kj = r8;
    reg64_t reg_kh = r9;
    reg64_t reg_oj = r15;
    reg64_t reg_tmp = r14;
    reg64_t reg_ih_shift = reg_tmp;
    reg64_t reg_long_offt = r14;
    reg64_t reg_icb = rbx;

    reg64_t ki = r11;
    reg64_t reg_oj_setup = r11;
    reg64_t reg_kd_count = r12;
    reg64_t reg_oi = r12;
    reg64_t reg_d_index = r13;
    reg64_t reg_src_d = r15;
    reg64_t reg_ddst_d = rbx;
    reg64_t aux_reg_src = r12;
    reg64_t aux_reg_kernel = r13;

    reg64_t reg_b_stride = reg_icb;
    reg64_t reg_a_stride = r10;

    Xbyak::Zmm vreg_bias_acc = Xbyak::Zmm(0);
    Xbyak::Zmm vreg_bias_unit = Xbyak::Zmm(1);
    Xbyak::Zmm vreg_bias_ddst = Xbyak::Zmm(2);

    enum {
        full_spat_opt_working_set_size = 48 * 1024,
        full_spat_max_working_set_size = 128 * 1024,
    };

    inline void maybe_zero_kernel(int nb_ic_blocking, int nb_oc_blocking);
    inline void od_step_comeback_pointers();
    inline void oh_step_comeback_pointers();
    inline void compute_ic_loop(
            int ic_block, int nb_ic_blocking, int nb_oc_blocking);
    inline void compute_full_spat_loop(int nb_ic_blocking, int nb_oc_blocking);
    inline void compute_oh_step_common(int nb_ic_blocking, int nb_oc_blocking);
    inline void compute_loop(int nb_ic_blocking, int nb_oc_blocking);
    inline void compute_oh_loop_common(
            int nb_ic_blocking, int nb_oc_blocking, bool partial = false);
    inline void compute_od_loop_common(
            int nb_ic_blocking, int nb_oc_blocking, bool partial = false);
    void compute_diff_bias_init(int ocb = 0);
    void compute_diff_bias_row(bool is_partial, int ocb);
    void maybe_compute_diff_bias(int nb_oc_blocking);
    void may_be_set_oc_tail_mask();
    void may_be_reset_oc_tail_mask();

    void generate() override;

    static void balance(const jit_conv_conf_t &j, int &nthr, int &nthr_mb,
            int &nthr_g, int &nthr_oc_b, int &nthr_ic_b);

    inline dim_t filter_w_to_src(int kw, int ow = 0, int pad_l = 0) {
        return static_cast<dim_t>(kw) * (jcp.dilate_w + 1) + ow - pad_l;
    }
    inline dim_t filter_h_to_src(int kh) { return kh * (jcp.dilate_h + 1); }
    inline dim_t filter_d_to_src(int kd) {
        return static_cast<dim_t>(kd) * (jcp.dilate_d + 1) * jcp.ih;
    }

    inline dim_t get_src_offset(dim_t ic_idx, dim_t w_idx, dim_t hd_idx = 0) {
        return static_cast<dim_t>(jcp.typesize_in)
                * (hd_idx * jcp.tr_iw * jcp.ic_block + jcp.tr_iw * ic_idx
                        + w_idx);
    }

    inline dim_t get_ddst_offset(dim_t w_idx, dim_t hd_idx = 0) {
        int ow_per_oc = 2;
        dim_t w_off = w_idx / ow_per_oc * ow_per_oc * jcp.oc_block
                + w_idx % ow_per_oc;
        return jcp.typesize_in * (w_off + jcp.tr_ow * jcp.oc_block * hd_idx);
    }

    inline dim_t get_kernel_offset(int ic_idx, dim_t ksp_idx) {
        return jcp.typesize_out * jcp.oc_block
                * (ksp_idx * jcp.ic_block + ic_idx);
    }
    inline dim_t get_full_kernel_offset(int ocb, int icb, int kh, int kw) {
        return jcp.typesize_out
                * (static_cast<dim_t>(ocb) * jcp.nb_ic * jcp.kd * jcp.kh
                                * jcp.kw * jcp.ic_block * jcp.oc_block
                        + static_cast<dim_t>(icb) * jcp.kd * jcp.kh * jcp.kw
                                * jcp.ic_block * jcp.oc_block
                        + static_cast<dim_t>(kh) * jcp.kw * jcp.ic_block
                                * jcp.oc_block
                        + static_cast<dim_t>(kw) * jcp.ic_block * jcp.oc_block);
    };

    inline void setup_stack_space();
    int ic_block_step_stack_size = 0;
    int stack_space_needed = 0;
    int kd_count_offset = 0;
    int src_d_offset = 0;
    int ddst_d_offset = 0;
    int d_index_offset = 0;
    int ih_dilate_offset = 0;
    int src_save_offset = 0;
    int ddst_save_offset = 0;
};

struct jit_avx512_core_amx_bwd_bias_kernel_t : public jit_generator {

    jit_avx512_core_amx_bwd_bias_kernel_t(const jit_conv_conf_t &ajcp)
        : jit_generator(jit_name(), avx512_core_amx), jcp(ajcp) {}

    ~jit_avx512_core_amx_bwd_bias_kernel_t() {}

    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_avx512_core_amx_bwd_bias_kernel_t)

    const jit_conv_conf_t &jcp;

private:
    using reg64_t = const Xbyak::Reg64;

    reg64_t param = abi_param1;
    reg64_t reg_ddst = rsi;
    reg64_t reg_oj = r15;
    reg64_t reg_tmp = r14;
    reg64_t reg_bias = r13;
    reg64_t reg_initial = r12;
    reg64_t reg_nrows = r11;

    Xbyak::Zmm vreg_bias_acc = Xbyak::Zmm(0);
    Xbyak::Zmm vreg_bias_unit = Xbyak::Zmm(1);
    Xbyak::Zmm vreg_bias_ddst = Xbyak::Zmm(2);
    Xbyak::Ymm yreg_bias_acc0 = Xbyak::Ymm(0);
    Xbyak::Ymm yreg_bias_acc1 = Xbyak::Ymm(3);
    Xbyak::Ymm yreg_bias_ddst0 = Xbyak::Ymm(2);
    Xbyak::Ymm yreg_bias_ddst1 = Xbyak::Ymm(4);

    void compute_diff_bias_row(int ocb);
    void compute_diff_bias(int nb_oc_blocking);

    void generate() override;

    inline dim_t get_ddst_offset(dim_t w_idx, dim_t hd_idx = 0) {
        int ow_per_oc = 2;
        dim_t w_off = w_idx / ow_per_oc * ow_per_oc * jcp.oc_block
                + w_idx % ow_per_oc;
        return jcp.typesize_in * (w_off + jcp.tr_ow * jcp.oc_block * hd_idx);
    }
};

} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif
