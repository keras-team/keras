/*******************************************************************************
* Copyright 2022-2024 Intel Corporation
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

#ifndef CPU_X64_JIT_BRGEMM_CONV_BWD_STRIDED_HPP
#define CPU_X64_JIT_BRGEMM_CONV_BWD_STRIDED_HPP

#include "common/c_types_map.hpp"
#include "common/dnnl_thread.hpp"
#include "common/memory_tracking.hpp"
#include "common/primitive.hpp"
#include "common/utils.hpp"

#include "cpu/cpu_convolution_pd.hpp"
#include "cpu/platform.hpp"

#include "cpu/x64/amx_tile_configure.hpp"
#include "cpu/x64/brgemm/brgemm.hpp"
#include "cpu/x64/brgemm/brgemm_containers.hpp"
#include "cpu/x64/jit_avx512_core_scale_precompute.hpp"
#include "cpu/x64/jit_brgemm_conv_bwd_copy_kernel.hpp"
#include "cpu/x64/jit_brgemm_conv_bwd_trans_kernel.hpp"
#include "cpu/x64/jit_brgemm_conv_comp_pad_kernel.hpp"
#include "cpu/x64/jit_brgemm_post_ops.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {

template <cpu_isa_t isa, bool is_deconv = false>
struct brgemm_convolution_bwd_strided_t : public primitive_t {

    struct pd_t : public cpu_convolution_bwd_data_pd_t {
        pd_t(const convolution_desc_t *adesc, const primitive_attr_t *attr,
                const typename pd_t::hint_class *hint_fwd_pd)
            : cpu_convolution_bwd_data_pd_t(adesc, attr, hint_fwd_pd) {}

        ~pd_t() = default;

        DECLARE_COMMON_PD_T(JIT_IMPL_NAME_HELPER("brgconv_strided:", isa, ""),
                brgemm_convolution_bwd_strided_t);

        status_t init(engine_t *engine);

        int brgs_sz_;
        std::shared_ptr<brgemm_containers::brgemm_desc_container_t> brgs_;

        jit_brgemm_conv_conf_t jcp_;
        // batch size info
        const int first_bs = 0;
        int get_brg_idx(int bs, int m, bool do_initialization, bool is_N_tail,
                bool is_K_tail) const {
            const int bs_c = 1;
            auto bs_idx = 0;
            return (((m * bs_c + bs_idx) * 2
                            + static_cast<int>(do_initialization))
                                   * 2
                           + static_cast<int>(is_N_tail))
                    * 2
                    + static_cast<int>(is_K_tail);
        }
    };

    brgemm_convolution_bwd_strided_t(const pd_t *apd)
        : primitive_t(apd), bias_d(pd()->weights_md(1)) {}

    ~brgemm_convolution_bwd_strided_t() = default;

    status_t execute(const exec_ctx_t &ctx) const override;

protected:
    status_t init(engine_t *engine) override;

private:
    //  brgemm convolution execution context
    struct brgemm_bwd_exec_ctx_t {
        brgemm_bwd_exec_ctx_t(const exec_ctx_t &ctx, const pd_t *pd)
            : diff_dst(CTX_IN_MEM(const char *, DNNL_ARG_DIFF_DST))
            , weights(CTX_IN_MEM(const char *, DNNL_ARG_WEIGHTS))
            , bias(CTX_IN_MEM(const char *, DNNL_ARG_BIAS))
            , diff_src(CTX_OUT_MEM(char *, DNNL_ARG_DIFF_SRC))
            , post_ops_binary_rhs_arg_vec(binary_injector::prepare_binary_args(
                      pd->attr()->post_ops_, ctx)) {}
        const char *const __restrict diff_dst;
        const char *const __restrict weights;
        const char *const __restrict bias;
        char *const __restrict diff_src;
        const std::vector<const void *> post_ops_binary_rhs_arg_vec;
    };

    struct brgemm_bwd_thread_ctx_t {
        brgemm_bwd_thread_ctx_t(brgemm_bwd_exec_ctx_t &brgemm_ctx_, int ithr_,
                brgemm_batch_element_t *__restrict brg_batch_, char *c_buffer_,
                char *out_buffer_, char *wsp_tile_)
            : brgemm_ctx(brgemm_ctx_)
            , ithr(ithr_)
            , brg_batch(brg_batch_)
            , c_buffer(c_buffer_)
            , out_buffer(out_buffer_)
            , wsp_tile(wsp_tile_)
            , cur_brg_idx(-1)
            , g(0)
            , n(0)
            , icb(0)
            , id(0)
            , idb(0)
            , ih(0)
            , ihb(0)
            , iwb(0)
            , occ(0)
            , sw(0)
            , oscales(nullptr)
            , dst_scales(nullptr)
            , src_zp_vals(0)
            , src_zp_comp_ptr(nullptr)
            , dst_zp_vals(nullptr)
            , s8s8_comp_ptr(nullptr) {}

        brgemm_bwd_exec_ctx_t &brgemm_ctx;
        int ithr;
        brgemm_batch_element_t *__restrict brg_batch;
        char *c_buffer;
        char *out_buffer;
        char *wsp_tile;
        int cur_brg_idx;
        int g, n, icb;
        int id, idb, ih, ihb, iwb;
        int occ;
        int sw;
        const float *oscales;
        const float *dst_scales;
        int32_t src_zp_vals;
        int32_t *src_zp_comp_ptr;
        int32_t *dst_zp_vals;
        int32_t *s8s8_comp_ptr;
    };

    inline static int get_ker_po_idx(int m, bool do_postwork, bool is_N_tail) {
        return (m * 2 + static_cast<int>(do_postwork)) * 2
                + static_cast<int>(is_N_tail);
    }

    inline int get_comp_iw(const int iw) const {
        return utils::div_up(IW, SW) * (iw % SW) + iw / SW;
    }

    void get_kw_range(int iw, int iw_raw, int &kw_s, int &kw_full_s,
            int &kw_full_e, int &kw_e) const;
    void get_iw_range(int iw, int iw_raw, int kw, int &ow_s, int &ow_e) const;

    void ker_base(brgemm_bwd_thread_ctx_t &btc) const;
    void ker_trans(brgemm_bwd_thread_ctx_t &btc, char *inp_buffer) const;

    void perform_outwork(char *dst_base, char *dst, char *c_buffer,
            const char *bias_w, int od, int oh, int ow, int iw_raw, int g_oc,
            bool is_oc_tail, int ker_ow_s, int ker_ow_f, int kd_l, int kh_l,
            const void *post_ops_binary_rhs_arg_vec, const float *oscales,
            int32_t src_zp_vals, int32_t *src_zp_ptr, int32_t *dst_zp_ptr,
            int32_t *s8s8_compensation, size_t comp_ker_offs,
            bool maybe_do_init, bool do_postwork, bool do_post_comp,
            const float *dst_scales) const;

    void call_brgemm_kernel(brgemm_bwd_thread_ctx_t &btc, int brg_idx,
            int batch_size, char *ptr_C, char *ptr_D, const char *bias_w,
            int g_ic, bool do_postops, const void *binary_post_ops_rhs,
            int32_t src_zp_vals, int32_t *src_zp_ptr, int32_t *dst_zp_ptr,
            int32_t *s8s8_comp, bool do_only_comp,
            bool is_first_call_postops) const;

    void maybe_trans_inp(int ithr, const char *__restrict input,
            char *__restrict inp_buffer, uint8_t *__restrict inp_buffer_mask,
            int g, int n, int icc, int odb, int ohb, int owb, int last_g,
            int last_n, int last_icc, int last_odb, int last_ohb,
            int last_owb) const;

    status_t add_po_kernel(brgemm_desc_t *bcfg, int ker_idx, bool is_init);
    void add_po_kernels(int i_N, int init_bcast_dim, int po_bcast_dim);
    status_t add_brg_kernel(int bs, int M, int i_N, int i_K, int i_init);

    void cal_compensation(const char *__restrict weights,
            int32_t *src_zp_buffer, int32_t *s8s8_comp_buffer) const;
    int get_comp_ker_idx(const int kd_b, const int kd_e, const int kh_b,
            const int kh_e, const int kw_b, const int kw_e) const;
    int get_comp_offset(const int g, const int icb, const int iw,
            const int kd_b, const int kd_e, const int kh_b, const int kh_e,
            const int kw_b, const int kw_e) const;
    void create_kernels();

    const pd_t *pd() const {
        return static_cast<const pd_t *>(primitive_t::pd().get());
    }

    brgemm_containers::brgemm_kernel_container_t brg_kernels_;
    brgemm_containers::brgemm_palette_container_t brgemm_palettes_;

    std::vector<std::unique_ptr<jit_brgemm_kernel_post_ops<isa>>> kernels_po_;

    using Vmm = typename cpu_isa_traits<isa>::Vmm;

    std::unique_ptr<jit_avx512_core_brgemm_conv_bwd_trans_kernel::
                    jit_avx512_core_brgemm_conv_bwd_trans_kernel_t<Vmm>>
            copy_to_pbuffer_;

    std::unique_ptr<jit_avx512_core_brgemm_conv_bwd_copy_kernel::
                    jit_avx512_core_brgemm_conv_bwd_copy_kernel_t<Vmm>>
            copy_to_output_buffer_;
    std::unique_ptr<jit_generator> comp_vpad_pbuffer_;
    std::unique_ptr<jit_avx512_core_scale_precompute_t> jit_scale_precompute_;

    size_t acc_dsz, bia_dsz, src_dsz, wei_dsz, dst_dsz;

    const memory_desc_wrapper bias_d;

    // precalculated values
    std::vector<dim_t> kd_bs, kd_es, kh_bs, kh_es, kw_bs, kw_es;

    int KD, KH, KW, EXT_KD, EXT_KH, EXT_KW, KS, KD_BLOCK, KH_BLOCK, KW_BLOCK,
            KD_BLOCK_PAD, KH_BLOCK_PAD, ID, IH, IW, ODP, OHP, OWP, OD, OH, OW,
            SD, SH, SW, FP, TP, LP, DD, DH, DW;
    dim_t src_w_sz, src_h_sz, src_d_sz, dst_w_sz, dst_h_sz, dst_d_sz, wei_oc_sz,
            wei_kw_sz, wei_kh_sz, wei_kd_sz, wei_icb_sz;
    dim_t pbuf_w_sz, pbuf_h_sz, pbuf_d_sz;
    dim_t comp_icb_sz, comp_ker_sz, comp_kw_sz, comp_iw_sz;

    int oc_chunks;
    bool need_postwork;
    bool need_compensation;
    bool is_amx;
};

} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif

// vim: et ts=4 sw=4 cindent cino+=l0,\:4,N-s
