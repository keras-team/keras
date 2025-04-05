/*******************************************************************************
* Copyright 2020-2023 Intel Corporation
* Copyright 2023 FUJITSU LIMITED
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
#ifndef CPU_AARCH64_BRGEMM_BRGEMM_TYPES_HPP
#define CPU_AARCH64_BRGEMM_BRGEMM_TYPES_HPP

#include "common/primitive_attr.hpp"
#include "cpu/aarch64/cpu_isa_traits.hpp"
#include "cpu/platform.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace aarch64 {

// The type defines organization of batch of matrices
typedef enum {
    // A and B arrays of pointers
    brgemm_addr = 1,
    // Base address and array of offsets from base address.
    brgemm_offs = 2,
    // Base address and fixed stride between matrices.
    brgemm_strd = 3,
    // Base address and static array of fixed offsets.
    brgemm_static_offs = 4,
} brgemm_batch_kind_t;

// The type defines the storage format of matrix
typedef enum {
    brgemm_layout_undef = 0,
    brgemm_col_major = 1,
    brgemm_row_major = 2,
} brgemm_layout_t;

typedef enum {
    none = 0,
    per_tensor = 1,
    per_m = 2,
    per_n = 3,
    per_k = 4,
} brgemm_broadcast_t;

struct brgemm_strides_t {
    // Stride between A matrices
    dim_t stride_a;
    // Stride between B matrices
    dim_t stride_b;
};

typedef enum {
    brgemm_lo_default = 0,
    brgemm_lo_bl_1load,
    brgemm_lo_bl_1bcst,
} brgemm_kernel_loop_order_t;

typedef enum {
    brgemm_prf_default = 1,
    brgemm_prf1,
    brgemm_prf2,
} brgemm_kernel_prefetching_t;

typedef enum {
    brgemm_innermost_undef = 0,
    brgemm_bd_loop_innermost,
    brgemm_ld_loop_innermost,
} brgemm_kernel_innermost_loop_t;

typedef enum {
    brgemm_hint_nt_undef = 0,
    brgemm_hint_nt_false,
    brgemm_hint_nt_true,
} brgemm_kernel_hint_nt_t;

struct brgemm_prf_t {
    int dist1 = -1;
    int dist2 = -1;
};

struct brgemm_batch_element_t {
    brgemm_batch_element_t() {
        ptr.A = ptr.B = nullptr;
        vvpad.top = vvpad.bottom = 0;
    }
    union {
        struct {
            const void *A;
            const void *B;
        } ptr;
        struct {
            dim_t A;
            dim_t B;
        } offset;
    };
    union {
        struct {
            dim_t top;
            dim_t bottom;
        } vvpad;
        struct {
            dim_t left;
            dim_t right;
        } hvpad;
    };
};

struct DNNL_API brgemm_attr_t {
    brgemm_attr_t();
    // if unrolled kernel is used (use_uker == true)
    // then "max_bs" is the the only batch size that can be used on kernel call
    // else "max_bs" is the maximum batch size that can be used
    int max_bs;
    int max_top_vpad, max_bottom_vpad;
    dim_t hint_expected_A_size, hint_expected_B_size, hint_expected_C_size;
    brgemm_kernel_innermost_loop_t hint_innermost_loop
            = brgemm_ld_loop_innermost;
    brgemm_kernel_loop_order_t hint_loop_order;
    brgemm_kernel_prefetching_t hint_prefetching
            = brgemm_kernel_prefetching_t::brgemm_prf_default;
    brgemm_prf_t hint_prfA, hint_prfB, hint_prfC;

    bool wary_tail_read;
    bool generate_skip_accumulation;
    // Value of bd_mask_level specifies how bd_mask is used in brgemm kernel
    // 0 – bd_mask is not used
    // 1 – bd_mask is used on storing stage only
    // 2 – bd_mask used both on reading and storing stages
    int bd_mask_level;
    // use_uker is a boolean value that determines whether to use the unrolled
    // kernel or not
    bool use_uker;
    // use_interleave_stores is a value that determines whether to use the
    // interleave stores or not
    bool use_interleave_stores;
    impl::fpmath_mode_t fpmath_mode = fpmath_mode::strict;
    // Second level leading dimension describing distance between 16-line
    // blocks in case of blocked layout. Used to calculate address of next
    // bd block. By default are equal to regular leading dimension parameters
    // specified on brgemm creation.
    // Supported by brgemm unrolled kernel for now.
    int LDA2 {0}, LDB2 {0}, LDC2_M {0}, LDC2_N {0};
    // If "true" then batchsize is allowed to change on each kernel call
    // and there is no unrolling by batchsize in kernel
    bool var_bs {false};
    bool postops_only {false};

    int hint_bd_block {0};
    int hint_ld_block {0};
    int hint_bd_block2 {0};
    int hint_ld_block2 {0};
    bool hint_ununroll_bd_loop {false};

    brgemm_kernel_hint_nt_t hint_load_nt_A {brgemm_hint_nt_undef};
    brgemm_kernel_hint_nt_t hint_load_nt_B {brgemm_hint_nt_undef};
    // this variable is used in tile decomposition heuristics
    // to calculate "effective" K value which may be different from just
    // K * batch_size for non-1x1 convolutions due to data overlap
    float K_koef = 1.f;
    // this flag may be used to omit some actions on calling from blocking
    // in convolutions init_conf
    bool test_call = false;
    // bd_mask is char array in which each element is a boolean value that
    // determines whether to write this row to the result matrix or skip
    const char *bd_mask;
    // static_offsets is a static array of fixed offsets used for
    // brgemm_static_offs batch kind
    const brgemm_batch_element_t *static_offsets;
};

struct brgemm_t {
    // Note: new added parameters must be taken into account in the brgemm
    // comparison function
    int bcast_dim = 0; // M;
    int load_dim = 0; // N;
    int reduce_dim = 0; // K;
    int LDA = 0;
    int LDB = 0;
    int LDC = 0;
    int LDD = 0;
    // we use two isa_ variables
    // isa_user to store the user provided isa value
    // isa_impl to store actual implementation. This can change until the kernel
    // is created, as calls to set_attr can affect this variable. Ex: bf32
    impl::cpu::aarch64::cpu_isa_t isa_user = isa_undef;
    impl::cpu::aarch64::cpu_isa_t isa_impl = isa_undef;
    float alpha = 0.0f;
    float beta = 0.0f;

    impl::data_type_t dt_a = data_type::undef;
    impl::data_type_t dt_c = data_type::undef;
    impl::data_type_t dt_b = data_type::undef;
    impl::data_type_t dt_d = data_type::undef;
    impl::data_type_t dt_bias = data_type::undef;

    dim_t stride_a = 0; // Offset in bytes
    dim_t stride_b = 0;

    brgemm_layout_t layout = brgemm_layout_undef;
    brgemm_batch_kind_t type;
    bool is_dgmm = false; // set to true in brdgmm_desc_init
    bool with_sum = false;
    bool req_cal_comp_pads = false;

    float sum_scale = 0.0f;
    int32_t sum_zp = 0;
    impl::data_type_t sum_dt;
    bool with_eltwise = false;
    bool with_binary = false;
    bool with_scales = false;

    brgemm_broadcast_t zp_type_a = brgemm_broadcast_t::none;
    brgemm_broadcast_t zp_type_b = brgemm_broadcast_t::none;
    brgemm_broadcast_t zp_type_c = brgemm_broadcast_t::none;

    int is_oc_scale = 0;
    bool with_dst_scales = false;

    brgemm_attr_t brgattr;

    // Derived  parameters
    int LDA2 {0}, LDB2 {0}, LDC2_M {0}, LDC2_N {0};
    bool is_blocked = false;

    int bdb = 0, bd_block = 0, bdb_tail = 0;
    int bdb2 = 0, bd_block2 = 0, bdb2_tail = 0;
    int ldb = 0, ld_block = 0, ldb_tail = 0;
    int ldb2 = 0, ld_block2 = 0, ldb2_tail = 0;
    int rdb = 0, rd_block = 0, rdb_tail = 0;
    int rd_step = 0, ld_step = 0;

    int typesize_A = 0;
    int typesize_B = 0;
    int typesize_C = 0;
    int typesize_D = 0;
    int typesize_bias = 0;

    bool is_ymm = false;
    bool is_zmm = false;

    bool is_int8 = false;
    bool is_bf16 = false, is_bf16_emu = false;
    bool is_f16 = false;

    bool is_f32 = false;
    bool is_bf32 = false;

    bool has_int8_vnni = false;

    bool load_nt_A = false;
    bool load_nt_B = false;

    bool embd_bcst = false;
    bool with_bias = false;
    bool req_s8s8_compensation = false;
    bool with_weights_scale_adjust = false;
    brgemm_kernel_innermost_loop_t innermost_loop = brgemm_ld_loop_innermost;
    int is_M_tail;
    bool interleave_tilestores_ = false;
    brgemm_prf_t prfA, prfB, prfC;

    static constexpr int MAX_VPAD = 100;

    const primitive_attr_t *attr = nullptr;
    const memory_desc_t *dst_md = nullptr;

    bool is_row_major() const {
        assert(layout != brgemm_layout_undef);
        return layout == brgemm_row_major;
    }

    int get_wsp_buffer_size() const noexcept {
        int sz = 0;
        return sz;
    }

    bool is_b_data_layout_vnni() { return true; }

    bool operator==(const brgemm_t &rhs) const;
    bool operator<(const brgemm_t &rhs) const;
};

struct brgemm_kernel_params_t {
    const void *ptr_A;
    const void *ptr_B;
    const brgemm_batch_element_t *batch;
    void *ptr_C;

    const void *ptr_bias;
    void *ptr_D;

    /* kernel takes single pointer scales, but configuration relies on a
     * combination of arg scales. This helps to reuse attributes from
     * primitives, but requires them to pre-compute
     * scales = src_scale * wei_scale[:]
     */
    const void *ptr_scales;
    void *ptr_buf;

    size_t do_post_ops;
    size_t do_apply_comp;
    size_t BS;

    /*
     * ptr to table of void * elements that are pointers to post_op binary
     * src1 tensors
     */
    const void *post_ops_binary_rhs_arg_vec;
    size_t oc_logical_off;
    size_t first_mb_matrix_addr_off;
    size_t dst_row_logical_off;

    const char *data_C_ptr_;

    const void *a_zp_compensations = nullptr;
    const void *b_zp_compensations = nullptr;
    const void *c_zp_values = nullptr;
    size_t skip_accm = 0;
    int32_t zp_a_val = 1;
    const void *ptr_dst_scales = nullptr;
};

struct jit_brgemm_kernel_t;
struct jit_brdgmm_kernel_base_t;
class jit_generator;

struct brgemm_kernel_t {
    brgemm_kernel_t() {};
    virtual ~brgemm_kernel_t() {};
    virtual status_t create_kernel() = 0;
    virtual void operator()(brgemm_kernel_params_t *) const = 0;
    virtual const jit_generator *get_jit_generator() const = 0;
};

struct brgemm_kernel_common_t : public brgemm_kernel_t {
    brgemm_kernel_common_t(const brgemm_t abrd);
    ~brgemm_kernel_common_t();

    status_t create_kernel();
    void operator()(brgemm_kernel_params_t *) const;
    virtual const jit_generator *get_jit_generator() const;

private:
    jit_brgemm_kernel_t *brgemm_kernel_ = nullptr;

    DNNL_DISALLOW_COPY_AND_ASSIGN(brgemm_kernel_common_t);
};

struct brdgmm_kernel_t : public brgemm_kernel_t {
    brdgmm_kernel_t(const brgemm_t abrd);
    ~brdgmm_kernel_t();

    status_t create_kernel();
    void operator()(brgemm_kernel_params_t *) const;
    virtual const jit_generator *get_jit_generator() const;

private:
    jit_brdgmm_kernel_base_t *brgemm_kernel_ = nullptr;

    DNNL_DISALLOW_COPY_AND_ASSIGN(brdgmm_kernel_t);
};

/// @param bias Vector of bias (vector length is N)
/// @param scales - Vector of scale factor values which represents combination
///     scale factors for matrixes A and B. If brgemm_t::is_oc_scale = true
///     vector length is N otherwise it must be broadcasted to vector of simd
///     width length
/// @param binary_post_ops_rhs - Ptr to table of pointers to tensors used as rhs
///     in binary post-operation { void* binary_op_tensor1, ...,
///      void* binary_op_tensor_n}
/// @param oc_logical_off - Used in binary postops in per_oc bcast strategy.
///     Offset to start oc processed by given thread in elements.
/// @param dst_row_logical_off - Used in binary postops in per_oc bcast
///     strategy. Offset to start oc processed by given thread in elements.
/// @param a_zp_compensations - Pre-computed compensations for A matrix zero
///     point values.
/// @param b_zp_compensations - Pre-computed compensations for B matrix zero
///     point values.
/// @param c_zp_values - C matrix zero point values.
/// @param skip_accumulation - specifies whether to skip accumulation when
///    computing post-ops. `Beta` value from descriptor affects final
///    accumulator values taken.
/// @param do_only_comp - specifies whether to perform accumulation only and skip
///    post-ops.
/// @param do_only_zp_a_val - specifies to apply pre-calculated compensation for
///     A zero point only and skip the rest post-ops.
/// @param zp_a_val - zero point value for A, required to adjust compensation
///     values if do_only_zp_a_val = true.
/// @param dst_scales - Vector of inverted scale factor values for matix C,
///     common scale vector type only is supported, it must be broadcasted to
///     vector of simd width length.
///
struct brgemm_post_ops_data_t {
    brgemm_post_ops_data_t() = default;
    brgemm_post_ops_data_t(const void *bias, const float *scales,
            const void *binary_post_ops_rhs, size_t oc_logical_off,
            const size_t dst_row_logical_off = 0,
            const char *data_C_ptr_ = nullptr,
            const size_t first_mb_matrix_addr_off = 0,
            const void *a_zp_compensations = nullptr,
            const void *b_zp_compensations = nullptr,
            const void *c_zp_values = nullptr, bool skip_accumulation = false,
            int32_t zp_a_val = 1, bool do_only_comp = false,
            bool do_only_zp_a_val = false, const float *dst_scales = nullptr)
        : bias(bias)
        , scales(scales)
        , binary_post_ops_rhs(binary_post_ops_rhs)
        , oc_logical_off(oc_logical_off)
        , dst_row_logical_off(dst_row_logical_off)
        , data_C_ptr_(data_C_ptr_)
        , first_mb_matrix_addr_off(first_mb_matrix_addr_off)
        , a_zp_compensations(a_zp_compensations)
        , b_zp_compensations(b_zp_compensations)
        , c_zp_values(c_zp_values)
        , skip_accumulation(skip_accumulation)
        , zp_a_val {zp_a_val}
        , do_only_comp {do_only_comp}
        , do_only_zp_a_val {do_only_zp_a_val}
        , dst_scales(dst_scales) {}

    const void *bias = nullptr;
    const float *scales = nullptr;
    const void *binary_post_ops_rhs = nullptr;
    size_t oc_logical_off = 0;
    size_t dst_row_logical_off = 0;
    const char *data_C_ptr_ = nullptr;
    size_t first_mb_matrix_addr_off = 0;
    const void *a_zp_compensations = nullptr;
    const void *b_zp_compensations = nullptr;
    const void *c_zp_values = nullptr;
    const bool skip_accumulation = false;
    int32_t zp_a_val = 1;
    const bool do_only_comp = false;
    const bool do_only_zp_a_val = false;
    const float *dst_scales = nullptr;
};

} // namespace aarch64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif

//vim: et ts=4 sw=4 cindent cino+=l0,\:4,N-s