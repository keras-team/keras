/*******************************************************************************
* Copyright 2022 Intel Corporation
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

#ifndef CPU_X64_JIT_UNI_CONVERT_XF16_HPP
#define CPU_X64_JIT_UNI_CONVERT_XF16_HPP

#include <assert.h>

#include "common/c_types_map.hpp"
#include "common/float16.hpp"
#include "common/nstl.hpp"
#include "common/type_helpers.hpp"

#include "cpu/x64/cpu_isa_traits.hpp"
#include "cpu/x64/jit_avx512_core_bf16cvt.hpp"
#include "cpu/x64/jit_generator.hpp"

#include "oneapi/dnnl/dnnl_debug.h"

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {

namespace cvt_xf16_support {
struct jit_call_t {
    void *inp;
    void *out;
    void *add;
    size_t nelems;
};
struct jit_cvt_xf16_to_ps_params_t {
    const void *inp;
    void *out;
    size_t nelems;
    size_t rows;
};
} // namespace cvt_xf16_support

template <cpu_isa_t isa>
struct jit_uni_cvt_ps_to_xf16_t : public jit_generator {

    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_uni_cvt_ps_to_xf16_t)

    jit_uni_cvt_ps_to_xf16_t(impl::data_type_t dt, size_t nelems = 0)
        : jit_generator(jit_name())
        , output_dt_(dt)
        , nelems_(nelems)
        , is_dynamic_size_(nelems_ == 0)
        , tail_size_(nelems_ % simd_w_) {}

    void generate() override;

protected:
    const impl::data_type_t output_dt_; // bf16 or f16
    const size_t nelems_;
    const bool is_dynamic_size_;
    const int tail_size_;

    constexpr static int simd_w_ = cpu_isa_traits<isa>::vlen / sizeof(float);
    using Vmm = typename cpu_isa_traits<isa>::Vmm;
    using Vmm_down_t = typename vreg_traits<Vmm>::Vmm_lower_t;

    const Vmm vmm_input = Vmm(0);
    const Vmm_down_t vmm_output = Vmm_down_t(1);

    // used in avx2_vnni_2
    const Vmm vmm_in_mask = Vmm(2);
    const Vmm_down_t vmm_out_mask = Vmm(3);
    // used in bf16 emulation
    const Vmm vmm_one = Vmm(2);
    const Vmm vmm_even = Vmm(3);
    const Vmm vmm_selector = Vmm(4);
    const Vmm vmm_fp32_tmp = Vmm(5);
    // used in avx512_core[_fp16]
    const Xbyak::Opmask ktail_f32_mask = Xbyak::Opmask(2);
    const Xbyak::Opmask ktail_xf16_mask = Xbyak::Opmask(3);

    Xbyak::Reg64 reg_input = rax;
    Xbyak::Reg64 reg_output = rbx;
    Xbyak::Reg64 reg_nelems = rdx;
    Xbyak::Reg64 reg_tail = rcx;
    Xbyak::Reg64 reg_tmp = r8;
    Xbyak::Reg64 reg_scratch = r9;

    void setup_mask();
    virtual void cvt_ps_to_xf16(const int idx, const bool is_tail);
    virtual void init_bf16() {} // unused for f16
};

struct jit_avx512_core_cvt_ps_to_bf16_t
    : public jit_uni_cvt_ps_to_xf16_t<avx512_core> {

    jit_avx512_core_cvt_ps_to_bf16_t(impl::data_type_t dt, size_t nelems = 0)
        : jit_uni_cvt_ps_to_xf16_t<avx512_core>(dt, nelems)
        , use_bf16_emu_(!mayiuse(avx512_core_bf16))
        , bf16_emu_(use_bf16_emu_ ? utils::make_unique<bf16_emulation_t>(this,
                            vmm_one, vmm_even, vmm_selector, reg_scratch,
                            vmm_fp32_tmp)
                                  : nullptr) {
        assert(dt == data_type::bf16);
    }

private:
    const bool use_bf16_emu_;
    std::unique_ptr<bf16_emulation_t> bf16_emu_;

    void cvt_ps_to_xf16(const int idx, const bool is_tail) override;
    void init_bf16() override {
        if (use_bf16_emu_) bf16_emu_->init_vcvtneps2bf16();
    }
};

struct jit_cvt_ps_to_xf16_t {

    jit_cvt_ps_to_xf16_t(impl::data_type_t data_type, size_t nelems = 0)
        : nelems_(nelems) {
        if (data_type == data_type::f16 && mayiuse(avx512_core_fp16))
            kernel_ = utils::make_unique<
                    jit_uni_cvt_ps_to_xf16_t<avx512_core_fp16>>(
                    data_type, nelems);
        else if (data_type == data_type::bf16 && mayiuse(avx512_core))
            kernel_ = utils::make_unique<jit_avx512_core_cvt_ps_to_bf16_t>(
                    data_type, nelems);
        else if (mayiuse(avx2_vnni_2))
            kernel_ = utils::make_unique<jit_uni_cvt_ps_to_xf16_t<avx2_vnni_2>>(
                    data_type, nelems);
        else {
            assert(!"unsupported ISA for converter");
            return;
        }
        kernel_->create_kernel();
    }

    void operator()(cvt_xf16_support::jit_call_t *params) const {
        (*kernel_)(params);
        msan_unpoison(params->out,
                (nelems_ ? nelems_ : params->nelems) * sizeof(float16_t));
    }

private:
    std::unique_ptr<jit_generator> kernel_;
    const size_t nelems_;
};

template <cpu_isa_t isa>
struct jit_uni_cvt_xf16_to_ps_t : public jit_generator {

    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_uni_cvt_xf16_to_ps_t)

    jit_uni_cvt_xf16_to_ps_t(
            impl::data_type_t dt, bool with_add, size_t row_stride)
        : jit_generator(jit_name())
        , input_dt_(dt)
        , with_add_(with_add)
        , row_stride_(row_stride) {
        create_kernel();
    }

    void generate() override;

protected:
    constexpr static int elem_granularity = isa == avx2_vnni_2 ? 2 : 1;
    constexpr static int simd_w_ = cpu_isa_traits<isa>::vlen / sizeof(float);
    using Vmm = typename cpu_isa_traits<isa>::Vmm;
    using Vmm_down_t = typename vreg_traits<Vmm>::Vmm_lower_t;

    const impl::data_type_t input_dt_;
    const bool with_add_;

    const size_t row_stride_;

    const Xbyak::Reg64 reg_input = rax;
    const Xbyak::Reg64 reg_output = rbx;
    const Xbyak::Reg64 reg_nelems = r8;
    const Xbyak::Reg64 reg_nrows = r9;

    const Xbyak::Reg64 reg_tail = rcx; //used for cl

    const Xbyak::Reg64 reg_long_row_stride = r10;
    const Xbyak::Reg64 reg_rollback = r11;
    const Xbyak::Reg64 reg_nelems_save = r12;

    const Xbyak::Reg64 reg_tmp = r13;

    const Xbyak::Opmask ktail_mask = Xbyak::Opmask(1);

    const Vmm vmm_tmp = Vmm(13);
    const Vmm vmm_dst = Vmm(14);
    const Vmm vmm_dst_2 = Vmm(15);
    const Vmm_down_t vmm_in_mask = Vmm_down_t(15);

    Vmm get_vmm_src(int idx) { return Vmm(get_even_src_idx(idx)); }
    int get_even_src_idx(int idx) {
        assert(idx < 4);
        return idx;
    }
    int get_odd_src_idx(int idx) {
        assert(idx < 4);
        return idx + 4;
    }

    void convert_xf16(const int idx, const bool handle_x2);
    void cvt_tail();
};

struct jit_cvt_xf16_to_ps_t {

    jit_cvt_xf16_to_ps_t(impl::data_type_t data_type, bool with_add = false,
            size_t row_stride = 0) {
        if (data_type == data_type::f16 && mayiuse(avx512_core_fp16))
            kernel_ = utils::make_unique<
                    jit_uni_cvt_xf16_to_ps_t<avx512_core_fp16>>(
                    data_type, with_add, row_stride);
        else if (data_type == data_type::bf16 && mayiuse(avx512_core))
            kernel_ = utils::make_unique<jit_uni_cvt_xf16_to_ps_t<avx512_core>>(
                    data_type, with_add, row_stride);
        else if (mayiuse(avx2_vnni_2)) {
            if (row_stride != 0) {
                assert(!"unsupported row_stride for avx2_vnni_2");
                return;
            } else if (with_add) {
                assert(!"untested implementation 'with_add' for avx2_vnni_2");
                return;
            }
            kernel_ = utils::make_unique<jit_uni_cvt_xf16_to_ps_t<avx2_vnni_2>>(
                    data_type, with_add, row_stride);
        } else {
            assert(!"unsupported configuration for converter");
            return;
        }
        kernel_->create_kernel();
    }

    void operator()(
            float *out, const void *inp, size_t nelems, size_t rows = 1) const {
        cvt_xf16_support::jit_cvt_xf16_to_ps_params_t p;
        p.inp = inp;
        p.out = (void *)out;
        p.nelems = nelems;
        p.rows = rows;
        (*kernel_)(&p);
        msan_unpoison(out, nelems * sizeof(float));
    }

    void operator()(float *out, const float16_t *inp, size_t nelems,
            size_t rows = 1) const {
        (*this)(out, (const void *)inp, nelems, rows);
    }

    void operator()(float *out, const bfloat16_t *inp, size_t nelems,
            size_t rows = 1) const {
        (*this)(out, (const void *)inp, nelems, rows);
    }

private:
    std::unique_ptr<jit_generator> kernel_;
};

} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif
