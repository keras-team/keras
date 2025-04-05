/*******************************************************************************
* Copyright 2019-2024 Intel Corporation
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

#ifndef CPU_X64_JIT_UNI_XF16_SUM_HPP
#define CPU_X64_JIT_UNI_XF16_SUM_HPP

#include "common/c_types_map.hpp"
#include "common/primitive.hpp"

#include "cpu/cpu_sum_pd.hpp"
#include "cpu/x64/jit_avx512_core_bf16cvt.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {

struct jit_sum_conf_t {
    int num_srcs;
    cpu_isa_t isa;
    data_type_t src_dt;
    data_type_t dst_dt;
    int unroll_reg_count;
    int is_bf16_dst;
    int typesize_in;
    int typesize_out;
    int loop_unroll;
    int size_blocking; /* minimum recommended data blocking size as this
                          number of elements computes main unrolled loop
                          in jit kernel per iteration */
};

struct jit_sum_call_t {
    const void **srcs;
    const void *dst;
    const void *scales;
    dim_t size;
};

template <typename Vmm>
struct jit_uni_xf16_sum_kernel_t : public jit_generator {
    jit_uni_xf16_sum_kernel_t(jit_sum_conf_t ajsp, unsigned int num_acc_iters)
        : jit_generator(jit_name())
        , jsp(ajsp)
        , reg_src {r8, r9, r10, r11, r12, r13, r14, r15}
        , num_acc_iters(num_acc_iters) {}

    ~jit_uni_xf16_sum_kernel_t() {}

    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_uni_xf16_sum_kernel_t)

    jit_sum_conf_t jsp;

protected:
    using reg64_t = const Xbyak::Reg64;
    using reg32_t = const Xbyak::Reg32;

    reg64_t reg_src[8];

    reg64_t param = abi_param1; /* may be rcx, note that cl is required
                                    for mask computation */
    reg64_t reg_srcs = abi_not_param1; /* may be rcx, note that cl is required
                                          for mask computation */
    reg64_t reg_dst = rax;
    reg64_t reg_scales = rbx;
    reg64_t reg_sz = rdx;

    const int num_acc_iters;

    Xbyak::Label exit_label;

    virtual int acc_vreg_idx(int i_unroll, int i_acc) = 0;
    virtual int scale_vreg_idx(int i_acc_iter) = 0;
    virtual int src_vreg_idx(int i_unroll, int i_inp) = 0;
    virtual int tmp_vreg_idx(int i_unroll, int i_acc_iter) = 0;
    virtual void pre_compute_init() = 0;
    virtual void broadcast_scale(int scale_iter) = 0;
    virtual void read_iter(int acc_iter, int u_idx, int src_shift) = 0;
    virtual void add_iter(int acc_iter, int u_idx) = 0;
    virtual void write_iter(int u_idx, int dst_shift) = 0;
    void loop_iteration(int current_unroll);
    virtual void tail_iteration() = 0;
    virtual void index_tables() = 0;
    void generate() override;
};

struct jit_avx512_core_bf16_sum_kernel_t
    : jit_uni_xf16_sum_kernel_t<Xbyak::Zmm> {
    jit_avx512_core_bf16_sum_kernel_t(jit_sum_conf_t ajsp)
        : jit_uni_xf16_sum_kernel_t<Xbyak::Zmm>(
                ajsp, utils::div_up(ajsp.num_srcs, 2))
        , max_vregs_available(cpu_isa_traits<avx512_core>::n_vregs
                  - (isa_has_bf16(jsp.isa) ? 1 : 6))
        , bf16_emu_(nullptr) {
        if (!mayiuse(avx512_core_bf16))
            bf16_emu_ = new bf16_emulation_t(this, bf16_emu_reserved_1,
                    bf16_emu_reserved_2, bf16_emu_reserved_3, bf16_emu_scratch,
                    bf16_emu_reserved_4, bf16_emu_reserved_5);
    }

    ~jit_avx512_core_bf16_sum_kernel_t() { delete bf16_emu_; }

    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_avx512_core_bf16_sum_kernel_t)

    static status_t init_conf(jit_sum_conf_t &jsp, const int num_srcs,
            const memory_desc_t &dst_d);

    static constexpr unsigned int max_num_arrs = 8;

protected:
    reg64_t reg_idx_table = abi_not_param1; /* may be rcx, note that cl is
                                               required for mask computation */
    reg64_t reg_mask = rsi;
    reg32_t reg32_mask = esi;

    const int max_vregs_available;

    int acc_vreg_idx(int i_unroll, int i_acc) override {
        // 2 accumulation registers per unroll iteration
        const int idx = 2 * i_unroll + i_acc;
        assert(idx < max_vregs_available);
        return idx;
    }

    int scale_vreg_idx(int i_acc_iter) override {
        const int scale_idx_start
                = 2 * jsp.loop_unroll; // reserved for acc registers
        const int idx = scale_idx_start + i_acc_iter;
        assert(idx < max_vregs_available);
        return idx;
    }

    int src_vreg_idx(int i_unroll, int i_inp) override {
        // reserved for acc and scale registers
        const int inp_idx_start
                = 2 * jsp.loop_unroll + utils::div_up(jsp.num_srcs, 2);
        const int idx = inp_idx_start
                + utils::rnd_up(jsp.num_srcs, 2) * i_unroll + i_inp;
        assert(idx < max_vregs_available);
        return idx;
    }

    int tmp_vreg_idx(int i_unroll, int i_acc_iter) override {
        const int num_acc_iters = utils::div_up(jsp.num_srcs, 2);
        // reserved for acc, scale and src registers
        const int tmp_idx_start = utils::div_up(jsp.num_srcs, 2)
                + (2 + utils::rnd_up(jsp.num_srcs, 2)) * jsp.loop_unroll;
        const int idx = tmp_idx_start + num_acc_iters * i_unroll + i_acc_iter;
        assert(idx < max_vregs_available);
        return idx;
    }

    static int num_vregs_required(int unroll, int num_srcs) {
        const int num_acc_iters = utils::div_up(num_srcs, 2);
        // reserved for acc, scale and src registers
        int num_regs = utils::div_up(num_srcs, 2)
                + (2 + utils::rnd_up(num_srcs, 2)) * unroll;
        // tmp registers
        num_regs += num_acc_iters * unroll;
        return num_regs;
    }

    bf16_emulation_t *bf16_emu_;

    Xbyak::Zmm bf16_emu_reserved_1 = Xbyak::Zmm(26);
    Xbyak::Zmm bf16_emu_reserved_2 = Xbyak::Zmm(27);
    Xbyak::Zmm bf16_emu_reserved_3 = Xbyak::Zmm(28);
    Xbyak::Zmm bf16_emu_reserved_4 = Xbyak::Zmm(29);
    Xbyak::Zmm bf16_emu_reserved_5 = Xbyak::Zmm(30);
    Xbyak::Reg64 bf16_emu_scratch = abi_not_param1;

    Xbyak::Zmm zmm_idx = Xbyak::Zmm(31);

    Xbyak::Label idx_table;

    const Xbyak::Opmask k_mask = k1;

    void pre_compute_init() override;
    void broadcast_scale(int scale_iter) override;
    void read_iter(int acc_iter, int u_idx, int shift) override;
    void add_iter(int acc_iter, int u_idx) override;
    void write_iter(int u_idx, int shift) override;
    void tail_iteration() override;
    void index_tables() override;
};

struct jit_avx2_vnni_2_xf16_sum_kernel_t
    : jit_uni_xf16_sum_kernel_t<Xbyak::Ymm> {
    jit_avx2_vnni_2_xf16_sum_kernel_t(jit_sum_conf_t ajsp)
        : jit_uni_xf16_sum_kernel_t<Xbyak::Ymm>(ajsp, ajsp.num_srcs) {}

    ~jit_avx2_vnni_2_xf16_sum_kernel_t() {}

    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_avx2_vnni_2_xf16_sum_kernel_t)

    static status_t init_conf(jit_sum_conf_t &jsp, const int num_srcs,
            const std::vector<memory_desc_t> &src_d,
            const memory_desc_t &dst_d);

    static constexpr unsigned int max_num_arrs = 4;

protected:
    int scale_vreg_idx(int i_acc_iter) override { return i_acc_iter; }

    int acc_vreg_idx(int i_unroll, int i_acc) override {
        return jsp.num_srcs
                + ((i_unroll * jsp.unroll_reg_count + i_acc)
                        % (16 - jsp.num_srcs));
    }

    int src_vreg_idx(int i_unroll, int i_inp) override {
        return jsp.num_srcs
                + ((i_unroll * jsp.unroll_reg_count + 2 + i_inp)
                        % (16 - jsp.num_srcs));
    }

    // max 2 tmp registers in a given unroll.
    int tmp_vreg_idx(int i_unroll, int i_acc_iter) override {
        // scale + unroll_window(max 12 registers i.e. 16 - num_srcs)
        return jsp.num_srcs
                + ((i_unroll * jsp.unroll_reg_count + 2 + 2 * jsp.num_srcs
                           + i_acc_iter)
                        % (16 - jsp.num_srcs));
    }

    void pre_compute_init() override {}
    void broadcast_scale(int scale_iter) override;
    void read_iter(int acc_iter, int u_idx, int shift) override;
    void add_iter(int acc_iter, int u_idx) override;
    void write_iter(int u_idx, int shift) override;
    void tail_iteration() override;
    void index_tables() override {}
};

template <data_type_t src_data_type, data_type_t dst_data_type, cpu_isa_t isa>
struct jit_xf16_sum_t : public primitive_t {
    struct pd_t : public cpu_sum_pd_t {
        using cpu_sum_pd_t::cpu_sum_pd_t;

        DECLARE_SUM_PD_T(JIT_IMPL_NAME_HELPER("jit_xf16_", jsp_.isa, ""),
                jit_xf16_sum_t);

        status_t init(engine_t *engine) {

            unsigned int max_num_arrs;
            if (!mayiuse(isa)) return status::unimplemented;
            if (is_superset(isa, avx512_core)) {
                max_num_arrs = jit_avx512_core_bf16_sum_kernel_t::max_num_arrs;
            } else {
                assert(isa == avx2_vnni_2);
                max_num_arrs = jit_avx2_vnni_2_xf16_sum_kernel_t::max_num_arrs;
            }

            VDISPATCH_SUM(cpu_sum_pd_t::init(engine) == status::success,
                    VERBOSE_BAD_ENGINE_KIND);
            VDISPATCH_SUM(src_mds_.size() <= (long unsigned int)max_num_arrs,
                    "number of inputs exceed max number of arrays");

            const memory_desc_wrapper o_d(&dst_md_);
            VDISPATCH_SUM(o_d.data_type() == dst_data_type,
                    VERBOSE_INCONSISTENT_DT, "o_d", "dst");
            VDISPATCH_SUM(o_d.is_dense(true), VERBOSE_UNSUPPORTED_SPARSE_CFG);

            for (size_t i = 0; i < src_mds_.size(); ++i) {
                const memory_desc_wrapper i_d(&src_mds_[i]);
                VDISPATCH_SUM(src_data_type == i_d.data_type(),
                        VERBOSE_INCONSISTENT_DT, "src", "i_d");
                VDISPATCH_SUM(o_d.similar_to(i_d, true, false, 0),
                        VERBOSE_INCONSISTENT_MDS, "o_d", "i_d");
                VDISPATCH_SUM(
                        i_d.is_dense(true), VERBOSE_UNSUPPORTED_SPARSE_CFG);
                // are scales representable in their respective xfloat16 datatype? scales will be down
                // converted to xf16.
                if (src_data_type == data_type::bf16)
                    VDISPATCH_SUM(scales_[i] == float(bfloat16_t(scales_[i])),
                            VERBOSE_UNSUPPORTED_SCALES_CFG);
                else
                    VDISPATCH_SUM(scales_[i] == float(float16_t(scales_[i])),
                            VERBOSE_UNSUPPORTED_SCALES_CFG);
            }

            return is_superset(isa, avx512_core)
                    ? jit_avx512_core_bf16_sum_kernel_t::init_conf(
                            jsp_, src_mds_.size(), dst_md_)
                    : jit_avx2_vnni_2_xf16_sum_kernel_t::init_conf(
                            jsp_, src_mds_.size(), src_mds_, dst_md_);
        }
        jit_sum_conf_t jsp_;
    };

    jit_xf16_sum_t(const pd_t *apd) : primitive_t(apd) {}

    status_t init(engine_t *engine) override {
        if (is_superset(isa, avx512_core)) {
            CHECK(safe_ptr_assign(kernel_,
                    new jit_avx512_core_bf16_sum_kernel_t(pd()->jsp_)));
        } else {
            assert(isa == avx2_vnni_2);
            CHECK(safe_ptr_assign(kernel_,
                    new jit_avx2_vnni_2_xf16_sum_kernel_t(pd()->jsp_)));
        }

        return kernel_->create_kernel();
    }

    status_t execute(const exec_ctx_t &ctx) const override;

    typedef typename prec_traits<src_data_type>::type src_data_t;
    typedef typename prec_traits<dst_data_type>::type dst_data_t;
    typedef typename prec_traits<data_type::f32>::type acc_data_t;

private:
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }
    std::unique_ptr<jit_generator> kernel_;
};

} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif
