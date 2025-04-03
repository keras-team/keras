/*******************************************************************************
* Copyright 2023-2024 Intel Corporation
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

#ifndef CPU_X64_JIT_AVX512_SPARSE_DECOMPRESS_KERNEL_HPP
#define CPU_X64_JIT_AVX512_SPARSE_DECOMPRESS_KERNEL_HPP

#include "common/c_types_map.hpp"
#include "cpu/x64/jit_generator.hpp"
#include "cpu/x64/matmul/brgemm_matmul_utils.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {

struct jit_avx512_sparse_decompress_kernel_t : public jit_generator {
    struct call_params_t {
        const void *src_ptr;
        const void *bitmask_ptr;
        const void *dst_ptr;
    };

    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_avx512_sparse_decompress_kernel_t)

    jit_avx512_sparse_decompress_kernel_t(
            const matmul::brgemm_matmul_conf_t &bgmmc)
        : jit_generator("brgemm_decompress", avx512_core_amx) {
        switch (bgmmc.wei_tag) {
            case format_tag::BA16a64b4a:
            case format_tag::aCB16b64c4b: b_blk_sz_ = 64; break;
            default:
                assert(!"unknown tag");
                ctor_status_ = status::invalid_arguments;
                return;
        }

        assert(a_outter_blk_sz_ == 16);
        assert(a_inner_blk_sz_ == 4);
        assert(b_blk_sz_ == 64);

        if (a_outter_blk_sz_ != 16 || a_inner_blk_sz_ != 4 || b_blk_sz_ != 64) {
            ctor_status_ = status::invalid_arguments;
            return;
        }

        blk_sz_ = a_outter_blk_sz_ * b_blk_sz_ * a_inner_blk_sz_;
        nblks_to_decompress_ = bgmmc.K_blk * b_blk_sz_ / blk_sz_;
    }

    void tile_configure(const char *palette) const { (*this)(palette); }

    status_t create_kernel() override {
        CHECK(ctor_status_);
        return jit_generator::create_kernel();
    }

private:
    status_t ctor_status_ = status::success;

    int nblks_to_decompress_ = 0;
    int blk_sz_ = 0;
    int b_blk_sz_ = 0;

    const int a_outter_blk_sz_ = 16;
    const int a_inner_blk_sz_ = 4;

    const Xbyak::Reg64 reg_src_ptr = r8;
    const Xbyak::Reg64 reg_dst_ptr = r9;
    const Xbyak::Reg64 reg_bitmask_ptr = r10;
    const Xbyak::Reg64 reg_tmp = r11;
    const Xbyak::Reg64 reg_popcnt_tmp = r12;
    const Xbyak::Reg64 reg_popcnt = rcx;

    Xbyak::Opmask get_opmask(int idx);
    Xbyak::Opmask get_load_mask(int idx);
    Xbyak::Opmask get_expand_mask(int idx);

    Xbyak::Reg64 get_reg_mask_tmp(int idx);
    Xbyak::Zmm get_zmm(int idx);

    int unroll_factor() const { return 4; }

    void generate() override;
};

} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif
