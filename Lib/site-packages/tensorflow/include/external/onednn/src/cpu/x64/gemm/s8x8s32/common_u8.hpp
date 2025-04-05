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

#ifndef CPU_X64_GEMM_S8X8S32_COMMON_U8_HPP
#define CPU_X64_GEMM_S8X8S32_COMMON_U8_HPP

#include "cpu/platform.hpp"

#include "cpu/x64/jit_generator.hpp"

#define PADD_BYTESIZE_ONPAGE(x, size) \
    (((x) * (size) + PAGE_4K - 1) / PAGE_4K) * PAGE_4K
#define NEXT_THR_STRIDE(x, size) (PADD_BYTESIZE_ONPAGE(x, size)) / size

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {

class jit_avx512_core_u8_copy_an_kern : public jit_generator {
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_avx512_core_u8_copy_an_kern);
    void generate() override ATTRIBUTE_OPTIMIZE;

public:
    jit_avx512_core_u8_copy_an_kern();
};

class jit_avx512_core_u8_copy_at_kern : public jit_generator {
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_avx512_core_u8_copy_at_kern);
    void generate() override ATTRIBUTE_OPTIMIZE;

public:
    jit_avx512_core_u8_copy_at_kern();
};

class jit_avx512_core_u8_copy_bn_kern : public jit_generator {
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_avx512_core_u8_copy_bn_kern);
    void generate() override ATTRIBUTE_OPTIMIZE;
    bool s8_case;

public:
    jit_avx512_core_u8_copy_bn_kern(bool s8 = false);
};

class jit_avx512_core_u8_copy_bt_kern : public jit_generator {
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_avx512_core_u8_copy_bt_kern);
    void generate() override ATTRIBUTE_OPTIMIZE;
    bool s8_case;

public:
    jit_avx512_core_u8_copy_bt_kern(bool s8 = false);
};

class jit_avx512_core_u8_copy_sum_an_kern : public jit_generator {
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_avx512_core_u8_copy_sum_an_kern);
    void generate() override ATTRIBUTE_OPTIMIZE;

public:
    jit_avx512_core_u8_copy_sum_an_kern();
};

class jit_avx512_core_u8_copy_sum_at_kern : public jit_generator {
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_avx512_core_u8_copy_sum_at_kern);
    void generate() override ATTRIBUTE_OPTIMIZE;

public:
    jit_avx512_core_u8_copy_sum_at_kern();
};

class jit_avx512_core_u8_copy_sum_bn_kern : public jit_generator {
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_avx512_core_u8_copy_sum_bn_kern);
    void generate() override ATTRIBUTE_OPTIMIZE;
    bool s8_case;

public:
    jit_avx512_core_u8_copy_sum_bn_kern(bool s8 = false);
};

class jit_avx512_core_u8_copy_sum_bt_kern : public jit_generator {
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_avx512_core_u8_copy_sum_bt_kern);
    void generate() override ATTRIBUTE_OPTIMIZE;
    bool s8_case;

public:
    jit_avx512_core_u8_copy_sum_bt_kern(bool s8 = false);
};

class jit_avx2_vnni_u8_copy_an_kern : public jit_generator {
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_avx2_vnni_u8_copy_an_kern);
    void generate() override ATTRIBUTE_OPTIMIZE;

public:
    jit_avx2_vnni_u8_copy_an_kern();
};

class jit_avx2_vnni_u8_copy_at_kern : public jit_generator {
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_avx2_vnni_u8_copy_at_kern);
    void generate() override ATTRIBUTE_OPTIMIZE;

public:
    jit_avx2_vnni_u8_copy_at_kern();
};

class jit_avx2_vnni_u8_copy_bn_kern : public jit_generator {
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_avx2_vnni_u8_copy_bn_kern);
    void generate() override ATTRIBUTE_OPTIMIZE;

public:
    jit_avx2_vnni_u8_copy_bn_kern();
};

class jit_avx2_vnni_u8_copy_bt_kern : public jit_generator {
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_avx2_vnni_u8_copy_bt_kern);
    void generate() override ATTRIBUTE_OPTIMIZE;

public:
    jit_avx2_vnni_u8_copy_bt_kern();
};

class jit_avx2_vnni_u8_copy_sum_an_kern : public jit_generator {
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_avx2_vnni_u8_copy_sum_an_kern);
    void generate() override ATTRIBUTE_OPTIMIZE;

public:
    jit_avx2_vnni_u8_copy_sum_an_kern();
};

class jit_avx2_vnni_u8_copy_sum_at_kern : public jit_generator {
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_avx2_vnni_u8_copy_sum_at_kern);
    void generate() override ATTRIBUTE_OPTIMIZE;

public:
    jit_avx2_vnni_u8_copy_sum_at_kern();
};

class jit_avx2_vnni_u8_copy_sum_bn_kern : public jit_generator {
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_avx2_vnni_u8_copy_sum_bn_kern);
    void generate() override ATTRIBUTE_OPTIMIZE;

public:
    jit_avx2_vnni_u8_copy_sum_bn_kern();
};

class jit_avx2_vnni_u8_copy_sum_bt_kern : public jit_generator {
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_avx2_vnni_u8_copy_sum_bt_kern);
    void generate() override ATTRIBUTE_OPTIMIZE;

public:
    jit_avx2_vnni_u8_copy_sum_bt_kern();
};

class jit_avx2_u8_copy_an_kern : public jit_generator {
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_avx2_u8_copy_an_kern);
    void generate() override ATTRIBUTE_OPTIMIZE;

public:
    jit_avx2_u8_copy_an_kern();
};

class jit_avx2_u8_copy_at_kern : public jit_generator {
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_avx2_u8_copy_at_kern);
    void generate() override ATTRIBUTE_OPTIMIZE;

public:
    jit_avx2_u8_copy_at_kern();
};

class jit_avx2_u8_copy_bn_kern : public jit_generator {
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_avx2_u8_copy_bn_kern);
    void generate() override ATTRIBUTE_OPTIMIZE;

public:
    jit_avx2_u8_copy_bn_kern();
};

class jit_avx2_u8_copy_bt_kern : public jit_generator {
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_avx2_u8_copy_bt_kern);
    void generate() override ATTRIBUTE_OPTIMIZE;

public:
    jit_avx2_u8_copy_bt_kern();
};

class jit_avx2_u8_copy_sum_an_kern : public jit_generator {
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_avx2_u8_copy_sum_an_kern);
    void generate() override ATTRIBUTE_OPTIMIZE;

public:
    jit_avx2_u8_copy_sum_an_kern();
};

class jit_avx2_u8_copy_sum_at_kern : public jit_generator {
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_avx2_u8_copy_sum_at_kern);
    void generate() override ATTRIBUTE_OPTIMIZE;

public:
    jit_avx2_u8_copy_sum_at_kern();
};

class jit_avx2_u8_copy_sum_bn_kern : public jit_generator {
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_avx2_u8_copy_sum_bn_kern);
    void generate() override ATTRIBUTE_OPTIMIZE;

public:
    jit_avx2_u8_copy_sum_bn_kern();
};

class jit_avx2_u8_copy_sum_bt_kern : public jit_generator {
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_avx2_u8_copy_sum_bt_kern);
    void generate() override ATTRIBUTE_OPTIMIZE;

public:
    jit_avx2_u8_copy_sum_bt_kern();
};

class jit_avx_u8_copy_an_kern : public jit_generator {
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_avx_u8_copy_an_kern);
    void generate() override ATTRIBUTE_OPTIMIZE;

public:
    jit_avx_u8_copy_an_kern();
};

class jit_avx_u8_copy_at_kern : public jit_generator {
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_avx_u8_copy_at_kern);
    void generate() override ATTRIBUTE_OPTIMIZE;

public:
    jit_avx_u8_copy_at_kern();
};

class jit_avx_u8_copy_bn_kern : public jit_generator {
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_avx_u8_copy_bn_kern);
    void generate() override ATTRIBUTE_OPTIMIZE;

public:
    jit_avx_u8_copy_bn_kern();
};

class jit_avx_u8_copy_bt_kern : public jit_generator {
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_avx_u8_copy_bt_kern);
    void generate() override ATTRIBUTE_OPTIMIZE;

public:
    jit_avx_u8_copy_bt_kern();
};

class jit_avx_u8_copy_sum_an_kern : public jit_generator {
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_avx_u8_copy_sum_an_kern);
    void generate() override ATTRIBUTE_OPTIMIZE;

public:
    jit_avx_u8_copy_sum_an_kern();
};

class jit_avx_u8_copy_sum_at_kern : public jit_generator {
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_avx_u8_copy_sum_at_kern);
    void generate() override ATTRIBUTE_OPTIMIZE;

public:
    jit_avx_u8_copy_sum_at_kern();
};

class jit_avx_u8_copy_sum_bn_kern : public jit_generator {
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_avx_u8_copy_sum_bn_kern);
    void generate() override ATTRIBUTE_OPTIMIZE;

public:
    jit_avx_u8_copy_sum_bn_kern();
};

class jit_avx_u8_copy_sum_bt_kern : public jit_generator {
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_avx_u8_copy_sum_bt_kern);
    void generate() override ATTRIBUTE_OPTIMIZE;

public:
    jit_avx_u8_copy_sum_bt_kern();
};

class jit_avx_kernel_b0_gemm_s8u8s32_kern : public jit_generator {
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_avx_kernel_b0_gemm_s8u8s32_kern);
    void generate() override ATTRIBUTE_OPTIMIZE;

public:
    jit_avx_kernel_b0_gemm_s8u8s32_kern();
};

class jit_avx_kernel_b0_b_gemm_s8u8s32_kern : public jit_generator {
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_avx_kernel_b0_b_gemm_s8u8s32_kern);
    void generate() override ATTRIBUTE_OPTIMIZE;

public:
    jit_avx_kernel_b0_b_gemm_s8u8s32_kern();
};

class jit_avx_kernel_b0_r_gemm_s8u8s32_kern : public jit_generator {
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_avx_kernel_b0_r_gemm_s8u8s32_kern);
    void generate() override ATTRIBUTE_OPTIMIZE;

public:
    jit_avx_kernel_b0_r_gemm_s8u8s32_kern();
};

class jit_avx_kernel_b0_c_gemm_s8u8s32_kern : public jit_generator {
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_avx_kernel_b0_c_gemm_s8u8s32_kern);
    void generate() override ATTRIBUTE_OPTIMIZE;

public:
    jit_avx_kernel_b0_c_gemm_s8u8s32_kern();
};

class jit_avx_kernel_gemm_s8u8s32_kern : public jit_generator {
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_avx_kernel_gemm_s8u8s32_kern);
    void generate() override ATTRIBUTE_OPTIMIZE;

public:
    jit_avx_kernel_gemm_s8u8s32_kern();
};

class jit_avx_kernel_b_gemm_s8u8s32_kern : public jit_generator {
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_avx_kernel_b_gemm_s8u8s32_kern);
    void generate() override ATTRIBUTE_OPTIMIZE;

public:
    jit_avx_kernel_b_gemm_s8u8s32_kern();
};

class jit_avx_kernel_r_gemm_s8u8s32_kern : public jit_generator {
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_avx_kernel_r_gemm_s8u8s32_kern);
    void generate() override ATTRIBUTE_OPTIMIZE;

public:
    jit_avx_kernel_r_gemm_s8u8s32_kern();
};

class jit_avx_kernel_c_gemm_s8u8s32_kern : public jit_generator {
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_avx_kernel_c_gemm_s8u8s32_kern);
    void generate() override ATTRIBUTE_OPTIMIZE;

public:
    jit_avx_kernel_c_gemm_s8u8s32_kern();
};

class jit_sse41_u8_copy_an_kern : public jit_generator {
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_sse41_u8_copy_an_kern);
    void generate() override ATTRIBUTE_OPTIMIZE;

public:
    jit_sse41_u8_copy_an_kern();
};

class jit_sse41_u8_copy_at_kern : public jit_generator {
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_sse41_u8_copy_at_kern);
    void generate() override ATTRIBUTE_OPTIMIZE;

public:
    jit_sse41_u8_copy_at_kern();
};

class jit_sse41_u8_copy_bn_kern : public jit_generator {
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_sse41_u8_copy_bn_kern);
    void generate() override ATTRIBUTE_OPTIMIZE;

public:
    jit_sse41_u8_copy_bn_kern();
};

class jit_sse41_u8_copy_bt_kern : public jit_generator {
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_sse41_u8_copy_bt_kern);
    void generate() override ATTRIBUTE_OPTIMIZE;

public:
    jit_sse41_u8_copy_bt_kern();
};

class jit_sse41_u8_copy_sum_an_kern : public jit_generator {
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_sse41_u8_copy_sum_an_kern);
    void generate() override ATTRIBUTE_OPTIMIZE;

public:
    jit_sse41_u8_copy_sum_an_kern();
};

class jit_sse41_u8_copy_sum_at_kern : public jit_generator {
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_sse41_u8_copy_sum_at_kern);
    void generate() override ATTRIBUTE_OPTIMIZE;

public:
    jit_sse41_u8_copy_sum_at_kern();
};

class jit_sse41_u8_copy_sum_bn_kern : public jit_generator {
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_sse41_u8_copy_sum_bn_kern);
    void generate() override ATTRIBUTE_OPTIMIZE;

public:
    jit_sse41_u8_copy_sum_bn_kern();
};

class jit_sse41_u8_copy_sum_bt_kern : public jit_generator {
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_sse41_u8_copy_sum_bt_kern);
    void generate() override ATTRIBUTE_OPTIMIZE;

public:
    jit_sse41_u8_copy_sum_bt_kern();
};

class jit_sse41_kernel_b0_gemm_s8u8s32_kern : public jit_generator {
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_sse41_kernel_b0_gemm_s8u8s32_kern);
    void generate() override ATTRIBUTE_OPTIMIZE;

public:
    jit_sse41_kernel_b0_gemm_s8u8s32_kern();
};

class jit_sse41_kernel_b0_b_gemm_s8u8s32_kern : public jit_generator {
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_sse41_kernel_b0_b_gemm_s8u8s32_kern);
    void generate() override ATTRIBUTE_OPTIMIZE;

public:
    jit_sse41_kernel_b0_b_gemm_s8u8s32_kern();
};

class jit_sse41_kernel_b0_r_gemm_s8u8s32_kern : public jit_generator {
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_sse41_kernel_b0_r_gemm_s8u8s32_kern);
    void generate() override ATTRIBUTE_OPTIMIZE;

public:
    jit_sse41_kernel_b0_r_gemm_s8u8s32_kern();
};

class jit_sse41_kernel_b0_c_gemm_s8u8s32_kern : public jit_generator {
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_sse41_kernel_b0_c_gemm_s8u8s32_kern);
    void generate() override ATTRIBUTE_OPTIMIZE;

public:
    jit_sse41_kernel_b0_c_gemm_s8u8s32_kern();
};

class jit_sse41_kernel_gemm_s8u8s32_kern : public jit_generator {
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_sse41_kernel_gemm_s8u8s32_kern);
    void generate() override ATTRIBUTE_OPTIMIZE;

public:
    jit_sse41_kernel_gemm_s8u8s32_kern();
};

class jit_sse41_kernel_b_gemm_s8u8s32_kern : public jit_generator {
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_sse41_kernel_b_gemm_s8u8s32_kern);
    void generate() override ATTRIBUTE_OPTIMIZE;

public:
    jit_sse41_kernel_b_gemm_s8u8s32_kern();
};

class jit_sse41_kernel_r_gemm_s8u8s32_kern : public jit_generator {
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_sse41_kernel_r_gemm_s8u8s32_kern);
    void generate() override ATTRIBUTE_OPTIMIZE;

public:
    jit_sse41_kernel_r_gemm_s8u8s32_kern();
};

class jit_sse41_kernel_c_gemm_s8u8s32_kern : public jit_generator {
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_sse41_kernel_c_gemm_s8u8s32_kern);
    void generate() override ATTRIBUTE_OPTIMIZE;

public:
    jit_sse41_kernel_c_gemm_s8u8s32_kern();
};

} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl
#endif // CPU_X64_GEMM_S8X8S32_COMMON_U8_HPP
