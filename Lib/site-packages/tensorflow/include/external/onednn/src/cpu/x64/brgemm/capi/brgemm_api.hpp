/*******************************************************************************
* Copyright 2024 Intel Corporation
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

#ifndef CPU_X64_BRGEMM_CAPI_BRGEMM_API_HPP
#define CPU_X64_BRGEMM_CAPI_BRGEMM_API_HPP

#include <memory>

#include "cpu/x64/matmul/brgemm_matmul_copy_utils.hpp"
#include "cpu/x64/matmul/brgemm_matmul_utils.hpp"

#include "cpu/x64/brgemm/brgemm_types.hpp"

struct dnnl_brgemm : public dnnl::impl::c_compatible {
    dnnl_brgemm() = default;

    // Just members here because brgemm API is C-based and should remain until
    // this new API becomes a production one.
    // Once becamoes, internal C API can be re-factored.
    dnnl::impl::cpu::x64::brgemm_desc_t brgemm_desc_;
    dnnl::impl::cpu::x64::brgemm_kernel_t *brgemm_kernel_;
};

struct dnnl_brgemm_pack_B : public dnnl::impl::c_compatible {
    dnnl_brgemm_pack_B() = default;

    // Ctor that follows a call to initialize matmul conf struct.
    dnnl_brgemm_pack_B(dnnl::impl::dim_t K, dnnl::impl::dim_t N,
            dnnl::impl::dim_t in_ld, dnnl::impl::dim_t out_ld,
            dnnl::impl::data_type_t in_type, dnnl::impl::data_type_t out_type);

    // Returns the flag is packing for VNNI is needed.
    // Note: not completely aligned with primitives logic.
    bool need_pack() const;

    // Generates a copy_b kernel.
    void generate();

    // Executes a copy_b kernel.
    void execute(const void *src, void *dst) const;

    dnnl::impl::cpu::x64::matmul::brgemm_matmul_conf_t bmc_;
    // unique_ptr is required by API that generates a kernel.
    std::unique_ptr<dnnl::impl::cpu::x64::matmul::jit_brgemm_matmul_copy_b_t>
            kernel_;
};

#endif

//vim: et ts=4 sw=4 cindent cino+=l0,\:4,N-s
