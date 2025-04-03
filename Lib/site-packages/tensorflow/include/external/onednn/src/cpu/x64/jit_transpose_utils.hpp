/*******************************************************************************
* Copyright 2017-2022 Intel Corporation
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

#ifndef CPU_X64_JIT_TRANSPOSE_UTILS_HPP
#define CPU_X64_JIT_TRANSPOSE_UTILS_HPP

#include "cpu/x64/cpu_barrier.hpp"
#include "cpu/x64/jit_primitive_conf.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {

struct jit_trans_src_t {
    struct ctx_t {
        const void *src;
        const void *tr_src;
        const void *src_prf;
        const void *tr_src_prf;
        int ch_work;
    };

    virtual void operator()(ctx_t *ctx) = 0;
    virtual status_t create_kernel() = 0;

    jit_trans_src_t(const jit_conv_conf_t *conf) : conf_(conf) {}
    virtual ~jit_trans_src_t() {}

    const jit_conv_conf_t *conf_;
};

struct jit_src_transpose_s {
    size_t size;
    const void *src;
    const void *tr_src;
    const void *src_prf;
    const void *tr_src_prf;
};

struct jit_trans_dst_t {
    struct ctx_t {
        const void *src;
        const void *tr_src;
        const void *src_prf;
        const void *tr_src_prf;
        int ch_work;
    };

    jit_trans_dst_t(const jit_conv_conf_t *conf) : conf_(conf) {}
    virtual ~jit_trans_dst_t() {}

    virtual void operator()(ctx_t *ctx) = 0;
    virtual status_t create_kernel() = 0;
    const jit_conv_conf_t *conf_;
};

struct jit_transpose4x16_src_t {
    int src_pf0_distance;
    int tr_src_pf0_distance;
    bool src_pf1;
    bool tr_src_pf1;
};

struct jit_transpose4x16_src : public jit_generator {
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_transpose4x16_src)

    jit_transpose4x16_src(const jit_1x1_conv_conf_t *aparams,
            jit_transpose4x16_src_t *tparams_)
        : jit_generator(jit_name()), params(aparams), tparams(tparams_) {}

    const jit_1x1_conv_conf_t *params;
    const jit_transpose4x16_src_t *tparams;

    static const int transpose_size = 4;

private:
    static const int typesize = sizeof(float);

    int src_stride = 0, tr_src_stride = 0;

    Xbyak::Reg64 imm_addr64 = rbx;

    Xbyak::Opmask kF0 = k1;
    Xbyak::Opmask kCC = k2;
    Xbyak::Opmask k33 = k3;
    Xbyak::Opmask kFFFF = k4;

    Xbyak::Zmm vidx01 = zmm31;
    Xbyak::Zmm vidx10 = zmm30;
    Xbyak::Zmm vidx1 = zmm29;
    Xbyak::Zmm vidxP = zmm28;

    Xbyak::Reg64 reg_src = r8;
    Xbyak::Reg64 reg_tr_src = r9;
    Xbyak::Reg64 reg_src_prf = r10;
    Xbyak::Reg64 reg_tr_src_prf = r11;
    Xbyak::Reg64 reg_loop = r12;
    Xbyak::Reg64 reg_tr_src_tmp = r13;
    Xbyak::Reg32 regw_tmp = r14d;

    void transpose_block(int ur, int nrows);
    void transpose(int nrows);
    void generate() override;
};

struct jit_diff_wei_trans_to_vnni_t : public jit_generator {
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_diff_wei_trans_to_vnni_t)

    jit_diff_wei_trans_to_vnni_t(const data_type_t dt, const int &kd,
            const int &kh, const int &kw, const int &ic_block,
            const int &oc_block)
        : jit_generator(jit_name())
        , out_dt_(dt)
        , kd_(kd)
        , kh_(kh)
        , kw_(kw)
        , ic_block_(ic_block)
        , oc_block_(oc_block) {}

    ~jit_diff_wei_trans_to_vnni_t() {}

    status_t create_kernel() override { return jit_generator::create_kernel(); }

    const data_type_t out_dt_;
    const int kd_, kh_, kw_;
    const int ic_block_, oc_block_;

private:
    void generate() override;
};

jit_trans_src_t *create_trans_src(const jit_conv_conf_t *conf);
jit_trans_dst_t *create_trans_dst(const jit_conv_conf_t *conf);

} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif
