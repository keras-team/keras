/*******************************************************************************
* Copyright 2020-2024 Intel Corporation
* Copyright 2023 Arm Ltd. and affiliates
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

#ifndef CPU_REORDER_CPU_REORDER_HPP
#define CPU_REORDER_CPU_REORDER_HPP

#include <map>
#include <vector>

#include "cpu/reorder/simple_reorder.hpp"
#include "cpu/reorder/simple_sparse_reorder.hpp"

#include "common/impl_list_item.hpp"
#include "common/memory.hpp"
#include "common/type_helpers.hpp"

#include "cpu/cpu_engine.hpp"
#include "cpu/reorder/cpu_reorder_pd.hpp"

#if DNNL_X64
#include "cpu/x64/jit_uni_reorder.hpp"
#include "cpu/x64/matmul/brgemm_matmul_reorders.hpp"
#elif DNNL_AARCH64
#include "cpu/aarch64/jit_uni_reorder.hpp"
#endif

#if DNNL_AARCH64 && DNNL_AARCH64_USE_ACL
#include "cpu/aarch64/acl_reorder.hpp"
#endif

#include "cpu/rnn/rnn_reorders.hpp"

namespace dnnl {
namespace impl {
namespace cpu {

using namespace dnnl::impl::data_type;
using namespace dnnl::impl::format_tag;

struct reorder_impl_key_t {
    data_type_t src_dt;
    data_type_t dst_dt; // data_type::undef if arbitrary
    int ndims; // 0 if arbitrary

    bool operator<(const reorder_impl_key_t &rhs) const {
        return value() < rhs.value();
    }

private:
    size_t value() const {
        const size_t dtm = data_type::data_type_max;
        const size_t m1 = static_cast<size_t>(ndims) * dtm;
        const size_t m2 = (m1 + static_cast<size_t>(src_dt)) * dtm;
        return m2 + static_cast<size_t>(dst_dt);
    }
};

using impl_list_map_t
        = std::map<reorder_impl_key_t, std::vector<impl_list_item_t>>;

/* regular reorders */
extern const impl_list_map_t &regular_f32_fp8_impl_list_map();
extern const impl_list_map_t &regular_f32_bf16_impl_list_map();
extern const impl_list_map_t &regular_f32_f16_impl_list_map();
extern const impl_list_map_t &regular_f32_f32_impl_list_map();
extern const impl_list_map_t &regular_f32_s32_impl_list_map();
extern const impl_list_map_t &regular_f32_s8_impl_list_map();
extern const impl_list_map_t &regular_f32_u8_impl_list_map();
extern const impl_list_map_t &regular_fp8_impl_list_map();
extern const impl_list_map_t &regular_bf16_impl_list_map();
extern const impl_list_map_t &regular_f16_impl_list_map();
extern const impl_list_map_t &regular_s32_impl_list_map();
extern const impl_list_map_t &regular_s8_impl_list_map();
extern const impl_list_map_t &regular_u8_impl_list_map();
extern const impl_list_map_t &regular_s4_impl_list_map();
extern const impl_list_map_t &regular_u4_impl_list_map();

/* conv reorders w/ compensation */
extern const impl_list_map_t &comp_f32_s8_impl_list_map();
extern const impl_list_map_t &comp_bf16_s8_impl_list_map();
extern const impl_list_map_t &comp_s8_s8_impl_list_map();

// clang-format off

// Some compilers do not allow guarding implementations with macros
// in the impl list.
#ifdef DNNL_EXPERIMENTAL_SPARSE

#if DNNL_X64
#define REG_SPARSE_SR_X64(idt, ifmt, odt, ofmt) \
    impl_list_item_t(impl_list_item_t::reorder_type_deduction_helper_t< \
            simple_sparse_reorder_t<idt, \
                    std::remove_const<decltype(ifmt)>::type, ifmt, odt, \
                    std::remove_const<decltype(ofmt)>::type, ofmt>::pd_t>()),
#else
#define REG_SPARSE_SR_X64(...)
#endif

#else
#define REG_SPARSE_SR_X64(...)
#endif

#define REG_SR(idt, ifmt, odt, ofmt, ...) \
    impl_list_item_t(impl_list_item_t::reorder_type_deduction_helper_t< \
            simple_reorder_t<idt, ifmt, odt, ofmt, __VA_ARGS__>::pd_t>()),

#define REG_SR_BIDIR(idt, ifmt, odt, ofmt) \
    REG_SR(idt, ifmt, odt, ofmt, fmt_order::keep) \
    REG_SR(idt, ifmt, odt, ofmt, fmt_order::reverse)

#define REG_SR_DIRECT_COPY(idt, odt)				  \
    REG_SR(idt, any, odt, any, fmt_order::any, spec::direct_copy) \
    REG_SR(idt, any, odt, any, fmt_order::any, spec::direct_copy_except_dim_0)

// clang-format on

#if defined(__INTEL_COMPILER) || (defined(__GNUC__) && !defined(__clang__))
/* Direct copy for icc which is faster than jitted code;
 * Direct copy for gcc which might or might not be faster than jitted
 * code, but still worth it because doesn't require jitting, i.e. much
 * faster creation time. This is tentative solution and should be
 * removed later (when we will cache jitted code?...). */
#define REG_FAST_DIRECT_COPY_F32_F32 REG_SR_DIRECT_COPY(f32, f32)
#else
#define REG_FAST_DIRECT_COPY_F32_F32
#endif

#ifdef __INTEL_COMPILER
/* direct copy for icc, which is faster than jitted code */
#define REG_FAST_DIRECT_COPY(sdt, ddt) REG_SR_DIRECT_COPY(sdt, ddt)
#else
#define REG_FAST_DIRECT_COPY(sdt, ddt)
#endif

#define CPU_REORDER_INSTANCE(...) \
    impl_list_item_t(impl_list_item_t::reorder_type_deduction_helper_t< \
            __VA_ARGS__::pd_t>()),

} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif
