/*******************************************************************************
* Copyright 2020-2022 Intel Corporation
* Copyright 2021-2024 FUJITSU LIMITED
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
#ifndef CPU_AARCH64_JIT_INJECTOR_UTILS_HPP
#define CPU_AARCH64_JIT_INJECTOR_UTILS_HPP

#include <array>
#include <cstddef>
#include <set>
#include <stack>

#include "cpu/aarch64/jit_generator.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace aarch64 {
namespace injector_utils {

using vmm_index_set_t = typename std::set<size_t>;
using vmm_index_set_iterator_t = typename std::set<size_t>::iterator;
template <cpu_isa_t isa>
struct vmm_size_t;

template <>
struct vmm_size_t<sve_512> {
    static constexpr std::size_t bytes = 64u;
};

template <>
struct vmm_size_t<sve_256> {
    static constexpr std::size_t bytes = 32u;
};

/*
template <>
struct vmm_size_t<sve_128> {
    static constexpr std::size_t bytes = 16u;
    };*/

enum class layout_t { ncsp, c_blocked, nspc, cspn, unsupported };

inline layout_t get_layout_type(const memory_desc_wrapper &dst_d) {
    const auto strides = dst_d.blocking_desc().strides;
    if (!dst_d.is_plain()) return layout_t::c_blocked;
    if (strides[0] >= strides[1]
            && IMPLICATION(dst_d.ndims() >= 3, strides[1] >= strides[2]))
        return layout_t::ncsp;
    if (strides[1] == 1) return layout_t::nspc;
    if (strides[0] == 1) return layout_t::cspn;
    return layout_t::unsupported;
}

/*
 * Scope guard for general purpose register and vector registers preservation.
 * Pushes registers to stack during construction and pops during destruction.
 */
template <cpu_isa_t isa>
class register_preserve_guard_t {

public:
    register_preserve_guard_t(jit_generator *host,
            std::initializer_list<Xbyak_aarch64::XReg> reg64_to_preserve,
            std::initializer_list<Xbyak_aarch64::VReg> vmm_to_preserve = {});
    register_preserve_guard_t(register_preserve_guard_t &&other) = default;
    register_preserve_guard_t &operator=(register_preserve_guard_t &&other)
            = default;
    DNNL_DISALLOW_COPY_AND_ASSIGN(register_preserve_guard_t);
    ~register_preserve_guard_t();
    size_t calc_vmm_to_preserve_size_bytes(
            const std::initializer_list<Xbyak_aarch64::VReg> &vmm_to_preserve)
            const;
    size_t stack_space_occupied() const;

private:
    jit_generator *host_;
    std::stack<Xbyak_aarch64::XReg> reg64_stack_;
    std::stack<Xbyak_aarch64::VReg> vmm_stack_;
    const uint64_t cpu_sveLen_ = get_sve_length();
    size_t vmm_to_preserve_size_bytes_;
};

template <cpu_isa_t isa>
class conditional_register_preserve_guard_t
    : public register_preserve_guard_t<isa> {
public:
    conditional_register_preserve_guard_t(bool condition_to_be_met,
            jit_generator *host,
            std::initializer_list<Xbyak_aarch64::XReg> reg64_to_preserve,
            std::initializer_list<Xbyak_aarch64::VReg> vmm_to_preserve = {});
    DNNL_DISALLOW_COPY_AND_ASSIGN(conditional_register_preserve_guard_t);
};

} // namespace injector_utils
} // namespace aarch64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif
