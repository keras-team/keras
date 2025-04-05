/*******************************************************************************
* Copyright 2020-2022 Intel Corporation
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

#ifndef CPU_X64_JIT_INJECTOR_UTILS_HPP
#define CPU_X64_JIT_INJECTOR_UTILS_HPP

#include <array>
#include <cstddef>
#include <set>
#include <stack>

#include "cpu/x64/jit_generator.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {
namespace injector_utils {

using vmm_index_set_t = typename std::set<size_t>;
using vmm_index_set_iterator_t = typename std::set<size_t>::iterator;

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
class register_preserve_guard_t {

public:
    register_preserve_guard_t(jit_generator *host,
            std::initializer_list<Xbyak::Reg64> reg64_to_preserve,
            std::initializer_list<Xbyak::Xmm> vmm_to_preserve = {});
    register_preserve_guard_t(register_preserve_guard_t &&other) = default;
    register_preserve_guard_t &operator=(register_preserve_guard_t &&other)
            = default;
    DNNL_DISALLOW_COPY_AND_ASSIGN(register_preserve_guard_t);
    ~register_preserve_guard_t();
    size_t stack_space_occupied() const;

private:
    jit_generator *host_;
    std::stack<Xbyak::Reg64> reg64_stack_;
    std::stack<Xbyak::Xmm> vmm_stack_;
    size_t vmm_to_preserve_size_bytes_;
};

class conditional_register_preserve_guard_t : public register_preserve_guard_t {
public:
    conditional_register_preserve_guard_t(bool condition_to_be_met,
            jit_generator *host,
            std::initializer_list<Xbyak::Reg64> reg64_to_preserve,
            std::initializer_list<Xbyak::Xmm> vmm_to_preserve = {});
    DNNL_DISALLOW_COPY_AND_ASSIGN(conditional_register_preserve_guard_t);
};

} // namespace injector_utils
} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif
