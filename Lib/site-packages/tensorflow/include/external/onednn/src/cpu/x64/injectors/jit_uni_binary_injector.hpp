/*******************************************************************************
* Copyright 2020-2024 Intel Corporation
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

#ifndef CPU_X64_JIT_UNI_BINARY_INJECTOR_HPP
#define CPU_X64_JIT_UNI_BINARY_INJECTOR_HPP

#include <array>
#include <cassert>
#include <functional>
#include <map>
#include <utility>
#include <vector>
#include <unordered_set>

#include "common/broadcast_strategy.hpp"
#include "common/c_types_map.hpp"
#include "common/primitive_attr.hpp"
#include "common/primitive_exec_types.hpp"
#include "cpu/binary_injector_utils.hpp"
#include "cpu/x64/cpu_isa_traits.hpp"
#include "cpu/x64/injectors/injector_utils.hpp"
#include "cpu/x64/jit_generator.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {
namespace binary_injector {
using dnnl::impl::cpu::binary_injector_utils::get_src1_desc;
using dnnl::impl::cpu::binary_injector_utils::prepare_binary_args;

bool binary_args_broadcast_supported(const post_ops_t &post_ops,
        const memory_desc_wrapper &dst_d,
        const bcast_set_t &supported_strategy_set);

bool any_binary_postop_rhs_non_scalar_broadcast(
        const post_ops_t &post_ops, const memory_desc_wrapper &dst_d);
bool any_binary_postop_rhs_per_oc_broadcast(const post_ops_t &post_ops,
        const memory_desc_wrapper &dst_d,
        const bcast_set_t &supported_strategy_set);

bool all_binary_postop_rhs_per_oc_broadcast(const post_ops_t &post_ops,
        const memory_desc_wrapper &dst_d,
        const bcast_set_t &supported_strategy_set,
        const std::function<bool(const memory_desc_wrapper &)> &predicate);

/*
 * Represents params related to all binary post-ops right-hand side arguments
 * (arg1) that don't change during jit_uni_binary_injector_t object lifetime
 * and between compute_vector_range calls.
 *
 * @param rhs_dt_helper_vmm_idx - index of vmm helper used when loading data for
 * calculations. Treated as hint from user. If inside compute_vector_range hint
 * turns out to be invalid, it will be overwriten by register preserving logic inside
 * binary injector.
 * @param rhs_addr_reg - gpr register, used as the currently processed address of
 * rhs tensor slice. Data of rhs(arg1) for the binary operation is loaded from address
 * stored inside rhs_addr_reg.
 * @param rhs_helper_reg - gpr register used as helper for calculations during data
 * loading phase.
 * @param rhs_addr_cache_reg - gpr register used for caching part of calculated
 * offset, this register is always preserved.
 * @param preserve_gpr_helpers - determines whether gpr registers specified above
 * should be preserved (pushed to stack and poped back afterwords) between
 * compute_vector_range calls.
 * @param preserve_vmm_helper - determines whether vmm helper register specified
 * above should be preserved between compute_vector_range calls.
 * @param abi_param_offset - offset to rhs tensor from first binary post-op operation
 * specified by user from runtime structure passed to kernel as abi param 1.
 * @param dst_orig_offset - offset 0 to destination tensor
 * @param dst_d - descriptor of destination tensor (result after applying all post-ops
 * operations).
 * @param tail_opmask - register with loaded by user mask, used in avx512 for load with
 * tail handling.
 * @param tail_size - size of processed tail in elements.
 * @param use_exact_tail_scalar_bcast - in case of scalar broadcast user can disable
 * loading data with tail, usually bcast through entire vector is faster (usually 1 instruction)
 * vs. broadcasting limited by tail size (potentially several instructions). In case
 * when user during storing ignores values from vmm above tail size, setting this option to
 * false can result in better performance.
 * @param reg_tail_size - register with loaded size of tail, used in sse41/avx/avx2
 * for load with tail in runtime.
 */
struct rhs_arg_static_params_t {
    rhs_arg_static_params_t(std::size_t rhs_dt_helper_vmm_idx,
            const Xbyak::Reg64 &rhs_addr_reg,
            const Xbyak::Reg64 &rhs_helper_reg,
            const Xbyak::Reg64 &rhs_addr_cache_reg, bool preserve_gpr_helpers,
            bool preserve_vmm_helper, std::size_t abi_param_offset,
            std::size_t dst_orig_offset, const memory_desc_wrapper &dst_d,
            std::size_t tail_size = 0u,
            bool use_exact_tail_scalar_bcast = false);
    rhs_arg_static_params_t(std::size_t rhs_dt_helper_vmm_idx,
            const Xbyak::Reg64 &rhs_addr_reg,
            const Xbyak::Reg64 &rhs_helper_reg,
            const Xbyak::Reg64 &rhs_addr_cache_reg, bool preserve_gpr_helpers,
            bool preserve_vmm_helper, std::size_t abi_param_offset,
            std::size_t dst_orig_offset, const memory_desc_wrapper &dst_d,
            std::size_t tail_size, const Xbyak::Opmask &tail_opmask,
            bool use_exact_tail_scalar_bcast);
    rhs_arg_static_params_t(std::size_t rhs_dt_helper_vmm_idx,
            const Xbyak::Reg64 &rhs_addr_reg,
            const Xbyak::Reg64 &rhs_helper_reg,
            const Xbyak::Reg64 &rhs_addr_cache_reg, bool preserve_gpr_helpers,
            bool preserve_vmm_helper, std::size_t abi_param_offset,
            std::size_t dst_orig_offset, const memory_desc_wrapper &dst_d,
            std::size_t tail_size, const Xbyak::Opmask &tail_opmask,
            const Xbyak::Reg64 &reg_tail_size,
            bool use_exact_tail_scalar_bcast);

    bool is_opmask_set() const noexcept { return is_opmask_set_; }

    mutable std::size_t rhs_dt_helper_vmm_idx;
    Xbyak::Reg64 rhs_addr_reg;
    Xbyak::Reg64 rhs_helper_reg;
    Xbyak::Reg64 rhs_addr_cache_reg;
    bool preserve_gpr_helpers;
    bool preserve_vmm_helper;
    std::size_t abi_param_offset;
    std::size_t dst_orig_offset;
    memory_desc_wrapper dst_d;
    std::size_t tail_size;
    Xbyak::Opmask tail_opmask;
    bool use_exact_tail_scalar_bcast;
    Xbyak::Reg64 reg_tail_size;
    bool is_tail;

private:
    rhs_arg_static_params_t(std::size_t rhs_dt_helper_vmm_idx,
            const Xbyak::Reg64 &rhs_addr_reg,
            const Xbyak::Reg64 &rhs_helper_reg,
            const Xbyak::Reg64 &rhs_addr_cache_reg, bool preserve_gpr_helpers,
            bool preserve_vmm_helper, std::size_t abi_param_offset,
            std::size_t dst_orig_offset, const memory_desc_wrapper &dst_d,
            std::size_t tail_size, const Xbyak::Opmask &tail_opmask,
            bool use_exact_tail_scalar_bcast, const Xbyak::Reg64 &reg_tail_size,
            bool is_opmask_set);

    bool is_opmask_set_;
};

/*
 * Represents params required by jit_uni_binary_injector_t that don't change
 * during it's entire lifetime.
 *
 * @param param1 - register storing abi param1. At the moment of calling
 * compute_vector_range method can be different than the default one defined
 * inside jit_generator.
 * @param bcast_set_t supported_strategy_set - set allowing disabling particular
 * bcast strategies
 * @param rhs_arg_static_params - params related to all binary post-ops right-hand side
 * arguments that don't change during entire lifetime of jit_uni_binary_injector_t
 * object.
 */
struct static_params_t {
    static_params_t(const Xbyak::Reg64 &param1,
            const bcast_set_t &supported_strategy_set,
            const rhs_arg_static_params_t &rhs_arg_static_params);
    static_params_t(const Xbyak::Reg64 &param1,
            const rhs_arg_static_params_t &rhs_arg_static_params);

    Xbyak::Reg64 param1;
    const bcast_set_t supported_strategy_set;
    rhs_arg_static_params_t rhs_arg_static_params;
};

/*
 * Mode of data load with tail for rhs:
 * STATIC - load based on given integer.
 * DYNAMIC - load based on opmask or 64-bit register.
 * DEFAULT - DYNAMIC for avx512, STATIC for others.
 */

enum class tail_lode_mode_t { STATIC, DYNAMIC, DEFAULT };

/*
 * Represents params passed to compute_vector_range method of
 * jit_uni_binary_injector_t that can be different for each call.
 * Contains configurable std::maps where key is vmm index and value is
 * offset in elements. The offset value identifies tensor slice in particular
 * vmm. This is utilized by broadcasting mechanism. Offset, depending on the
 * implementation particular kernels, can be passed as value (usually during
 * unrolling), inside operand, under memory address.
 *
 * @param vmm_idx_to_out_addr - vmm mapped to address of destination tensor with offset,
 * used to calculate offset in no_broadcast strategy, but also in other strategies whose
 * calculations are based on no_broadcast strategy.
 * @param vmm_idx_to_out_reg - vmm mapped to register containing address of destination
 * with offset, used to calculate offset in no_broadcast strategy, but also in other
 * strategies whose calculations are based on no_broadcast strategy.
 * @param vmm_idx_to_out_elem_off_val - vmm mapped to offset in elements passed as raw
 * value intended to use in no_broadcast strategy, but also in other
 * strategies whose calculations are based on no_broadcast strategy.
 * @param vmm_tail_idx - vmm indices that contains data don't fill the whole vector (tail).
 * @param is_dynamic_tail_load - determines whether to load with tail in
 * runtime (based on the value from reg_tail_size or opmask) or based on given
 * integer.
 */

struct rhs_arg_dynamic_params_t {
    std::map<int, Xbyak::Address> vmm_idx_to_out_addr;
    std::map<int, Xbyak::Reg64> vmm_idx_to_out_reg;
    std::map<int, size_t> vmm_idx_to_out_elem_off_val;

    std::unordered_set<int> vmm_tail_idx_;
    tail_lode_mode_t tail_load_mode = tail_lode_mode_t::DEFAULT;
};

/*
 * Checks if src1 data type is supported by binary injector.
 */
bool is_data_supported(cpu_isa_t isa, data_type_t data_type);

/*
 * Checks if broadcast of src1 is supported by binary injector.
 */
bool is_bcast_supported(const dnnl::impl::memory_desc_t &src1_desc,
        const memory_desc_wrapper &dst_d,
        const bcast_set_t &supported_strategy_set);

/*
 * Checks if binary injection for given args is supported.
 */
bool is_supported(cpu_isa_t isa, const dnnl::impl::memory_desc_t &src1_desc,
        const memory_desc_wrapper &dst_d,
        const bcast_set_t &supported_strategy_set);

/*
 * Main mechanism responsible for injecting binary postops supporting various
 * isa: sse41, avx, avx2, avx512 with core, bf16 extensions as well as data
 * types: f32, bf16, s32, u8, s8.
 */
template <cpu_isa_t isa, typename Vmm = typename cpu_isa_traits<isa>::Vmm>
class jit_uni_binary_injector_t {
public:
    jit_uni_binary_injector_t(
            jit_generator *host, const static_params_t &static_params);

    /*
     * Generates code of binary post_op injected to host primitive. Applied to
     * ordered set of vector registers' indexes. Function loads appropriate
     * slice of rhs tensor for computations based on internally determined
     * broadcast strategy and information about stored data in particular vmm
     * described inside rhs_arg_params.
     */
    void compute_vector_range(const injector_utils::vmm_index_set_t &vmm_idxs,
            std::size_t rhs_arg_idx, const dnnl_post_ops::entry_t &post_op,
            const rhs_arg_dynamic_params_t &rhs_arg_params) const;

    /*
     * Generates code of binary post_op injected to host primitive. Applied to
     * range <start_idx, end_idx) of vector registers' indexes. Function loads
     * appropriate slice of rhs tensor for computations based on internally
     * determined broadcast strategy and information about stored data in particular
     * vmm described inside rhs_arg_params.
     */
    void compute_vector_range(size_t start_idx, size_t end_idx,
            std::size_t rhs_arg_idx, const dnnl_post_ops::entry_t &post_op,
            const rhs_arg_dynamic_params_t &rhs_arg_params) const;

    /*
     * Generates code of binary post_op injected to host primitive. Applied to
     * a single vector register index. Function loads appropriate slice of rhs tensor
     * for computations based on internally determined broadcast strategy and information
     * about stored data in particular vmm described inside rhs_arg_params.
     */
    void compute_vector(size_t idx, std::size_t rhs_arg_idx,
            const dnnl_post_ops::entry_t &post_op,
            const rhs_arg_dynamic_params_t &rhs_arg_params) const;

private:
    /*
     * Determines if hint passed by user is valid (is inside range
     * <start_idx, end_idx>). If not it returns new vmm idx value that will be
     * used as temporary vmm in future computations.
     */
    int adjust_temp_vmm_hint(
            int user_hint, int start_idx, int end_idx, int max_vmm_idx) const;
    /*
     * Taking into account rhs_broadcasting_strategy and information from user
     * about tensor slice (rhs_arg_params) stored in Vmm(vmm_idx) calculates
     * address of rhs tensor slice needed for binary operation and returns
     * ptr to it.
     */
    Xbyak::Address prepare_rhs_arg_addr(std::size_t vmm_idx,
            std::size_t rhs_arg_idx, const dnnl_post_ops::entry_t &post_op,
            const rhs_arg_dynamic_params_t &rhs_arg_params,
            const broadcasting_strategy_t rhs_broadcasting_strategy,
            bool is_first) const;
    /*
     * Loads data and applies particular binary operation.
     */
    void inject_binary(const dnnl_post_ops::entry_t &post_op, Vmm dst,
            const Xbyak::Address &rhs_addr, bool with_tail,
            const tail_lode_mode_t tail_load_mode) const;

    /*
     * Helper functions responsible for preparing rhs tensor slice address.
     */
    void append_no_broadcast_offset(
            const std::map<int, Xbyak::Address> &vmm_idx_to_out_addr,
            const std::map<int, Xbyak::Reg64> &vmm_idx_to_out_reg,
            const std::map<int, size_t> &vmm_idx_to_out_elem_off_val,
            int vmm_idx, const Xbyak::Reg64 &addr_reg,
            const Xbyak::Reg64 &tmp_reg, std::size_t elem_size_bytes,
            bool is_first) const;
    void calculate_no_broadcast_base(
            Xbyak::Address addr, const Xbyak::Reg64 &out_reg) const;
    void calculate_no_broadcast_partial(const std::size_t offset,
            const Xbyak::Reg64 &out_reg, std::size_t elem_size_bytes) const;

    void append_oc_offset(
            const std::map<int, Xbyak::Address> &vmm_idx_to_out_addr,
            const std::map<int, Xbyak::Reg64> &vmm_idx_to_out_reg,
            const std::map<int, size_t> &vmm_idx_to_out_elem_off_val,
            int vmm_idx, const Xbyak::Reg64 &addr_reg,
            const Xbyak::Reg64 &tmp_reg, std::size_t elem_size_bytes,
            bool is_first) const;
    void calculate_oc_ncsp_base(
            const dim_t *strides, const Xbyak::Reg64 &tmp_reg) const;
    void calculate_oc_ncsp_partial(const dim_t *strides,
            const std::size_t offset, const Xbyak::Reg64 &tmp_reg,
            std::size_t elem_size_bytes) const;
    void calculate_oc_blocked_base(
            const dim_t *strides, const Xbyak::Reg64 &tmp_reg) const;
    void calculate_oc_blocked_partial(const dim_t *strides,
            const std::size_t offset, const Xbyak::Reg64 &tmp_reg,
            std::size_t elem_size_bytes) const;
    void calculate_oc_nspc_base(
            const dim_t *strides, const Xbyak::Reg64 &tmp_reg) const;
    void calculate_oc_nspc_partial(const dim_t *strides,
            const std::size_t offset, const Xbyak::Reg64 &tmp_reg,
            std::size_t elem_size_bytes) const;
    void calculate_oc_cspn_base(
            const dim_t *strides, const Xbyak::Reg64 &tmp_reg) const;
    void calculate_oc_cspn_partial(const dim_t *strides,
            const std::size_t offset, const Xbyak::Reg64 &tmp_reg,
            std::size_t elem_size_bytes) const;

    void append_mb_sp_offset(
            const std::map<int, Xbyak::Address> &vmm_idx_to_out_addr,
            const std::map<int, Xbyak::Reg64> &vmm_idx_to_out_reg,
            const std::map<int, size_t> &vmm_idx_to_out_elem_off_val,
            int vmm_idx, const Xbyak::Reg64 &addr_reg,
            const Xbyak::Reg64 &tmp_reg, std::size_t elem_size_bytes,
            bool is_first) const;
    void calculate_mb_sp_ncsp_base(
            const dim_t *strides, const Xbyak::Reg64 &tmp_reg) const;
    void calculate_mb_sp_ncsp_partial(const dim_t *strides,
            const std::size_t offset, const Xbyak::Reg64 &tmp_reg,
            std::size_t elem_size_bytes) const;
    void calculate_mb_sp_blocked_base(
            const dim_t *strides, const Xbyak::Reg64 &tmp_reg) const;
    void calculate_mb_sp_blocked_partial(const dim_t *strides,
            const std::size_t offset, const Xbyak::Reg64 &tmp_reg,
            std::size_t elem_size_bytes) const;
    void calculate_mb_sp_nspc_base(
            const dim_t *strides, const Xbyak::Reg64 &tmp_reg) const;
    void calculate_mb_sp_nspc_partial(const dim_t *strides,
            const std::size_t offset, const Xbyak::Reg64 &tmp_reg,
            std::size_t elem_size_bytes) const;
    void calculate_mb_sp_cspn_base(
            const dim_t *strides, const Xbyak::Reg64 &tmp_reg) const;
    void calculate_mb_sp_cspn_partial(const dim_t *strides,
            const std::size_t offset, const Xbyak::Reg64 &tmp_reg,
            std::size_t elem_size_bytes) const;

    void append_mb_w_offset(
            const std::map<int, Xbyak::Address> &vmm_idx_to_out_addr,
            const std::map<int, Xbyak::Reg64> &vmm_idx_to_out_reg,
            const std::map<int, size_t> &vmm_idx_to_out_elem_off_val,
            int vmm_idx, const Xbyak::Reg64 &addr_reg,
            const Xbyak::Reg64 &tmp_reg, std::size_t elem_size_bytes,
            bool is_first) const;
    void calculate_mb_w_ncsp_base(
            const dim_t *strides, const Xbyak::Reg64 &tmp_reg) const;
    void calculate_mb_w_ncsp_partial(const dim_t *strides,
            const std::size_t offset, const Xbyak::Reg64 &tmp_reg,
            std::size_t elem_size_bytes) const;
    void calculate_mb_w_blocked_base(
            const dim_t *strides, const Xbyak::Reg64 &tmp_reg) const;
    void calculate_mb_w_blocked_partial(const dim_t *strides,
            const std::size_t offset, const Xbyak::Reg64 &tmp_reg,
            std::size_t elem_size_bytes) const;
    void calculate_mb_w_nspc_base(
            const dim_t *strides, const Xbyak::Reg64 &tmp_reg) const;
    void calculate_mb_w_nspc_partial(const dim_t *strides,
            const std::size_t offset, const Xbyak::Reg64 &tmp_reg,
            std::size_t elem_size_bytes) const;
    void calculate_mb_w_cspn_base(
            const dim_t *strides, const Xbyak::Reg64 &tmp_reg) const;
    void calculate_mb_w_cspn_partial(const dim_t *strides,
            const std::size_t offset, const Xbyak::Reg64 &tmp_reg,
            std::size_t elem_size_bytes) const;

    void append_w_offset(
            const std::map<int, Xbyak::Address> &vmm_idx_to_out_addr,
            const std::map<int, Xbyak::Reg64> &vmm_idx_to_out_reg,
            const std::map<int, size_t> &vmm_idx_to_out_elem_off_val,
            int vmm_idx, const Xbyak::Reg64 &addr_reg,
            const Xbyak::Reg64 &tmp_reg, std::size_t elem_size_bytes,
            bool is_first) const;
    void calculate_w_ncsp_base(
            const dim_t *strides, const Xbyak::Reg64 &tmp_reg) const;
    void calculate_w_ncsp_partial(const dim_t *strides,
            const std::size_t offset, const Xbyak::Reg64 &tmp_reg,
            std::size_t elem_size_bytes) const;
    void calculate_w_blocked_base(
            const dim_t *strides, const Xbyak::Reg64 &tmp_reg) const;
    void calculate_w_blocked_partial(const dim_t *strides,
            const std::size_t offset, const Xbyak::Reg64 &tmp_reg,
            std::size_t elem_size_bytes) const;
    void calculate_w_nspc_base(
            const dim_t *strides, const Xbyak::Reg64 &tmp_reg) const;
    void calculate_w_nspc_partial(const dim_t *strides,
            const std::size_t offset, const Xbyak::Reg64 &tmp_reg,
            std::size_t elem_size_bytes) const;
    void calculate_w_cspn_base(
            const dim_t *strides, const Xbyak::Reg64 &tmp_reg) const;
    void calculate_w_cspn_partial(const dim_t *strides,
            const std::size_t offset, const Xbyak::Reg64 &tmp_reg,
            std::size_t elem_size_bytes) const;

    void append_mb_offset(
            const std::map<int, Xbyak::Address> &vmm_idx_to_out_addr,
            const std::map<int, Xbyak::Reg64> &vmm_idx_to_out_reg,
            const std::map<int, size_t> &vmm_idx_to_out_elem_off_val,
            int vmm_idx, const Xbyak::Reg64 &addr_reg,
            const Xbyak::Reg64 &tmp_reg, std::size_t elem_size_bytes,
            bool is_first) const;
    void calculate_mb_ncsp_base(
            const dim_t *strides, const Xbyak::Reg64 &tmp_reg) const;
    void calculate_mb_ncsp_partial(const dim_t *strides,
            const std::size_t offset, const Xbyak::Reg64 &tmp_reg,
            std::size_t elem_size_bytes) const;
    void calculate_mb_nspc_base(
            const dim_t *strides, const Xbyak::Reg64 &tmp_reg) const;
    void calculate_mb_nspc_partial(const dim_t *strides,
            const std::size_t offset, const Xbyak::Reg64 &tmp_reg,
            std::size_t elem_size_bytes) const;
    void calculate_mb_cspn_base(
            const dim_t *strides, const Xbyak::Reg64 &tmp_reg) const;
    void calculate_mb_cspn_partial(const dim_t *strides,
            const std::size_t offset, const Xbyak::Reg64 &tmp_reg,
            std::size_t elem_size_bytes) const;

    void append_oc_spatial_offset(
            const std::map<int, Xbyak::Address> &vmm_idx_to_out_addr,
            const std::map<int, Xbyak::Reg64> &vmm_idx_to_out_reg,
            const std::map<int, size_t> &vmm_idx_to_out_elem_off_val,
            int vmm_idx, const Xbyak::Reg64 &addr_reg,
            const Xbyak::Reg64 &tmp_reg, std::size_t elem_size_bytes,
            bool is_first) const;
    void calculate_oc_spatial_ncsp_base(
            const dim_t *strides, const Xbyak::Reg64 &tmp_reg) const;
    void calculate_oc_spatial_ncsp_partial(const dim_t *strides,
            const std::size_t offset, const Xbyak::Reg64 &tmp_reg,
            std::size_t elem_size_bytes) const;
    void calculate_oc_spatial_nspc_base(
            const dim_t *strides, const Xbyak::Reg64 &tmp_reg) const;
    void calculate_oc_spatial_nspc_partial(const dim_t *strides,
            const std::size_t offset, const Xbyak::Reg64 &tmp_reg,
            std::size_t elem_size_bytes) const;
    void calculate_oc_spatial_cspn_base(
            const dim_t *strides, const Xbyak::Reg64 &tmp_reg) const;
    void calculate_oc_spatial_cspn_partial(const dim_t *strides,
            const std::size_t offset, const Xbyak::Reg64 &tmp_reg,
            std::size_t elem_size_bytes) const;

    void append_mb_oc_offset(
            const std::map<int, Xbyak::Address> &vmm_idx_to_out_addr,
            const std::map<int, Xbyak::Reg64> &vmm_idx_to_out_reg,
            const std::map<int, size_t> &vmm_idx_to_out_elem_off_val,
            int vmm_idx, const Xbyak::Reg64 &addr_reg,
            const Xbyak::Reg64 &tmp_reg, std::size_t elem_size_bytes,
            bool is_first) const;
    void calculate_mb_oc_ncsp_base(
            const dim_t *strides, const Xbyak::Reg64 &tmp_reg) const;
    void calculate_mb_oc_ncsp_partial(const dim_t *strides,
            const std::size_t offset, const Xbyak::Reg64 &tmp_reg,
            std::size_t elem_size_bytes) const;
    void calculate_mb_oc_nspc_base(
            const dim_t *strides, const Xbyak::Reg64 &tmp_reg) const;
    void calculate_mb_oc_nspc_partial(const dim_t *strides,
            const std::size_t offset, const Xbyak::Reg64 &tmp_reg,
            std::size_t elem_size_bytes) const;
    void calculate_mb_oc_cspn_base(
            const dim_t *strides, const Xbyak::Reg64 &tmp_reg) const;
    void calculate_mb_oc_cspn_partial(const dim_t *strides,
            const std::size_t offset, const Xbyak::Reg64 &tmp_reg,
            std::size_t elem_size_bytes) const;

    template <typename T>
    typename std::enable_if<std::is_same<T, Xbyak::Zmm>::value
            || std::is_same<T, Xbyak::Address>::value>::type
    execute_cmp_binary(const Vmm &dst, const Vmm &lhs, const T &rhs,
            const unsigned int cmp_predicate) const;
    template <typename T>
    typename std::enable_if<!(std::is_same<T, Xbyak::Zmm>::value
            || std::is_same<T, Xbyak::Address>::value)>::type
    execute_cmp_binary(const Vmm &dst, const Vmm &lhs, const T &rhs,
            const unsigned int cmp_predicate) const;
    template <typename T>
    void execute_binary(alg_kind_t binary_alg, const Vmm &dst, const Vmm &lhs,
            const T &rhs) const;
    void execute_prelu(const Vmm &dst, const Xbyak::Operand &rhs) const;

    /*
     * Used in scalar broadcast strategy, broadcasting single value of given
     * data type over entire vector Vmm register.
     */
    void execute_broadcast(const data_type_t &data_type, const Vmm &tmp_reg,
            const Xbyak::Address &rhs_addr,
            const tail_lode_mode_t tail_load_mode,
            bool with_tail = false) const;
    void load_rhs(const data_type_t &data_type, const Vmm &tmp_reg,
            const Xbyak::Address &rhs_addr,
            const tail_lode_mode_t tail_load_mode,
            bool with_tail = false) const;
    void execute_broadcast_tail_with_opmask(const data_type_t &data_type,
            const Vmm &tmp_reg, const Xbyak::Address &rhs_addr) const;
    void execute_broadcast_tail_statically(const data_type_t &data_type,
            const Vmm &tmp_reg, const Xbyak::Address &rhs_addr,
            const std::size_t tail_size) const;
    void execute_broadcast_tail_with_gpr(const data_type_t &data_type,
            const Vmm &tmp_reg, const Xbyak::Address &rhs_addr) const;
    void load_rhs_tail_dynamically_with_opmask(const data_type_t &data_type,
            const Vmm &tmp_vmm, const Xbyak::Address &rhs_addr) const;
    void load_rhs_tail_dynamically_with_gpr(
            const data_type_t &data_type, const Vmm &tmp_vmm) const;
    void load_rhs_tail_statically(const data_type_t &data_type,
            const Vmm &tmp_vmm, const Xbyak::Address &rhs_addr) const;
    void execute_broadcast_no_tail(const data_type_t &data_type,
            const Vmm &tmp_vmm, const Xbyak::Address &rhs_addr) const;
    void execute_broadcast_s8u8_no_tail(const data_type_t &data_type,
            const Vmm &tmp_vmm, const Xbyak::Address &rhs_addr) const;
    void load_rhs_no_tail(const data_type_t &data_type, const Vmm &tmp_reg,
            const Xbyak::Address &rhs_addr) const;
    void load_rhs_i8_no_tail(const data_type_t &data_type, const Vmm &tmp_reg,
            const Xbyak::Address &rhs_addr) const;
    void cvt_to_f32(const Vmm &tmp_reg) const;
    /*
     * Returns pair consisting of flag indication preservation is needed for vmm
     * index in second member that should be used as temporary vmm inside inject
     * binary.
     */
    std::pair<bool, int> should_preserve_vmm(int curr_idx, int vmm_hint,
            int max_vmm_idx, bool dt_helper_vmm_needed) const;
    /*
     * Used in isa != avx512 where m32bcst is not supported, replaces ptr_b
     * with ptr.
     */
    Xbyak::Address remove_bcast_bit(const Xbyak::Address &rhs_addr) const;

    /*
    * Returns an Opmask register, that can be used from temporary use.
    * The returned register will be different from tail_opmask.
    * Note: The Opmask register returned may need to be preserved.
    */
    Xbyak::Opmask get_aux_kmask() const;

    jit_generator *host_;
    const rhs_arg_static_params_t rhs_arg_static_params_;
    const Xbyak::Reg64 param1_;
    const bcast_set_t supported_strategy_set_;
    const bool is_avx512_ = is_superset(isa, avx512_core);
    const bool is_avx512_core_fp16_ = is_superset(isa, avx512_core_fp16);

    static constexpr int sizeof_reg64 = 8;
    /*
     * Instructions from SSE/AVX used to compute binary result like vaddps where
     * second operand is memory, require mem operand to be 16/32 byte explicitly
     * aligned. (Intel Manual chapter 2.4).
     * Rule is relaxed from AVX2 (Intel Manual chapter 14.9).
     * When using benchdnn zmalloc_protect doesn't guarantee that tensor memory
     * address is 64 byte aligned, which can cause segmentation fault.
     */
    static constexpr bool binary_op_with_unaligned_mem_operand_allowed_
            = !utils::one_of(isa, avx, sse41);
};

} // namespace binary_injector
} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif
