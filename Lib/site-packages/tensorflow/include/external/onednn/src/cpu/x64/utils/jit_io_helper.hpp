/*******************************************************************************
* Copyright 2021-2024 Intel Corporation
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

#ifndef CPU_X64_UTILS_JIT_IO_HELPER_HPP
#define CPU_X64_UTILS_JIT_IO_HELPER_HPP

#include <map>
#include <memory>
#include <unordered_set>

#include "common/optional.hpp"

#include "cpu/x64/cpu_isa_traits.hpp"
#include "cpu/x64/jit_generator.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {

struct bf16_emulation_t;
struct fp8_emulation_base_t;

namespace io {

class io_conf_t {
public:
    io_conf_t() = default;
    io_conf_t(const bool nt_stores_enabled);
    io_conf_t(const io_conf_t &other) = default;

    io_conf_t &operator=(const io_conf_t &other) = default;

    bool nt_stores_enabled_ = false;
};

class io_tail_conf_t {
public:
    io_tail_conf_t(const std::size_t simd_w, const std::size_t tail_size,
            const Xbyak::Opmask &tail_opmask, const int tail_vmm_mask_idx,
            const Xbyak::Reg64 &reg_tmp);
    io_tail_conf_t(const std::size_t simd_w, const std::size_t tail_size,
            int tail_opmask_idx, const int tail_vmm_mask_idx,
            const Xbyak::Reg64 &reg_tmp);
    io_tail_conf_t(const io_tail_conf_t &other) = default;

    io_tail_conf_t &operator=(const io_tail_conf_t &other) = default;

    std::size_t simd_w_ = 0;
    std::size_t tail_size_ = 0;
    Xbyak::Opmask tail_opmask_ = Xbyak::Opmask();
    int tail_vmm_mask_idx_ = 0;
    Xbyak::Reg64 reg_tmp_ = Xbyak::Reg64();
};

class io_emu_bf16_conf_t {
public:
    io_emu_bf16_conf_t() = default;
    io_emu_bf16_conf_t(const Xbyak::Zmm &bf16_emu_reserv_1,
            const Xbyak::Zmm &bf16_emu_reserv_2,
            const Xbyak::Zmm &bf16_emu_reserv_3, const Xbyak::Reg64 &reg_tmp,
            const Xbyak::Zmm &bf16_emu_reserv_4);
    io_emu_bf16_conf_t(int bf16_emu_reserv_1_idx, int bf16_emu_reserv_2_idx,
            int bf16_emu_reserv_3_idx, const Xbyak::Reg64 &reg_tmp,
            int bf16_emu_reserv_4_idx);
    io_emu_bf16_conf_t(const io_emu_bf16_conf_t &other) = default;

    io_emu_bf16_conf_t &operator=(const io_emu_bf16_conf_t &other) = default;

    Xbyak::Zmm bf16_emu_reserv_1_ = Xbyak::Zmm(28);
    Xbyak::Zmm bf16_emu_reserv_2_ = Xbyak::Zmm(29);
    Xbyak::Zmm bf16_emu_reserv_3_ = Xbyak::Zmm(30);
    Xbyak::Reg64 reg_tmp_ = Xbyak::Reg64(Xbyak::Operand::RAX);
    Xbyak::Zmm bf16_emu_reserv_4_ = Xbyak::Zmm(31);
};

class io_emu_fp8_conf_t {
public:
    io_emu_fp8_conf_t() = default;
    io_emu_fp8_conf_t(const Xbyak::Zmm &fp8_emu_reserv_1,
            const Xbyak::Zmm &fp8_emu_reserv_2,
            const Xbyak::Zmm &fp8_emu_reserv_3,
            const Xbyak::Zmm &fp8_emu_reserv_4,
            const Xbyak::Zmm &fp8_emu_reserv_5, const Xbyak::Opmask &kmask_aux,
            const Xbyak::Reg64 &reg_tmp);
    io_emu_fp8_conf_t(int fp8_emu_reserv_1_idx, int fp8_emu_reserv_2_idx,
            int fp8_emu_reserv_3_idx, int fp8_emu_reserv_4_idx,
            int fp8_emu_reserv_5_idx, int kmask_aux_idx_,
            const Xbyak::Reg64 &reg_tmp);
    io_emu_fp8_conf_t(const io_emu_fp8_conf_t &other) = default;

    io_emu_fp8_conf_t &operator=(const io_emu_fp8_conf_t &other) = default;

    Xbyak::Zmm fp8_emu_reserv_1_ = Xbyak::Zmm(27);
    Xbyak::Zmm fp8_emu_reserv_2_ = Xbyak::Zmm(28);
    Xbyak::Zmm fp8_emu_reserv_3_ = Xbyak::Zmm(29);
    Xbyak::Zmm fp8_emu_reserv_4_ = Xbyak::Zmm(30);
    Xbyak::Zmm fp8_emu_reserv_5_ = Xbyak::Zmm(31);
    Xbyak::Opmask kmask_aux_ = Xbyak::Opmask(2);
    Xbyak::Reg64 reg_tmp_ = Xbyak::util::rax;
};

class io_saturation_conf_t {
public:
    io_saturation_conf_t(const int vreg_zero_saturation_idx,
            const int vreg_saturation_ubound_idx, const Xbyak::Reg64 &reg_tmp);
    io_saturation_conf_t(const io_saturation_conf_t &other) = default;

    io_saturation_conf_t &operator=(const io_saturation_conf_t &other)
            = default;

    int vreg_zero_saturation_idx_ = 0;
    int vreg_saturation_ubound_idx_ = 0;
    Xbyak::Reg64 reg_tmp_ = Xbyak::Reg64();
};

class io_gather_conf_t {
public:
    io_gather_conf_t(const std::size_t simd_w, const Xbyak::Opmask &full_opmask,
            const int full_vmm_mask_idx, const Xbyak::Reg64 &reg_tmp,
            const Xbyak::Reg64 &reg_tmp1,
            const utils::optional_t<int> &vmm_tmp_idx = utils::nullopt);
    io_gather_conf_t(const io_gather_conf_t &other) = default;

    io_gather_conf_t &operator=(const io_gather_conf_t &other) = default;

    std::size_t simd_w_ = 0;
    Xbyak::Opmask full_opmask_ = Xbyak::Opmask();
    int full_vmm_mask_idx_ = 0;
    Xbyak::Reg64 reg_tmp_ = Xbyak::Reg64();
    Xbyak::Reg64 reg_tmp1_ = Xbyak::Reg64();
    // It is needed, when io_helper use emulation for gather
    // and it is not needed for sse.
    utils::optional_t<int> vmm_tmp_idx_ = utils::nullopt;
};

template <typename Vmm>
class jit_io_multi_dt_helper_t;

template <typename Vmm>
class jit_io_helper_t {
public:
    friend class jit_io_multi_dt_helper_t<Vmm>;

    jit_io_helper_t() = default;
    jit_io_helper_t(jit_generator *host, const cpu_isa_t &isa,
            const data_type_t &data_type, const io_conf_t &io_conf,
            const utils::optional_t<io_tail_conf_t> &tail_conf = utils::nullopt,
            const utils::optional_t<io_emu_bf16_conf_t> &bf16_conf
            = utils::nullopt,
            const utils::optional_t<io_saturation_conf_t> &saturation_conf
            = utils::nullopt,
            const utils::optional_t<io_gather_conf_t> &gather_conf
            = utils::nullopt,
            const utils::optional_t<io_emu_fp8_conf_t> &fp8_conf
            = utils::nullopt);
    jit_io_helper_t(jit_io_helper_t &&) = default;
    jit_io_helper_t &operator=(jit_io_helper_t &&) = default;

    ~jit_io_helper_t();
    void prepare_tail_mask();
    void prepare_full_mask();
    /*
     * Sometimes the values in the register can be nan at the
     * beginning of the kernel, then using vcmpps(vmm, vmm, vmm)
     * will not set all bits to 1, instead of that this instruction will
     * return zero. At the beginning, it is worth to zeroing
     * full mask vmm to be sure, that vcmpps work properly.
     */
    void init_full_mask();
    void init_saturate_f32() const;
    void init_bf16();
    void prepare_table_fp8();
    void gather(const Xbyak::Reg64 &src_reg, const Vmm &indices_vmm,
            const Vmm &dst_vmm, const bool tail);
    void broadcast(const Xbyak::Address &src_addr, const Vmm &dst_vmm);
    void load(const Xbyak::Address &src_addr, const Vmm &dst_vmm,
            const bool tail);
    void store(const Vmm &src_vmm, const Xbyak::Address &dst_addr,
            const bool tail);
    // Load odd and even data into two registers with ne_convert instructions
    // Then we can call merge_interleaved_to_plain() to merge them into
    // plain layouts when needed
    void load_two_simdw_xf16(const Xbyak::Address &src_addr,
            const Vmm &dst_even_vmm, const Vmm &dst_odd_vmm);
    void merge_interleaved_to_plain(
            const Vmm &vmm_even, const Vmm &vmm_odd, const Vmm &vmm_aux0);

private:
    bool is_data_type_supported(const data_type_t dt);
    void prepare_opmask(const std::size_t how_many_bits_to_set,
            const Xbyak::Reg64 &reg_tmp, const Xbyak::Opmask &mask);
    void prepare_vmm_mask(const std::size_t how_many_bits_to_set,
            const std::size_t simd_w, const Xbyak::Reg64 &reg_tmp,
            const Vmm &mask);
    void prepare_i8_data_to_store(const Vmm &i8_vmm);
    void prepare_xf16_data_to_store(const Vmm &vmm);
    // Emulates the behavior of vgatherdps for architectures
    // that do not support this instruction.
    void emu_gather(const Xbyak::Reg64 &src_reg, const Vmm &indices_vmm,
            const Vmm &dst_vmm, const bool tail);
    void load_byte_by_byte(const Xbyak::Address &src_addr, const Vmm &dst_vmm,
            const int load_size);
    void load_f32(const Xbyak::Address &src_addr, const Vmm &dst_vmm,
            const bool tail);
    void load_s32(const Xbyak::Address &src_addr, const Vmm &dst_vmm,
            const bool tail);
    void load_bf16(const Xbyak::Address &src_addr, const Vmm &dst_vmm);
    void load_f16(const Xbyak::Address &src_addr, const Vmm &dst_vmm);
    void load_f8(const Xbyak::Address &src_addr, const Vmm &dst_vmm);
    void load_i8(const Xbyak::Address &src_addr, const Vmm &dst_vmm);
    void saturate(const Vmm &vmm);
    void store_byte_by_byte(const Vmm &src_vmm, const Xbyak::Address &dst_addr,
            const int store_size);
    void store_f32(const Vmm &src_vmm, const Xbyak::Address &dst_addr,
            const bool tail);
    void store_bf16(const Vmm &src_vmm, const Xbyak::Address &dst_addr);
    void store_f16(const Vmm &src_vmm, const Xbyak::Address &dst_addr);
    void store_f8(const Vmm &src_vmm, const Xbyak::Address &dst_addr);
    void store_i8(const Vmm &src_vmm, const Xbyak::Address &dst_addr);
    void convert_to_f32(const Vmm &dst_vmm, const Xbyak::Xmm &src_vmm,
            const data_type_t src_data_type);

    jit_generator *host_;
    const cpu_isa_t isa_;
    const data_type_t data_type_;
    const bool bf16_supported_;
    const bool f16_supported_;
    const bool fp8_supported_;
    std::unique_ptr<bf16_emulation_t> bf16_emu_;
    std::unique_ptr<fp8_emulation_base_t> fp8_emu_;
    const io_conf_t io_conf_;
    const utils::optional_t<io_tail_conf_t> tail_conf_;
    const utils::optional_t<io_emu_bf16_conf_t> bf16_conf_;
    const utils::optional_t<io_saturation_conf_t> saturation_conf_;
    const utils::optional_t<io_gather_conf_t> gather_conf_;
    const utils::optional_t<io_emu_fp8_conf_t> fp8_conf_;
};

template <typename Vmm>
class jit_io_multi_dt_helper_t {
public:
    using data_types_t = std::unordered_set<data_type_t, std::hash<int>>;
    using saturation_map_t = std::map<data_type_t, io_saturation_conf_t>;

    jit_io_multi_dt_helper_t();
    jit_io_multi_dt_helper_t(jit_generator *host, const cpu_isa_t &isa,
            const data_types_t &data_types, const io_conf_t &io_conf,
            const utils::optional_t<io_tail_conf_t> &tail_conf = utils::nullopt,
            const utils::optional_t<io_emu_bf16_conf_t> &bf16_conf
            = utils::nullopt,
            const saturation_map_t &saturation_confs = saturation_map_t {},
            const utils::optional_t<io_gather_conf_t> &gather_conf
            = utils::nullopt,
            const utils::optional_t<io_emu_fp8_conf_t> &fp8_conf
            = utils::nullopt);
    virtual ~jit_io_multi_dt_helper_t();
    void prepare_tail_mask();
    void prepare_full_mask();
    void init_saturate_f32(const data_types_t &store_data_types);
    void init_full_mask();
    void init_bf16();

    std::shared_ptr<jit_io_helper_t<Vmm>> at(const data_type_t dt) const;
    bool empty() const;
    std::shared_ptr<jit_io_helper_t<Vmm>> operator[](
            const data_type_t dt) const;

private:
    std::unordered_map<data_type_t, std::shared_ptr<jit_io_helper_t<Vmm>>,
            std::hash<int>>
            storage_;
};

template <>
void jit_io_helper_t<Xbyak::Zmm>::emu_gather(const Xbyak::Reg64 &src_reg,
        const Xbyak::Zmm &indices_vmm, const Xbyak::Zmm &dst_vmm,
        const bool tail);
template <>
void jit_io_helper_t<Xbyak::Ymm>::emu_gather(const Xbyak::Reg64 &src_reg,
        const Xbyak::Ymm &indices_vmm, const Xbyak::Ymm &dst_vmm,
        const bool tail);
template <>
void jit_io_helper_t<Xbyak::Xmm>::emu_gather(const Xbyak::Reg64 &src_reg,
        const Xbyak::Xmm &indices_vmm, const Xbyak::Xmm &dst_vmm,
        const bool tail);

extern template class jit_io_helper_t<Xbyak::Zmm>;
extern template class jit_io_helper_t<Xbyak::Ymm>;
extern template class jit_io_helper_t<Xbyak::Xmm>;

extern template class jit_io_multi_dt_helper_t<Xbyak::Zmm>;
extern template class jit_io_multi_dt_helper_t<Xbyak::Ymm>;
extern template class jit_io_multi_dt_helper_t<Xbyak::Xmm>;

} // namespace io
} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif
