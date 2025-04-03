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

#ifndef CPU_X64_BRGEMM_BRGEMM_CONTAINERS_HPP
#define CPU_X64_BRGEMM_BRGEMM_CONTAINERS_HPP

#include <set>
#include "common/rw_mutex.hpp"
#include "cpu/x64/amx_tile_configure.hpp"
#include "cpu/x64/brgemm/brgemm.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {

namespace brgemm_containers {

// These containers are intended to be used as local objects in brgemm
// primitives to ensure that references are unique and correct.

struct brgemm_desc_container_t {
public:
    brgemm_desc_container_t() {}
    brgemm_desc_container_t(size_t ns) { resize(ns); }
    void resize(size_t ns) { refs_.resize(ns); }
    inline const brgemm_desc_t *operator[](int idx) const { return refs_[idx]; }

    bool insert(int idx, brgemm_desc_t &brg) {
        std::vector<char> dummy_bd_mask;
        std::vector<brgemm_batch_element_t> dummy_static_offsets;
        return insert(idx, brg, dummy_bd_mask, dummy_static_offsets);
    }

    bool insert(int idx, brgemm_desc_t &brg, const std::vector<char> &bd_mask,
            const std::vector<brgemm_batch_element_t> &static_offsets);

    int insert(brgemm_desc_t &brg) {
        std::vector<char> dummy_bd_mask;
        std::vector<brgemm_batch_element_t> dummy_static_offsets;
        return insert(brg, dummy_bd_mask, dummy_static_offsets);
    }

    int insert(brgemm_desc_t &brg, const std::vector<char> &bd_mask,
            const std::vector<brgemm_batch_element_t> &static_offsets);

    size_t refs_size() { return refs_.size(); }

private:
    std::vector<const brgemm_desc_t *> refs_;

    std::map<brgemm_desc_t, int> map_;
    std::vector<std::vector<char>> bd_mask_list_;
    std::vector<std::vector<brgemm_batch_element_t>> static_offsets_list_;
};

// global storage disabled for now
// #define BRGEMM_KERNEL_GLOBAL_STORAGE

struct brgemm_kernel_container_t {
    brgemm_kernel_container_t() {}
    brgemm_kernel_container_t(size_t ns) { resize(ns); }
    void resize(size_t ns) { refs_.resize(ns); }
    inline const brgemm_kernel_t *operator[](int idx) const {
        return refs_[idx];
    }

    status_t insert(int idx, const brgemm_desc_t *brg);
    static bool brgemm_kernel_cmp(const std::shared_ptr<brgemm_kernel_t> &lhs,
            const std::shared_ptr<brgemm_kernel_t> &rhs);

private:
    std::vector<const brgemm_kernel_t *> refs_;
#ifdef BRGEMM_KERNEL_GLOBAL_STORAGE
    static utils::rw_mutex_t &rw_mutex() {
        static utils::rw_mutex_t mutex;
        return mutex;
    }

    void lock_write() { rw_mutex().lock_write(); }
    void unlock_write() { rw_mutex().unlock_write(); }

#else
    std::set<std::shared_ptr<brgemm_kernel_t>,
            decltype(brgemm_kernel_container_t::brgemm_kernel_cmp) *>
            set_ {std::set<std::shared_ptr<brgemm_kernel_t>,
                    decltype(brgemm_kernel_container_t::brgemm_kernel_cmp) *>(
                    brgemm_kernel_container_t::brgemm_kernel_cmp)};

    void lock_write() {}
    void unlock_write() {}
#endif
    std::set<std::shared_ptr<brgemm_kernel_t>,
            decltype(brgemm_kernel_container_t::brgemm_kernel_cmp) *> &
    get_set();

    std::map<const brgemm_desc_t *, const brgemm_kernel_t *> brgemm_map_;
};

struct brgemm_palette_container_t {
    typedef std::array<char, AMX_PALETTE_SIZE> S_t;

    brgemm_palette_container_t() {}
    brgemm_palette_container_t(size_t ns) { resize(ns); }
    void resize(size_t ns) { refs_.resize(ns); }

    inline const char *operator[](int idx) const { return refs_[idx]->data(); }

    bool insert(int idx, const brgemm_desc_t *brg);
    bool insert(int idx, const brgemm_desc_t &brg) { return insert(idx, &brg); }

    inline void maybe_tile_configure(bool is_amx, int &idx, int new_idx) const {
        if (idx == new_idx) return;
        if (is_amx && (idx < 0 || refs_[idx] != refs_[new_idx]))
            amx_tile_configure(refs_[new_idx]->data());
        idx = new_idx;
    }

private:
    std::vector<const S_t *> refs_;
    std::set<S_t> set_;
};

} // namespace brgemm_containers

} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif // CPU_X64_BRGEMM_BRGEMM_CONTAINERS_HPP

//vim: et ts=4 sw=4 cindent cino+=l0,\:4,N-s
