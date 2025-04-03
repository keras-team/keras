/*******************************************************************************
* Copyright 2023 Intel Corporation
* Copyright 2024 FUJITSU LIMITED
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

#ifndef CPU_AARCH64_BRGEMM_BRGEMM_CONTAINERS_HPP
#define CPU_AARCH64_BRGEMM_BRGEMM_CONTAINERS_HPP

#include <set>
#include "common/rw_mutex.hpp"
#include "cpu/aarch64/brgemm/brgemm.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace aarch64 {

namespace brgemm_containers {

// These containers are intended to be used as local objects in brgemm
// primitives to ensure that references are unique and correct.

struct brgemm_desc_container_t {
public:
    brgemm_desc_container_t() {}
    brgemm_desc_container_t(size_t ns) { resize(ns); }
    void resize(size_t ns) { refs_.resize(ns); }
    inline const brgemm_t *operator[](int idx) const { return refs_[idx]; }

    bool insert(int idx, brgemm_t &brg) {
        std::vector<char> dummy_bd_mask;
        std::vector<brgemm_batch_element_t> dummy_static_offsets;
        return insert(idx, brg, dummy_bd_mask, dummy_static_offsets);
    }

    bool insert(int idx, brgemm_t &brg, const std::vector<char> &bd_mask,
            const std::vector<brgemm_batch_element_t> &static_offsets);

private:
    std::vector<const brgemm_t *> refs_;

    std::set<brgemm_t> set_;
    std::vector<std::vector<char>> bd_mask_list_;
    std::vector<std::vector<brgemm_batch_element_t>> static_offsets_list_;
};

#define BRGEMM_KERNEL_GLOBAL_STORAGE

struct brgemm_kernel_container_t {
    brgemm_kernel_container_t() {}
    brgemm_kernel_container_t(size_t ns) { resize(ns); }
    void resize(size_t ns) { refs_.resize(ns); }
    inline const brgemm_kernel_t *operator[](int idx) const {
        return refs_[idx];
    }

    status_t insert(int idx, const brgemm_t *brg);
    static bool brgemm_kernel_cmp(const std::shared_ptr<brgemm_kernel_t> &lhs,
            const std::shared_ptr<brgemm_kernel_t> &rhs);

private:
    std::vector<const brgemm_kernel_t *> refs_;
#ifdef BRGEMM_KERNEL_GLOBAL_STORAGE
    static std::set<std::shared_ptr<brgemm_kernel_t>,
            decltype(brgemm_kernel_container_t::brgemm_kernel_cmp) *>
            set_;

    static utils::rw_mutex_t &rw_mutex() {
        static utils::rw_mutex_t mutex;
        return mutex;
    }

    void lock_write() { rw_mutex().lock_write(); }
    void unlock_write() { rw_mutex().unlock_write(); }

#else
    std::set<std::shared_ptr<brgemm_kernel_t>,
            decltype(brgemm_kernel_container_t::brgemm_kernel_cmp) *>
            set_;
    void lock_write() {}
    void unlock_write() {}
#endif
    std::map<const brgemm_t *, const brgemm_kernel_t *> brgemm_map_;
};

} // namespace brgemm_containers

} // namespace aarch64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif // CPU_AARCH64_BRGEMM_BRGEMM_CONTAINERS_HPP

//vim: et ts=4 sw=4 cindent cino+=l0,\:4,N-s
