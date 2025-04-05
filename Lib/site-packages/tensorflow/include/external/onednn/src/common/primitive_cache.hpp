/*******************************************************************************
* Copyright 2019-2023 Intel Corporation
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

#ifndef COMMON_PRIMITIVE_CACHE_HPP
#define COMMON_PRIMITIVE_CACHE_HPP

#include "c_types_map.hpp"
#include "oneapi/dnnl/dnnl.h"
#include "primitive_hashing.hpp"
#include "type_helpers.hpp"

namespace dnnl {
namespace impl {

struct primitive_t;
struct primitive_cache_t;

struct primitive_cache_iface_t {
    using key_t = primitive_hashing::key_t;
    struct result_t {
        result_t() : status(status::success) {};
        result_t(std::shared_ptr<primitive_t> p, status_t s)
            : value(std::move(p)), status(s) {}
        bool is_empty() const { return value == nullptr; }
        primitive_t &get_value() const { return *value; }
        std::shared_ptr<primitive_t> value;
        status_t status;
    };
    using create_func_t = result_t (&)(void *);
    using create_func_ptr_t = result_t (*)(void *);

    primitive_cache_iface_t(primitive_cache_t &cache) : cache_(cache) {};

    ~primitive_cache_iface_t() = default;

    status_t set_capacity(int capacity);
    int get_capacity() const;
    int get_size() const;

    std::shared_ptr<primitive_desc_t> get_pd(const key_t &key);
    result_t get_or_create(
            const key_t &key, create_func_t create, void *create_context);

private:
    primitive_cache_t &cache_;
};

primitive_cache_iface_t primitive_cache();

// Undocumented API for testing.
status_t DNNL_API get_primitive_cache_size(int *size);
bool DNNL_API is_primitive_in_cache(const primitive_iface_t *p_iface);
bool DNNL_API is_pd_in_cache(const primitive_desc_iface_t *pd_iface);
size_t DNNL_API set_primitive_cache_capacity_without_clearing(size_t capacity);

} // namespace impl
} // namespace dnnl
#endif

// vim: et ts=4 sw=4 cindent cino^=l0,\:0,N-s
