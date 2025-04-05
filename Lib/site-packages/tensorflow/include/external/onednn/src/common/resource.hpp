/*******************************************************************************
* Copyright 2022-2024 Intel Corporation
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

#ifndef COMMON_RESOURCE_HPP
#define COMMON_RESOURCE_HPP

#include <assert.h>
#include <memory>
#include <unordered_map>

#include "common/nstl.hpp"
#include "common/utils.hpp"

namespace dnnl {
namespace impl {

// The resource_t abstraction is a base class for all resource classes.
// Those are responsible for holding a part of a primitive implementation that
// cannot be stored in the primitive cache as part of the implementation.
// Currently, there are two such things:
// 1. Any memory (memory_t, memory_storage_t, etc...), because it contains
// an engine.
// 2. (for GPU only) compiled kernels, because they are context dependent.
//
// The idea is that each primitive implementation should be able to create
// a resource and put there everything it needs to run, which cannot be stored
// in the cache as part of the primitive implementation. To create the resource
// each primitive implementation can override a function `create_resource`.
//
// This abstraction takes ownership of all content it holds hence it should be
// responsible for destroying it as well.
struct resource_t : public c_compatible {
    virtual ~resource_t() = default;
};

// The resource_mapper_t is an abstraction for holding resources for
// a particular primitive implementation and providing corresponding mapping.
//
// Interacting with the mapper happens in two steps:
// 1. Initialization. Each derived from impl::primitive_t class may define
// `create_resource` member function that is responsible for creating a
// certain derived from resource_t object and filling it with some content,
// e.g. memory for scales, OpenCL kernels etc...
// 2. Passing it to the execution function which extracts needed resources and
// uses them at execution time. The mapper is passed to the execution function
// with the execution context.
//
// The resource_mapper_t takes ownership of all resources hence it should be
// responsible for destroying them as well.
struct primitive_t;
struct resource_mapper_t {
    using key_t = const primitive_t;
    using mapped_t = std::unique_ptr<resource_t>;

    resource_mapper_t() = default;

    bool has_resource(const primitive_t *p) const {
        return primitive_to_resource_.count(p);
    }

    void add(key_t *p, mapped_t &&r) {
        assert(primitive_to_resource_.count(p) == 0);
        primitive_to_resource_.emplace(p, std::move(r));
    }

    template <typename T>
    const T *get(key_t *p) const {
        assert(primitive_to_resource_.count(p));
        return utils::downcast<T *>(primitive_to_resource_.at(p).get());
    }

    DNNL_DISALLOW_COPY_AND_ASSIGN(resource_mapper_t);

private:
    std::unordered_map<key_t *, mapped_t> primitive_to_resource_;
};

} // namespace impl
} // namespace dnnl

#endif
