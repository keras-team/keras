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

#ifndef GRAPH_BACKEND_DNNL_COMMON_HPP
#define GRAPH_BACKEND_DNNL_COMMON_HPP

#include <memory>
#include <string>
#include <utility>
#include <vector>
#include <unordered_map>

#include "oneapi/dnnl/dnnl.hpp"
#include "oneapi/dnnl/dnnl_graph_types.h"

#include "graph/interface/allocator.hpp"
#include "graph/interface/constant_tensor_cache.hpp"
#include "graph/interface/logical_tensor.hpp"
#include "graph/interface/value.hpp"

#include "graph/utils/any.hpp"

#define DNNL_GRAPH_ARG_POST_SRC (-1)

namespace dnnl {
namespace impl {
namespace graph {
namespace dnnl_impl {

using memory = dnnl::memory;
using desc = memory::desc;
using format_tag = memory::format_tag;
using format_kind = memory::format_kind;
using tag = memory::format_tag;
using data_type = typename memory::data_type;
using dims = typename memory::dims;
using dim = memory::dim;
using query = dnnl::query;
using prop_kind = dnnl::prop_kind;
using algorithm = dnnl::algorithm;
using exec_args = std::unordered_map<int, memory>;

using constant_cache_t = graph::constant_tensor_cache_t;

using pd_cache_t = std::unordered_map<op_t *, graph::utils::any_t>;
struct dnnl_allocator_t {
    static void *malloc(size_t size, const dnnl::engine &p_engine,
            const allocator_t *alc, allocator_t::mem_type_t type);

    static void free(
            void *p, const dnnl::engine &p_engine, const allocator_t *alc);

#ifdef DNNL_WITH_SYCL
    static void free(void *p, const dnnl::engine &p_engine,
            const allocator_t *alc, const ::sycl::event &deps);
#endif

#if DNNL_GPU_RUNTIME == DNNL_RUNTIME_OCL
    static void free(void *p, const dnnl::engine &p_engine,
            const allocator_t *alc, const cl_event &deps);
#endif
};

format_tag get_ncx_format(size_t ndim);

format_tag get_ncx_format(const dims &adims);

dims get_compatible_dilates(const dims &dilates, size_t input_size = 4);

dims group_dims(const dims &adims, dim groups);

engine make_dnnl_engine(const engine_t &g_engine);

stream make_dnnl_stream(const engine &p_engine, const stream_t &g_stream);

memory::desc make_dnnl_memory_desc(const logical_tensor_t &lt);

memory make_dnnl_memory(const tensor_t &atensor, const engine &p_engine);

dnnl::memory make_dnnl_memory(const dnnl::memory::desc &md,
        const dnnl::engine &p_engine, void *handle);

memory::desc expand(const memory::desc &adesc, int tgt_ndims);

std::vector<int64_t> get_permutation(int ndims, const std::string &from_format,
        const std::string &to_format);

std::vector<int64_t> get_last_two_dims_permutation(int ndims);

memory::desc transpose(const memory::desc &adesc, dim dim0, dim dim1);

memory::desc to_grouped(const memory::desc &adesc, dim groups);

memory::desc from_grouped(const memory::desc &adesc);

memory::desc to_format_any(const memory::desc &adesc);

dims get_ncx_strides(const dims &shape);

dims get_nxc_strides(const dims &shape);

dims get_dense_strides(const dims &shape);

memory::desc to_nxc_format(const memory::desc &adesc);

bool is_format(const memory::desc &adesc, memory::format_tag tag);

bool is_format(const memory::desc &adesc, const std::string &tag);

bool is_4c_blocked(const memory::desc &adesc);

bool is_plain(const memory::desc &adesc);

memory::desc to_ncx_format(const memory::desc &adesc);

void set_all_layout_to_any(std::vector<std::shared_ptr<op_t>> &subgraph);

status_t fill_layout_info(logical_tensor_t *lt, const memory::desc &md);

status_t fill_layout_info(
        const std::shared_ptr<value_t> &val, const memory::desc &md);

std::shared_ptr<value_t> insert_empty_scratchpad(std::shared_ptr<op_t> &op);

std::shared_ptr<value_t> insert_empty_workspace(std::shared_ptr<op_t> &op);

std::string get_format_tag_str(const dnnl::memory::desc &md);

dnnl::memory::format_tag get_format_tag(const dnnl::memory::desc &md);

size_t generate_constant_cache_key(
        size_t part_id, const std::vector<dnnl::memory::desc> &const_mds);

#ifndef NDEBUG
#define BACKEND_DNNL_ENFORCE(condition, message) \
    do { \
        error::wrap_c_api((condition) ? dnnl_success : dnnl_invalid_arguments, \
                (message)); \
    } while (false)
#else
#define BACKEND_DNNL_ENFORCE(condition, message)
#endif

#define BACKEND_DNNL_CHECK(statement) \
    do { \
        status_t ret = (statement); \
        if (ret != status::success) return ret; \
    } while (false)

} // namespace dnnl_impl
} // namespace graph
} // namespace impl
} // namespace dnnl

#endif
