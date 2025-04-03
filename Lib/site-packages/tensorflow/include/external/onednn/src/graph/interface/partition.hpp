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

#ifndef GRAPH_INTERFACE_PARTITION_HPP
#define GRAPH_INTERFACE_PARTITION_HPP

#include <cstring>
#include <future>
#include <memory>
#include <sstream>
#include <string>
#include <utility>
#include <vector>
#include <unordered_map>
#include <unordered_set>

#include "graph/interface/c_types_map.hpp"
#include "graph/interface/logical_tensor.hpp"
#include "graph/interface/op.hpp"
#include "graph/interface/partition_impl.hpp"

#include "graph/utils/id.hpp"
#include "graph/utils/utils.hpp"
#include "graph/utils/verbose.hpp"

#if DNNL_GPU_RUNTIME == DNNL_RUNTIME_OCL
#include <CL/cl.h>
#endif

namespace dnnl {
namespace impl {
namespace graph {
class backend_t;
} // namespace graph
} // namespace impl
} // namespace dnnl

namespace graph = dnnl::impl::graph;

struct dnnl_graph_partition : public dnnl::impl::graph::utils::id_t {
public:
    friend struct dnnl_graph_compiled_partition;
    friend struct graph::utils::partition_info_t;

    dnnl_graph_partition() = default;

    // deep copy
    dnnl_graph_partition(const dnnl_graph_partition &other) = default;

    // disable assign
    dnnl_graph_partition &operator=(const dnnl_graph_partition &other) = delete;

    ~dnnl_graph_partition() = default;

    void init(const std::shared_ptr<graph::partition_impl_t> &pimpl) {
        pimpl_ = pimpl;
        const_cast<graph::partition_impl_t *>(pimpl_.get())->set_id(id());
    }

    bool is_initialized() const {
        return (pimpl_ != nullptr) && pimpl_->is_initialized();
    }

    bool is_supported() const;

    const graph::partition_impl_t *get_pimpl() const { return pimpl_.get(); }

    const graph::backend_t *get_assigned_backend() const {
        return pimpl_->get_assigned_backend();
    }

    graph::engine_kind_t get_engine_kind() const {
        return pimpl_->get_engine_kind();
    }

    graph::fpmath_mode_t get_fpmath_mode() const {
        return pimpl_->get_fpmath_mode();
    }

    graph::partition_kind_t get_kind() const { return pimpl_->get_kind(); }

    const std::vector<std::shared_ptr<graph::op_t>> &get_ops() const {
        return pimpl_->get_ops();
    }

    size_t num_ops() const { return pimpl_->get_ops().size(); }

    std::vector<size_t> get_op_ids() const {
        std::vector<size_t> ids;
        auto ops = pimpl_->get_ops();
        ids.reserve(ops.size());
        for (auto &op : ops) {
            ids.emplace_back(op->get_id());
        }
        return ids;
    }

    const std::vector<graph::logical_tensor_t> &get_inputs() const {
        return pimpl_->get_inputs();
    }

    const std::vector<graph::logical_tensor_t> &get_outputs() const {
        return pimpl_->get_outputs();
    }

    size_t get_inputs_num() const { return pimpl_->get_inputs().size(); }

    size_t get_outputs_num() const { return pimpl_->get_outputs().size(); }

    graph::status_t compile(graph::compiled_partition_t *compiled_partition,
            std::vector<const graph::logical_tensor_t *> &inputs,
            std::vector<const graph::logical_tensor_t *> &outputs,
            const graph::engine_t *e = nullptr) const;

    graph::status_t compile(
            std::pair<graph::compiled_partition_t *, bool> &compiled_partition,
            std::vector<const graph::logical_tensor_t *> &inputs,
            std::vector<const graph::logical_tensor_t *> &outputs,
            const graph::engine_t *aengine) const;

    graph::status_t infer_shape(
            std::vector<const graph::logical_tensor_t *> &inputs,
            std::vector<graph::logical_tensor_t *> &outputs);

private:
    std::shared_ptr<const graph::partition_impl_t> pimpl_;
};

///
/// \brief dnnl_graph_compiled_partition_t
///
struct dnnl_graph_compiled_partition : public dnnl::impl::graph::utils::id_t {
public:
    friend struct dnnl_graph_partition;
    friend struct graph::utils::partition_info_t;

    dnnl_graph_compiled_partition(const graph::partition_t &src_partition)
        : src_partition_ {src_partition} {}

    ~dnnl_graph_compiled_partition() = default;

    const graph::partition_t &src_partition() const { return src_partition_; }

    void init(const std::shared_ptr<graph::compiled_partition_impl_t> &pimpl) {
        pimpl_ = pimpl;
    }

    bool is_initialized() const { return pimpl_ != nullptr; }

    const graph::compiled_partition_impl_t *get_pimpl() const {
        return pimpl_.get();
    }

    const std::vector<graph::inplace_pair_t> &get_inplace_pairs() const {
        static std::vector<graph::inplace_pair_t> empty = {};
        if (!pimpl_) {
            assertm(false, "pimpl_ is nullptr");
            return empty;
        }

        return pimpl_->get_inplace_pairs();
    }

    graph::status_t execute(const graph::stream_t *astream,
            const std::vector<graph::tensor_t> &inputs,
            const std::vector<graph::tensor_t> &outputs) const;

#ifdef DNNL_WITH_SYCL
    graph::status_t execute_sycl(const graph::stream_t *astream,
            const std::vector<graph::tensor_t> &inputs,
            const std::vector<graph::tensor_t> &outputs,
            const std::vector<::sycl::event> &sycl_deps,
            ::sycl::event *sycl_event) const;
#endif

#if DNNL_GPU_RUNTIME == DNNL_RUNTIME_OCL
    graph::status_t execute_ocl(const graph::stream_t *astream,
            const std::vector<graph::tensor_t> &inputs,
            const std::vector<graph::tensor_t> &outputs,
            const std::vector<cl_event> &sycl_deps, cl_event *sycl_event) const;
#endif

    graph::status_t query_logical_tensor(
            size_t tid, graph::logical_tensor_t *lt) const {
        if (!pimpl_) {
            *lt = graph::empty_logical_tensor_with_default_id();
            return graph::status::success;
        }
        return pimpl_->query_logical_tensor(tid, lt);
    }

    const graph::engine_t *get_engine() const { return pimpl_->get_engine(); }

    std::vector<graph::logical_tensor_t> &get_mutable_inputs() {
        return pimpl_->get_mutable_inputs();
    }

    std::vector<graph::logical_tensor_t> &get_mutable_outputs() {
        return pimpl_->get_mutable_outputs();
    }

    const std::vector<graph::logical_tensor_t> &get_inputs() const {
        return pimpl_->get_inputs();
    }

    const std::vector<graph::logical_tensor_t> &get_outputs() const {
        return pimpl_->get_outputs();
    }

    const char *info() const {
        auto eng = pimpl_->get_engine();
        if (!info_.is_initialized()) info_.init(eng, this);
        return info_.c_str();
    }

private:
    std::shared_ptr<graph::compiled_partition_impl_t> pimpl_;

    const graph::partition_t src_partition_;

    // Partition information
    mutable graph::utils::partition_info_t info_;
};

#endif
