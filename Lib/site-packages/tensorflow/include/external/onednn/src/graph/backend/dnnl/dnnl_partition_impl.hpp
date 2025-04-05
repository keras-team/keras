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

#ifndef GRAPH_BACKEND_DNNL_DNNL_PARTITION_IMPL_HPP
#define GRAPH_BACKEND_DNNL_DNNL_PARTITION_IMPL_HPP

#include <memory>
#include <string>
#include <utility>
#include <vector>
#include <unordered_set>

#include "graph/interface/backend.hpp"
#include "graph/interface/partition.hpp"

#include "graph/backend/dnnl/dnnl_backend.hpp"
#include "graph/backend/dnnl/internal_ops.hpp"

namespace dnnl {
namespace impl {
namespace graph {
namespace dnnl_impl {

namespace {

inline status_t get_ordered_inputs_outputs(
        const std::vector<logical_tensor_t> &expected,
        const std::vector<logical_tensor_t> &given,
        std::vector<logical_tensor_t> &ordered) {
    ordered.reserve(expected.size());
    for (size_t i = 0; i < expected.size(); i++) {
        for (size_t j = 0; j < given.size(); j++) {
            if (expected[i].id == given[j].id) {
                ordered.emplace_back(given[j]);
                break;
            }
        }
    }

    if (ordered.size() != expected.size()) return status::invalid_arguments;
    return status::success;
}

} // namespace

class dnnl_compiled_partition_impl_t : public compiled_partition_impl_t {
    friend class dnnl_backend;
    friend class dnnl_partition_impl_t;

public:
    dnnl_compiled_partition_impl_t(const engine_t &engine,
            const std::vector<logical_tensor_t> &inputs,
            const std::vector<logical_tensor_t> &outputs, kernel_ptr &kernel)
        : compiled_partition_impl_t(
                engine, inputs, outputs, kernel->inplace_pairs_)
        , kernel_(kernel) {}

    status_t execute(const stream_t *g_stream,
            const std::vector<tensor_t> &inputs,
            const std::vector<tensor_t> &outputs) override {
        // We don't need to resort the inputs and outputs
        return kernel_->execute(g_stream, inputs, outputs);
    }

#ifdef DNNL_WITH_SYCL
    status_t execute_sycl(const stream_t *g_stream,
            const std::vector<tensor_t> &inputs,
            const std::vector<tensor_t> &outputs,
            const std::vector<::sycl::event> &sycl_deps,
            ::sycl::event *sycl_event) override {
        // We don't need to resort the inputs and outputs
        return kernel_->execute_sycl(
                g_stream, inputs, outputs, sycl_deps, sycl_event);
    }
#endif

#if DNNL_GPU_RUNTIME == DNNL_RUNTIME_OCL
    // It looks very similar to execute_sycl. Consider to merge them in the
    // future.
    status_t execute_ocl(const stream_t *g_stream,
            const std::vector<tensor_t> &inputs,
            const std::vector<tensor_t> &outputs,
            const std::vector<cl_event> &ocl_deps,
            cl_event *ocl_event) override {
        return kernel_->execute_ocl(
                g_stream, inputs, outputs, ocl_deps, ocl_event);
    }
#endif

private:
    kernel_ptr kernel_;
};

class dnnl_partition_impl_t : public partition_impl_t {
    friend class dnnl_backend;

public:
    dnnl_partition_impl_t(engine_kind_t engine_kind, fpmath_mode_t fpmath_mode,
            partition_kind_t pkind)
        : partition_impl_t(engine_kind, fpmath_mode, pkind) {}

    ~dnnl_partition_impl_t() override = default;

    ///// The following are used only in backend for constructing object

    void init(FCreateKernel kernel_creator) {
        init_inputs_outputs();

        // init kernel
        kernel_creator_ = std::move(kernel_creator);
    }

    void add_op(const std::shared_ptr<op_t> &op) { ops_.emplace_back(op); }

    // init backend partition's input/output logical tensors
    // based on ops in the partition
    void init_inputs_outputs() {
        inputs_.clear();
        outputs_.clear();
        std::unordered_set<op_t *> visit;
        for (auto &cur_op : ops_) {
            visit.insert(cur_op.get());
        }

        for (auto &cur_op : ops_) {
            for (size_t j = 0; j < cur_op->num_inputs(); ++j) {
                auto in_value = cur_op->get_input_value(j);
                if (!in_value->has_producer()
                        || !visit.count(&in_value->get_producer())) {
                    inputs_.push_back(in_value->get_logical_tensor());
                }
            }
            for (size_t j = 0; j < cur_op->num_outputs(); ++j) {
                auto out_value = cur_op->get_output_value(j);
                // if out_value has no consumer
                // OR any of its consumers are not inside the pattern
                // it is output tensor
                bool is_output = out_value->get_consumers().empty();
                for (auto &consumer : out_value->get_consumers()) {
                    if (!visit.count(&consumer.get_op())) {
                        is_output = true;
                        break;
                    }
                }
                if (is_output) {
                    outputs_.push_back(out_value->get_logical_tensor());
                }
            }
        }
    }

    FCreateKernel get_kernel_creator() const { return kernel_creator_; }

    /////////////// the followings are the implementation of interface

    bool is_initialized() const override { return kernel_creator_ != nullptr; }

    std::shared_ptr<partition_impl_t> clone() const override {
        auto ret = std::make_shared<dnnl_partition_impl_t>(
                get_engine_kind(), get_fpmath_mode(), get_kind());
        ret->ops_ = graph_t::deep_copy(ops_);
        ret->inputs_ = inputs_;
        ret->outputs_ = outputs_;
        ret->kernel_creator_ = kernel_creator_;
        ret->id_ = id_;
        ret->can_use_blocked_layout_ = can_use_blocked_layout_;
        return ret;
    }

    const backend_t *get_assigned_backend() const override {
        return &dnnl_backend::get_singleton();
    }

    status_t compile(compiled_partition_t *compiled_partition,
            const std::vector<logical_tensor_t> &inputs,
            const std::vector<logical_tensor_t> &outputs,
            const engine_t *g_engine) const override {
        // compile will transform the subgraph in partition, so we make
        // a copy
        auto part = std::dynamic_pointer_cast<dnnl_partition_impl_t>(
                this->clone());

        // get kernel creator
        auto kernel_creator = part->get_kernel_creator();

        // This internal env var is used for test purpose. When setting
        // _DNNL_USE_LARGE_PARTITION_KERNEL to 1, all partitions will be
        // dispatched to the large partition kernel.
        if (graph::utils::getenv_int_internal(
                    "USE_LARGE_PARTITION_KERNEL", 0)) {
            kernel_creator = large_partition_kernel_creator;
        }

        // Dispatch to fake kernel if one of the output dimensions is zero.
        const std::vector<std::shared_ptr<op_t>> &fused_op = part->get_ops();
        auto agraph = graph_t(fused_op, get_engine_kind(), get_fpmath_mode());
        agraph.set_user_inputs_outputs(inputs, outputs);
        agraph.infer_shape();
        for (const auto &val : agraph.get_output_values()) {
            if (logical_tensor_wrapper_t(val->get_logical_tensor())
                            .has_zero_dim()) {
                kernel_creator = dummy_kernel_creator;
                break;
            }
        }

        kernel_ptr kernel = kernel_creator();
        if (!kernel) return status::unimplemented;

        status_t ret;

        // compile kernel.
        // FIXME(qun) will modify the outputs inside the compile, which
        // break the constant semantics
        ret = kernel->compile(part.get(), g_engine, inputs, outputs);
        if (ret != status::success) return ret;

        std::vector<logical_tensor_t> ordered_inputs;
        std::vector<logical_tensor_t> ordered_outputs;
        ret = get_ordered_inputs_outputs(inputs_, inputs, ordered_inputs);
        if (status::success != ret) return ret;

        ret = get_ordered_inputs_outputs(outputs_, outputs, ordered_outputs);
        if (status::success != ret) return ret;

        // wrapper kernel to dnnl_compiled_partition_impl_t
        auto pimpl = std::make_shared<dnnl_compiled_partition_impl_t>(
                *g_engine, ordered_inputs, ordered_outputs, kernel);
        compiled_partition->init(pimpl);

        return status::success;
    }

    status_t infer_shape(std::vector<const logical_tensor_t *> &inputs,
            std::vector<logical_tensor_t *> &outputs) const override {
        UNUSED(inputs);
        UNUSED(outputs);
        return status::success;
    }

private:
    FCreateKernel kernel_creator_;
};

} // namespace dnnl_impl
} // namespace graph
} // namespace impl
} // namespace dnnl

#endif
