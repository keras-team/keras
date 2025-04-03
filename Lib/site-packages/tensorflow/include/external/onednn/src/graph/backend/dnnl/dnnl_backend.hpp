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

#ifndef GRAPH_BACKEND_DNNL_DNNL_BACKEND_HPP
#define GRAPH_BACKEND_DNNL_DNNL_BACKEND_HPP

#include <algorithm>
#include <limits>
#include <memory>
#include <string>
#include <vector>
#include <unordered_map>
#include <unordered_set>

#include "graph/interface/backend.hpp"
#include "graph/interface/c_types_map.hpp"
#include "graph/interface/logical_tensor.hpp"

#include "graph/utils/any.hpp"
#include "graph/utils/pm/pass_manager.hpp"
#include "graph/utils/utils.hpp"

#include "graph/backend/dnnl/common.hpp"
#include "graph/backend/dnnl/internal_ops.hpp"
#include "graph/backend/dnnl/layout_id_mgr.hpp"
#include "graph/backend/dnnl/utils.hpp"

namespace dnnl {
namespace impl {
namespace graph {
namespace dnnl_impl {

class dnnl_partition_impl_t;

// gcc4.8.5 can 't support enum class as key
struct enum_hash_t {
    template <typename T>
    size_t operator()(const T &t) const {
        return static_cast<size_t>(t);
    }
};

struct kernel_base_t {
    virtual ~kernel_base_t() = default;

    status_t compile(const dnnl_partition_impl_t *part, const engine_t *aengine,
            const std::vector<logical_tensor_t> &inputs,
            const std::vector<logical_tensor_t> &outputs) {
        auto ret = compile_impl(part, aengine, inputs, outputs);
        if (ret != status::success) return ret;
        return prepare_inplace_pairs_impl();
    }

    status_t execute(const stream_t *astream,
            const std::vector<tensor_t> &inputs,
            const std::vector<tensor_t> &outputs) {
        return execute_impl(astream, inputs, outputs);
    }

#ifdef DNNL_WITH_SYCL
    status_t execute_sycl(const stream_t *astream,
            const std::vector<tensor_t> &inputs,
            const std::vector<tensor_t> &outputs,
            const std::vector<::sycl::event> &sycl_deps,
            ::sycl::event *sycl_event) {
        return sycl_execute_impl(
                astream, inputs, outputs, sycl_deps, sycl_event);
    }

    virtual status_t sycl_execute_impl(const stream_t *astream,
            const std::vector<tensor_t> &inputs,
            const std::vector<tensor_t> &outputs,
            const std::vector<::sycl::event> &sycl_deps,
            ::sycl::event *sycl_event)
            = 0;
#endif

#if DNNL_GPU_RUNTIME == DNNL_RUNTIME_OCL
    status_t execute_ocl(const stream_t *astream,
            const std::vector<tensor_t> &inputs,
            const std::vector<tensor_t> &outputs,
            const std::vector<cl_event> &ocl_deps, cl_event *ocl_event) {
        return ocl_execute_impl(astream, inputs, outputs, ocl_deps, ocl_event);
    }

    virtual status_t ocl_execute_impl(const stream_t *astream,
            const std::vector<tensor_t> &inputs,
            const std::vector<tensor_t> &outputs,
            const std::vector<cl_event> &ocl_deps, cl_event *ocl_event)
            = 0;
#endif

    virtual status_t compile_impl(const dnnl_partition_impl_t *part,
            const engine_t *aengine,
            const std::vector<logical_tensor_t> &inputs,
            const std::vector<logical_tensor_t> &outputs)
            = 0;

    virtual status_t execute_impl(const stream_t *astream,
            const std::vector<tensor_t> &inputs,
            const std::vector<tensor_t> &outputs)
            = 0;

    virtual status_t prepare_inplace_pairs_impl() { return status::success; };

    bool enabled_constant_cache() const;

    std::vector<inplace_pair_t> inplace_pairs_;
    dnnl::engine p_engine_;
};

using kernel_ptr = std::shared_ptr<kernel_base_t>;
using FCreateKernel = std::function<kernel_ptr(void)>;

kernel_ptr large_partition_kernel_creator();
kernel_ptr dummy_kernel_creator();

class dnnl_backend : public backend_t {
    friend class dnnl_partition_impl_t;

public:
    static dnnl_backend &get_singleton() {
        static dnnl_backend ins("dnnl_backend", /*priority*/ 1.f);
        return ins;
    }

    // Used by DNNL backend to cache memory descriptor and get layout id
    graph::utils::optional_t<size_t> set_mem_desc(const memory::desc &md);

    graph::utils::optional_t<memory::desc> get_mem_desc(
            const size_t &layout_id) const;

    graph::pass::pass_registry_t &get_pass_registry() { return pass_registry_; }

    dnnl_layout_id_manager_t &get_layout_id_manager() {
        return layout_id_manager_;
    }

    size_t get_mem_size(const logical_tensor_t &lt) const override;

    bool compare_logical_tensor(const logical_tensor_t &lhs,
            const logical_tensor_t &rhs) const override;

    bool support_engine_kind(engine_kind_t kind) const override {
        static const std::unordered_set<engine_kind_t, enum_hash_t>
                supported_kind = {
#if DNNL_CPU_RUNTIME != DNNL_RUNTIME_NONE
                    engine_kind::cpu,
#endif

#if DNNL_GPU_RUNTIME == DNNL_RUNTIME_SYCL \
        || DNNL_GPU_RUNTIME == DNNL_RUNTIME_OCL
                    engine_kind::gpu,
#endif
                };
        return supported_kind.count(kind);
    }

    status_t get_partitions(
            graph_t &agraph, partition_policy_t policy) override {
        // Note: This environment variable is internal and for test purpose. It
        // can be changed or removed without prior notice. Users should avoid
        // using it in their applications. Enabling the environment variable may
        // cause some tests and examples to fail.
        const bool disable_dnnl_bkd
                = graph::utils::getenv_int_internal("DISABLE_DNNL_BACKEND", 0)
                > 0;
        if (disable_dnnl_bkd) return status::success;

        // Note: This environment variable is internal and for test/debug
        // purpose. It can be changed or removed without prior notice. Users
        // should avoid using it in their applications. Enabled by default.
        const bool enable_large_partition
                = graph::utils::getenv_int_internal("ENABLE_LARGE_PARTITION", 1)
                > 0;

        // FIXME(xx): Here we only changes the passes in registry. If json file
        // existed, pm will run passes according to the json file, the env var
        // will not take effect.
        // - priority == 50.f: data type check pass (fixed highest priority)
        // - 50.f > priority > 20.f: large fusion pattern
        // - 20.f >= priority > 8.f: normal fusion pattern
        // - priority <= 8.f: debug fusion pattern (single op fusion)
        const float priority_ths = (policy == graph::partition_policy::fusion
                                           && enable_large_partition)
                ? std::numeric_limits<float>::max()
                : policy == graph::partition_policy::fusion ? 20.0f
                                                            : 8.0f;

        const auto &dnnl_pass_filter
                = [priority_ths](const graph::pass::pass_base_ptr &pass,
                          partition_policy_t policy) -> bool {
            UNUSED(policy);
            return pass->get_priority() <= priority_ths;
        };

        auto &pass_registry = get_pass_registry();
        graph::pass::pass_manager_t pm(pass_registry);

#ifdef DNNL_ENABLE_GRAPH_DUMP
        std::string pass_config_json = "dnnl_graph_passes.json";
        std::ifstream fs(pass_config_json.c_str());
        if (fs) {
            printf("onednn_graph_verbose,info,pattern,load,%s\n",
                    pass_config_json.c_str());
            fflush(stdout);
        } else {
            if (getenv_int_user("GRAPH_DUMP", 0) > 0
                    || graph::utils::check_verbose_string_user(
                            "GRAPH_DUMP", "pattern")) {
                printf("onednn_graph_verbose,info,pattern,dump,%s\n",
                        pass_config_json.c_str());
                fflush(stdout);
                pm.print_passes(pass_config_json);
            }
        }
        pm.run_passes(agraph, &fs, policy, dnnl_pass_filter);
#else
        pm.run_passes(agraph, "", policy, dnnl_pass_filter);
#endif
        return status::success;
    }

private:
    dnnl_backend(const std::string &name, float priority);

    static graph::pass::pass_registry_t register_passes();
    bool register_op_schemas();

    dnnl_layout_id_manager_t layout_id_manager_;
    static graph::pass::pass_registry_t pass_registry_;
};

} // namespace dnnl_impl
} // namespace graph
} // namespace impl
} // namespace dnnl

#endif
