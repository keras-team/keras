/*******************************************************************************
 * Copyright 2021-2023 Intel Corporation
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

#ifndef GRAPH_BACKEND_FAKE_FAKE_BACKEND_HPP
#define GRAPH_BACKEND_FAKE_FAKE_BACKEND_HPP

#include <memory>
#include <string>
#include <vector>
#include <unordered_map>

#include "graph/interface/backend.hpp"
#include "graph/interface/c_types_map.hpp"
#include "graph/interface/logical_tensor.hpp"

#include "graph/utils/pm/pass_manager.hpp"

namespace dnnl {
namespace impl {
namespace graph {
namespace fake_impl {

class fake_backend_t : public backend_t {
    friend class fake_partition_impl_t;

public:
    static fake_backend_t &get_singleton() {
        static fake_backend_t ins("fake_backend", /*priority*/ 0.f);
        return ins;
    }

    pass::pass_registry_t &get_pass_registry() { return pass_registry_; }

    size_t get_mem_size(const logical_tensor_t &lt) const override {
        UNUSED(lt);
        return static_cast<size_t>(-1);
    }

    status_t get_partitions(
            graph_t &agraph, partition_policy_t policy) override {
        pass::pass_manager_t pm(get_pass_registry());
        pm.run_passes(agraph, "", policy);
        return status::success;
    }

    bool support_engine_kind(engine_kind_t kind) const override {
        UNUSED(kind);
        return true;
    }

private:
    fake_backend_t(const std::string &name, float priority)
        : backend_t(name, priority) {};
    static graph::pass::pass_registry_t register_passes();
    static graph::pass::pass_registry_t pass_registry_;
};

} // namespace fake_impl
} // namespace graph
} // namespace impl
} // namespace dnnl

#endif
