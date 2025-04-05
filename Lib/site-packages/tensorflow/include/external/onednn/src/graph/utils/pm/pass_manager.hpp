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

#ifndef GRAPH_UTILS_PM_PASS_MANAGER_HPP
#define GRAPH_UTILS_PM_PASS_MANAGER_HPP

#include <list>
#include <string>
#include <unordered_map>

#include "graph/interface/c_types_map.hpp"

#include "graph/utils/json.hpp"
#include "graph/utils/pm/pass_base.hpp"

namespace dnnl {
namespace impl {
namespace graph {
namespace pass {

// The pass filter function returns true if the pass is desired under the
// given policy. Each backend should define its own filter function.
using pass_filter_fn
        = std::function<bool(const pass_base_ptr &, partition_policy_t)>;

bool default_pass_filter(const pass_base_ptr &pass, partition_policy_t policy);

/*!
 * \brief pass_registry is a registry class that
 *        is responsible for registering pass
 */
class pass_registry_t {
    using pass_create_fn = pass_base_ptr (*)(std::string, std::string);

public:
    // register a pass
    pass_base &register_pass(const std::string &backend_name,
            const std::string &pass_name, pass_create_fn fn);
    pass_base &register_pass(const pass_base_ptr &pass);
    // get registered passes
    const std::list<pass_base_ptr> &get_passes() const { return passes_; }

    // sort passes based on their priorities, passes with high priority
    // will be executed before passes with low priority
    void sort_passes();

    pass_base_ptr &get_pass_ptr(const std::string &pass_name) {
        auto it = passes_map_.find(pass_name);
        assert(it != passes_map_.end() && "pass not exists!");
        return it->second;
    }

    pass_registry_t() = default;
    pass_registry_t(const pass_registry_t &) = delete;
    pass_registry_t(pass_registry_t &&) = default;
    pass_registry_t &operator=(const pass_registry_t &) = delete;
    pass_registry_t &operator=(pass_registry_t &&) = delete;

private:
    std::list<pass_base_ptr> passes_;
    std::unordered_map<std::string, pass_base_ptr> passes_map_;
};

/*!
 * \brief pass_manager_t manages the registered passes
 *        as well as backends. It supports pass registration,
 *        pass execution, partition compilation, etc.
 */
class pass_manager_t {
private:
    pass_registry_t &pass_registry_;

public:
    pass_manager_t(pass_registry_t &registry) : pass_registry_(registry) {}

    // get all registered passes
    const std::list<pass_base_ptr> &get_passes() {
        return pass_registry_.get_passes();
    }

    // get pass by pass_name
    pass_base_ptr &get_pass_ptr(const std::string &pass_name) {
        return pass_registry_.get_pass_ptr(pass_name);
    }
    // print all passes info to json,  e.g. pass name, pass enabled,
    // pass type, pass backend, pass execution priority, etc.
    void print_passes(const std::string &pass_config_json);
    void print_passes(std::ostream *os);

    // run all passes enabled according to passConfig
    impl::status_t run_passes(graph_t &agraph,
            const std::string &pass_config_json,
            partition_policy_t policy = partition_policy::fusion,
            pass_filter_fn filter_fn = default_pass_filter);

    impl::status_t run_passes(graph_t &agraph, std::istream *fs,
            partition_policy_t policy = partition_policy::fusion,
            pass_filter_fn filter_fn = default_pass_filter);
};

} // namespace pass
} // namespace graph
} // namespace impl
} // namespace dnnl

#endif
