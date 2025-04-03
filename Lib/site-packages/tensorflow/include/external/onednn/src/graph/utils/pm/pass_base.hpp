/*******************************************************************************
* Copyright 2020-2023 Intel Corporation
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

#ifndef GRAPH_UTILS_PM_PASS_BASE_HPP
#define GRAPH_UTILS_PM_PASS_BASE_HPP

#include <functional>
#include <list>
#include <memory>
#include <queue>
#include <string>
#include <utility>
#include <vector>
#include <unordered_map>
#include <unordered_set>

#include "graph/interface/graph.hpp"

#include "graph/utils/debug.hpp"
#include "graph/utils/json.hpp"
#include "graph/utils/pm/pbuilder.hpp"

namespace dnnl {
namespace impl {
namespace graph {
namespace pass {

using pb_graph_t = utils::pm::pb_graph_t;

class pass_base;
using pass_base_ptr = std::shared_ptr<pass_base>;

// FCreatePattern: a function for defining pattern.
// One pass can have several FCreatePattern functions.
using FCreatePattern
        = std::function<void(const std::shared_ptr<pb_graph_t> &pattern_graph)>;
using Pattern = std::shared_ptr<pb_graph_t>;

/*!
 * \brief pass_base provides a base class for pass creation.
 *        A pass is used to do pattern matching on a given graph,
 *        and reconstruct a new graph based on optimized patterns.
 */
class pass_base {
public:
    pass_base(std::string pbackend, std::string pname)
        : backend_(std::move(pbackend)), name_(std::move(pname)) {}

    pass_base() = default;

    // the criteria of pass execution
    virtual impl::status_t run(graph_t &agraph) {
        UNUSED(agraph);
        return impl::status::success;
    }
    // save pass basic information into json
    virtual void save(utils::json::json_writer_t *writer) {
        writer->begin_object();
        writer->write_keyvalue("pass_name", name_);
        writer->write_keyvalue("pass_backend", backend_);
        writer->write_keyvalue("priority", priority_);
        writer->write_keyvalue("enable", enable_);
        writer->write_keyvalue("kind", utils::partition_kind2str(pkind_));
        writer->end_object();
    }

    // load pass basic information from json
    virtual void load(utils::json::json_reader_t *reader) {
        utils::json::read_helper_t helper;
        std::string kind;
        helper.declare_field("pass_name", &name_);
        helper.declare_field("pass_backend", &backend_);
        helper.declare_field("priority", &priority_);
        helper.declare_field("enable", &enable_);
        helper.declare_field("kind", &kind);
        helper.read_fields(reader);

        pkind_ = utils::str2partition_kind(kind);
    }

    virtual ~pass_base() = default;

    std::string get_pass_backend() { return backend_; }

    std::string get_pass_name() { return name_; }

    // set pass priority, passes with high priority
    // will be executed before passes with low priority
    pass_base &set_priority(float priority) {
        priority_ = priority;
        return *this;
    }
    float get_priority() const { return priority_; }

    // set enable status
    // can be used for override default value of enable_
    pass_base &set_enable(bool enable) {
        enable_ = enable;
        return *this;
    }

    bool get_enable() const { return enable_; }

    pass_base &set_kind(partition_kind_t pkind) {
        pkind_ = pkind;
        return *this;
    }

    partition_kind_t get_kind() const { return pkind_; }

    pass_base &set_engine_kind(engine_kind_t kind) {
        engine_kind_ = kind;
        return *this;
    }

    engine_kind_t get_engine_kind() const { return engine_kind_; }

    /*!
    * \brief Register additional attributes.
    * \param attr_name The name of the attribute.
    * \param value The value to be set.
    * \tparam value_type The type of the value to be set.
    */
    template <typename value_type>
    pass_base &set_attr(const std::string &attr_name, // NOLINT(*)
            const value_type &value) {
        attrs_.insert(make_pair(attr_name, value));
        return *this;
    }

    /*!
    * \brief Get additional registered attribute.
    * \param attr_name The name of the attribute.
    * \return An attribute of specified attr_name.
    * \tparam value_type The type of the attribute.
    */
    template <typename value_type>
    std::vector<value_type> get_attr(const std::string &attr_name) {
        std::vector<value_type> attr_vec;
        for (auto it = attrs_.begin(); it != attrs_.end(); ++it) {
            if (it->first == attr_name) {
                attr_vec.push_back(utils::any_cast<value_type>(it->second));
            }
        }
        return attr_vec;
    }

    /*!
    * \brief check if pass has a specific attribute.
    * \param attr_name The name of the attribute.
    * \return bool: if this attribute exist in pass.
    */
    bool has_attr(const std::string &attr_name) {
        return attrs_.find(attr_name) != attrs_.end();
    }

protected:
    std::unordered_multimap<std::string, utils::any_t> attrs_;

private:
    std::string backend_ {};
    std::string name_ {};
    float priority_ {5.0f};
    bool enable_ {true};
    partition_kind_t pkind_ {partition_kind_t::undef};
    engine_kind_t engine_kind_ {engine_kind::any_engine};
};

template <>
pass_base &pass_base::set_attr<FCreatePattern>(
        const std::string &attr_name, // NOLINT(*)
        const FCreatePattern &func);

} // namespace pass
} // namespace graph
} // namespace impl
} // namespace dnnl

#endif
