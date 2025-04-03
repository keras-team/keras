// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/frontend/tensorflow/hash_table.hpp"
#include "openvino/frontend/tensorflow/variable.hpp"

namespace ov {
namespace frontend {
namespace tensorflow {

// a container of Variables state for each operation node in a graph
class VariableMap {
public:
    using Ptr = std::shared_ptr<VariableMap>;
    bool get_variable_state(const std::string& node_name,
                            const std::string& variable_name,
                            Variable::Ptr& found_variable) const {
        if (m_variables_state.count(node_name) > 0) {
            for (const auto& variable : m_variables_state.at(node_name)) {
                if (variable && variable->get_name() == variable_name && variable->is_initialized()) {
                    found_variable = variable;
                    return true;
                }
            }
        } else {
            return false;
        }
        return false;
    }

    void initialize_variable_state_map_for_node(const std::vector<std::string>& control_dependencies,
                                                const std::vector<std::string>& data_dependencies,
                                                const std::string& node_name) {
        m_variables_state[node_name] = {};
        for (const auto& dependency : control_dependencies) {
            for (const auto& dependency_variable : m_variables_state[dependency]) {
                update_variable_state_map_for_node(node_name, dependency_variable);
            }
        }

        for (const auto& dependency : data_dependencies) {
            for (const auto& dependency_variable : m_variables_state[dependency]) {
                update_variable_state_map_for_node(node_name, dependency_variable);
            }
        }
    }

    void update_variable_state_map_for_node(const std::string& node_name, const Variable::Ptr& update_variable) {
        if (!update_variable->is_initialized()) {
            m_uninitialized_variables.insert(update_variable);
            return;
        }
        auto variable_name = update_variable->get_name();

        // update uninitialized variables of variable_name
        // with alternative values
        for (auto& uninitialized_variable : m_uninitialized_variables) {
            auto uninitialized_variable_name = uninitialized_variable->get_name();
            if (uninitialized_variable_name != variable_name) {
                continue;
            }

            auto hash_table = ov::as_type_ptr<HashTable>(update_variable);
            auto uninitialized_hash_table = ov::as_type_ptr<HashTable>(uninitialized_variable);
            if (hash_table && uninitialized_hash_table) {
                uninitialized_hash_table->add_other_keys_values(hash_table->get_keys(), hash_table->get_values());
            }
        }

        size_t remove_ind = 0;
        bool remove_old_variable = false;
        bool found_better = false;
        // remove old variable state if exists
        for (size_t ind = 0; ind < m_variables_state[node_name].size(); ++ind) {
            auto checked_variable = m_variables_state[node_name][ind];
            if (checked_variable->get_name() == variable_name && checked_variable->is_initialized() &&
                checked_variable->get_init_counter() < update_variable->get_init_counter()) {
                remove_ind = ind;
                remove_old_variable = true;
                break;
            } else if (checked_variable->get_name() == variable_name && checked_variable->is_initialized() &&
                       checked_variable->get_init_counter() >= update_variable->get_init_counter()) {
                found_better = true;
            }
        }

        if (remove_old_variable) {
            // update the variable map with new variable
            m_variables_state[node_name].erase(m_variables_state[node_name].begin() + remove_ind);
        }

        if (!found_better) {
            m_variables_state[node_name].push_back(update_variable);
        }
    }

private:
    // stores a map of variables values at each point (node) in a graph
    // a node name maps a vector of initialized variables
    std::unordered_map<std::string, std::vector<Variable::Ptr>> m_variables_state;

    // stores a set of uninitialized variables
    std::set<Variable::Ptr> m_uninitialized_variables;
};

}  // namespace tensorflow
}  // namespace frontend
}  // namespace ov
