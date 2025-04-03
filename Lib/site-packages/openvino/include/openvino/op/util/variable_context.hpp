// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <unordered_map>

#include "openvino/core/node_vector.hpp"
#include "openvino/core/runtime_attribute.hpp"
#include "openvino/op/util/variable.hpp"
#include "openvino/op/util/variable_value.hpp"

namespace ov {
namespace op {
namespace util {
using VariableMap = std::unordered_map<Variable::Ptr, VariableValue::Ptr>;

/// VariableContext stores and manages a evaluation context for Variables.
class OPENVINO_API VariableContext {
public:
    /// \brief Constructs an uninitialized VariableContext.
    VariableContext() = default;

    /// \brief Constructor for VariableContext.
    /// \param variable_values The values associated with a particular Variables.
    explicit VariableContext(const VariableMap& variable_values) : m_variable_values(variable_values) {}

    /// \brief Sets the reset flags for all stored Variables to true.
    void reset_variable_context() const {
        for (const auto& el : m_variable_values) {
            el.second->set_reset(true);
        }
    }

    /// \brief Sets the new values for Variables.
    /// \param variable_values The new values associated with a particular Variable.
    void set_variable_values(const VariableMap& variable_values) {
        m_variable_values = variable_values;
    }

    /// \brief Changes/sets the values for Variable.
    /// \param variable New or stored Variable.
    /// \param variable_value The values associated with the variable.
    void set_variable_value(const Variable::Ptr& variable, const VariableValue::Ptr& variable_value) {
        m_variable_values[variable] = variable_value;
    }

    /// \brief Removes context for a particular Variable.
    /// \param variable The variable for which the context will be cleared.
    void remove_variable_value(const Variable::Ptr& variable) {
        m_variable_values.erase(variable);
    }

    /// \brief Returns the current values for Variables.
    const VariableMap& get_variable_values() const {
        return m_variable_values;
    }

    /// \brief Returns the value for specified Variable.
    VariableValue::Ptr get_variable_value(const Variable::Ptr& variable) const {
        auto var_value = m_variable_values.find(variable);
        if (var_value != m_variable_values.end()) {
            return (*var_value).second;
        }
        return VariableValue::Ptr();
    }

private:
    /// The values associated with a particular Variable.
    VariableMap m_variable_values;
};
}  // namespace util
}  // namespace op
}  // namespace ov
