// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <utility>

#include "openvino/core/core_visibility.hpp"
#include "openvino/op/util/variable.hpp"

namespace ov {
namespace op {
namespace util {
class OPENVINO_API VariableExtension {
public:
    VariableExtension() = default;

    /// \brief Returns variable connected to this node.
    virtual std::shared_ptr<Variable> get_variable() const {
        return m_variable;
    }

    /// \brief Sets a new variable to be connected to this node.
    ///
    /// \param variable New variable to be connected to this node.
    virtual void set_variable(const std::shared_ptr<Variable>& variable) {
        m_variable = variable;
    }

    /// \brief Sets the identifier to a variable
    ///
    /// \param variable_id New identifier of the variable.
    virtual void set_variable_id(const std::string& variable_id) {
        m_variable->get_info().variable_id = variable_id;
    };

    /// \brief Returns the identifier of corresponding variable.
    virtual std::string get_variable_id() const = 0;

protected:
    virtual ~VariableExtension();

protected:
    std::shared_ptr<Variable> m_variable;
};
}  // namespace util
}  // namespace op
}  // namespace ov
