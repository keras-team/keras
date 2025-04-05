// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/core/extension.hpp"
#include "openvino/frontend/exception.hpp"
#include "openvino/frontend/visibility.hpp"
#include "openvino/pass/graph_rewrite.hpp"
#include "openvino/pass/manager.hpp"
#include "openvino/pass/pass.hpp"

namespace ov {
namespace frontend {

class FRONTEND_API NodeContext {
public:
    // TODO: Why this ctor is explicit when get_op_type is virtual so m_op_type looks to be a custom implementation
    explicit NodeContext(const std::string& op_type) : m_op_type(op_type) {}
    virtual ~NodeContext();

    /// \brief  Returns a number of inputs
    virtual size_t get_input_size() const {
        FRONT_END_NOT_IMPLEMENTED(get_input_size);
    };

    /// \brief Returns a number of inputs
    virtual size_t get_input_size(const std::string& port_name) const {
        FRONT_END_NOT_IMPLEMENTED(get_input_size);
    }

    /// \brief Returns exactly one input with a given idx; throws if there is no inputs or
    /// there are more than one input
    virtual Output<Node> get_input(int idx) const {
        FRONT_END_NOT_IMPLEMENTED(get_input);
    }

    /// \brief Returns exactly one input with a given name and idx; throws if there is no inputs or
    /// there are more than one input
    virtual Output<Node> get_input(const std::string& name, int idx) const {
        FRONT_END_NOT_IMPLEMENTED(get_input);
    }

    /// \brief Returns exactly one input with a given name; throws if there is no inputs or
    /// there are more than one input
    virtual Output<Node> get_input(const std::string& name) const {
        FRONT_END_NOT_IMPLEMENTED(get_input);
    }

    /// \brief Returns output of Variable node (or Variable value).
    /// Variable is a special node that stores a value represented with a sub-graph.
    /// Variable has a concrete value at each conversion step.
    /// The current (consuming) operation node can change its value
    /// so consumers of this Variable will have a new value at next conversion steps.
    /// See ov::frontend::Variable class for more details.
    virtual Output<Node> get_input_by_reference(int idx) const {
        FRONT_END_NOT_IMPLEMENTED(get_input_by_reference);
    }

    /// \brief Returns values from Constant input with the given index as ov::Any.
    ///        Throws an exception if the input cannot be represented as Constant.
    virtual Any get_values_from_const_input(int idx) const {
        FRONT_END_NOT_IMPLEMENTED(get_values_from_const_input);
    }

    virtual const std::string& get_op_type() const {
        return m_op_type;
    }

    virtual const std::string& get_name() const {
        FRONT_END_NOT_IMPLEMENTED(get_name);
    }

    /// \brief Returns node attribute by name.
    template <class T>
    T get_attribute(const std::string& name) const {
        auto any = get_attribute_as_any(name);
        FRONT_END_GENERAL_CHECK(!any.empty(), "Attribute with name '", name, "' does not exist");

        // sometimes we can't unambiguously recognize types in protobuf, e.g.
        // int we can interpret as int or as enum inherited from int, so
        // we have to apply additional rules based on the type (T) passed from the user.
        auto res = apply_additional_conversion_rules(any, typeid(T));
        return res.as<T>();
    }

    /// \brief Returns node attribute by name. Returns 'def' value if attribute does not exist
    template <class T>
    T get_attribute(const std::string& name, const T& def) const {
        auto any = get_attribute_as_any(name);

        // sometimes we can't unambiguously recognize types in protobuf, e.g.
        // int we can interpret as int or as enum inherited from int, so
        // we have to apply additional rules based on the type (T) passed from the user.
        auto res = apply_additional_conversion_rules(any, typeid(T));
        if (!res.empty()) {
            return res.as<T>();
        }
        return def;
    }

    /// \brief Check if an attribute of a given name exist
    bool has_attribute(const std::string& name) const {
        return !get_attribute_as_any(name).empty();
    }

    /// \brief Returns node attribute by name as ov::Any.
    virtual ov::Any get_attribute_as_any(const std::string& name) const = 0;

    /// \brief Returns the number of sub-graphs that can be enumerated with get_subgraph
    virtual size_t get_subgraph_size() const {
        FRONT_END_NOT_IMPLEMENTED(get_subgraph_size);
    }

    /// \brief Returns subgraph converted on demand by the first access
    /// If there is no query for specific sub-graph it shouldn't be converted
    /// idx should be in range 0..get_subgraph_size()-1
    virtual std::shared_ptr<Model> get_subgraph(int idx) const {
        FRONT_END_NOT_IMPLEMENTED(get_subgraph);
    }

private:
    virtual ov::Any apply_additional_conversion_rules(const ov::Any& data, const std::type_info& type_info) const {
        return data;
    }
    std::string m_op_type;
};

struct NamedOutput {
    NamedOutput(const Output<Node>& _port) : port(_port) {}
    NamedOutput(const std::string& _name, const Output<Node>& _port) : name(_name), port(_port) {}

    std::string name;
    Output<Node> port;
};

using NamedOutputVector = std::vector<NamedOutput>;

inline OutputVector indexed_from_named(const NamedOutputVector& outputs) {
    OutputVector result;
    result.reserve(outputs.size());
    std::transform(outputs.begin(), outputs.end(), std::back_inserter(result), [](const NamedOutput& x) {
        return x.port;
    });
    return result;
}

inline NamedOutputVector named_from_indexed(const OutputVector& outputs) {
    return NamedOutputVector(outputs.begin(), outputs.end());
}

using CreatorFunction = std::function<OutputVector(const NodeContext&)>;
using CreatorFunctionNamed = std::function<std::map<std::string, OutputVector>(const NodeContext&)>;
using CreatorFunctionNamedAndIndexed = std::function<NamedOutputVector(const NodeContext&)>;

}  // namespace frontend
}  // namespace ov
