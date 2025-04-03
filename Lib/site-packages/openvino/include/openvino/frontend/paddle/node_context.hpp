// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
#include "openvino/core/any.hpp"
#include "openvino/frontend/paddle/decoder.hpp"
#include "openvino/frontend/paddle/exception.hpp"
#include "openvino/frontend/paddle/visibility.hpp"

namespace ov {
namespace frontend {
namespace paddle {
using InPortName = std::string;
using OutPortName = std::string;
using TensorName = std::string;
using NamedOutputs = std::map<OutPortName, OutputVector>;
using NamedInputs = std::map<InPortName, OutputVector>;

/// Keep necessary data for a single node in the original FW graph to facilitate
/// conversion process in the rules code.
class NodeContext : public ov::frontend::NodeContext {
public:
    using Ptr = std::shared_ptr<NodeContext>;
    NodeContext(const std::shared_ptr<DecoderBase>& _decoder, const NamedInputs& _name_map)
        : ov::frontend::NodeContext(_decoder->get_op_type()),
          decoder(_decoder),
          name_map(_name_map) {}

    /// Detects if there is at least one input attached with a given name
    bool has_input(const std::string& name) const {
        auto found = name_map.find(name);
        if (found != name_map.end())
            return !found->second.empty();
        return false;
    }

    /// Returns exactly one input with a given name; throws if there is no inputs or
    /// there are more than one input
    Output<Node> get_input(const std::string& name) const override {
        FRONT_END_GENERAL_CHECK(name_map.at(name).size() == 1);
        return name_map.at(name).at(0);
    }

    /// Returns all inputs with a given name
    OutputVector get_ng_inputs(const std::string& name) const {
        return name_map.at(name);
    }

    /// Returns all inputs in order they appear in map. This is used for FrameworkNode
    /// creation
    OutputVector get_all_ng_inputs() const {
        OutputVector res;
        for (const auto& entry : name_map) {
            res.insert(res.end(), entry.second.begin(), entry.second.end());
        }
        return res;
    }

    Output<Node> get_input(const std::string& name, int idx) const override {
        return name_map.at(name).at(idx);
    }

    size_t get_input_size(const std::string& name) const override {
        return name_map.at(name).size();
    }

    std::vector<OutPortName> get_output_names() const {
        return decoder->get_output_names();
    }

    std::vector<TensorName> get_output_var_names(const std::string& var_name) const {
        return decoder->get_output_var_names(var_name);
    }

    std::vector<TensorName> get_input_var_names(const std::string& var_name) const {
        return decoder->get_input_var_names(var_name);
    }

    ov::element::Type get_out_port_type(const std::string& port_name) const {
        return decoder->get_out_port_type(port_name);
    }

    NamedOutputs default_single_output_mapping(const std::shared_ptr<Node>& node,
                                               const std::vector<OutPortName>& required_pdpd_out_names) const;

    ov::Any get_attribute_as_any(const std::string& name) const override {
        auto res = decoder->get_attribute(name);
        return res;
    }

    size_t get_output_size(const std::string& port_name) const {
        return decoder->get_output_size(port_name);
    }

    std::vector<std::pair<ov::element::Type, ov::PartialShape>> get_output_port_infos(
        const std::string& port_name) const {
        return decoder->get_output_port_infos(port_name);
    }

    int64_t get_version() const {
        return decoder->get_version();
    }

private:
    ov::Any apply_additional_conversion_rules(const ov::Any& any, const std::type_info& type_info) const override {
        auto res = decoder->convert_attribute(any, type_info);
        return res;
    }

    const std::shared_ptr<DecoderBase> decoder;
    const NamedInputs& name_map;
};

inline NamedOutputs NodeContext::default_single_output_mapping(
    const std::shared_ptr<Node>& node,
    const std::vector<OutPortName>& required_pdpd_out_names) const {
    NamedOutputs named_outputs;
    const auto& outputs = node->outputs();
    const auto& pdpd_op_output_names = this->get_output_names();
    FRONT_END_GENERAL_CHECK(outputs.size() == 1, "OV node must have exactly one output");
    for (const auto& pdpd_name : pdpd_op_output_names) {
        if (std::find(required_pdpd_out_names.begin(), required_pdpd_out_names.end(), pdpd_name) !=
            required_pdpd_out_names.end())
            named_outputs[pdpd_name] = {outputs[0]};
    }
    return named_outputs;
}

using CreatorFunction = std::function<NamedOutputs(const NodeContext&)>;
using TranslatorDictionaryType = std::map<std::string, CreatorFunction>;
}  // namespace paddle
}  // namespace frontend
}  // namespace ov
