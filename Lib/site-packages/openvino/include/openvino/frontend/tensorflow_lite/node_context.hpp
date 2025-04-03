// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/core/any.hpp"
#include "openvino/frontend/node_context.hpp"
#include "openvino/frontend/tensorflow_lite/decoder.hpp"

namespace ov {
namespace frontend {
namespace tensorflow_lite {

using SubGraphFuncs = std::vector<std::function<std::shared_ptr<ov::Model>()>>;

/// Keep necessary data for a single node in the original FW graph to facilitate
/// conversion process in the rules code.
class TENSORFLOW_LITE_FRONTEND_API NodeContext : public ov::frontend::NodeContext {
public:
    using Ptr = std::shared_ptr<NodeContext>;
    NodeContext(const std::shared_ptr<DecoderBase>& decoder, const OutputVector& inputs)
        : ov::frontend::NodeContext(decoder->get_op_type()),
          m_decoder(decoder),
          m_inputs(inputs),
          m_subgraph_functions(m_empty_vector) {}

    NodeContext(const std::shared_ptr<DecoderBase>& decoder,
                const OutputVector& inputs,
                const SubGraphFuncs& subgraph_functions)
        : ov::frontend::NodeContext(decoder->get_op_type()),
          m_decoder(decoder),
          m_inputs(inputs),
          m_subgraph_functions(subgraph_functions) {}

    /// \brief  Returns a number of inputs
    size_t get_input_size() const override {
        return m_inputs.size();
    }

    /// \brief Returns exactly one input with a given idx; throws if there is no inputs or
    /// there are more than one input
    Output<Node> get_input(int port_index) const override {
        return m_inputs.at(port_index);
    }

    /// Detects if there is at least one input attached with a given name
    bool has_input(const size_t& port_index) const {
        return port_index < m_inputs.size();
    }

    /// \brief Get a node name
    const std::string& get_name() const override {
        return m_decoder->get_op_name();
    }

    OutputVector get_inputs() const {
        return m_inputs;
    }

    /// \brief Returns node attribute by name as ov::Any.
    ov::Any get_attribute_as_any(const std::string& name) const override {
        return m_decoder->get_attribute(name);
    }

    /// \brief Returns the number of sub-graphs that can be enumerated with get_subgraph
    size_t get_subgraph_size() const override {
        return m_subgraph_functions.size();
    }

    /// \brief Returns subgraph converted on demand by the first access
    /// If there is no query for specific sub-graph it shouldn't be converted
    /// idx should be in range 0..get_subgraph_size()-1
    std::shared_ptr<Model> get_subgraph(int idx) const override {
        int size = static_cast<int>(get_subgraph_size());
        FRONT_END_GENERAL_CHECK(idx >= 0 && idx < size,
                                "Incorrect subgraph idx ",
                                idx,
                                ". There are only ",
                                get_subgraph_size(),
                                "subgraphs currently");
        return m_subgraph_functions[idx]();
    }

    /// \brief Get a decoder
    std::shared_ptr<DecoderBase> get_decoder() const {
        return m_decoder;
    }

private:
    std::shared_ptr<DecoderBase> m_decoder;
    const OutputVector& m_inputs;
    const SubGraphFuncs& m_subgraph_functions;
    const SubGraphFuncs m_empty_vector = {};
};

using CreatorFunction = std::function<ov::OutputVector(const ov::frontend::tensorflow_lite::NodeContext&)>;
using TranslatorDictionaryType = std::map<std::string, CreatorFunction>;

}  // namespace tensorflow_lite
}  // namespace frontend
}  // namespace ov
