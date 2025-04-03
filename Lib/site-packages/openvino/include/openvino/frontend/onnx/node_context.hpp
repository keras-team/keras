// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/frontend/extension/conversion.hpp"
#include "openvino/frontend/node_context.hpp"
#include "openvino/frontend/onnx/visibility.hpp"

namespace ov {
namespace frontend {
namespace onnx {
class Node;

class ONNX_FRONTEND_API NodeContext : public ov::frontend::NodeContext {
public:
    using Ptr = std::shared_ptr<NodeContext>;
    explicit NodeContext(const ov::frontend::onnx::Node& context);
    size_t get_input_size() const override;

    Output<ov::Node> get_input(int port_idx) const override;

    ov::Any get_attribute_as_any(const std::string& name) const override;

protected:
    const ov::frontend::onnx::Node& m_context;
    ov::OutputVector m_inputs;

private:
    ov::Any apply_additional_conversion_rules(const ov::Any& data, const std::type_info& type_info) const override;
};
using CreatorFunction = std::function<ov::OutputVector(const ov::frontend::onnx::Node&)>;
}  // namespace onnx
}  // namespace frontend
}  // namespace ov
