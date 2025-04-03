// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/op/op.hpp"

namespace ov {
namespace op {
namespace v0 {
/// \brief Performs a SELU activation function on all elements of the input node
/// \ingroup ov_ops_cpp_api
class OPENVINO_API Selu : public Op {
public:
    OPENVINO_OP("Selu", "opset1");

    Selu() = default;
    /// \brief Constructs a Selu node.
    ///
    /// \param data - Node producing the input tensor
    /// \param alpha - Alpha coefficient of SELU operation
    /// \param lambda - Lambda coefficient of SELU operation
    Selu(const Output<Node>& data, const Output<Node>& alpha, const Output<Node>& lambda);

    void validate_and_infer_types() override;

    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& new_args) const override;
};
}  // namespace v0
}  // namespace op
}  // namespace ov
