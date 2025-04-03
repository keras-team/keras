// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/op/op.hpp"

namespace ov {
namespace op {
namespace v0 {
/// \brief Parametrized Relu
/// x <  0 => f(x) = x * slope
/// x >= 0 => f(x) = x
///
/// \ingroup ov_ops_cpp_api
class OPENVINO_API PRelu : public Op {
public:
    OPENVINO_OP("PRelu", "opset1");
    PRelu();
    /// \brief Constructs a PRelu operation.
    ///
    /// \param data Input tensor
    /// \param slope Multipliers for negative values
    PRelu(const Output<Node>& data, const Output<Node>& slope);

    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& new_args) const override;

    void validate_and_infer_types() override;
    bool evaluate(TensorVector& outputs, const TensorVector& inputs) const override;
    bool has_evaluate() const override;
};
}  // namespace v0
}  // namespace op
}  // namespace ov
