// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/op/util/unary_elementwise_arithmetic.hpp"

namespace ov {
namespace op {
namespace v9 {
class OPENVINO_API SoftSign : public util::UnaryElementwiseArithmetic {
public:
    OPENVINO_OP("SoftSign", "opset9", util::UnaryElementwiseArithmetic);

    SoftSign() = default;
    /// \brief Constructs a SoftSign operation.
    ///
    /// \param arg Node that produces the input tensor.<br>
    /// `[d0, ...]`
    /// Output `[d0, ...]`
    ///
    SoftSign(const Output<Node>& arg);

    void validate_and_infer_types() override;

    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& new_args) const override;
    bool evaluate(TensorVector& outputs,
                  const TensorVector& inputs,
                  const EvaluationContext& evaluation_context) const override;
    bool has_evaluate() const override;
};
}  // namespace v9
}  // namespace op
}  // namespace ov
