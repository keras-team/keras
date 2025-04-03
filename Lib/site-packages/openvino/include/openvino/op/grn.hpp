// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>

#include "openvino/op/util/unary_elementwise_arithmetic.hpp"

namespace ov {
namespace op {
namespace v0 {
/// \brief  Global Response Normalization with L2 norm (across channels only).
///
class OPENVINO_API GRN : public util::UnaryElementwiseArithmetic {
public:
    OPENVINO_OP("GRN", "opset1", util::UnaryElementwiseArithmetic);

    GRN() = default;
    /// \brief      Constructs a GRN operation.
    ///
    /// \param      data  - Node producing the input tensor
    /// \param      bias  - The bias added to the variance.
    ///
    GRN(const Output<Node>& data, float bias);

    void validate_and_infer_types() override;
    bool visit_attributes(AttributeVisitor& visitor) override;
    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& new_args) const override;

    float get_bias() const {
        return m_bias;
    }
    void set_bias(const float& bias) {
        m_bias = bias;
    }

protected:
    float m_bias = 1.0f;
};
}  // namespace v0
}  // namespace op
}  // namespace ov
