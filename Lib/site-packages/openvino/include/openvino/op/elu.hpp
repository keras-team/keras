// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/op/util/unary_elementwise_arithmetic.hpp"

namespace ov {
namespace op {
namespace v0 {
/// \brief Exponential Linear Unit
/// x <  0 => f(x) = alpha * (exp(x) - 1.)
/// x >= 0 => f(x) = x
///
/// \ingroup ov_ops_cpp_api
class OPENVINO_API Elu : public util::UnaryElementwiseArithmetic {
public:
    OPENVINO_OP("Elu", "opset1", util::UnaryElementwiseArithmetic);

    Elu() = default;
    /// \brief Constructs an Elu operation.
    ///
    /// \param data Input tensor
    /// \param alpha Multiplier for negative values
    Elu(const Output<Node>& data, const double alpha);

    bool visit_attributes(AttributeVisitor& visitor) override;
    void validate_and_infer_types() override;

    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& new_args) const override;

    double get_alpha() const {
        return m_alpha;
    }
    void set_alpha(const double& alpha) {
        m_alpha = alpha;
    }

private:
    double m_alpha{0};
};
}  // namespace v0
}  // namespace op
}  // namespace ov
