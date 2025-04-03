// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>

#include "openvino/op/op.hpp"

namespace ov {
namespace op {
namespace v0 {
/// \brief BatchNormInference operation.
///
/// \ingroup ov_ops_cpp_api
class OPENVINO_API BatchNormInference : public Op {
public:
    OPENVINO_OP("BatchNormInference", "opset1");
    BatchNormInference() = default;
    /// \param input [., C, ...]
    /// \param gamma gamma scaling for normalized value. [C]
    /// \param beta bias added to the scaled normalized value [C]
    /// \param mean value for mean normalization [C]
    /// \param variance value for variance normalization [C]
    /// \param epsilon Avoids divsion by 0 if input has 0 variance
    BatchNormInference(const Output<Node>& input,
                       const Output<Node>& gamma,
                       const Output<Node>& beta,
                       const Output<Node>& mean,
                       const Output<Node>& variance,
                       double epsilon);

    bool visit_attributes(AttributeVisitor& visitor) override;

    void validate_and_infer_types() override;

    double get_eps_value() const {
        return m_epsilon;
    }
    void set_eps_value(double epsilon) {
        m_epsilon = epsilon;
    }
    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& new_args) const override;

private:
    static constexpr size_t INPUT_GAMMA = 0;
    static constexpr size_t INPUT_BETA = 1;
    static constexpr size_t INPUT_DATA = 2;
    static constexpr size_t INPUT_MEAN = 3;
    static constexpr size_t INPUT_VARIANCE = 4;

    double m_epsilon{0};
};
}  // namespace v0
namespace v5 {
/// \brief BatchNormInference operation.
///
/// \ingroup ov_ops_cpp_api
class OPENVINO_API BatchNormInference : public Op {
public:
    OPENVINO_OP("BatchNormInference", "opset5", op::Op);
    BatchNormInference() = default;
    /// \param input [., C, ...]
    /// \param gamma gamma scaling for normalized value. [C]
    /// \param beta bias added to the scaled normalized value [C]
    /// \param mean value for mean normalization [C]
    /// \param variance value for variance normalization [C]
    /// \param epsilon Avoids divsion by 0 if input has 0 variance
    BatchNormInference(const Output<Node>& input,
                       const Output<Node>& gamma,
                       const Output<Node>& beta,
                       const Output<Node>& mean,
                       const Output<Node>& variance,
                       double epsilon);

    bool visit_attributes(AttributeVisitor& visitor) override;

    void validate_and_infer_types() override;

    double get_eps_value() const {
        return m_epsilon;
    }
    void set_eps_value(double epsilon) {
        m_epsilon = epsilon;
    }
    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& new_args) const override;

private:
    static constexpr size_t INPUT_DATA = 0;
    static constexpr size_t INPUT_GAMMA = 1;
    static constexpr size_t INPUT_BETA = 2;
    static constexpr size_t INPUT_MEAN = 3;
    static constexpr size_t INPUT_VARIANCE = 4;

    double m_epsilon{0};
};
}  // namespace v5
}  // namespace op
}  // namespace ov
