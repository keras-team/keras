// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/op/op.hpp"

namespace ov {
namespace op {
namespace v12 {
/// \brief GroupNormalization operation over the input tensor.
///
/// \ingroup ov_ops_cpp_api
class OPENVINO_API GroupNormalization : public Op {
public:
    OPENVINO_OP("GroupNormalization", "opset12");
    GroupNormalization();
    /// \param data The input tensor to be normalized
    /// \param scale The tensor containing scale values for each channel
    /// \param bias The tensor containing bias values for each channel
    /// \param num_groups The number of groups that the channel dimension will be divided into
    /// \param epsilon The value that prevents divisions by zero in GroupNormalization formula
    GroupNormalization(const Output<Node>& data,
                       const Output<Node>& scale,
                       const Output<Node>& bias,
                       int64_t num_groups,
                       double epsilon);

    bool visit_attributes(AttributeVisitor& visitor) override;

    void validate_and_infer_types() override;

    int64_t get_num_groups() const {
        return m_num_groups;
    }

    void set_num_groups(int64_t num_groups) {
        m_num_groups = num_groups;
    }

    double get_epsilon() const {
        return m_epsilon;
    }
    void set_epsilon(double epsilon) {
        m_epsilon = epsilon;
    }
    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& new_args) const override;

private:
    int64_t m_num_groups;
    double m_epsilon;
};
}  // namespace v12
}  // namespace op
}  // namespace ov
