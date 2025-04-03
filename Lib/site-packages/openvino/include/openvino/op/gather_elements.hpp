// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/op/op.hpp"

namespace ov {
namespace op {
namespace v6 {
/// \brief GatherElements operation
///
/// \ingroup ov_ops_cpp_api
class OPENVINO_API GatherElements : public Op {
public:
    OPENVINO_OP("GatherElements", "opset6", op::Op);
    GatherElements() = default;

    /// \brief Constructs a GatherElements operation.
    ///
    /// \param data Node producing data that are gathered
    /// \param indices Node producing indices by which the operation gathers elements
    /// \param axis specifies axis along which indices are specified
    GatherElements(const Output<Node>& data, const Output<Node>& indices, const int64_t axis);

    void validate_and_infer_types() override;
    bool visit_attributes(AttributeVisitor& visitor) override;
    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& new_args) const override;

    int64_t get_axis() const {
        return m_axis;
    }
    void set_axis(int64_t axis) {
        m_axis = axis;
    }

private:
    int64_t m_axis{0};
};
}  // namespace v6
}  // namespace op
}  // namespace ov
