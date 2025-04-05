// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/op/op.hpp"

namespace ov {
namespace op {
namespace v5 {
/// \brief LogSoftmax operation.
///
/// \ingroup ov_ops_cpp_api
class OPENVINO_API LogSoftmax : public Op {
public:
    OPENVINO_OP("LogSoftmax", "opset5", op::Op);
    LogSoftmax() = default;
    /// \brief Constructs a LogSoftmax operation.
    ///
    /// \param arg Node that produces the first input tensor.<br>
    /// `[d0, ...]`
    /// \param axis The axis position (0-based) on which to calculate the LogSoftmax.
    ///
    /// Output `[d0, ...]`
    ///
    LogSoftmax(const Output<Node>& arg, const int64_t axis);

    bool visit_attributes(AttributeVisitor& visitor) override;
    void validate_and_infer_types() override;

    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& new_args) const override;

    int64_t get_axis() const {
        return m_axis;
    }
    void set_axis(const int64_t axis) {
        m_axis = axis;
    }

private:
    int64_t m_axis = 1;
};
}  // namespace v5
}  // namespace op
}  // namespace ov
