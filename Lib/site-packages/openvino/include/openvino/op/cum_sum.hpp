// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/core/axis_set.hpp"
#include "openvino/op/op.hpp"

namespace ov {
namespace op {
namespace v0 {
/// \brief Tensor cumulative sum operation.
///
/// Compute the cumulative sum of the input tensor along the axis specified.
///
/// \ingroup ov_ops_cpp_api
class OPENVINO_API CumSum : public Op {
public:
    OPENVINO_OP("CumSum", "opset3");

    /// \brief Constructs a cumulative summation operation.
    CumSum() = default;

    /// \brief Constructs a cumulative summation operation.
    ///
    /// \param arg The tensor to be summed.
    /// \param axis zero dimension tensor specifying axis position along which
    /// cumulative sum must be performed
    /// \param exclusive if set to true, the top element is not included
    /// \param reverse if set to true, will perform the sums in reverse direction
    CumSum(const Output<Node>& arg, const Output<Node>& axis, const bool exclusive = false, const bool reverse = false);

    /// \brief Constructs a cumulative summation operation with axis = 0
    ///
    /// \param arg The tensor to be summed
    CumSum(const Output<Node>& arg, const bool exclusive = false, const bool reverse = false);

    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& new_args) const override;

    bool evaluate(TensorVector& outputs, const TensorVector& inputs) const override;
    bool has_evaluate() const override;

    bool visit_attributes(AttributeVisitor& visitor) override;
    void validate_and_infer_types() override;

    bool is_exclusive() const {
        return m_exclusive;
    }
    bool is_reverse() const {
        return m_reverse;
    }

private:
    bool m_exclusive = false;
    bool m_reverse = false;
};
}  // namespace v0
}  // namespace op
}  // namespace ov
