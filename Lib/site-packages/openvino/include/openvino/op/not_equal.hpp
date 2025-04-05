// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/op/util/binary_elementwise_comparison.hpp"

namespace ov {
namespace op {
namespace v1 {
/// \brief Elementwise not-equal operation.
/// \ingroup ov_ops_cpp_api
class OPENVINO_API NotEqual : public util::BinaryElementwiseComparison {
public:
    OPENVINO_OP("NotEqual", "opset1", op::util::BinaryElementwiseComparison);
    /// \brief Constructs a not-equal operation.
    NotEqual() : util::BinaryElementwiseComparison(AutoBroadcastType::NUMPY) {}
    /// \brief Constructs a not-equal operation.
    ///
    /// \param arg0 Node that produces the first input tensor.
    /// \param arg1 Node that produces the second input tensor.
    /// \param auto_broadcast Auto broadcast specification
    NotEqual(const Output<Node>& arg0,
             const Output<Node>& arg1,
             const AutoBroadcastSpec& auto_broadcast = AutoBroadcastSpec(AutoBroadcastType::NUMPY));

    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& new_args) const override;

    bool evaluate(TensorVector& outputs, const TensorVector& inputs) const override;
    bool has_evaluate() const override;
};
}  // namespace v1
}  // namespace op
}  // namespace ov
