// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>

#include "openvino/op/util/binary_elementwise_logical.hpp"

namespace ov {
namespace op {
namespace v1 {
/// \brief Elementwise logical-and operation.
///
/// \ingroup ov_ops_cpp_api
class OPENVINO_API LogicalAnd : public util::BinaryElementwiseLogical {
public:
    OPENVINO_OP("LogicalAnd", "opset1", util::BinaryElementwiseLogical);
    /// \brief Constructs a logical-and operation.
    LogicalAnd() = default;

    /// \brief Constructs a logical-and operation.
    ///
    /// \param arg0 Output that produces the first input tensor.<br>
    /// `[d0, ...]`
    /// \param arg1 Output that produces the second input tensor.<br>
    /// `[d0, ...]`
    /// \param auto_broadcast Auto broadcast specification
    ///
    /// Output `[d0, ...]`
    ///
    LogicalAnd(const Output<Node>& arg0,
               const Output<Node>& arg1,
               const AutoBroadcastSpec& auto_broadcast = AutoBroadcastSpec(AutoBroadcastType::NUMPY));

    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& new_args) const override;
    bool evaluate(TensorVector& outputs, const TensorVector& inputs) const override;
    bool has_evaluate() const override;
};
}  // namespace v1
}  // namespace op
}  // namespace ov
