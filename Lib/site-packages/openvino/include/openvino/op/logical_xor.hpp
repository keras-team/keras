// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>

#include "openvino/op/util/binary_elementwise_logical.hpp"

namespace ov {
namespace op {
namespace v1 {
/// \brief Elementwise logical-xor operation.
///
/// \ingroup ov_ops_cpp_api
class OPENVINO_API LogicalXor : public util::BinaryElementwiseLogical {
public:
    OPENVINO_OP("LogicalXor", "opset2", util::BinaryElementwiseLogical);
    LogicalXor() = default;
    /// \brief Constructs a logical-xor operation.
    ///
    /// \param arg0 Node that produces the first input tensor.<br>
    /// `[d0, ...]`
    /// \param arg1 Node that produces the second input tensor.<br>
    /// `[d0, ...]`
    /// \param auto_broadcast Auto broadcast specification
    ///
    /// Output `[d0, ...]`
    ///
    LogicalXor(const Output<Node>& arg0,
               const Output<Node>& arg1,
               const AutoBroadcastSpec& auto_broadcast = AutoBroadcastSpec(AutoBroadcastType::NUMPY));

    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& new_args) const override;

    bool evaluate(TensorVector& outputs, const TensorVector& inputs) const override;
    bool has_evaluate() const override;
};
}  // namespace v1
}  // namespace op
}  // namespace ov
