// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/op/op.hpp"
#include "openvino/op/util/binary_elementwise_bitwise.hpp"

namespace ov {
namespace op {
namespace v13 {
/// \brief Elementwise bitwise XOR operation.
/// \ingroup ov_ops_cpp_api
class OPENVINO_API BitwiseXor : public util::BinaryElementwiseBitwise {
public:
    OPENVINO_OP("BitwiseXor", "opset13", util::BinaryElementwiseBitwise);
    /// \brief Constructs a bitwise XOR operation.
    BitwiseXor() = default;
    /// \brief Constructs a bitwise XOR operation.
    ///
    /// \param arg0 Output that produces the first input tensor.<br>
    /// `[d0, ...]`
    /// \param arg1 Output that produces the second input tensor.<br>
    /// `[d0, ...]`
    /// \param auto_broadcast Auto broadcast specification. Default is Numpy-style
    ///                       implicit broadcasting.
    ///
    /// Output `[d0, ...]`
    ///
    BitwiseXor(const Output<Node>& arg0,
               const Output<Node>& arg1,
               const AutoBroadcastSpec& auto_broadcast = AutoBroadcastSpec(AutoBroadcastType::NUMPY));

    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& new_args) const override;
};
}  // namespace v13
}  // namespace op
}  // namespace ov
