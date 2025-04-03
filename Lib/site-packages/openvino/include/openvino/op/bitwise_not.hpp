// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/op/op.hpp"

namespace ov {
namespace op {
namespace v13 {
/// \brief Elementwise bitwise negation operation.
/// \ingroup ov_ops_cpp_api
class OPENVINO_API BitwiseNot : public op::Op {
public:
    OPENVINO_OP("BitwiseNot", "opset13", op::Op);
    /// \brief Constructs a bitwise negation operation.
    BitwiseNot() = default;
    /// \brief Constructs a bitwise negation operation.
    ///
    /// \param arg Node that produces the input tensor.
    BitwiseNot(const Output<Node>& arg);

    void validate_and_infer_types() override;

    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& new_args) const override;
};
}  // namespace v13
}  // namespace op
}  // namespace ov
