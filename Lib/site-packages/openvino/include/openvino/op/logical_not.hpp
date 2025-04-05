// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/op/op.hpp"

namespace ov {
namespace op {
namespace v1 {
/// \brief Elementwise logical negation operation.
/// \ingroup ov_ops_cpp_api
class OPENVINO_API LogicalNot : public Op {
public:
    OPENVINO_OP("LogicalNot", "opset1", op::Op);
    /// \brief Constructs a logical negation operation.
    LogicalNot() = default;
    /// \brief Constructs a logical negation operation.
    ///
    /// \param arg Node that produces the input tensor.
    LogicalNot(const Output<Node>& arg);

    void validate_and_infer_types() override;

    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& new_args) const override;
    bool evaluate(TensorVector& outputs, const TensorVector& inputs) const override;
    bool has_evaluate() const override;
};
}  // namespace v1
}  // namespace op
}  // namespace ov
