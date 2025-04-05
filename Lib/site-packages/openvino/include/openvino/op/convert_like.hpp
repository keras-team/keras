// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/op/op.hpp"

namespace ov {
namespace op {
namespace v1 {
/// \brief Elementwise type conversion operation.
/// \ingroup ov_ops_cpp_api
class OPENVINO_API ConvertLike : public Op {
public:
    OPENVINO_OP("ConvertLike", "opset1", op::Op);

    /// \brief Constructs a conversion operation.
    ConvertLike() = default;
    /// \brief Constructs a conversion operation.
    /// \param data  Node that produces the input tensor.
    /// \param like  Node which provides the target type information for the conversion.
    ConvertLike(const Output<Node>& data, const Output<Node>& like);

    void validate_and_infer_types() override;

    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& new_args) const override;

    bool constant_fold(OutputVector& output_values, const OutputVector& input_values) override;
    bool can_constant_fold(const OutputVector& inputs_values) const override;
};
}  // namespace v1
}  // namespace op
}  // namespace ov
