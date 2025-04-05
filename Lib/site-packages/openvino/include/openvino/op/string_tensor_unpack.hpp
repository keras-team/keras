// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/op/op.hpp"

namespace ov {
namespace op {
namespace v15 {
/// \brief Operator unpacking a batch of strings into three tensors.
/// \ingroup ov_ops_cpp_api
class OPENVINO_API StringTensorUnpack : public ov::op::Op {
public:
    OPENVINO_OP("StringTensorUnpack", "opset15", ov::op::Op);

    StringTensorUnpack() = default;
    /// \brief Constructs a StringTensorUnpack operation.
    ///
    /// \param data Input of type element::string
    StringTensorUnpack(const Output<Node>& data);

    void validate_and_infer_types() override;
    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& new_args) const override;
};

}  // namespace v15
}  // namespace op
}  // namespace ov
