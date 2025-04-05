// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/op/op.hpp"

namespace ov {
namespace op {
namespace v15 {
/// \brief Operator packing a concatenated batch of strings into a batched string tensor.
/// \ingroup ov_ops_cpp_api
class OPENVINO_API StringTensorPack : public ov::op::Op {
public:
    OPENVINO_OP("StringTensorPack", "opset15", ov::op::Op);

    StringTensorPack() = default;
    /// \brief Constructs a StringTensorPack operation.
    ///
    /// \param begins Indices of each string's beginnings
    /// \param ends Indices of each string's endings
    /// \param symbols Concatenated input strings encoded in utf-8 bytes
    StringTensorPack(const Output<Node>& begins, const Output<Node>& ends, const Output<Node>& symbols);

    void validate_and_infer_types() override;
    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& new_args) const override;
};

}  // namespace v15
}  // namespace op
}  // namespace ov
