// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>

#include "openvino/op/op.hpp"

namespace ov {
namespace op {
namespace v13 {
/// \brief Scaled dot product attention operation from PyTorch
///
/// \ingroup ov_ops_cpp_api

class OPENVINO_API ScaledDotProductAttention : public Op {
public:
    OPENVINO_OP("ScaledDotProductAttention", "opset13", op::Op);

    /// \brief Constructs a ScaledDotProductAttention operation.
    ScaledDotProductAttention() = default;

    ScaledDotProductAttention(const OutputVector& inputs, bool causal);

    ScaledDotProductAttention(const Output<Node>& query,
                              const Output<Node>& key,
                              const Output<Node>& value,
                              const Output<Node>& attn_mask,
                              const Output<Node>& scale,
                              bool causal);

    ScaledDotProductAttention(const Output<Node>& query,
                              const Output<Node>& key,
                              const Output<Node>& value,
                              const Output<Node>& attn_mask,
                              bool causal);

    ScaledDotProductAttention(const Output<Node>& query,
                              const Output<Node>& key,
                              const Output<Node>& value,
                              bool causal);

    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& new_args) const override;
    bool visit_attributes(AttributeVisitor& visitor) override;
    void validate_and_infer_types() override;

    bool get_causal() const {
        return m_causal;
    }

    void set_causal(bool causal) {
        m_causal = causal;
    }

private:
    bool m_causal = false;
};

}  // namespace v13
}  // namespace op
}  // namespace ov
