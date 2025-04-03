// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <string>
#include <vector>

#include "openvino/op/util/rnn_cell_base.hpp"

namespace ov {
namespace op {
namespace v5 {
/// \brief RNNSequence operation.
///
/// \ingroup ov_ops_cpp_api
class OPENVINO_API RNNSequence : public util::RNNCellBase {
public:
    OPENVINO_OP("RNNSequence", "opset5", util::RNNCellBase);

    RNNSequence();

    RNNSequence(const Output<Node>& X,
                const Output<Node>& H_t,
                const Output<Node>& sequence_lengths,
                const Output<Node>& W,
                const Output<Node>& R,
                const Output<Node>& B,
                size_t hidden_size,
                op::RecurrentSequenceDirection direction,
                const std::vector<std::string>& activations = std::vector<std::string>{"tanh"},
                const std::vector<float>& activations_alpha = {},
                const std::vector<float>& activations_beta = {},
                float clip = 0.f);

    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& new_args) const override;

    void validate_and_infer_types() override;

    bool visit_attributes(AttributeVisitor& visitor) override;

    op::RecurrentSequenceDirection get_direction() const {
        return m_direction;
    }
    void set_direction(const RecurrentSequenceDirection& direction) {
        m_direction = direction;
    }

protected:
    op::RecurrentSequenceDirection m_direction;
};
}  // namespace v5
}  // namespace op
}  // namespace ov
