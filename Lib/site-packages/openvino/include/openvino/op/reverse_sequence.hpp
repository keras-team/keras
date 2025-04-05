// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/op/op.hpp"

namespace ov {
namespace op {
namespace v0 {
/// \brief ReverseSequence operation.
///
/// \ingroup ov_ops_cpp_api
class OPENVINO_API ReverseSequence : public Op {
public:
    OPENVINO_OP("ReverseSequence", "opset1");

    ReverseSequence() = default;
    /// \brief Constructs a ReverseSequence operation.
    ///
    /// \param arg         tensor with input data to reverse
    /// \param seq_lengths 1D tensor of integers with sequence lengths in the input
    /// tensor.
    /// \param batch_axis  index of the batch dimension.
    /// \param seq_axis    index of the sequence dimension.
    ReverseSequence(const Output<Node>& arg,
                    const Output<Node>& seq_lengths,
                    int64_t batch_axis = 0,
                    int64_t seq_axis = 1);

    bool visit_attributes(AttributeVisitor& visitor) override;
    void validate_and_infer_types() override;

    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& new_args) const override;

    size_t get_batch_axis() const;
    int64_t get_origin_batch_axis() const {
        return m_batch_axis;
    }
    void set_batch_axis(int64_t batch_axis);

    size_t get_sequence_axis() const {
        return m_normalized_seq_axis;
    }
    int64_t get_origin_sequence_axis() const {
        return m_seq_axis;
    }
    void set_sequence_axis(int64_t sequence_axis);

private:
    int64_t m_batch_axis{};
    int64_t m_seq_axis{1};
    size_t m_normalized_seq_axis{};
};
}  // namespace v0
}  // namespace op
}  // namespace ov
