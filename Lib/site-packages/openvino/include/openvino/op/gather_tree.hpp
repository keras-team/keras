// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/op/op.hpp"

namespace ov {
namespace op {
namespace v1 {
/// \brief Generates the complete beams from the ids per each step and the parent beam
/// ids.
/// \ingroup ov_ops_cpp_api
class OPENVINO_API GatherTree : public Op {
public:
    OPENVINO_OP("GatherTree", "opset1", op::Op);

    GatherTree() = default;
    /// \param step_ids     Tensor of shape [MAX_TIME, BATCH_SIZE, BEAM_WIDTH] with
    ///                     indices from per each step
    /// \param parent_idx   Tensor of shape [MAX_TIME, BATCH_SIZE, BEAM_WIDTH] with
    ///                     parent beam indices
    /// \param max_seq_len  Tensor of shape [BATCH_SIZE] with maximum lengths for each
    ///                     sequence in the batch
    /// \param end_token    Tensor of shape [MAX_TIME, BATCH_SIZE, BEAM_WIDTH]
    GatherTree(const Output<Node>& step_ids,
               const Output<Node>& parent_idx,
               const Output<Node>& max_seq_len,
               const Output<Node>& end_token);

    void validate_and_infer_types() override;

    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& new_args) const override;
};
}  // namespace v1
}  // namespace op
}  // namespace ov
