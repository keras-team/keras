// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/core/axis_set.hpp"
#include "openvino/op/util/embeddingbag_packed_base.hpp"
#include "openvino/op/util/index_reduction.hpp"

namespace ov {
namespace op {
namespace v3 {
/// \brief Returns embeddings for given indices
/// \ingroup ov_ops_cpp_api
class OPENVINO_API EmbeddingBagPackedSum : public util::EmbeddingBagPackedBase {
public:
    OPENVINO_OP("EmbeddingBagPackedSum", "opset3", util::EmbeddingBagPackedBase);
    /// \brief Constructs a EmbeddingBagPackedSum operation.
    EmbeddingBagPackedSum() = default;
    /// \brief Constructs a EmbeddingBagPackedSum operation.
    ///
    /// EmbeddingBagPackedSum constructs an output tensor by replacing every index in a
    /// given
    /// input tensor with a row (from the weights matrix) at that index
    ///
    /// \param emb_table Tensor containing the embedding lookup table of the module of
    /// shape [num_emb, emb_dim1, emb_dim2, ...] and  of type T
    /// \param  indices Tensor of shape `[batch, indices_per_bag]` and of type *T_IND*.
    /// Required.
    /// \param per_sample_weigths tensor of the same shape as indices and of type T.
    /// Each value in this tensor are multiplied with each
    /// value pooled from embedding table for each index. Optional.

    EmbeddingBagPackedSum(const Output<Node>& emb_table,
                          const Output<Node>& indices,
                          const Output<Node>& per_sample_weights);

    EmbeddingBagPackedSum(const Output<Node>& emb_table, const Output<Node>& indices);

    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& new_args) const override;

    bool visit_attributes(AttributeVisitor& visitor) override;
};
}  // namespace v3
}  // namespace op
}  // namespace ov
