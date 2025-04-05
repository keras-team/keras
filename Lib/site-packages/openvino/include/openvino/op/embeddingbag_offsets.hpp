// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/core/axis_set.hpp"
#include "openvino/op/util/embeddingbag_offsets_base.hpp"
#include "openvino/op/util/index_reduction.hpp"

namespace ov {
namespace op {
namespace v15 {
/// \brief Returns embeddings for given indices
/// \ingroup ov_ops_cpp_api
class OPENVINO_API EmbeddingBagOffsets : public util::EmbeddingBagOffsetsBase {
public:
    OPENVINO_OP("EmbeddingBagOffsets", "opset15", util::EmbeddingBagOffsetsBase);
    /// \brief Constructs a EmbeddingBagOffsets operation.
    EmbeddingBagOffsets() = default;
    /// \brief Constructs a EmbeddingBagOffsets operation.
    ///
    /// EmbeddingBagOffsets constructs an output tensor by replacing every index in a
    /// given
    /// input tensor with a row (from the weights matrix) at that index
    ///
    /// \param emb_table tensor containing the embedding lookup table of the module of
    /// shape [num_emb, emb_dim1, emb_dim2, ...] and  of type T
    /// \param indices tensor of shape [num_indices] and of type T_IND. Required
    /// \param offsets tensor of shape [batch] and of type T_IND containing the starting
    /// index positions of each "bag" in indices. Required.
    /// \param default_index scalar of type T_IND containing default index in embedding
    /// table to fill empty "bags". If set to value -1 or not provided, empty "bags"
    /// are filled with zeros. Reverse indexing using negative values is not supported. Optional.
    /// \param per_sample_weights tensor of the same shape as indices and of type T.
    /// Each value in this tensor are multiplied with each
    /// value pooled from embedding table for each index. Optional.
    /// \param reduction enum to select algorithm used to perform reduction of elements in bag. Optional.
    EmbeddingBagOffsets(const Output<Node>& emb_table,
                        const Output<Node>& indices,
                        const Output<Node>& offsets,
                        const Output<Node>& default_index,
                        const Output<Node>& per_sample_weights,
                        const Reduction& reduction = Reduction::SUM);

    EmbeddingBagOffsets(const Output<Node>& emb_table,
                        const Output<Node>& indices,
                        const Output<Node>& offsets,
                        const Output<Node>& default_index,
                        const Reduction& reduction = Reduction::SUM);

    EmbeddingBagOffsets(const Output<Node>& emb_table,
                        const Output<Node>& indices,
                        const Output<Node>& offsets,
                        const Reduction& reduction = Reduction::SUM);

    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& new_args) const override;
};
}  // namespace v15
}  // namespace op
}  // namespace ov
