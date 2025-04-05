// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/core/axis_set.hpp"
#include "openvino/op/util/index_reduction.hpp"

namespace ov {
namespace op {
namespace util {
/// \brief Returns embeddings for given indices
class OPENVINO_API EmbeddingBagOffsetsBase : public Op {
public:
    OPENVINO_OP("EmbeddingBagOffsetsBase", "util");

    enum class Reduction { SUM, MEAN };

    /// \brief Constructs a EmbeddingBagOffsetsBase operation.
    EmbeddingBagOffsetsBase() = default;
    /// \brief Constructs a EmbeddingBagOffsetsBase operation.
    ///
    /// EmbeddingBagOffsetsBase constructs an output tensor by replacing every index in
    /// a given input tensor with a row (from the weights matrix) at that index
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
    EmbeddingBagOffsetsBase(const Output<Node>& emb_table,
                            const Output<Node>& indices,
                            const Output<Node>& offsets,
                            const Output<Node>& default_index,
                            const Output<Node>& per_sample_weights);

    EmbeddingBagOffsetsBase(const Output<Node>& emb_table,
                            const Output<Node>& indices,
                            const Output<Node>& offsets,
                            const Output<Node>& default_index);

    EmbeddingBagOffsetsBase(const Output<Node>& emb_table, const Output<Node>& indices, const Output<Node>& offsets);

    EmbeddingBagOffsetsBase(const Output<Node>& emb_table,
                            const Output<Node>& indices,
                            const Output<Node>& offsets,
                            const Output<Node>& default_index,
                            const Output<Node>& per_sample_weights,
                            const Reduction& reduction);

    EmbeddingBagOffsetsBase(const Output<Node>& emb_table,
                            const Output<Node>& indices,
                            const Output<Node>& offsets,
                            const Output<Node>& default_index,
                            const Reduction& reduction);

    EmbeddingBagOffsetsBase(const Output<Node>& emb_table,
                            const Output<Node>& indices,
                            const Output<Node>& offsets,
                            const Reduction& reduction);

    void validate_and_infer_types() override;
    bool visit_attributes(AttributeVisitor& visitor) override;

    const Reduction& get_reduction() {
        return m_reduction;
    }

private:
    static constexpr int EMB_TABLE = 0;
    static constexpr int INDICES = 1;
    static constexpr int OFFSETS = 2;
    static constexpr int DEFAULT_INDEX = 3;
    static constexpr int PER_SAMPLE_WEIGHTS = 4;

protected:
    Reduction m_reduction = Reduction::SUM;
};
}  // namespace util
}  // namespace op
template <>
class OPENVINO_API AttributeAdapter<op::util::EmbeddingBagOffsetsBase::Reduction>
    : public EnumAttributeAdapterBase<op::util::EmbeddingBagOffsetsBase::Reduction> {
public:
    AttributeAdapter(op::util::EmbeddingBagOffsetsBase::Reduction& value)
        : EnumAttributeAdapterBase<op::util::EmbeddingBagOffsetsBase::Reduction>(value) {}

    OPENVINO_RTTI("AttributeAdapter<ov::op::util::EmbeddingBagOffsetsBase::Reduction>");
};

OPENVINO_API
std::ostream& operator<<(std::ostream& s, const op::util::EmbeddingBagOffsetsBase::Reduction& reduction);
}  // namespace ov
