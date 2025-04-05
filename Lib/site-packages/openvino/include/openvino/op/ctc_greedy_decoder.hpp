// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/op/op.hpp"

namespace ov {
namespace op {
namespace v0 {
/// \brief CTCGreedyDecoder operation.
///
/// \ingroup ov_ops_cpp_api
class OPENVINO_API CTCGreedyDecoder : public Op {
public:
    OPENVINO_OP("CTCGreedyDecoder", "opset1");

    CTCGreedyDecoder() = default;
    /// \brief Constructs a CTCGreedyDecoder operation
    ///
    /// \param input              Logits on which greedy decoding is performed
    /// \param seq_len            Sequence lengths
    /// \param ctc_merge_repeated Whether to merge repeated labels
    CTCGreedyDecoder(const Output<Node>& input, const Output<Node>& seq_len, const bool ctc_merge_repeated);

    void validate_and_infer_types() override;
    bool visit_attributes(AttributeVisitor& visitor) override;
    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& new_args) const override;

    bool get_ctc_merge_repeated() const {
        return m_ctc_merge_repeated;
    }

    void set_ctc_merge_repeated(bool ctc_merge_repeated) {
        m_ctc_merge_repeated = ctc_merge_repeated;
    }

private:
    bool m_ctc_merge_repeated{true};
};
}  // namespace v0
}  // namespace op
}  // namespace ov
