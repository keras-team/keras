// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <vector>

#include "openvino/op/op.hpp"

namespace ov {
namespace op {
namespace v1 {

/// \brief Splits the input tensor into a list of equal sized tensors
/// \ingroup ov_ops_cpp_api
class OPENVINO_API Split : public Op {
public:
    OPENVINO_OP("Split", "opset1", op::Op);

    /// \brief Constructs a split operation.
    Split() = default;
    /// \brief Constructs a split operation.
    /// \param data        The tensor to be split.
    /// \param axis        The index of an axis in "data" along which to perform
    ///                    the split.
    /// \param num_splits  The number of pieces that the data tensor should be
    ///                    split into.
    Split(const Output<Node>& data, const Output<Node>& axis, const size_t num_splits);

    bool visit_attributes(AttributeVisitor& visitor) override;
    void validate_and_infer_types() override;
    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& new_args) const override;

    size_t get_num_splits() const {
        return m_num_splits;
    }
    void set_num_splits(const size_t num_splits) {
        m_num_splits = num_splits;
    }

    bool evaluate(TensorVector& outputs, const TensorVector& inputs) const override;
    bool evaluate_lower(TensorVector& outputs) const override;
    bool evaluate_upper(TensorVector& outputs) const override;
    bool has_evaluate() const override;
    bool evaluate_symbol(TensorSymbolVector& output_symbols) const override;

protected:
    size_t m_num_splits;
};
}  // namespace v1
}  // namespace op
}  // namespace ov
