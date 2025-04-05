// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/op/op.hpp"

namespace ov {
namespace op {
namespace v9 {
/// \brief Tensor Eye operation.
/// \ingroup ov_ops_cpp_api
class OPENVINO_API Eye : public Op {
public:
    OPENVINO_OP("Eye", "opset9");

    Eye() = default;

    /// \brief      Constructs a Eye operation.
    ///
    /// \param      num_rows          Node producing the tensor with row number.
    /// \param      num_columns       Node producing the tensor with column number.
    /// \param      diagonal_index    Node producing the tensor with the index of diagonal with ones.
    /// \param      batch_shape       Node producing the tensor with batch shape.
    /// \param      out_type          Output type of the tensor.
    Eye(const Output<Node>& num_rows,
        const Output<Node>& num_columns,
        const Output<Node>& diagonal_index,
        const Output<Node>& batch_shape,
        const ov::element::Type& out_type);

    /// \brief      Constructs a Eye operation without batch_shape.
    ///
    /// \param      num_rows          Node producing the tensor with row number.
    /// \param      num_columns       Node producing the tensor with column number.
    /// \param      diagonal_index    Node producing the tensor with the index of diagonal with ones.
    /// \param      out_type          Output type of the tensor.
    Eye(const Output<Node>& num_rows,
        const Output<Node>& num_columns,
        const Output<Node>& diagonal_index,
        const ov::element::Type& out_type);

    void validate_and_infer_types() override;

    bool visit_attributes(AttributeVisitor& visitor) override;

    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& new_args) const override;

    /// \return The output tensor type.
    const ov::element::Type& get_out_type() const {
        return m_output_type;
    }
    void set_out_type(const ov::element::Type& output_type) {
        m_output_type = output_type;
    }

    bool evaluate(TensorVector& outputs, const TensorVector& inputs) const override;
    bool has_evaluate() const override;

protected:
    ov::element::Type m_output_type;
};
}  // namespace v9
}  // namespace op
}  // namespace ov
