// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/op/op.hpp"

namespace ov {
namespace op {
namespace v1 {
/// \brief Tensor dynamic reshape operation.
///
/// "Converts" an input tensor into a new shape with the same number of elements.
/// This op does not touch the actual data. If needed, use Transpose for that purpose.
///
/// \ingroup ov_ops_cpp_api
class OPENVINO_API Reshape : public Op {
public:
    OPENVINO_OP("Reshape", "opset1", op::Op);
    Reshape() = default;
    /// \brief Constructs a dynamic reshape operation. This operation does not perform
    ///        transpose.
    ///
    /// \param arg The tensor to be reshaped.
    /// \param shape_pattern The node that defines output shape shape_pattern.
    ///        If the input shape is \f$(a_0,\dots,a_{k-1})\f$ then the output shape
    ///        must
    ///        be of the form \f$(b_0,\dots,b_{j-1})\f$ where \f$\Pi(a_i) = \Pi(b_i)\f$.
    ///        A value of -1 is allowed for at most one dimension, in which case the
    ///        dimension size is inferred based on element count of input tensor.
    /// \param special_zero Treats zeros in `shape_pattern` as wildcard flags indicating
    /// a
    ///        copy from input shape at the same index.
    ///
    Reshape(const Output<Node>& arg, const Output<Node>& shape_pattern, bool special_zero);

    bool visit_attributes(AttributeVisitor& visitor) override;
    void validate_and_infer_types() override;

    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& new_args) const override;

    bool get_special_zero() const {
        return m_special_zero;
    }
    void set_special_zero(bool special_zero) {
        m_special_zero = special_zero;
    }
    bool evaluate(ov::TensorVector& outputs, const ov::TensorVector& inputs) const override;
    bool has_evaluate() const override;
    bool evaluate_upper(TensorVector& outputs) const override;
    bool evaluate_lower(TensorVector& outputs) const override;
    bool evaluate_symbol(TensorSymbolVector& output_symbols) const override;
    bool constant_fold(OutputVector& output_values, const OutputVector& inputs_values) override;
    bool can_constant_fold(const OutputVector& inputs_values) const override;

protected:
    bool m_special_zero;
    bool evaluate_reshape(ov::TensorVector& outputs, const ov::TensorVector& inputs) const;

private:
    void calculate_output_shape(std::vector<Dimension>& reshape_pattern,
                                const int64_t& minus_one_idx,
                                const PartialShape& input_pshape,
                                std::vector<Dimension>& output_shape) const;
};
}  // namespace v1
}  // namespace op
}  // namespace ov
