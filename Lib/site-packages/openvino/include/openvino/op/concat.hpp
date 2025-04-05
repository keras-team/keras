// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>

#include "openvino/op/op.hpp"

namespace ov {
namespace op {
namespace v0 {
/// \brief Concatenation operation.
/// \ingroup ov_ops_cpp_api
class OPENVINO_API Concat : public Op {
public:
    OPENVINO_OP("Concat", "opset1");

    /// \brief Constructs a concatenation operation.
    Concat() = default;
    /// \brief Constructs a concatenation operation.
    ///
    /// \param args               The outputs producing the input tensors.
    /// \param axis The axis along which to concatenate the input tensors.
    Concat(const OutputVector& args, int64_t axis);

    /// \brief Constructs a concatenation operation.
    ///
    /// \param args               The nodes producing the input tensors.
    /// \param axis The axis along which to concatenate the input tensors.
    Concat(const NodeVector& args, int64_t axis);

    bool visit_attributes(AttributeVisitor& visitor) override;
    void validate_and_infer_types() override;

    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& new_args) const override;

    /// \return The concatenation axis.
    int64_t get_axis() const {
        return m_axis;
    }
    void set_axis(int64_t axis) {
        m_axis = axis;
    }
    bool has_evaluate() const override;
    bool evaluate(TensorVector& outputs, const TensorVector& inputs) const override;
    bool evaluate_lower(TensorVector& output_values) const override;
    bool evaluate_upper(TensorVector& output_values) const override;
    bool evaluate_symbol(TensorSymbolVector& output_symbol) const override;

protected:
    /// \ brief m_axis stores default value for all iterations
    int64_t m_axis;
    /// \brief m_concat_axis stores m_axis plus the number of rank for each iteration
    int64_t m_concat_axis = -1;
};
}  // namespace v0
}  // namespace op
}  // namespace ov
