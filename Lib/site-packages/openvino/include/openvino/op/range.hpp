// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/op/op.hpp"

namespace ov {
namespace op {
namespace v4 {
/// \brief Range operation, analogous to `arange()` in Numpy.
/// \ingroup ov_ops_cpp_api
class OPENVINO_API Range : public Op {
public:
    OPENVINO_OP("Range", "opset4", op::Op);
    /// \brief Constructs an unitialized range operation.
    Range() = default;

    /// \brief Constructs a range operation.
    ///
    /// \param start The tensor producing the start value. Must be a scalar of numeric
    ///              element type.
    /// \param stop The tensor producing the stop value. Must be a scalar of numeric
    ///             element type.
    /// \param step The tensor producing the step value. Must be a scalar of numeric
    ///             element type.
    /// \param output_type The type of the output.
    Range(const Output<Node>& start, const Output<Node>& stop, const Output<Node>& step, element::Type output_type);

    bool visit_attributes(AttributeVisitor& visitor) override;
    void validate_and_infer_types() override;

    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& new_args) const override;
    bool evaluate(TensorVector& outputs, const TensorVector& inputs) const override;
    bool has_evaluate() const override;
    void set_output_type(element::Type output_type) {
        m_output_type = output_type;
    }

    const element::Type& get_output_type() const {
        return m_output_type;
    }

    // Overload collision with method on Node
    using Node::set_output_type;

private:
    element::Type m_output_type;
};
}  // namespace v4
namespace v0 {
/// \brief Range operation, analogous to `range()` in Python.
/// \ingroup ov_ops_cpp_api
class OPENVINO_API Range : public Op {
public:
    OPENVINO_OP("Range", "opset1");

    /// \brief Constructs an unitialized range operation.
    Range() = default;

    /// \brief Constructs a range operation.
    ///
    /// \param start The tensor producing the start value. Must be a scalar of integer
    ///              element type, and same element type as `stop` and `step`.
    /// \param stop The tensor producing the stop value. Must be a scalar of integer
    ///             element type, and same element type as `start` and `step`.
    /// \param step The tensor producing the step value. Must be a scalar of integer
    ///             element type, and same element type as `start` and `stop`.
    Range(const Output<Node>& start, const Output<Node>& stop, const Output<Node>& step);

    bool visit_attributes(AttributeVisitor& visitor) override;
    void validate_and_infer_types() override;

    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& new_args) const override;
    bool evaluate(TensorVector& outputs, const TensorVector& inputs) const override;
    bool has_evaluate() const override;
};
}  // namespace v0
}  // namespace op
}  // namespace ov
