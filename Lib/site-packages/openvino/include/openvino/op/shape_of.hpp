// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/op/util/shape_of_base.hpp"

namespace ov {
namespace op {
namespace v3 {
/// \brief Operation that returns the shape of its input argument as a tensor.
/// \ingroup ov_ops_cpp_api
class OPENVINO_API ShapeOf : public util::ShapeOfBase {
public:
    OPENVINO_OP("ShapeOf", "opset3", util::ShapeOfBase);
    ShapeOf() = default;
    /// \brief Constructs a shape-of operation.
    ShapeOf(const Output<Node>& arg, const element::Type output_type = element::i64);

    bool visit_attributes(AttributeVisitor& visitor) override;
    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& new_args) const override;

    void validate_and_infer_types() override;

    element::Type get_output_type() const {
        return m_output_type;
    }
    void set_output_type(element::Type output_type) {
        m_output_type = output_type;
    }
    // Overload collision with method on Node
    using Node::set_output_type;

    bool has_evaluate() const override;
    bool evaluate(TensorVector& output_values, const TensorVector& input_values) const override;
    bool evaluate_lower(TensorVector& output_values) const override;
    bool evaluate_upper(TensorVector& output_values) const override;
    bool evaluate_symbol(TensorSymbolVector& output_symbols) const override;
    bool constant_fold(OutputVector& output_values, const OutputVector& input_values) override;
    bool can_constant_fold(const OutputVector& inputs_values) const override;

private:
    element::Type m_output_type;
};
}  // namespace v3

namespace v0 {
/// \brief Operation that returns the shape of its input argument as a tensor.
/// \ingroup ov_ops_cpp_api
class OPENVINO_API ShapeOf : public util::ShapeOfBase {
public:
    OPENVINO_OP("ShapeOf", "opset1", util::ShapeOfBase);
    ShapeOf() = default;
    /// \brief Constructs a shape-of operation.
    ShapeOf(const Output<Node>& arg);

    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& new_args) const override;

    void validate_and_infer_types() override;

    bool has_evaluate() const override;
    bool evaluate(TensorVector& output_values, const TensorVector& input_values) const override;
    bool evaluate_lower(TensorVector& output_values) const override;
    bool evaluate_upper(TensorVector& output_values) const override;
    bool evaluate_symbol(TensorSymbolVector& output_symbols) const override;
    bool constant_fold(OutputVector& output_values, const OutputVector& input_values) override;
    bool can_constant_fold(const OutputVector& inputs_values) const override;
};
}  // namespace v0
}  // namespace op
}  // namespace ov
