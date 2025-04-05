// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/op/op.hpp"
#include "openvino/op/util/unary_elementwise_arithmetic.hpp"

namespace ov {
namespace op {
namespace v0 {
/// \brief Elementwise type conversion operation.
/// \ingroup ov_ops_cpp_api
class OPENVINO_API Convert : public Op {
public:
    OPENVINO_OP("Convert", "opset1");

    /// \brief Constructs a conversion operation.
    Convert() = default;
    /// \brief Constructs a conversion operation.
    ///
    /// \param arg          Node that produces the input tensor.
    /// \param destination_type  Element type for the output tensor.
    Convert(const Output<Node>& arg, const ov::element::Type& destination_type);

    void validate_and_infer_types() override;
    bool visit_attributes(AttributeVisitor& visitor) override;
    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& new_args) const override;
    const element::Type& get_destination_type() const {
        return m_destination_type;
    }
    void set_destination_type(const element::Type& destination_type) {
        m_destination_type = destination_type;
    }
    const element::Type& get_convert_element_type() const {
        return m_destination_type;
    }
    void set_convert_element_type(const element::Type& destination_type) {
        m_destination_type = destination_type;
    }

    bool evaluate(TensorVector& outputs, const TensorVector& inputs) const override;
    bool has_evaluate() const override;
    bool evaluate_lower(TensorVector& outputs) const override;
    bool evaluate_upper(TensorVector& outputs) const override;
    bool evaluate_symbol(TensorSymbolVector& output_symbols) const override;

protected:
    ov::element::Type m_destination_type;
};
}  // namespace v0
}  // namespace op
}  // namespace ov
