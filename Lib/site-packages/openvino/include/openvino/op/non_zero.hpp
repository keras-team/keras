// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/op/op.hpp"

namespace ov {
namespace op {
namespace v3 {
/// \brief NonZero operation returning indices of non-zero elements in the input tensor.
///
/// \note The indices are returned by-dimension in row-major order. For example
///       the following output contains 3 indices of a 3D input tensor elements:
///       [[0, 0, 2],
///        [0, 1, 1],
///        [0, 1, 2]]
///       The values point to input elements at [0,0,0], [0,1,1] and [2,1,2]
/// \ingroup ov_ops_cpp_api
class OPENVINO_API NonZero : public Op {
public:
    OPENVINO_OP("NonZero", "opset3", op::Op);
    /// \brief Constructs a NonZero operation.
    NonZero() = default;
    /// \brief Constructs a NonZero operation.
    ///
    /// \note The output type is int64.
    ///
    /// \param arg Node that produces the input tensor.
    NonZero(const Output<Node>& arg);
    /// \brief Constructs a NonZero operation.
    ///
    /// \param arg Node that produces the input tensor.
    /// \param output_type produce indices. Currently, only 'int64' or 'int32'
    /// are
    ///                           supported
    NonZero(const Output<Node>& arg, const std::string& output_type);
    /// \brief Constructs a NonZero operation.
    ///
    /// \param arg Node that produces the input tensor.
    /// \param output_type produce indices. Currently, only int64 or int32 are
    ///                           supported
    NonZero(const Output<Node>& arg, const element::Type& output_type);

    bool visit_attributes(AttributeVisitor& visitor) override;
    void validate_and_infer_types() override;

    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& new_args) const override;

    element::Type get_output_type() const {
        return m_output_type;
    }
    void set_output_type(element::Type output_type) {
        m_output_type = output_type;
    }
    // Overload collision with method on Node
    using Node::set_output_type;

    bool evaluate(TensorVector& outputs, const TensorVector& inputs) const override;
    bool has_evaluate() const override;

protected:
    element::Type m_output_type = element::i64;
};
}  // namespace v3
}  // namespace op
}  // namespace ov
