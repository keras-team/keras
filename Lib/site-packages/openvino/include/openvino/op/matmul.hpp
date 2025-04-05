// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/op/op.hpp"

namespace ov {
namespace op {
namespace v0 {
/// \brief Operator performing Matrix Multiplication.
/// \ingroup ov_ops_cpp_api
class OPENVINO_API MatMul : public Op {
public:
    OPENVINO_OP("MatMul", "opset1");
    MatMul() = default;
    /// \brief Constructs an Matrix Multiplication operation.
    ///
    /// \param A Matrix A
    /// \param B Matrix B
    /// \param transpose_a If matrix A should be transposed.
    /// \param transpose_b If matrix B should be transposed.
    MatMul(const Output<Node>& A,
           const Output<Node>& B,
           const bool& transpose_a = false,
           const bool& transpose_b = false);

    bool visit_attributes(AttributeVisitor& visitor) override;
    void validate_and_infer_types() override;

    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& new_args) const override;

    bool evaluate(TensorVector& outputs, const TensorVector& inputs) const override;
    bool has_evaluate() const override;

    bool get_transpose_a() const {
        return m_transpose_a;
    }
    bool get_transpose_b() const {
        return m_transpose_b;
    }
    void set_transpose_a(bool transpose_a) {
        m_transpose_a = transpose_a;
    }
    void set_transpose_b(bool transpose_b) {
        m_transpose_b = transpose_b;
    }

private:
    bool m_transpose_a{false};
    bool m_transpose_b{false};
};
}  // namespace v0
}  // namespace op
}  // namespace ov
