// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/op/op.hpp"

namespace ov {
namespace op {
namespace v14 {
/// \brief Inverse operation computes the inverse of the input tensor.
///
/// \ingroup ov_ops_cpp_api
class OPENVINO_API Inverse : public Op {
public:
    OPENVINO_OP("Inverse", "opset14");
    Inverse() = default;
    /**
     * @brief Inverse operation computes the inverse of the input matrices. The inverse is computed for each MxM matrix
     * separetely, preserving all batch dimensions.
     *
     * @param data Input matrices to compute the inverse for. Last two tensor dimensions must be of the same size.
     * @param adjoint Boolean that determines whether to return a normal inverse or adjoint (conjugate transpose) of the
     * input matrices.
     */
    Inverse(const Output<Node>& data, const bool adjoint = false);

    bool visit_attributes(AttributeVisitor& visitor) override;
    void validate_and_infer_types() override;
    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& new_args) const override;

    bool get_adjoint() const;
    void set_adjoint(const bool adjoint);

private:
    bool m_adjoint = false;
};
}  // namespace v14
}  // namespace op
}  // namespace ov
