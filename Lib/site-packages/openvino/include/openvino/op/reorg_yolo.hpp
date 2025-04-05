// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/op/op.hpp"

namespace ov {
namespace op {
namespace v0 {
/// \brief ReorgYolo operation.
///
/// \ingroup ov_ops_cpp_api
class OPENVINO_API ReorgYolo : public Op {
public:
    OPENVINO_OP("ReorgYolo", "opset2");

    ReorgYolo() = default;
    /// \brief Constructs a ReorgYolo operation
    ///
    /// \param input          Input
    /// \param stride         Stride to reorganize input by
    ReorgYolo(const Output<Node>& input, const size_t stride);

    // Constructor with `strides` for backward compatibility
    ReorgYolo(const Output<Node>& input, const Strides& strides);

    void validate_and_infer_types() override;

    bool visit_attributes(AttributeVisitor& visitor) override;

    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& new_args) const override;

    Strides get_strides() const {
        return m_strides;
    }

    void set_strides(const size_t stride);

private:
    Strides m_strides;
};
}  // namespace v0
}  // namespace op
}  // namespace ov
