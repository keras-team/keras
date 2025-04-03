// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/op/op.hpp"

namespace ov {
namespace op {
namespace v10 {
/// \brief Boolean mask that maps NaN and Infinity values to false and other values to true.
/// \ingroup ov_ops_cpp_api
class OPENVINO_API IsFinite : public Op {
public:
    OPENVINO_OP("IsFinite", "opset10");
    IsFinite() = default;
    /// \brief Constructs a IsFinite operation.
    ///
    /// \param data   Input data tensor
    IsFinite(const Output<Node>& data);

    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& new_args) const override;

    bool visit_attributes(AttributeVisitor& visitor) override;

    void validate_and_infer_types() override;
};
}  // namespace v10
}  // namespace op
}  // namespace ov
