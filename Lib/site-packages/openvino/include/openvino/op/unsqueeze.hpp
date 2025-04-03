// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>

#include "openvino/op/op.hpp"

namespace ov {
namespace op {
namespace v0 {
/// \brief Unsqueeze operation.
///
/// \ingroup ov_ops_cpp_api
class OPENVINO_API Unsqueeze : public Op {
public:
    OPENVINO_OP("Unsqueeze", "opset1");

    Unsqueeze() = default;
    Unsqueeze(const Output<Node>& data, const Output<Node>& axes);

    void validate_and_infer_types() override;
    bool visit_attributes(AttributeVisitor& visitor) override;
    bool evaluate(ov::TensorVector& outputs, const ov::TensorVector& inputs) const override;
    bool has_evaluate() const override;
    bool evaluate_lower(TensorVector& output_values) const override;
    bool evaluate_upper(TensorVector& output_values) const override;
    bool evaluate_symbol(TensorSymbolVector& output_symbols) const override;

    bool constant_fold(OutputVector& output_values, const OutputVector& inputs_values) override;
    bool can_constant_fold(const OutputVector& inputs_values) const override;

    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& new_args) const override;
};
}  // namespace v0
}  // namespace op
}  // namespace ov
