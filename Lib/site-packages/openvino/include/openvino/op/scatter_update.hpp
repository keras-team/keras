// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/op/util/scatter_base.hpp"

namespace ov {
namespace op {
namespace v3 {
///
/// \brief      Set new values to slices from data addressed by indices
///
/// \ingroup ov_ops_cpp_api
class OPENVINO_API ScatterUpdate : public util::ScatterBase {
public:
    OPENVINO_OP("ScatterUpdate", "opset3", util::ScatterBase);
    ScatterUpdate() = default;
    ///
    /// \brief      Constructs ScatterUpdate operator object.
    ///
    /// \param      data     The input tensor to be updated.
    /// \param      indices  The tensor with indexes which will be updated.
    /// \param      updates  The tensor with update values.
    /// \param[in]  axis     The axis at which elements will be updated.
    ///
    ScatterUpdate(const Output<Node>& data,
                  const Output<Node>& indices,
                  const Output<Node>& updates,
                  const Output<Node>& axis);

    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& inputs) const override;

    bool evaluate(TensorVector& outputs, const TensorVector& inputs) const override;
    bool evaluate_lower(TensorVector& outputs) const override;
    bool evaluate_upper(TensorVector& outputs) const override;
    bool evaluate_symbol(TensorSymbolVector& output_symbols) const override;
    bool has_evaluate() const override;
};
}  // namespace v3
}  // namespace op
}  // namespace ov
