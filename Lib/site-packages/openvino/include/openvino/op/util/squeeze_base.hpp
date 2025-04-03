// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/op/op.hpp"

namespace ov {
namespace op {
namespace util {
/// \brief Squeeze operation.
///
/// \ingroup ov_ops_cpp_api
class OPENVINO_API SqueezeBase : public Op {
public:
    OPENVINO_OP("Squeeze", "util");
    SqueezeBase() = default;
    /// \brief Constructs a squeeze operation.
    ///
    /// \param data Input tensor with data
    SqueezeBase(const Output<Node>& data);
    /// \brief Constructs a squeeze operation.
    ///
    /// \param data Input tensor with data
    /// \param axis The axis along which to squeeze the input tensor.
    SqueezeBase(const Output<Node>& data, const Output<Node>& axes);

    bool has_evaluate() const override;
    bool evaluate_lower(TensorVector& outputs) const override;
    bool evaluate_upper(TensorVector& outputs) const override;
    bool evaluate_symbol(TensorSymbolVector& output_symbols) const override;
    bool can_constant_fold(const OutputVector& inputs_values) const override;
    bool constant_fold(OutputVector& output_values, const OutputVector& inputs_values) override;
    bool is_dynamic() const override;
};
}  // namespace util
}  // namespace op
}  // namespace ov
