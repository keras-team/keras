// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/op/util/squeeze_base.hpp"

namespace ov {
namespace op {
namespace v0 {
/// \brief Squeeze operation.
///
/// \ingroup ov_ops_cpp_api
class OPENVINO_API Squeeze : public util::SqueezeBase {
public:
    OPENVINO_OP("Squeeze", "opset1");

    Squeeze();
    /// \brief Constructs a squeeze v0 operation.
    ///
    /// \param data Input tensor with data
    Squeeze(const Output<Node>& data);
    /// \brief Constructs a squeeze v0 operation.
    ///
    /// \param data Input tensor with data
    /// \param axis The axis along which to squeeze the input tensor.
    Squeeze(const Output<Node>& data, const Output<Node>& axes);

    void validate_and_infer_types() override;
    bool evaluate(TensorVector& outputs, const TensorVector& inputs) const override;

    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& new_args) const override;

private:
    Output<Node> get_default_axes_input() const;
};
}  // namespace v0

namespace v15 {
/// \brief Squeeze operation.
///
/// \ingroup ov_ops_cpp_api
class OPENVINO_API Squeeze : public util::SqueezeBase {
public:
    OPENVINO_OP("Squeeze", "opset15");

    Squeeze();
    /// \brief Constructs a squeeze v15 operation.
    ///
    /// \param data Input tensor with data
    /// \param allow_axis_skip Shape inference result dynamic rank if selected axis has 1 in range of its dynamic
    Squeeze(const Output<Node>& data, const bool allow_axis_skip = false);
    /// \brief Constructs a squeeze v15 operation.
    ///
    /// \param data Input tensor with data
    /// \param axis The axis along which to squeeze the input tensor.
    /// \param allow_axis_skip Shape inference result dynamic rank if selected axis has 1 in range of its dynamic
    Squeeze(const Output<Node>& data, const Output<Node>& axes, const bool allow_axis_skip = false);

    void validate_and_infer_types() override;
    bool evaluate(TensorVector& outputs, const TensorVector& inputs) const override;

    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& new_args) const override;

    bool visit_attributes(AttributeVisitor& visitor) override;

    bool get_allow_axis_skip() const;

private:
    Output<Node> get_default_axes_input() const;
    bool m_allow_axis_skip{};
};
}  // namespace v15
}  // namespace op
}  // namespace ov
