// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/op/op.hpp"

namespace ov {
namespace op {
namespace v1 {
// clang-format off
/// \brief Elementwise selection operation.
///
/// ## Inputs
///
/// |        | Type                                          | Description                                                  |
/// | ------ | --------------------------------------------- | ------------------------------------------------------------ |
/// | `arg0` | \f$\texttt{bool}[d_1,\dots,d_n]~(n \geq 0)\f$ | A tensor of any shape, with element `bool`.                  |
/// | `arg1` | \f$E[d_1,\dots,d_n]~(n \geq 0)\f$             | A tensor of a shape that is broadcast-compatible with `arg0`, with any element type. |
/// | `arg2` | \f$E[d_1,\dots,d_n]~(n \geq 0)\f$             | A tensor of a shape that is broadcast-compatible with `arg0`, and same element type as `arg1`. |
/// | `auto_broadcast`| AutoBroadcastSpec                             | Auto broadcast specification.                                |
///
/// ## Output
///
/// | Type                   | Description                                                                                                                                                             |
/// | ---------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
/// | \f$E[d_1,\dots,d_n]\f$ | The tensor \f$T\f$, where \f$T[i_1,\dots,i_n] = \texttt{arg1}[i_1,\dots,i_n]\text{ if }\texttt{arg0}[i_1,\dots,i_n] \neq 0\text{, else }\texttt{arg2}[i_1,\dots,i_n]\f$ |
/// \ingroup ov_ops_cpp_api
// clang-format on
class OPENVINO_API Select : public Op {
public:
    OPENVINO_OP("Select", "opset1", op::Op);
    /// \brief Constructs a selection operation.
    Select() : m_auto_broadcast(AutoBroadcastSpec(AutoBroadcastType::NUMPY)) {}

    /// \brief Constructs a selection operation.
    ///
    /// \param arg0 Node that produces the first input tensor.
    /// \param arg1 Node that produces the second input tensor.
    /// \param arg2 Node that produces the third input tensor.
    /// \param auto_broadcast Auto broadcast specification. Default is Numpy-style
    ///                       implicit broadcasting.
    Select(const Output<Node>& arg0,
           const Output<Node>& arg1,
           const Output<Node>& arg2,
           const AutoBroadcastSpec& auto_broadcast = AutoBroadcastSpec(AutoBroadcastType::NUMPY));

    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& new_args) const override;
    bool visit_attributes(AttributeVisitor& visitor) override;
    void validate_and_infer_types() override;

    const AutoBroadcastSpec& get_auto_broadcast() const {
        return m_auto_broadcast;
    }
    void set_auto_broadcast(const AutoBroadcastSpec& auto_broadcast) {
        m_auto_broadcast = auto_broadcast;
    }
    // TODO: Move all uses of get_autob to get_auto_broadcast() and remove this.
    const AutoBroadcastSpec& get_autob() const override {
        return m_auto_broadcast;
    }
    bool evaluate(TensorVector& outputs, const TensorVector& inputs) const override;
    bool evaluate_upper(TensorVector& outputs) const override;
    bool evaluate_lower(TensorVector& outputs) const override;
    bool has_evaluate() const override;

private:
    AutoBroadcastSpec m_auto_broadcast;
};
}  // namespace v1
}  // namespace op
}  // namespace ov
