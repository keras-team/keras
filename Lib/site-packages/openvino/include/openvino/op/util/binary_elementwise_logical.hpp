// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/op/op.hpp"

namespace ov {
namespace op {
namespace util {
// clang-format off
/// \brief Abstract base class for elementwise binary logical operations, i.e.,
///        operations where the same scalar binary logical operation is applied to
///        each corresponding pair of elements in two boolean input tensors. Implicit
///        broadcast of input tensors is supported through one of the AutoBroadcast
///        modes.
///
/// For example, if the underlying operation (determined by the subclass) is
/// \f$\mathit{op}(x,y)\f$, the input tensors \f$[[x_0,y_0],[z_0,w_0]]\f$ and
/// \f$[[x_1,y_1],[z_1,w_1]]\f$ will be mapped to
/// \f$[[\mathit{op}(x_0,x_1),\mathit{op}(y_0,y_1)],[\mathit{op}(z_0,z_1),\mathit{op}(w_0,w_1)]]\f$.
///
/// ## Inputs
///
/// |        | Type                                          | Description                                            |
/// | ------ | --------------------------------------------- | ------------------------------------------------------ |
/// | `arg0` | \f$\texttt{bool}[d_1,\dots,d_n]~(n \geq 0)\f$ | A tensor of any shape, with element type `bool`.       |
/// | `arg1` | \f$\texttt{bool}[d_1,\dots,d_n]~(n \geq 0)\f$ | A tensor of the same shape and element type as `arg0`. |
/// | `autob`| AutoBroadcastSpec                             | Auto broadcast specification.                          |
///
/// ## Output
///
/// | Type                               | Description                                                                                                                                                                                                        |
/// | ---------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
/// | \f$\texttt{bool}[d_1,\dots,d_n]\f$ | The tensor \f$T\f$, where \f$T[i_1,\dots,i_n] = \mathit{op}(\texttt{arg0}[i_1,\dots,i_n],\texttt{arg1}[i_1,\dots,i_n])\f$. This will always have the same shape as the input tensors, and the element type `bool`. |
// clang-format on
class OPENVINO_API BinaryElementwiseLogical : public Op {
protected:
    BinaryElementwiseLogical();

    /// \brief Constructs a binary elementwise logical operation.
    ///
    /// \param arg0 Output that produces the first input tensor.
    /// \param arg1 Output that produces the second input tensor.
    BinaryElementwiseLogical(const Output<Node>& arg0,
                             const Output<Node>& arg1,
                             const AutoBroadcastSpec& autob = AutoBroadcastSpec());

public:
    OPENVINO_OP("BinaryElementwiseLogical", "util");

    void validate_and_infer_types() override;

    const AutoBroadcastSpec& get_autob() const override {
        return m_autob;
    }
    void set_autob(const AutoBroadcastSpec& autob) {
        m_autob = autob;
    }
    bool visit_attributes(AttributeVisitor& visitor) override;

private:
    AutoBroadcastSpec m_autob = AutoBroadcastType::NUMPY;
};
}  // namespace util
}  // namespace op
}  // namespace ov
