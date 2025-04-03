// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/op/op.hpp"
#include "openvino/op/util/attr_types.hpp"

namespace ov {
namespace op {
namespace util {
// clang-format off
/// \brief Abstract base class for elementwise binary arithmetic operations, i.e.,
///        operations where the same scalar binary arithmetic operation is applied to
///        each corresponding pair of elements in the two input tensors. Implicit
///        broadcast of input tensors is supported through one of the AutoBroadcast
///        modes.
///
/// For example, if the underlying arithmetic operation (determined by the subclass) is
/// \f$\mathit{op}(x,y)\f$, the input tensors
/// \f$[[x_0,y_0],[z_0,w_0]]\f$ and \f$[[x_1,y_1],[z_1,w_1]]\f$ will be mapped to
/// \f$[[\mathit{op}(x_0,x_1),\mathit{op}(y_0,y_1)],[\mathit{op}(z_0,z_1),\mathit{op}(w_0,w_1)]]\f$.
///
/// ## Inputs
///
/// |        | Type                              | Description                                                              |
/// | ------ | --------------------------------- | ------------------------------------------------------------------------ |
/// | `arg0` | \f$N[d_1,\dots,d_n]~(n \geq 0)\f$ | A tensor of any shape. The element type \f$N\f$ may be any numeric type. |
/// | `arg1` | \f$N[d_1,\dots,d_n]~(n \geq 0)\f$ | A tensor of the same element type as `arg0`.                             |
/// | `autob`| AutoBroadcastSpec                 | Auto broadcast specification.                                            |
///
/// ## Output
///
/// | Type                   | Description                                                                                                                                                                                                                      |
/// | ---------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
/// | \f$N[d_1,\dots,d_n]\f$ | The tensor \f$T\f$, where \f$T[i_1,\dots,i_n] = \mathit{op}(\texttt{arg0}[i_1,\dots,i_n],\texttt{arg1}[i_1,\dots,i_n])\f$. This will always have the same shape and element type as the input tensors (after auto broadcasting). |
// clang-format on
class OPENVINO_API BinaryElementwiseArithmetic : public Op {
protected:
    BinaryElementwiseArithmetic(const AutoBroadcastSpec& autob);

    /// \brief Constructs a binary elementwise arithmetic operation.
    ///
    /// \param arg0 Output that produces the first input tensor.
    /// \param arg1 Output that produces the second input tensor.
    BinaryElementwiseArithmetic(const Output<Node>& arg0, const Output<Node>& arg1, const AutoBroadcastSpec& autob);

public:
    OPENVINO_OP("BinaryElementwiseArithmetic", "util");

    void validate_and_infer_types() override;

    const AutoBroadcastSpec& get_autob() const override {
        return m_autob;
    }
    void set_autob(const AutoBroadcastSpec& autob) {
        m_autob = autob;
    }
    bool visit_attributes(AttributeVisitor& visitor) override;
    bool evaluate_lower(TensorVector& outputs) const override;
    bool evaluate_upper(TensorVector& outputs) const override;

private:
    AutoBroadcastSpec m_autob;
    void validate_and_infer_elementwise_arithmetic();
};
}  // namespace util
}  // namespace op
}  // namespace ov
