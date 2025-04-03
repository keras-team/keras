// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/op/op.hpp"

namespace ov {
namespace op {
namespace util {
// clang-format off
/// \brief Abstract base class for elementwise unary arithmetic operations, i.e.,
///        operations where the same scalar arithmetic operation is applied to each
///        element.
///
/// For example, if the underlying operation (determined by the subclass) is
/// \f$\mathit{op}(x)\f$, the input tensor \f$[[x,y],[z,w]]\f$ will be mapped to
/// \f$[[\mathit{op}(x),\mathit{op}(y)],[\mathit{op}(z),\mathit{op}(w)]]\f$.
///
/// ## Inputs
///
/// |       | Type                              | Description                                                              |
/// | ----- | --------------------------------- | ------------------------------------------------------------------------ |
/// | `arg` | \f$N[d_1,\dots,d_n]~(n \geq 0)\f$ | A tensor of any shape. The element type \f$N\f$ may be any numeric type. |
///
/// ## Output
///
/// | Type                   | Description                                                                                                                                                             |
/// | ---------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
/// | \f$N[d_1,\dots,d_n]\f$ | The tensor \f$T\f$, where \f$T[i_1,\dots,i_n] = \mathit{op}(\texttt{arg}[i_1,\dots,i_n])\f$. This will always have the same shape and element type as the input tensor. |
// clang-format on
class OPENVINO_API UnaryElementwiseArithmetic : public Op {
protected:
    /// \brief Constructs a unary elementwise arithmetic operation.
    UnaryElementwiseArithmetic();
    /// \brief Constructs a unary elementwise arithmetic operation.
    ///
    /// \param arg Output that produces the input tensor.
    UnaryElementwiseArithmetic(const Output<Node>& arg);

public:
    OPENVINO_OP("UnaryElementwiseArithmetic", "util");

    void validate_and_infer_types() override;
    bool visit_attributes(AttributeVisitor& visitor) override;

private:
    void validate_and_infer_elementwise_arithmetic();
};
}  // namespace util
}  // namespace op
}  // namespace ov
