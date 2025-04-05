// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/op/constant.hpp"
#include "openvino/op/op.hpp"

namespace ov {
namespace op {
namespace v8 {
/// \brief Slice operation.
///
/// \ingroup ov_ops_cpp_api
class OPENVINO_API Slice : public Op {
public:
    OPENVINO_OP("Slice", "opset8");

    Slice() = default;

    /// \brief    Constructs Slice operation (default axes).
    ///
    /// \param data             The tensor to be sliced.
    /// \param start            1D tensor with start indices of the slice.
    /// \param stop             1D tensor with end indices of the slice.
    /// \param step             1D tensor specifies the increment to use in slicing along corresponding axes.
    Slice(const Output<Node>& data, const Output<Node>& start, const Output<Node>& stop, const Output<Node>& step);

    /// \brief    Constructs Slice operation.
    ///
    /// \param data             The tensor to be sliced.
    /// \param start            1D tensor with start indices of the slice.
    /// \param stop             1D tensor with end indices of the slice.
    /// \param step             1D tensor specifies the increment to use in slicing along corresponding axes.
    /// \param axes             1D tensor indicating which dimensions the values in the `start` and `stop` apply to.
    Slice(const Output<Node>& data,
          const Output<Node>& start,
          const Output<Node>& stop,
          const Output<Node>& step,
          const Output<Node>& axes);

    void validate_and_infer_types() override;

    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& new_args) const override;
    bool has_evaluate() const override;
    bool evaluate(TensorVector&, const TensorVector&) const override;
    bool evaluate_lower(TensorVector& outputs) const override;
    bool evaluate_upper(TensorVector& outputs) const override;
    bool evaluate_symbol(TensorSymbolVector& output_symbols) const override;

    std::shared_ptr<v0::Constant> get_default_const_axes(const Output<Node>& start) const;
};
}  // namespace v8
}  // namespace op
}  // namespace ov
