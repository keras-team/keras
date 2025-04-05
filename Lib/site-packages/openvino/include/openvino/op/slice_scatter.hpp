// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/op/op.hpp"

namespace ov {
namespace op {
namespace v15 {
/// \brief SliceScatter operation.
///
/// \ingroup ov_ops_cpp_api
class OPENVINO_API SliceScatter : public Op {
public:
    OPENVINO_OP("SliceScatter", "opset15");

    SliceScatter() = default;

    /// \brief    Constructs SliceScatter operation (default axes).
    ///
    /// \param data             The tensor to be updated.
    /// \param updates          Tensor containing update values.
    /// \param start            1D tensor with start indices of the update slice.
    /// \param stop             1D tensor with end indices of the update slice.
    /// \param step             1D tensor specifies the increment to use in slicing along corresponding axes.
    SliceScatter(const Output<Node>& data,
                 const Output<Node>& updates,
                 const Output<Node>& start,
                 const Output<Node>& stop,
                 const Output<Node>& step);

    /// \brief    Constructs SliceScatter operation.
    ///
    /// \param data             The tensor to be updated.
    /// \param updates          Tensor containing update values.
    /// \param start            1D tensor with start indices of the update slice.
    /// \param stop             1D tensor with end indices of the update slice.
    /// \param step             1D tensor specifies the increment to use in slicing along corresponding axes.
    /// \param axes             1D tensor indicating which dimensions the values in the `start` and `stop` apply to.
    SliceScatter(const Output<Node>& data,
                 const Output<Node>& updates,
                 const Output<Node>& start,
                 const Output<Node>& stop,
                 const Output<Node>& step,
                 const Output<Node>& axes);

    void validate_and_infer_types() override;

    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& new_args) const override;
};
}  // namespace v15
}  // namespace op
}  // namespace ov
