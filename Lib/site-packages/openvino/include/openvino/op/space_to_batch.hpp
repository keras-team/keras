// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/op/op.hpp"

namespace ov {
namespace op {
namespace v1 {
/// \brief SpaceToBatch permutes data tensor blocks of spatial data into batch
/// dimension.
///
/// \note  Values from spatial blocks dimensions are moved in the batch dimension.
///
///        Output node produces a tensor with shape: tensor with shape
///        `[batch * block_shape[0] * block_shape[1] * ... * block_shape[N - 1],
///         (pads_begin[1] + D_1 + pads_end[1]) / block_shape[1],
///         (pads_begin[2] + D_2 + pads_end[2]) / block_shape[2], ...,
///         (pads_begin[N - 1] + D_{N - 1} + pads_end[N - 1]) / block_shape[N - 1]`
///         of the same type as `data` input.
/// \ingroup ov_ops_cpp_api
class OPENVINO_API SpaceToBatch : public Op {
public:
    OPENVINO_OP("SpaceToBatch", "opset2", op::Op);

    SpaceToBatch() = default;

    /// \brief Constructs a SpaceToBatch operation.
    ///
    /// \param data Node producing the data tensor
    /// \param block_shape The sizes of the block of values to be moved
    /// \param pads_begin Specifies the padding for the beginning along each axis of
    /// `data` input
    /// \param pads_end Specifies the padding for the ending along each axis of `data`
    /// input.
    SpaceToBatch(const Output<Node>& data,
                 const Output<Node>& block_shape,
                 const Output<ov::Node>& pads_begin,
                 const Output<ov::Node>& pads_end);

    void validate_and_infer_types() override;
    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& new_args) const override;
    bool visit_attributes(AttributeVisitor& visitor) override;

    bool evaluate(TensorVector& outputs, const TensorVector& inputs) const override;
    bool has_evaluate() const override;
};
}  // namespace v1
}  // namespace op
}  // namespace ov
