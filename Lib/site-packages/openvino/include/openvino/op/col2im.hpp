// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/op/op.hpp"

namespace ov {
namespace op {
namespace v15 {
/// \brief Operator combining sliding blocks into an image tensor
/// \ingroup ov_ops_cpp_api
class OPENVINO_API Col2Im : public ov::op::Op {
public:
    OPENVINO_OP("Col2Im", "opset15", ov::op::Op);

    Col2Im() = default;
    /// \brief Constructs a Col2Im operation.
    ///
    /// \param data Input tensor with data
    /// \param output_size Shape of the spatial dimensions of the output image
    /// \param kernel_size Size of the sliding blocks
    /// \param strides Stride in the sliding blocks in the input spatial dimensions
    /// \param dilations Local stride of the elements
    /// \param pads_begin Paddings at the beginning of each spatial axis, if undefined no padding is applied
    /// \param pads_end Paddings at the end of each spatial axis, if undefined no padding is applied
    Col2Im(const Output<Node>& data,
           const Output<Node>& output_size,
           const Output<Node>& kernel_size,
           const Strides& strides = Strides{1, 1},
           const Strides& dilations = Strides{1, 1},
           const Shape& pads_begin = Shape{0, 0},
           const Shape& pads_end = Shape{0, 0});

    bool visit_attributes(AttributeVisitor& visitor) override;
    void validate_and_infer_types() override;
    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& new_args) const override;

    const Strides& get_strides() const;
    const Strides& get_dilations() const;
    const Shape& get_pads_begin() const;
    const Shape& get_pads_end() const;

private:
    Strides m_strides;
    Strides m_dilations;
    Shape m_pads_begin;
    Shape m_pads_end;
};

}  // namespace v15
}  // namespace op
}  // namespace ov
