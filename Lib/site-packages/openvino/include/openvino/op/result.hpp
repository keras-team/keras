// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/core/layout.hpp"
#include "openvino/op/op.hpp"

namespace ov {
namespace op {
namespace v0 {
/// \brief Result operation.
///
/// \ingroup ov_ops_cpp_api
///
/// The Result output tensor is special, it shares tensor with Result's input but requires to have dedicated properties
/// like:
/// - tensor names.
///
/// Setting/adding Result's output names modify this specific tensor names.
/// Result's specific tensor names are added to input descriptor and transferred to new descriptor if Result's input
/// has been replaced.
///
/// Examples 1: No specific names on Result's output
///
///  set output names:
///        [N1]
///         ↓
/// |----------------|        [names: N1]         |-----------------|
/// |      Node      |--------------------------->|     Result      |   -> Model output names: N1
/// |----------------|                            |-----------------|
///
///
/// Examples 2: Result's has got specific names
///
///  set output names:                             set output names:
///        [N1]                                         [R1, R2]
///         ↓                                              ↓
/// |----------------|    [names: N1, R1, R2]     |-----------------|
/// |      Node      |--------------------------->|     Result      |   -> Model output names: R1, R2
/// |----------------|                            |-----------------|
///
///
/// Examples 3: Result from example 2 connected to new node
///
///  set output names:                             set output names:
///        [N2]                                         [R1, R2]
///         ↓                                              ↓
/// |----------------|    [names: N2, R1, R2]     |-----------------|
/// |      Node      |--------------------------->|     Result      |   -> Model output names: R1, R2
/// |----------------|                            |-----------------|
///
///  set output names:
///        [N1]
///         ↓
/// |----------------|    [names: N1]
/// |      Node      |----------------->
/// |----------------|
///
class OPENVINO_API Result : public Op {
public:
    OPENVINO_OP("Result", "opset1");

    /// \brief Allows a value to be used as a function result.
    Result() = default;
    /// \brief Allows a value to be used as a function result.
    ///
    /// \param arg Node that produces the input tensor.
    Result(const Output<Node>& arg);

    /// \brief Allows a value to be used as a function result.
    ///
    /// \param arg Node that produces the input tensor.
    /// \param use_input_names  When true Result will use input node tensor names as Result's output names.
    Result(const Output<Node>& arg, bool use_input_names);

    void validate_and_infer_types() override;

    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& new_args) const override;

    bool evaluate(ov::TensorVector& outputs, const ov::TensorVector& inputs) const override;
    bool has_evaluate() const override;
    bool can_constant_fold(const OutputVector& inputs_values) const override;

    /// \brief Returns current layout, or empty Layout if it is not set
    Layout get_layout() const;

    /// \brief Sets layout runtime information to tensor.
    ///
    /// \param layout Layout to set. If empty (default constructed), layout runtime information is erased.
    void set_layout(const Layout& layout);
};
}  // namespace v0
}  // namespace op
using ResultVector = std::vector<std::shared_ptr<op::v0::Result>>;

template <>
class OPENVINO_API AttributeAdapter<ResultVector> : public VisitorAdapter {
public:
    AttributeAdapter(ResultVector& ref);

    bool visit_attributes(AttributeVisitor& visitor) override;

    OPENVINO_RTTI("AttributeAdapter<ResultVector>");

protected:
    ResultVector& m_ref;
};

}  // namespace ov
