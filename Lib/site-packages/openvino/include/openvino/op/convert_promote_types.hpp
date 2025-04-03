// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/op/op.hpp"

namespace ov {
namespace op {
namespace v14 {
/// \brief Elementwise operation that promote and convert input types to one common datatype.
/// \ingroup ov_ops_cpp_api
class OPENVINO_API ConvertPromoteTypes : public Op {
public:
    OPENVINO_OP("ConvertPromoteTypes", "opset14", op::Op);

    /// \brief Constructs operation that promote and convert input types to one common datatype.
    ConvertPromoteTypes() = default;
    /// \brief Constructs operation that promote and convert input types to one common datatype.
    /// \param input_0  Node with datatype to be promoted.
    /// \param input_1  Node with datatype to be promoted.
    /// \param promote_unsafe  Bool attribute whether to allow promotions that might result in bit-widening,
    /// precision loss and undefined behaviors.
    /// \param pytorch_scalar_promotion  Bool attribute whether to promote scalar input to type provided by non-scalar
    /// input when number format is matching. \param u64_integer_promotion_target  Element type attribute to select
    /// promotion result for u64 and signed integers.
    ConvertPromoteTypes(const Output<Node>& input_0,
                        const Output<Node>& input_1,
                        const bool promote_unsafe = false,
                        const bool pytorch_scalar_promotion = false,
                        const element::Type& u64_integer_promotion_target = element::f32);
    void validate_and_infer_types() override;
    bool visit_attributes(AttributeVisitor& visitor) override;

    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& new_args) const override;

    /// \brief Get bool attribute whether to promote scalar input to type provided by non-scalar input when number
    /// format is matching.
    bool get_pytorch_scalar_promotion() const;

    /// \brief Set bool attribute whether to promote scalar input to type provided by non-scalar input when number
    /// format is matching.
    void set_pytorch_scalar_promotion(bool pytorch_scalar_promotion);

    /// \brief Get bool attribute whether to allow promotions that might result in bit-widening, precision loss and
    /// undefined behaviors.
    bool get_promote_unsafe() const;

    /// \brief Set bool attribute whether to allow promotions that might result in bit-widening, precision loss and
    /// undefined behaviors.
    void set_promote_unsafe(bool promote_unsafe);

    /// \brief Get element type attribute to select promotion result for u64 and signed integers.
    const element::Type& get_u64_integer_promotion_target() const;

    /// \brief Set element type attribute to select promotion result for u64 and signed integers.
    void set_u64_integer_promotion_target(const element::Type& u64_integer_promotion_target);

private:
    bool m_promote_unsafe = false;
    bool m_pytorch_scalar_promotion = false;
    element::Type m_u64_integer_promotion_target = element::f32;
};
}  // namespace v14
}  // namespace op
}  // namespace ov
