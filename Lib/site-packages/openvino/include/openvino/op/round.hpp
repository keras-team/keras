// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/op/util/unary_elementwise_arithmetic.hpp"

namespace ov {
namespace op {
namespace v5 {
/// \brief Elementwise round operation. The output is round to the nearest integer
/// for each value. In case of halfs, the rule is defined in attribute 'mode':
///     'HALF_TO_EVEN' - round halfs to the nearest even integer.
///     'HALF_AWAY_FROM_ZERO': - round in such a way that the result heads away from
/// zero.
/// \ingroup ov_ops_cpp_api
class OPENVINO_API Round : public util::UnaryElementwiseArithmetic {
public:
    enum class RoundMode { HALF_TO_EVEN, HALF_AWAY_FROM_ZERO };
    OPENVINO_OP("Round", "opset5", util::UnaryElementwiseArithmetic);

    /// \brief Constructs a round operation.
    Round() = default;

    /// \brief Constructs a round operation.
    ///
    /// \param arg Node that produces the input tensor.
    /// \param mode Rule to resolve halfs
    Round(const Output<Node>& arg, const RoundMode mode);

    bool visit_attributes(AttributeVisitor& visitor) override;
    void validate_and_infer_types() override;

    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& new_args) const override;

    bool evaluate(TensorVector& outputs, const TensorVector& inputs) const override;
    bool has_evaluate() const override;

    RoundMode get_mode() const {
        return m_mode;
    }
    void set_mode(const RoundMode& mode) {
        m_mode = mode;
    }

private:
    RoundMode m_mode{RoundMode::HALF_TO_EVEN};
};
}  // namespace v5
}  // namespace op
OPENVINO_API
std::ostream& operator<<(std::ostream& s, const op::v5::Round::RoundMode& type);

template <>
class OPENVINO_API AttributeAdapter<op::v5::Round::RoundMode>
    : public EnumAttributeAdapterBase<op::v5::Round::RoundMode> {
public:
    AttributeAdapter(op::v5::Round::RoundMode& value) : EnumAttributeAdapterBase<op::v5::Round::RoundMode>(value) {}

    OPENVINO_RTTI("AttributeAdapter<ov::op::v5::Round::RoundMode>");
};

}  // namespace ov
