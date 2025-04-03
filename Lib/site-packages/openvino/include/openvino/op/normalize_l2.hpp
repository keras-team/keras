// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>

#include "openvino/op/op.hpp"
#include "openvino/op/util/attr_types.hpp"

namespace ov {
namespace op {
namespace v0 {
/// \brief  Normalization with L2 norm.
///
/// \ingroup ov_ops_cpp_api
class OPENVINO_API NormalizeL2 : public Op {
public:
    OPENVINO_OP("NormalizeL2", "opset1");

    NormalizeL2() = default;
    ///
    /// \brief      Constructs a NormalizeL2 operation.
    ///
    /// \param      data            - Node producing the input tensor
    /// \param      axes            - Node indicating axes along which reduction is
    ///                               calculated
    /// \param      eps             - The epsilon added to L2 norm.
    /// \param      eps_mode        - Specifies how eps is combined with L2 value
    ///                               calculated before division
    ///
    NormalizeL2(const Output<Node>& data, const Output<Node>& axes, float eps, EpsMode eps_mode);

    void validate_and_infer_types() override;
    bool visit_attributes(AttributeVisitor& visitor) override;

    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& new_args) const override;

    float get_eps() const {
        return m_eps;
    }
    EpsMode get_eps_mode() const {
        return m_eps_mode;
    }
    AxisSet get_reduction_axes() const;

protected:
    float m_eps;
    EpsMode m_eps_mode;
};
}  // namespace v0
}  // namespace op
}  // namespace ov
