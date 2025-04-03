// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/op/op.hpp"
#include "openvino/op/util/attr_types.hpp"

namespace ov {
namespace op {
namespace v0 {
///
/// \brief      Class performing element-wise linear quantization.
///
/// \note       Input floating point values are quantized into a discrete
///             set of floating point values.
///
/// \paragraph Implementation This class creates a node which performs the following
///            operation:
///
///            round((data - input_low) / (input_high - input_low) * (levels-1)) /
///                 (levels-1) * (output_high - output_low) + output_low
///
///
/// \ingroup ov_ops_cpp_api
class OPENVINO_API FakeQuantize : public Op {
public:
    OPENVINO_OP("FakeQuantize", "opset1");

    FakeQuantize();
    ///
    /// \brief      Constructs a FakeQuantize operation node.
    ///
    /// \param[in]  data            The input data tensor.
    /// \param[in]  input_low       The minimum limit for input values.
    /// \param[in]  input_high      The maximum limit for input values.
    /// \param[in]  output_low      The minimum quantized value.
    /// \param[in]  output_high     The maximum quantized value.
    /// \param[in]  levels          The number of quantization levels.
    /// \param[in]  auto_broadcast  AutoBroadcast mode to be used for broadcasting
    ///                             limit values
    ///
    FakeQuantize(const Output<Node>& data,
                 const Output<Node>& input_low,
                 const Output<Node>& input_high,
                 const Output<Node>& output_low,
                 const Output<Node>& output_high,
                 std::size_t levels,
                 const AutoBroadcastSpec& auto_broadcast = AutoBroadcastSpec(AutoBroadcastType::NUMPY));

    bool visit_attributes(AttributeVisitor& visitor) override;
    void validate_and_infer_types() override;

    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& new_args) const override;

    std::size_t get_levels() const {
        return m_levels;
    }
    void set_levels(std::size_t levels) {
        m_levels = levels;
    }
    const AutoBroadcastSpec& get_auto_broadcast() const {
        return m_auto_broadcast;
    }
    void set_auto_broadcast(const AutoBroadcastSpec& auto_broadcast) {
        m_auto_broadcast = auto_broadcast;
    }

    bool evaluate(TensorVector& outputs, const TensorVector& inputs) const override;
    bool has_evaluate() const override;

    bool can_constant_fold(const OutputVector& inputs_values) const override {
        return false;
    }

private:
    std::size_t m_levels;
    AutoBroadcastSpec m_auto_broadcast = op::AutoBroadcastType::NUMPY;
};
}  // namespace v0
}  // namespace op
}  // namespace ov
