// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <vector>

#include "openvino/core/model.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/util/sub_graph_base.hpp"
#include "openvino/runtime/tensor.hpp"

namespace ov {
namespace op {
namespace v5 {
/// \brief  Iterate a body over tensors, accumulating into tensors.
/// \ingroup ov_ops_cpp_api
class OPENVINO_API Loop : public op::util::SubGraphOp {
public:
    /// \brief  Allows to define the purpose of inputs/outputs in the body
    struct SpecialBodyPorts {
        SpecialBodyPorts() = default;
        SpecialBodyPorts(int64_t in_current_iteration_input_idx, int64_t in_body_condition_output_idx)
            : current_iteration_input_idx(in_current_iteration_input_idx),
              body_condition_output_idx(in_body_condition_output_idx) {}
        // -1 means the input is not provided, this input is optional
        int64_t current_iteration_input_idx = -1;
        // -1 means the output is not provided,
        // this output is required, throw an exception if not provided
        int64_t body_condition_output_idx = -1;
    };

    OPENVINO_OP("Loop", "opset5", op::util::SubGraphOp);

    /// \brief Constructs a Loop operation.
    Loop() = default;

    /// \brief Constructs a Loop operation.
    ///
    /// \param trip_count Node specifies the maximum number of iterations.
    /// \param execution_condition Node determines whether to execute the first
    /// iteration or not.
    Loop(const Output<Node>& trip_count, const Output<Node>& execution_condition);

    Output<Node> get_concatenated_slices(const Output<Node>& value,
                                         int64_t start,
                                         int64_t stride,
                                         int64_t part_size,
                                         int64_t end,
                                         int64_t axis) override;

    void set_special_body_ports(const SpecialBodyPorts& special_body_ports) {
        m_special_body_ports = special_body_ports;
    }

    SpecialBodyPorts get_special_body_ports() const {
        return m_special_body_ports;
    }
    void validate_and_infer_types() override;
    bool visit_attributes(AttributeVisitor& visitor) override;
    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& new_args) const override;

    bool evaluate(ov::TensorVector& outputs, const ov::TensorVector& inputs) const override;
    bool evaluate(ov::TensorVector& outputs,
                  const ov::TensorVector& inputs,
                  const EvaluationContext& evaluation_context) const override;
    bool has_evaluate() const override;

protected:
    Loop(const Loop&);

private:
    void clone_to(Loop& dst, const OutputVector& new_args) const;

    SpecialBodyPorts m_special_body_ports;
};
}  // namespace v5
}  // namespace op

template <>
class OPENVINO_API AttributeAdapter<op::v5::Loop::SpecialBodyPorts>
    : public DirectValueAccessor<op::v5::Loop::SpecialBodyPorts> {
public:
    AttributeAdapter(op::v5::Loop::SpecialBodyPorts& value)
        : DirectValueAccessor<op::v5::Loop::SpecialBodyPorts>(value) {}

    OPENVINO_RTTI("AttributeAdapter<ov::op::v5::Loop::SpecialBodyPorts>");
};

}  // namespace ov
