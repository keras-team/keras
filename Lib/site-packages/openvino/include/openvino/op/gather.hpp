// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/op/util/gather_base.hpp"

namespace ov {
namespace op {
namespace v1 {
/// \brief Gather slices from axis of data according to indices
/// \ingroup ov_ops_cpp_api
class OPENVINO_API Gather : public op::util::GatherBase {
public:
    OPENVINO_OP("Gather", "opset1", op::util::GatherBase);
    static constexpr int64_t AXIS_NOT_SET_VALUE = std::numeric_limits<int64_t>::max();
    Gather() = default;
    /// \param data The tensor from which slices are gathered
    /// \param indices Tensor with indexes to gather
    /// \param axis The tensor is a dimension index to gather data from
    Gather(const Output<Node>& params, const Output<Node>& indices, const Output<Node>& axis);

    int64_t get_axis() const override;

    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& new_args) const override;
};
}  // namespace v1

namespace v7 {
/// \brief Gather slices from axis of data according to indices
/// \ingroup ov_ops_cpp_api
class OPENVINO_API Gather : public op::util::GatherBase {
public:
    OPENVINO_OP("Gather", "opset7", op::util::GatherBase);
    Gather() = default;

    /// \param data The tensor from which slices are gathered
    /// \param indices Tensor with indexes to gather
    /// \param axis The tensor is a dimension index to gather data from
    /// \param batch_dims The number of batch dimension in data and indices tensors.
    /// If batch_dims = 0 Gather v7 is identical to Gather v1.
    Gather(const Output<Node>& data,
           const Output<Node>& indices,
           const Output<Node>& axis,
           const int64_t batch_dims = 0);

    bool visit_attributes(AttributeVisitor& visitor) override;
    void validate_and_infer_types() override;
    int64_t get_batch_dims() const;

    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& new_args) const override;
};
}  // namespace v7

namespace v8 {
/// \brief Gather slices from axis of data according to indices. Negative indices
/// are supported and indicate reverse indexing from the end
/// \ingroup ov_ops_cpp_api
class OPENVINO_API Gather : public op::util::GatherBase {
public:
    OPENVINO_OP("Gather", "opset8", op::util::GatherBase);
    Gather() = default;

    /// \param data The tensor from which slices are gathered
    /// \param indices Tensor with indexes to gather
    /// \param axis The tensor is a dimension index to gather data from
    /// \param batch_dims The number of batch dimension in data and indices tensors.
    Gather(const Output<Node>& data,
           const Output<Node>& indices,
           const Output<Node>& axis,
           const int64_t batch_dims = 0);

    bool visit_attributes(AttributeVisitor& visitor) override;
    void validate_and_infer_types() override;
    int64_t get_batch_dims() const;
    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& new_args) const override;
};
}  // namespace v8
}  // namespace op
}  // namespace ov
