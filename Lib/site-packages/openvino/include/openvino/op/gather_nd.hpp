// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/op/util/gather_nd_base.hpp"

namespace ov {
namespace op {
namespace v5 {
/// \brief GatherND operation
/// \ingroup ov_ops_cpp_api
class OPENVINO_API GatherND : public op::util::GatherNDBase {
public:
    OPENVINO_OP("GatherND", "opset5", op::util::GatherNDBase);
    GatherND() = default;

    /// \brief Constructs a GatherND operation.
    ///
    /// \param data Node producing data that are gathered
    /// \param indices Node producing indices by which the operation gathers elements
    /// or slices from data
    /// \param batch_dims Specifies a number of batch dimensions
    GatherND(const Output<Node>& data, const Output<Node>& indices, const size_t batch_dims = 0);

    void validate_and_infer_types() override;
    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& new_args) const override;
};
}  // namespace v5

namespace v8 {
/// \brief GatherND operation
///
/// \ingroup ov_ops_cpp_api
class OPENVINO_API GatherND : public op::util::GatherNDBase {
public:
    OPENVINO_OP("GatherND", "opset8", op::util::GatherNDBase);
    GatherND() = default;

    /// \brief Constructs a GatherND operation.
    ///
    /// \param data Node producing data that are gathered
    /// \param indices Node producing indices by which the operation gathers elements
    /// or slices from data
    /// \param batch_dims Specifies a number of batch dimensions
    GatherND(const Output<Node>& data, const Output<Node>& indices, const size_t batch_dims = 0);

    void validate_and_infer_types() override;
    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& new_args) const override;
};
}  // namespace v8
}  // namespace op
}  // namespace ov
