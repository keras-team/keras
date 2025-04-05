// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/op/op.hpp"

namespace ov {
namespace op {
namespace util {
/// \brief GatherBase basic class for Gather v1 and v7
class OPENVINO_API GatherBase : public Op {
public:
    OPENVINO_OP("GatherBase", "util");
    GatherBase() = default;

    /// \param data The tensor from which slices are gathered
    /// \param indices Tensor with indexes to gather
    /// \param axis The tensor is a dimension index to gather data from
    /// \param batch_dims The number of batch dimension in data and indices tensors
    GatherBase(const Output<Node>& data,
               const Output<Node>& indices,
               const Output<Node>& axis,
               const int64_t batch_dims = 0);

    void validate_and_infer_types() override;
    virtual int64_t get_axis() const;

    bool evaluate(TensorVector& outputs, const TensorVector& inputs) const override;
    bool has_evaluate() const override;
    bool evaluate_lower(TensorVector& outputs) const override;
    bool evaluate_upper(TensorVector& outputs) const override;
    bool evaluate_symbol(TensorSymbolVector& output_symbols) const override;

    bool constant_fold(OutputVector& output_values, const OutputVector& inputs_values) override;
    bool can_constant_fold(const OutputVector& inputs_values) const override;
    const int64_t& get_batch_dims() const;
    void set_batch_dims(int64_t batch_dims);

protected:
    int64_t m_batch_dims = 0;
};
}  // namespace util
}  // namespace op
}  // namespace ov
