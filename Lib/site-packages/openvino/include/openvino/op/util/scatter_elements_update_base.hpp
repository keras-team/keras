// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/op/op.hpp"
#include "openvino/op/util/attr_types.hpp"

namespace ov {
namespace op {
namespace util {
class OPENVINO_API ScatterElementsUpdateBase : public Op {
public:
    OPENVINO_OP("ScatterElementsUpdateBase", "util");
    ScatterElementsUpdateBase() = default;

    /// \brief The common base class for all ScatterElementsUpdate operator versions
    ///
    /// \param data            Input data
    /// \param indices         Data entry index that will be updated
    /// \param updates         Update values
    /// \param axis            Axis to scatter on
    ScatterElementsUpdateBase(const Output<Node>& data,
                              const Output<Node>& indices,
                              const Output<Node>& updates,
                              const Output<Node>& axis);

    void validate_and_infer_types() override;

    bool has_evaluate() const override;
    bool evaluate_lower(TensorVector& output_values) const override;
    bool evaluate_upper(TensorVector& output_values) const override;
    bool evaluate_symbol(TensorSymbolVector& output_symbols) const override;

protected:
    bool is_supported_index_input_element_type() const;
    int64_t get_normalized_axis(const TensorVector& inputs) const;
};
}  // namespace util
}  // namespace op
}  // namespace ov
