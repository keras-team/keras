// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/core/core_visibility.hpp"
#include "openvino/op/op.hpp"

namespace ov {
namespace op {
namespace util {
class OPENVINO_API ReductionBase : public Op {
protected:
    /// \brief Constructs a reduction operation.
    ReductionBase();

    /// \brief Constructs a reduction operation.
    ///
    /// \param arg Output that produces the first input tensor.
    /// \param reduction_axes The axis positions (0-based) to be eliminated.
    ReductionBase(const Output<Node>& arg, const Output<Node>& reduction_axes);

    /// \brief      Infers reduction operations output shape.
    ///
    /// \param[in] keep_dims    Reduction operation keeps dimensions.
    ///
    /// \return Partial shape of the output.
    PartialShape infer_reduction_output_shape(const bool keep_dims);

public:
    OPENVINO_OP("ReductionBase", "util");

    /// \return true if reduction axes are constant else false.
    bool reduction_axes_constant() const;

    /// \return The axis positions (0-based) to be eliminated through reduction.
    /// \throws CheckFailure if the reduction axes are not constant. (Use
    ///           reduction_axes_constant to check.)
    const AxisSet get_reduction_axes() const;

    /// \brief Change the reduction axes
    void set_reduction_axes(const AxisSet& reduction_axes);

    // \brief Returns true if keep_dims is set to true explicitly.
    // Otherwise, (also keep_dims not handled) returns false.
    virtual bool get_keep_dims() const {
        return false;
    }
};
}  // namespace util
}  // namespace op
}  // namespace ov
