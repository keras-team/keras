// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstddef>
#include <vector>

#include "openvino/op/op.hpp"
#include "openvino/op/util/fft_base.hpp"

namespace ov {
namespace op {
namespace v7 {
/// \brief An operation IDFT that computes the inverse discrete Fourier transformation.
/// \ingroup ov_ops_cpp_api
class OPENVINO_API IDFT : public util::FFTBase {
public:
    OPENVINO_OP("IDFT", "opset7", util::FFTBase);
    IDFT() = default;

    /// \brief Constructs a IDFT operation. IDFT is performed for full size axes.
    ///
    /// \param data  Input data
    /// \param axes Axes to perform IDFT
    IDFT(const Output<Node>& data, const Output<Node>& axes);

    /// \brief Constructs a IDFT operation.
    ///
    /// \param data  Input data
    /// \param axes Axes to perform IDFT
    /// \param signal_size Signal sizes for 'axes'
    IDFT(const Output<Node>& data, const Output<Node>& axes, const Output<Node>& signal_size);

    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& new_args) const override;
};
}  // namespace v7
}  // namespace op
}  // namespace ov
