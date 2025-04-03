// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/op/op.hpp"

namespace ov {
namespace op {
namespace v15 {
/// \brief An operation STFT that computes the Short Time Fourier Transform.
/// \ingroup ov_ops_cpp_api
class OPENVINO_API STFT : public Op {
public:
    OPENVINO_OP("STFT", "opset15");
    STFT() = default;

    /// \brief Constructs a STFT operation.
    ///
    /// \param data  Input data
    /// \param window Window to perform STFT
    /// \param frame_size Scalar value representing the size of Fourier Transform
    /// \param frame_step The distance (number of samples) between successive window frames
    /// \param transpose_frames Flag to set output shape layout. If true the `frames` dimension is at out_shape[2],
    ///                         otherwise it is at out_shape[1].
    STFT(const Output<Node>& data,
         const Output<Node>& window,
         const Output<Node>& frame_size,
         const Output<Node>& frame_step,
         const bool transpose_frames);

    bool visit_attributes(AttributeVisitor& visitor) override;
    void validate_and_infer_types() override;
    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& new_args) const override;

    bool get_transpose_frames() const;
    void set_transpose_frames(const bool transpose_frames);

private:
    bool m_transpose_frames = false;
};
}  // namespace v15
}  // namespace op
}  // namespace ov
