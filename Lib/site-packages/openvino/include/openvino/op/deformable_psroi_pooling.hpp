// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/op/op.hpp"

namespace ov {
namespace op {
namespace v1 {
/// \brief DeformablePSROIPooling operation.
///
/// \ingroup ov_ops_cpp_api
class OPENVINO_API DeformablePSROIPooling : public Op {
public:
    OPENVINO_OP("DeformablePSROIPooling", "opset1", op::Op);

    DeformablePSROIPooling() = default;
    /// \brief Constructs a DeformablePSROIPooling operation
    ///
    /// \param input           Input tensor with position sensitive score maps
    /// \param coords          Input tensor with list of five element tuples
    ///                        describing ROI coordinates
    /// \param offsets         Input tensor with transformation values
    /// \param output_dim      Pooled output channel number
    /// \param group_size      Number of horizontal bins per row to divide ROI area,
    ///                        it defines output width and height
    /// \param spatial_scale   Multiplicative spatial scale factor to translate ROI
    ///                        coordinates from their input scale to the scale used when
    ///                        pooling
    /// \param mode            Specifies mode for pooling.
    /// \param spatial_bins_x  Specifies numbers of bins to divide ROI single
    ///                        bin over width
    /// \param spatial_bins_y  Specifies numbers of bins to divide ROI single
    ///                        bin over height
    /// \param no_trans        The flag that specifies whenever third input exists
    ///                        and contains transformation (offset) values
    /// \param trans_std       The value that all transformation (offset) values are
    ///                        multiplied with
    /// \param part_size       The number of parts the output tensor spatial dimensions
    ///                        are divided into. Basically it is the height
    ///                        and width of the third input
    DeformablePSROIPooling(const Output<Node>& input,
                           const Output<Node>& coords,
                           const Output<Node>& offsets,
                           const int64_t output_dim,
                           const float spatial_scale,
                           const int64_t group_size = 1,
                           const std::string mode = "bilinear_deformable",
                           int64_t spatial_bins_x = 1,
                           int64_t spatial_bins_y = 1,
                           float trans_std = 1,
                           int64_t part_size = 1);

    DeformablePSROIPooling(const Output<Node>& input,
                           const Output<Node>& coords,
                           const int64_t output_dim,
                           const float spatial_scale,
                           const int64_t group_size = 1,
                           const std::string mode = "bilinear_deformable",
                           int64_t spatial_bins_x = 1,
                           int64_t spatial_bins_y = 1,
                           float trans_std = 1,
                           int64_t part_size = 1);

    bool visit_attributes(AttributeVisitor& visitor) override;

    void validate_and_infer_types() override;

    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& new_args) const override;

    void set_output_dim(int64_t output_dim);
    int64_t get_output_dim() const {
        return m_output_dim;
    }
    void set_group_size(int64_t group_size);
    int64_t get_group_size() const {
        return m_group_size;
    }
    float get_spatial_scale() const {
        return m_spatial_scale;
    }
    const std::string& get_mode() const {
        return m_mode;
    }
    int64_t get_spatial_bins_x() const {
        return m_spatial_bins_x;
    }
    int64_t get_spatial_bins_y() const {
        return m_spatial_bins_y;
    }
    float get_trans_std() const {
        return m_trans_std;
    }
    int64_t get_part_size() const {
        return m_part_size;
    }

private:
    int64_t m_output_dim{0};
    float m_spatial_scale{0};
    int64_t m_group_size = 1;
    std::string m_mode = "bilinear_deformable";
    int64_t m_spatial_bins_x = 1;
    int64_t m_spatial_bins_y = 1;
    float m_trans_std = 1.f;
    int64_t m_part_size = 1;
};
}  // namespace v1
}  // namespace op
}  // namespace ov
