// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/op/op.hpp"

namespace ov {
namespace op {
namespace v0 {
/// \brief RegionYolo operation.
///
/// \ingroup ov_ops_cpp_api
class OPENVINO_API RegionYolo : public Op {
public:
    OPENVINO_OP("RegionYolo", "opset1");

    RegionYolo() = default;
    ///
    /// \brief      Constructs a RegionYolo operation
    ///
    /// \param[in]  input        Input
    /// \param[in]  coords       Number of coordinates for each region
    /// \param[in]  classes      Number of classes for each region
    /// \param[in]  regions      Number of regions
    /// \param[in]  do_softmax   Compute softmax
    /// \param[in]  mask         Mask
    /// \param[in]  axis         Axis to begin softmax on
    /// \param[in]  end_axis     Axis to end softmax on
    /// \param[in]  anchors      A flattened list of pairs `[width, height]` that
    /// describes
    ///                          prior box sizes.
    ///
    RegionYolo(const Output<Node>& input,
               const size_t coords,
               const size_t classes,
               const size_t regions,
               const bool do_softmax,
               const std::vector<int64_t>& mask,
               const int axis,
               const int end_axis,
               const std::vector<float>& anchors = std::vector<float>{});

    bool visit_attributes(AttributeVisitor& visitor) override;
    void validate_and_infer_types() override;

    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& new_args) const override;

    size_t get_num_coords() const {
        return m_num_coords;
    }
    void set_num_coords(const size_t num_coords) {
        m_num_coords = num_coords;
    }
    size_t get_num_classes() const {
        return m_num_classes;
    }
    void set_num_classes(const size_t num_classes) {
        m_num_classes = num_classes;
    }
    size_t get_num_regions() const {
        return m_num_regions;
    }
    void set_num_regions(const size_t num_regions) {
        m_num_regions = num_regions;
    }
    bool get_do_softmax() const {
        return m_do_softmax;
    }
    void set_do_softmax(const bool do_softmax) {
        m_do_softmax = do_softmax;
    }
    const std::vector<int64_t>& get_mask() const {
        return m_mask;
    }
    void set_mask(const std::vector<int64_t>& mask) {
        m_mask = mask;
    }
    const std::vector<float>& get_anchors() const {
        return m_anchors;
    }
    void set_anchors(const std::vector<float>& anchors) {
        m_anchors = anchors;
    }
    int get_axis() const {
        return m_axis;
    }
    void set_axis(const int axis) {
        m_axis = axis;
    }
    int get_end_axis() const {
        return m_end_axis;
    }
    void set_end_axis(const int end_axis) {
        m_end_axis = end_axis;
    }

private:
    size_t m_num_coords;
    size_t m_num_classes;
    size_t m_num_regions;
    bool m_do_softmax;
    std::vector<int64_t> m_mask;
    std::vector<float> m_anchors{};
    int m_axis;
    int m_end_axis;
};
}  // namespace v0
}  // namespace op
}  // namespace ov
