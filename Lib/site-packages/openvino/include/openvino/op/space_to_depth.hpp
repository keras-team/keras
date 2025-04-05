// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/op/op.hpp"

namespace ov {
namespace op {
namespace v0 {
/// \brief SpaceToDepth permutes input tensor blocks of spatial data into depth
/// dimension.
///
/// \note  Values from the height and width dimensions are moved to the depth dimension.
///
///        Output node produces a tensor with shape:
///        [N, C * blocksize * blocksize, H / blocksize, W / blocksize]
/// \ingroup ov_ops_cpp_api
class OPENVINO_API SpaceToDepth : public Op {
public:
    OPENVINO_OP("SpaceToDepth", "opset1");

    enum class SpaceToDepthMode {
        // The output depth is gathered from [block_size, ..., block_size, C]
        BLOCKS_FIRST,
        // The output depth is gathered from [C, block_size, ..., block_size]
        DEPTH_FIRST
    };

    SpaceToDepth() = default;
    /// \brief Constructs a SpaceToDepth operation.
    ///
    /// \param data - Node producing the input tensor
    /// \param mode Specifies how the output depth dimension is gathered
    /// from block coordinates and the old depth dimension.
    /// \param block_size - the size of the block of values to be moved
    SpaceToDepth(const Output<Node>& data, const SpaceToDepthMode& mode, std::size_t block_size = 1);

    SpaceToDepth(const Output<Node>& data, const std::string& mode, std::size_t block_size = 1);

    bool visit_attributes(AttributeVisitor& visitor) override;

    void set_block_size(size_t block_size);

    const std::size_t& get_block_size() const {
        return m_blocksize;
    }

    void set_mode(SpaceToDepthMode mode);

    SpaceToDepthMode get_mode() const {
        return m_mode;
    }
    void validate_and_infer_types() override;
    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& new_args) const override;

    bool evaluate(TensorVector& outputs, const TensorVector& inputs) const override;
    bool has_evaluate() const override;

protected:
    std::size_t m_blocksize;
    SpaceToDepthMode m_mode;
};
}  // namespace v0
}  // namespace op

OPENVINO_API
std::ostream& operator<<(std::ostream& s, const op::v0::SpaceToDepth::SpaceToDepthMode& type);

template <>
class OPENVINO_API AttributeAdapter<op::v0::SpaceToDepth::SpaceToDepthMode>
    : public EnumAttributeAdapterBase<op::v0::SpaceToDepth::SpaceToDepthMode> {
public:
    AttributeAdapter(op::v0::SpaceToDepth::SpaceToDepthMode& value)
        : EnumAttributeAdapterBase<op::v0::SpaceToDepth::SpaceToDepthMode>(value) {}

    OPENVINO_RTTI("AttributeAdapter<ov::op::v0::SpaceToDepth::SpaceToDepthMode>");
};

}  // namespace ov
