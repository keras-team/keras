// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/core/attribute_visitor.hpp"
#include "openvino/core/core_visibility.hpp"
#include "openvino/core/layout.hpp"
#include "openvino/core/preprocess/color_format.hpp"
#include "openvino/core/type/element_type.hpp"
#include "openvino/runtime/tensor.hpp"

namespace ov {
namespace preprocess {

class OPENVINO_API TensorInfoMemoryType : public RuntimeAttribute {
public:
    OPENVINO_RTTI("memory_type", "0", RuntimeAttribute);

    TensorInfoMemoryType() = default;

    explicit TensorInfoMemoryType(const std::string& value) : value(value) {}

    bool visit_attributes(AttributeVisitor& visitor) override {
        visitor.on_attribute("value", value);
        return true;
    }

    std::string value;
};

/// \brief Information about user's input tensor. By default, it will be initialized to same data (type/shape/etc) as
/// model's input parameter. User application can override particular parameters (like 'element_type') according to
/// application's data and specify appropriate conversions in pre-processing steps
///
/// \code{.cpp}
/// auto proc = PrePostProcessor(function);
/// proc.input().tensor().set_element_type(ov::element::u8);
/// \endcode
class OPENVINO_API InputTensorInfo final {
    class InputTensorInfoImpl;
    std::unique_ptr<InputTensorInfoImpl> m_impl;
    friend class InputInfo;

    /// \brief Default internal empty constructor
    InputTensorInfo();

public:
    /// \brief Default destructor
    ~InputTensorInfo();

    /// \brief Set element type for user's input tensor
    ///
    /// \param type Element type for user's input tensor.
    ///
    /// \return Reference to 'this' to allow chaining with other calls in a builder-like manner
    InputTensorInfo& set_element_type(const ov::element::Type& type);

    /// \brief Set layout for user's input tensor
    ///
    /// \param layout Layout for user's input tensor.
    ///
    /// \return Reference to 'this' to allow chaining with other calls in a builder-like manner
    InputTensorInfo& set_layout(const ov::Layout& layout);

    /// \brief By default, input image shape is inherited from model input shape. This method specifies that user's
    /// input image has dynamic spatial dimensions (width & height). This can be useful for adding resize preprocessing
    /// from any input image to model's expected dimensions.
    ///
    /// \return Reference to 'this' to allow chaining with other calls in a builder-like manner.
    InputTensorInfo& set_spatial_dynamic_shape();

    /// \brief By default, input image shape is inherited from model input shape. Use this method to specify different
    /// width and height of user's input image. In case if input image size is not known, use
    /// `set_spatial_dynamic_shape` method.
    ///
    /// \param height Set fixed user's input image height.
    ///
    /// \param width Set fixed user's input image width.
    ///
    /// \return Reference to 'this' to allow chaining with other calls in a builder-like manner.
    InputTensorInfo& set_spatial_static_shape(size_t height, size_t width);

    /// \brief Set color format for user's input tensor.
    ///
    /// In general way, some formats support multi-plane input, e.g. NV12 image can be represented as 2 separate tensors
    /// (planes): Y plane and UV plane. set_color_format API also allows to set sub_names for such parameters for
    /// convenient usage of plane parameters. During build stage, new parameters for each plane will be inserted to the
    /// place of original parameter. This means that all parameters located after will shift their positions accordingly
    /// (e.g. {param1, param2} will become {param1/Y, param1/UV, param2})
    ///
    /// \param format Color format of input image.
    ///
    /// \param sub_names Optional list of sub-names assigned for each plane (e.g. {"Y", "UV"}). If specified, number of
    /// sub-names shall match with number of planes. If not specified, friendly name and tensor name for plane
    /// parameters will be empty. It is not allowed to specify sub-names for single-plane inputs.
    ///
    /// \return Reference to 'this' to allow chaining with other calls in a builder-like manner.
    InputTensorInfo& set_color_format(const ov::preprocess::ColorFormat& format,
                                      const std::vector<std::string>& sub_names = {});

    /// \brief Set memory type runtime information for user's input tensor
    ///
    /// \param memory_type Memory type. Refer to specific plugin's documentation for exact string format
    ///
    /// \return Reference to 'this' to allow chaining with other calls in a builder-like manner
    InputTensorInfo& set_memory_type(const std::string& memory_type);

    /// \brief By default, input shape is inherited from model's input shape. Use this method to specify different
    /// input data shape. If it is needed to change only input height & width of input image, consider define layout and
    /// use `set_spatial_static_shape' or 'set_spatial_dynamic_shape' instead. This method allows defining any custom
    /// input shape and can be useful for custom preprocessing operations
    ///
    /// \note Methods 'set_spatial_dynamic_shape', 'set_spatial_static_shape' are also intended to modify input shape,
    /// using those methods together will throw ov::AssertFailure exception
    ///
    /// \param shape New shape for input tensor.
    ///
    /// \return Reference to 'this' to allow chaining with other calls in a builder-like manner.
    InputTensorInfo& set_shape(const ov::PartialShape& shape);

    /// \brief Helper function to reuse element type and shape from user's created tensor. Use this only in case if
    /// input tensor is already known and available before. Overwrites previously set element type & shape via
    /// `set_element_type` and `set_shape`. Tensor's memory type is not reused, so if `runtime_tensor` represents remote
    /// tensor with particular memory type - you should still specify appropriate memory type manually using
    /// `set_memory_type`
    ///
    /// \note As for `InputTensorInfo::set_shape`, this method shall not be used together with methods
    /// 'set_spatial_dynamic_shape' and 'set_spatial_static_shape', otherwise ov::AssertFailure exception will be thrown
    ///
    /// \param runtime_tensor User's created tensor.
    ///
    /// \return Reference to 'this' to allow chaining with other calls in a builder-like manner.
    InputTensorInfo& set_from(const ov::Tensor& runtime_tensor);
};

}  // namespace preprocess
}  // namespace ov
