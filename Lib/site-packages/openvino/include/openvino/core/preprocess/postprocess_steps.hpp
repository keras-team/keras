// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/core/core_visibility.hpp"
#include "openvino/core/layout.hpp"
#include "openvino/core/type/element_type.hpp"

namespace ov {

class Node;

namespace preprocess {

/// \brief Postprocessing steps. Each step typically intends adding of some operation to output parameter
/// User application can specify sequence of postprocessing steps in a builder-like manner
/// \code{.cpp}
///     auto proc = PrePostProcessor(function);
///     proc.output().postprocess().convert_element_type(element::u8);
///     function = proc.build();
/// \endcode
class OPENVINO_API PostProcessSteps final {
    class PostProcessStepsImpl;
    std::unique_ptr<PostProcessStepsImpl> m_impl;
    friend class OutputInfo;

    /// \brief Default empty internal constructor
    PostProcessSteps();

public:
    /// \brief Default destructor
    ~PostProcessSteps();

    /// \brief Add convert element type post-process operation
    ///
    /// \param type Desired type of output. If not specified, type will be obtained from 'tensor' output information
    ///
    /// \return Reference to 'this' to allow chaining with other calls in a builder-like manner
    PostProcessSteps& convert_element_type(const ov::element::Type& type = {});

    /// \brief Add 'convert layout' operation to specified layout.
    ///
    /// \param dst_layout New layout after conversion. If not specified - destination layout is obtained from
    /// appropriate tensor output properties.
    ///
    /// \return Reference to 'this' to allow chaining with other calls in a builder-like manner.
    ///
    /// Adds appropriate 'transpose' operation between model layout and user's desired layout.
    /// Current implementation requires source and destination layout to have same number of dimensions
    ///
    /// Example: when model data has output in 'NCHW' layout ([1, 3, 224, 224]) but user needs
    /// interleaved output image ('NHWC', [1, 224, 224, 3]). Post-processing may look like this:
    ///
    /// \code{.cpp}
    ///
    /// auto proc = PrePostProcessor(function);
    /// proc.output().model(OutputTensorInfo().set_layout("NCHW"); // model output is NCHW
    /// proc.output().postprocess().convert_layout("NHWC"); // User needs output as NHWC
    /// \endcode
    PostProcessSteps& convert_layout(const Layout& dst_layout = {});

    /// \brief Add convert layout operation by direct specification of transposed dimensions.
    ///
    /// \param dims Dimensions array specifying places for new axis. If not empty, array size (N) must match to input
    /// shape rank. Array values shall contain all values from 0 to N-1. If empty, no actual conversion will be added.
    ///
    /// \return Reference to 'this' to allow chaining with other calls in a builder-like manner.
    ///
    /// Example: model produces output with shape [1, 3, 480, 640] and user's needs
    /// interleaved output image [1, 480, 640, 3]. Post-processing may look like this:
    ///
    /// \code{.cpp} auto proc = PrePostProcessor(function);
    /// proc.output().postprocess().convert_layout({0, 2, 3, 1});
    /// function = proc.build();
    /// \endcode
    PostProcessSteps& convert_layout(const std::vector<uint64_t>& dims);

    /// \brief Signature for custom postprocessing operation. Custom postprocessing operation takes one output node and
    /// produces one output node. For more advanced cases, client's code can use transformation passes over ov::Model
    /// directly
    ///
    /// \param node Output node for custom post-processing operation
    ///
    /// \return New node after applying custom post-processing operation
    using CustomPostprocessOp = std::function<ov::Output<ov::Node>(const ov::Output<ov::Node>& node)>;

    /// \brief Add custom post-process operation.
    /// Client application can specify callback function for custom action
    ///
    /// \param postprocess_cb Client's custom postprocess operation.
    ///
    /// \return Reference to 'this' to allow chaining with other calls in a builder-like manner
    PostProcessSteps& custom(const CustomPostprocessOp& postprocess_cb);

    /// \brief Converts color format for user's output tensor. Requires destinantion color format to be specified by
    /// OutputTensorInfo::set_color_format.
    ///
    /// \param dst_format Destination color format of input image
    ///
    /// \return Reference to 'this' to allow chaining with other calls in a builder-like manner
    PostProcessSteps& convert_color(const ov::preprocess::ColorFormat& dst_format);
};

}  // namespace preprocess
}  // namespace ov
