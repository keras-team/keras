// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/core/core_visibility.hpp"
#include "openvino/core/layout.hpp"
#include "openvino/core/preprocess/color_format.hpp"

namespace ov {
namespace preprocess {

/// \brief Information about model's output tensor. If all information is already included to loaded model, this
/// info may not be needed. However it can be set to specify additional information about model's output, like 'layout'.
///
/// Example of usage of model's 'layout':
/// Suppose model has output result with shape {1, 3, 224, 224} and `NHWC` layout. User may need to transpose
/// output picture to interleaved format {1, 224, 224, 3}. This can be done with the following code
///
/// \code{.cpp}
///     <model has output result with shape {1, 3, 224, 224}>
///     auto proc = PrePostProcessor(function);
///     proc.output().model().set_layout("NCHW");
///     proc.output().postprocess().convert_layout("NHWC");
///     function = proc.build();
/// \endcode
class OPENVINO_API OutputModelInfo final {
    class OutputModelInfoImpl;
    std::unique_ptr<OutputModelInfoImpl> m_impl;
    friend class OutputInfo;

    /// \brief Default internal empty constructor
    OutputModelInfo();

public:
    /// \brief Default destructor
    ~OutputModelInfo();

    /// \brief Set layout for model's output tensor
    ///
    /// \param layout Layout for model's output tensor.
    ///
    /// \return Reference to 'this' to allow chaining with other calls in a builder-like manner
    OutputModelInfo& set_layout(const ov::Layout& layout);

    /// \brief Set color format for model's output tensor
    ///
    /// \param format Color format for model's output tensor.
    ///
    /// \param sub_names Optional list of sub-names, not used, placeholder for future.
    ///
    /// \return Reference to 'this' to allow chaining with other calls in a builder-like manner
    OutputModelInfo& set_color_format(const ov::preprocess::ColorFormat& format,
                                      const std::vector<std::string>& sub_names = {});
};

}  // namespace preprocess
}  // namespace ov
