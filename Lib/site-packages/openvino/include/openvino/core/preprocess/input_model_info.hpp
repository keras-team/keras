// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/core/core_visibility.hpp"
#include "openvino/core/layout.hpp"
#include "openvino/core/type/element_type.hpp"

namespace ov {
namespace preprocess {

/// \brief Information about model's input tensor. If all information is already included to loaded model, this info
/// may not be needed. However it can be set to specify additional information about model, like 'layout'.
///
/// Example of usage of model 'layout':
/// Support model has input parameter with shape {1, 3, 224, 224} and user needs to resize input image to model's
/// dimensions. It can be done like this
///
/// \code{.cpp}
/// <model has input parameter with shape {1, 3, 224, 224}>
/// auto proc = PrePostProcessor(function);
/// proc.input().preprocess().resize(ResizeAlgorithm::RESIZE_LINEAR);
/// proc.input().model().set_layout("NCHW");
/// \endcode
class OPENVINO_API InputModelInfo final {
    class InputModelInfoImpl;
    std::unique_ptr<InputModelInfoImpl> m_impl;
    friend class InputInfo;

    /// \brief Default empty constructor
    InputModelInfo();

public:
    /// \brief Default destructor
    ~InputModelInfo();

    /// \brief Set layout for model's input tensor
    /// This version allows chaining for Lvalue objects
    ///
    /// \param layout Layout for model's input tensor.
    ///
    /// \return Reference to 'this' to allow chaining with other calls in a builder-like manner
    InputModelInfo& set_layout(const ov::Layout& layout);
};

}  // namespace preprocess
}  // namespace ov
