// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/core/core_visibility.hpp"
#include "openvino/core/preprocess/input_model_info.hpp"
#include "openvino/core/preprocess/input_tensor_info.hpp"
#include "openvino/core/preprocess/preprocess_steps.hpp"

namespace ov {
namespace preprocess {

/// \brief Class holding preprocessing information for one input
/// From preprocessing pipeline perspective, each input can be represented as:
///    - User's input parameter info (InputInfo::tensor)
///    - Preprocessing steps applied to user's input (InputInfo::preprocess)
///    - Model's input info, which is a final input's info after preprocessing (InputInfo::model)
///
class OPENVINO_API InputInfo final {
    struct InputInfoImpl;
    std::unique_ptr<InputInfoImpl> m_impl;
    friend class PrePostProcessor;

    /// \brief Empty constructor for internal usage
    InputInfo();

public:
    /// \brief Move constructor
    InputInfo(InputInfo&& other) noexcept;

    /// \brief Move assignment operator
    InputInfo& operator=(InputInfo&& other) noexcept;

    /// \brief Default destructor
    ~InputInfo();

    /// \brief Get current input tensor information with ability to change specific data
    ///
    /// \return Reference to current input tensor structure
    InputTensorInfo& tensor();

    /// \brief Get current input preprocess information with ability to add more preprocessing steps
    ///
    /// \return Reference to current preprocess steps structure
    PreProcessSteps& preprocess();

    /// \brief Get current input model information with ability to change original model's input data
    ///
    /// \return Reference to current model's input information structure
    InputModelInfo& model();
};

}  // namespace preprocess
}  // namespace ov
