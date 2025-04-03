// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/core/core_visibility.hpp"
#include "openvino/core/preprocess/output_model_info.hpp"
#include "openvino/core/preprocess/output_tensor_info.hpp"
#include "openvino/core/preprocess/postprocess_steps.hpp"

namespace ov {
namespace preprocess {

/// \brief Class holding postprocessing information for one output
/// From postprocessing pipeline perspective, each output can be represented as:
///    - Model's output info,  (OutputInfo::model)
///    - Postprocessing steps applied to user's input (OutputInfo::postprocess)
///    - User's desired output parameter information, which is a final one after preprocessing (OutputInfo::tensor)
class OPENVINO_API OutputInfo final {
    struct OutputInfoImpl;
    std::unique_ptr<OutputInfoImpl> m_impl;
    friend class PrePostProcessor;

    /// \brief Empty internal default constructor
    OutputInfo();

public:
    /// \brief Move constructor
    OutputInfo(OutputInfo&& other) noexcept;

    /// \brief Move assignment operator
    OutputInfo& operator=(OutputInfo&& other) noexcept;

    /// \brief Default destructor
    ~OutputInfo();

    /// \brief Get current output model information with ability to change original model's output data
    ///
    /// \return Reference to current model's output information structure
    OutputModelInfo& model();

    /// \brief Get current output post-process information with ability to add more post-processing steps
    ///
    /// \return Reference to current preprocess steps structure
    PostProcessSteps& postprocess();

    /// \brief Get current output tensor information with ability to change specific data
    ///
    /// \return Reference to current output tensor structure
    OutputTensorInfo& tensor();
};

}  // namespace preprocess
}  // namespace ov
