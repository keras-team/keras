// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/core/core_visibility.hpp"
#include "openvino/core/layout.hpp"
#include "openvino/core/type/element_type.hpp"

namespace ov {
namespace preprocess {

/// \brief Information about user's desired output tensor. By default, it will be initialized to same data
/// (type/shape/etc) as model's output parameter. User application can override particular parameters (like
/// 'element_type') according to application's data and specify appropriate conversions in post-processing steps
///
/// \code{.cpp}
///     auto proc = PrePostProcessor(function);
///     auto& output = proc.output();
///     output.postprocess().<add steps + conversion to user's output element type>;
///     output.tensor().set_element_type(ov::element::u8);
///     function = proc.build();
/// \endcode
class OPENVINO_API OutputTensorInfo final {
    class OutputTensorInfoImpl;
    std::unique_ptr<OutputTensorInfoImpl> m_impl;
    friend class OutputInfo;

    /// \brief Default empty internal constructor
    OutputTensorInfo();

public:
    /// \brief Default destructor
    ~OutputTensorInfo();

    /// \brief Set element type for user's desired output tensor.
    ///
    /// \param type Element type for user's output tensor.
    ///
    /// \return Reference to 'this' to allow chaining with other calls in a builder-like manner.
    OutputTensorInfo& set_element_type(const ov::element::Type& type);

    /// \brief Set layout for user's output tensor.
    ///
    /// \param layout Layout for user's output tensor.
    ///
    /// \return Reference to 'this' to allow chaining with other calls in a builder-like manner
    OutputTensorInfo& set_layout(const ov::Layout& layout);
};

}  // namespace preprocess
}  // namespace ov
