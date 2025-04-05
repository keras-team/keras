// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/core/core_visibility.hpp"
#include "openvino/core/preprocess/input_info.hpp"
#include "openvino/core/preprocess/output_info.hpp"

namespace ov {

class Model;

namespace preprocess {

/// \brief Main class for adding pre- and post- processing steps to existing ov::Model
///
/// This is a helper class for writing easy pre- and post- processing operations on ov::Model object assuming that
/// any preprocess operation takes one input and produces one output.
///
/// For advanced preprocessing scenarios, like combining several functions with multiple inputs/outputs into one,
/// client's code can use transformation passes over ov::Model
///
/// \ingroup ov_model_cpp_api
class OPENVINO_API PrePostProcessor final {
    struct PrePostProcessorImpl;
    std::unique_ptr<PrePostProcessorImpl> m_impl;

public:
    /// \brief Default constructor
    ///
    /// \param function Existing function representing loaded model
    explicit PrePostProcessor(const std::shared_ptr<Model>& function);

    /// \brief Default move constructor
    PrePostProcessor(PrePostProcessor&&) noexcept;

    /// \brief Default move assignment operator
    PrePostProcessor& operator=(PrePostProcessor&&) noexcept;

    /// \brief Default destructor
    ~PrePostProcessor();

    /// \brief Gets input pre-processing data structure. Should be used only if model/function has only one input
    /// Using returned structure application's code is able to set user's tensor data (e.g layout), preprocess steps,
    /// target model's data
    ///
    /// \return Reference to model's input information structure
    InputInfo& input();

    /// \brief Gets input pre-processing data structure for input identified by it's tensor name
    ///
    /// \param tensor_name Tensor name of specific input. Throws if tensor name is not associated with any input in a
    /// model
    ///
    /// \return Reference to model's input information structure
    InputInfo& input(const std::string& tensor_name);

    /// \brief Gets input pre-processing data structure for input identified by it's order in a model
    ///
    /// \param input_index Input index of specific input. Throws if input index is out of range for associated function
    ///
    /// \return Reference to model's input information structure
    InputInfo& input(size_t input_index);

    /// \brief Gets output post-processing data structure. Should be used only if model/function has only one output
    /// Using returned structure application's code is able to set model's output data, post-process steps, user's
    /// tensor data (e.g layout)
    ///
    /// \return Reference to model's output information structure
    OutputInfo& output();

    /// \brief Gets output post-processing data structure for output identified by it's tensor name
    ///
    /// \param tensor_name Tensor name of specific output. Throws if tensor name is not associated with any input in a
    /// model
    ///
    /// \return Reference to model's output information structure
    OutputInfo& output(const std::string& tensor_name);

    /// \brief Gets output post-processing data structure for output identified by it's order in a model
    ///
    /// \param output_index Output index of specific output. Throws if output index is out of range for associated
    /// function
    ///
    /// \return Reference to model's output information structure
    OutputInfo& output(size_t output_index);

    /// \brief Adds pre/post-processing operations to function passed in constructor
    ///
    /// \return Function with added pre/post-processing operations
    std::shared_ptr<Model> build();

private:
    friend OPENVINO_API std::ostream& operator<<(std::ostream& str, const PrePostProcessor& prePostProcessor);
    void dump(std::ostream&) const;
};

/// \brief Inserts a human-readable representation of a PrePostProcessors into an output stream. The output to the
/// stream is in "informal" notation and can be used for debugging purposes
///
/// \param str The output stream targeted for insertion.
///
/// \param prePostProcessor The shape to be inserted into output stream.
///
/// \return A reference to same output stream after insertion.
OPENVINO_API std::ostream& operator<<(std::ostream& str, const PrePostProcessor& prePostProcessor);

}  // namespace preprocess
}  // namespace ov
