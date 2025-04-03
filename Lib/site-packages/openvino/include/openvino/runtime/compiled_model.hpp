// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @brief A header file that provides the CompiledModel class.
 *
 * @file openvino/runtime/compiled_model.hpp
 */

#pragma once

#include <map>
#include <memory>
#include <ostream>
#include <string>
#include <vector>

#include "openvino/core/model.hpp"
#include "openvino/runtime/infer_request.hpp"
#include "openvino/runtime/properties.hpp"
#include "openvino/runtime/remote_context.hpp"

namespace ov {

class Core;
class InferRequest;
class ICompiledModel;

/**
 * @brief This class represents a compiled model.
 * @ingroup ov_runtime_cpp_api
 * A model is compiled by a specific device by applying multiple optimization
 * transformations, then mapping to compute kernels.
 */
class OPENVINO_RUNTIME_API CompiledModel {
    std::shared_ptr<ov::ICompiledModel> _impl;
    std::shared_ptr<void> _so;

    /**
     * @brief Constructs CompiledModel from the initialized std::shared_ptr.
     * @param impl Initialized shared pointer.
     * @param so Plugin to use. This parameter is required to ensure that CompiledModel can work properly even if a
     * plugin object is destroyed.
     */
    CompiledModel(const std::shared_ptr<ov::ICompiledModel>& impl, const std::shared_ptr<void>& so);
    friend class ov::Core;
    friend class ov::InferRequest;

public:
    /**
     * @brief Default constructor.
     */
    CompiledModel() = default;

    /**
     * @brief Destructor that preserves unloading order of an implementation object and reference to library.
     */
    ~CompiledModel();

    /**
     * @brief Gets runtime model information from a device.
     * This object represents an internal device-specific model that is optimized for a particular
     * accelerator. It contains device-specific nodes, runtime information and can be used only
     * to understand how the source model is optimized and which kernels, element types, and layouts
     * are selected for optimal inference.
     *
     * @return A model containing Executable Graph Info.
     */
    std::shared_ptr<const Model> get_runtime_model() const;

    /**
     * @brief Gets all inputs of a compiled model.
     * Inputs are represented as a vector of outputs of the ov::op::v0::Parameter operations.
     * They contain information about input tensors such as tensor shape, names, and element type.
     * @return std::vector of model inputs.
     */
    const std::vector<ov::Output<const ov::Node>>& inputs() const;

    /**
     * @brief Gets a single input of a compiled model.
     * The input is represented as an output of the ov::op::v0::Parameter operation.
     * The input contains information about input tensor such as tensor shape, names, and element type.
     * @return Compiled model input.
     * @note If a model has more than one input, this method throws ov::Exception.
     */
    const ov::Output<const ov::Node>& input() const;

    /**
     * @brief Gets input of a compiled model identified by @p i.
     * The input contains information about input tensor such as tensor shape, names, and element type.
     * @param i Index of input.
     * @return Compiled model input.
     * @note The method throws ov::Exception if input with the specified index @p i is not found.
     */
    const ov::Output<const ov::Node>& input(size_t i) const;

    /**
     * @brief Gets input of a compiled model identified by @p tensor_name.
     * The input contains information about input tensor such as tensor shape, names, and element type.
     * @param tensor_name The input tensor name.
     * @return Compiled model input.
     * @note The method throws ov::Exception if input with the specified tensor name @p tensor_name is not found.
     */
    const ov::Output<const ov::Node>& input(const std::string& tensor_name) const;

    /**
     * @brief Get all outputs of a compiled model.
     * Outputs are represented as a vector of output from the ov::op::v0::Result operations.
     * Outputs contain information about output tensors such as tensor shape, names, and element type.
     * @return std::vector of model outputs.
     */
    const std::vector<ov::Output<const ov::Node>>& outputs() const;

    /**
     * @brief Gets a single output of a compiled model.
     * The output is represented as an output from the ov::op::v0::Result operation.
     * The output contains information about output tensor such as tensor shape, names, and element type.
     * @return Compiled model output.
     * @note If a model has more than one output, this method throws ov::Exception.
     */
    const ov::Output<const ov::Node>& output() const;

    /**
     * @brief Gets output of a compiled model identified by @p index.
     * The output contains information about output tensor such as tensor shape, names, and element type.
     * @param i Index of input.
     * @return Compiled model output.
     * @note The method throws ov::Exception if output with the specified index @p index is not found.
     */
    const ov::Output<const ov::Node>& output(size_t i) const;

    /**
     * @brief Gets output of a compiled model identified by @p tensor_name.
     * The output contains information about output tensor such as tensor shape, names, and element type.
     * @param tensor_name Output tensor name.
     * @return Compiled model output.
     * @note The method throws ov::Exception if output with the specified tensor name @p tensor_name is not found.
     */
    const ov::Output<const ov::Node>& output(const std::string& tensor_name) const;

    /**
     * @brief Creates an inference request object used to infer the compiled model.
     * The created request has allocated input and output tensors (which can be changed later).
     *
     * @return InferRequest object
     */
    InferRequest create_infer_request();

    /**
     * @brief Exports the current compiled model to an output stream `std::ostream`.
     * The exported model can also be imported via the ov::Core::import_model method.
     * @see ov::Core::import_model
     * @param model_stream Output stream to store the model to.
     */
    void export_model(std::ostream& model_stream);

    /**
     * @brief Sets properties for the current compiled model.
     *
     * @param properties Map of pairs: (property name, property value).
     */
    void set_property(const AnyMap& properties);

    /**
     * @brief Sets properties for the current compiled model.
     *
     * @tparam Properties Should be the pack of `std::pair<std::string, ov::Any>` types.
     * @param properties Optional pack of pairs: (property name, property value).
     */
    template <typename... Properties>
    util::EnableIfAllStringAny<void, Properties...> set_property(Properties&&... properties) {
        set_property(AnyMap{std::forward<Properties>(properties)...});
    }

    /** @brief Gets properties for current compiled model
     *
     * The method is responsible for extracting information
     * that affects compiled model inference. The list of supported configuration values can be extracted via
     * CompiledModel::get_property with the ov::supported_properties key, but some of these keys cannot be changed
     * dynamically, for example, ov::device::id cannot be changed if a compiled model has already been compiled for a
     * particular device.
     *
     * @param name Property key, can be found in openvino/runtime/properties.hpp.
     * @return Property value.
     */
    Any get_property(const std::string& name) const;

    /**
     * @brief Gets properties related to device behaviour.
     *
     * The method extracts information that can be set via the set_property method.
     *
     * @tparam T Type of a returned value.
     * @param property  Property  object.
     * @return Value of property.
     */
    template <typename T, PropertyMutability mutability>
    T get_property(const ov::Property<T, mutability>& property) const {
        return get_property(property.name()).template as<T>();
    }

    /**
     * @brief Release intermediate memory.
     *
     * This method forces the Compiled model to release memory allocated for intermediate structures, e.g. caches,
     * tensors, temporal buffers etc., when possible
     *
     */
    void release_memory();

    /**
     * @brief Returns pointer to device-specific shared context
     * on a remote accelerator device that was used to create this CompiledModel.
     * @return A context.
     */
    RemoteContext get_context() const;

    /**
     * @brief Checks if the current CompiledModel object is not initialized.
     * @return `true` if the current CompiledModel object is not initialized; `false`, otherwise.
     */
    bool operator!() const noexcept;

    /**
     * @brief Checks if the current CompiledModel object is initialized.
     * @return `true` if the current CompiledModel object is initialized; `false`, otherwise.
     */
    explicit operator bool() const noexcept;
};

}  // namespace ov
