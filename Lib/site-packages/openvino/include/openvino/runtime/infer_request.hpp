// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @brief A header file that provides InferRequest.
 *
 * @file openvino/runtime/infer_request.hpp
 */
#pragma once

#include <map>
#include <memory>
#include <string>

#include "openvino/core/node_output.hpp"
#include "openvino/runtime/common.hpp"
#include "openvino/runtime/profiling_info.hpp"
#include "openvino/runtime/tensor.hpp"
#include "openvino/runtime/variable_state.hpp"

namespace ov {

class CompiledModel;
class IAsyncInferRequest;

/**
 * @brief This is a class of infer request that can be run in asynchronous or synchronous manners.
 * @ingroup ov_runtime_cpp_api
 */
class OPENVINO_RUNTIME_API InferRequest {
    std::shared_ptr<ov::IAsyncInferRequest> _impl;
    std::shared_ptr<void> _so;

    /**
     * @brief Constructs InferRequest from the initialized std::shared_ptr.
     * @param impl Initialized shared pointer.
     * @param so Plugin to use. This is required to ensure that InferRequest can work properly even if a plugin object
     * is destroyed.
     */
    InferRequest(const std::shared_ptr<ov::IAsyncInferRequest>& impl, const std::shared_ptr<void>& so);
    friend class ov::CompiledModel;

public:
    /**
     * @brief Default constructor.
     */
    InferRequest() = default;

    /**
     * @brief Default copy constructor.
     * @param other Another InferRequest object.
     */
    InferRequest(const InferRequest& other) = default;

    /**
     * @brief Default copy assignment operator.
     * @param other Another InferRequest object.
     * @return Reference to the current object.
     */
    InferRequest& operator=(const InferRequest& other) = default;

    /**
     * @brief Default move constructor.
     * @param other Another InferRequest object.
     */
    InferRequest(InferRequest&& other) = default;

    /**
     * @brief Default move assignment operator.
     * @param other Another InferRequest object.
     * @return Reference to the current object.
     */
    InferRequest& operator=(InferRequest&& other) = default;

    /**
     * @brief Destructor that preserves unloading order of implementation object and reference to the library.
     * @note To preserve destruction order inside the default generated assignment operator, `_impl` is stored before
     *       `_so`. Use the destructor to remove implementation object before referencing to the library explicitly.
     */
    ~InferRequest();

    /**
     * @brief Sets an input/output tensor to infer on.
     *
     * @param tensor_name Name of the input or output tensor.
     * @param tensor Reference to the tensor. The element_type and shape of the tensor must match
     * the model's input/output element_type and size.
     */
    void set_tensor(const std::string& tensor_name, const Tensor& tensor);

    /**
     * @brief Sets an input/output tensor to infer.
     * @param port Port of the input or output tensor. Use the following methods to get the ports:
     * - ov::Model::input()
     * - ov::Model::inputs()
     * - ov::Model::outputs()
     * - ov::Model::outputs()
     * - ov::CompiledModel::input()
     * - ov::CompiledModel::inputs()
     * - ov::CompiledModel::outputs()
     * - ov::CompiledModel::outputs()
     * @param tensor Reference to a tensor. The element_type and shape of a tensor must match
     * the model's input/output element_type and size.
     */
    void set_tensor(const ov::Output<const ov::Node>& port, const Tensor& tensor);

    /**
     * @brief Sets an input/output tensor to infer.
     * @param port Port of the input or output tensor. Use the following methods to get the ports:
     * - ov::Model::input()
     * - ov::Model::inputs()
     * - ov::Model::outputs()
     * - ov::Model::outputs()
     * - ov::CompiledModel::input()
     * - ov::CompiledModel::inputs()
     * - ov::CompiledModel::outputs()
     * - ov::CompiledModel::outputs()
     * @param tensor Reference to a tensor. The element_type and shape of a tensor must match
     * the model's input/output element_type and size.
     */
    void set_tensor(const ov::Output<ov::Node>& port, const Tensor& tensor);

    /**
     * @brief Sets a batch of tensors for input data to infer by tensor name.
     * Model input must have batch dimension, and the number of @p tensors must match the batch size.
     * The current version supports setting tensors to model inputs only. If @p tensor_name is associated
     * with output (or any other non-input node), an exception is thrown.
     *
     * @param tensor_name Name of the input tensor.
     * @param tensors Input tensors for batched infer request. The type of each tensor must match the model
     * input element type and shape (except batch dimension). Total size of tensors must match the input size.
     */
    void set_tensors(const std::string& tensor_name, const std::vector<Tensor>& tensors);

    /**
     * @brief Sets a batch of tensors for input data to infer by input port.
     * Model input must have batch dimension, and the number of @p tensors must match the batch size.
     * The current version supports setting tensors to model inputs only. If @p port is associated
     * with output (or any other non-input node), an exception is thrown.
     *
     * @param port Port of the input tensor.
     * @param tensors Input tensors for batched infer request. The type of each tensor must match the model
     * input element type and shape (except batch dimension). Total size of tensors must match the input size.
     */
    void set_tensors(const ov::Output<const ov::Node>& port, const std::vector<Tensor>& tensors);

    /**
     * @brief Sets an input tensor to infer.
     *
     * @param idx Index of the input tensor. If @p idx is greater than the number of model inputs, an exception
     * is thrown.
     * @param tensor Reference to the tensor. The element_type and shape of the tensor must match
     * the model's input/output element_type and size.
     */
    void set_input_tensor(size_t idx, const Tensor& tensor);

    /**
     * @brief Sets an input tensor to infer models with single input.
     * @note If model has several inputs, an exception is thrown.
     * @param tensor Reference to the input tensor.
     */
    void set_input_tensor(const Tensor& tensor);

    /**
     * @brief Sets a batch of tensors for single input data.
     * Model input must have batch dimension, and the number of @p tensors must match the batch size.
     *
     * @param tensors Input tensors for batched infer request. The type of each tensor must match the model
     * input element type and shape (except batch dimension). Total size of tensors must match the input size.
     */
    void set_input_tensors(const std::vector<Tensor>& tensors);

    /**
     * @brief Sets a batch of tensors for input data to infer by input name.
     * Model input must have batch dimension, and number of @p tensors must match the batch size.
     *
     * @param idx Name of the input tensor.
     * @param tensors Input tensors for batched infer request. The type of each tensor must match the model
     * input element type and shape (except batch dimension). Total size of tensors must match the input size.
     */
    void set_input_tensors(size_t idx, const std::vector<Tensor>& tensors);

    /**
     * @brief Sets an output tensor to infer.
     * @note Index of the input preserved accross ov::Model, ov::CompiledModel, and ov::InferRequest.
     * @param idx Index of the output tensor.
     * @param tensor Reference to the output tensor. The type of the tensor must match the model output element type and
     * shape.
     */
    void set_output_tensor(size_t idx, const Tensor& tensor);

    /**
     * @brief Sets an output tensor to infer models with single output.
     * @note If model has several outputs, an exception is thrown.
     * @param tensor Reference to the output tensor.
     */
    void set_output_tensor(const Tensor& tensor);

    /**
     * @brief Gets an input/output tensor for inference by tensor name.
     * @param tensor_name Name of a tensor to get.
     * @return The tensor with name @p tensor_name. If the tensor is not found, an exception is thrown.
     */
    Tensor get_tensor(const std::string& tensor_name);

    /**
     * @brief Gets an input/output tensor for inference.
     * @note If the tensor with the specified @p port is not found, an exception is thrown.
     * @param port Port of the tensor to get.
     * @return Tensor for the port @p port.
     */
    Tensor get_tensor(const ov::Output<const ov::Node>& port);

    /**
     * @brief Gets an input/output tensor for inference.
     * @note If the tensor with the specified @p port is not found, an exception is thrown.
     * @param port Port of the tensor to get.
     * @return Tensor for the port @p port.
     */
    Tensor get_tensor(const ov::Output<ov::Node>& port);

    /**
     * @brief Gets an input tensor for inference.
     *
     * @param idx Index of the tensor to get.
     * @return Tensor with the input index @p idx. If the tensor with the specified @p idx is not found, an exception
     * is thrown.
     */
    Tensor get_input_tensor(size_t idx);

    /**
     * @brief Gets an input tensor for inference.
     *
     * @return The input tensor for the model. If model has several inputs, an exception is thrown.
     */
    Tensor get_input_tensor();

    /**
     * @brief Gets an output tensor for inference.
     *
     * @param idx Index of the tensor to get.
     * @return Tensor with the output index @p idx. If the tensor with the specified @p idx is not found, an exception
     * is thrown.
     */
    Tensor get_output_tensor(size_t idx);

    /**
     * @brief Gets an output tensor for inference.
     *
     * @return Output tensor for the model. If model has several outputs, an exception is thrown.
     */
    Tensor get_output_tensor();

    /**
     * @brief Infers specified input(s) in synchronous mode.
     * @note It blocks all methods of InferRequest while request is ongoing (running or waiting in a queue).
     *       Calling any method leads to throwing the ov::Busy exception.
     */
    void infer();

    /**
     * @brief Cancels inference request.
     */
    void cancel();

    /**
     * @brief Queries performance measures per layer to identify the most time consuming operation.
     * @note Not all plugins provide meaningful data.
     * @return Vector of profiling information for operations in a model.
     */
    std::vector<ProfilingInfo> get_profiling_info() const;

    /**
     * @brief Starts inference of specified input(s) in asynchronous mode.
     * @note It returns immediately. Inference starts also immediately.
     *       Calling any method while the request in a running state leads to throwing the ov::Busy exception.
     */
    void start_async();

    /**
     * @brief Waits for the result to become available. Blocks until the result
     * becomes available.
     */
    void wait();

    /**
     * @brief Waits for the result to become available. Blocks until the specified timeout has elapsed or the result
     * becomes available, whichever comes first.
     *
     * @param timeout Maximum duration, in milliseconds, to block for.
     * @return True if inference request is ready and false, otherwise.
     */
    bool wait_for(const std::chrono::milliseconds timeout);

    /**
     * @brief Sets a callback std::function that is called on success or failure of an asynchronous request.
     * @param callback callback object which will be called on when inference finish.
     * @warning Do not capture strong references to OpenVINO runtime objects into callback.
     * Following objects should not be captured like:
     *  - ov::InferRequest
     *  - ov::ExecutableNetwork
     *  - ov::Core
     * As specified objects implement shared reference concept do not capture this objects by value.
     * It can lead to memory leaks or undefined behaviour!
     * Try to use weak references or pointers.
     */
    void set_callback(std::function<void(std::exception_ptr)> callback);

    /**
     * @brief Gets state control interface for the given infer request.
     *
     * State control essential for recurrent models.
     * @return Vector of Variable State objects.
     */
    std::vector<VariableState> query_state();

    /**
     * @brief Resets all internal variable states for relevant infer request to a value specified as
     * default for the corresponding `ReadValue` node
     */
    void reset_state();

    /**
     * @brief Returns a compiled model that creates this inference request.
     * @return Compiled model object.
     */
    CompiledModel get_compiled_model();

    /**
     * @brief Checks if the current InferRequest object is not initialized.
     * @return True if the current InferRequest object is not initialized; false, otherwise.
     */
    bool operator!() const noexcept;

    /**
     * @brief Checks if the current InferRequest object is initialized.
     * @return True if the current InferRequest object is initialized; false, otherwise.
     */
    explicit operator bool() const noexcept;

    /**
     * @brief Compares whether this request wraps the same impl underneath.
     * @param other Another inference request.
     * @return True if the current InferRequest object does not wrap the same impl as the operator's arg.
     */
    bool operator!=(const InferRequest& other) const noexcept;

    /**
     * @brief Compares whether this request wraps the same impl underneath.
     * @param other Another inference request.
     * @return True if the current InferRequest object wraps the same impl as the operator's arg.
     */
    bool operator==(const InferRequest& other) const noexcept;
};

}  // namespace ov
