// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @brief This is a header file for the ov_infer_request C API, which is a C wrapper for ov::InferRequest class
 * This is a class of infer request that can be run in asynchronous or synchronous manners.
 * @file ov_infer_request.h
 */

#pragma once

#include "openvino/c/ov_common.h"
#include "openvino/c/ov_node.h"
#include "openvino/c/ov_tensor.h"

/**
 * @struct ov_infer_request_t
 * @ingroup ov_infer_request_c_api
 * @brief type define ov_infer_request_t from ov_infer_request
 */
typedef struct ov_infer_request ov_infer_request_t;

/**
 * @struct ov_callback_t
 * @ingroup ov_infer_request_c_api
 * @brief Completion callback definition about the function and args
 */
typedef struct {
    void(OPENVINO_C_API_CALLBACK* callback_func)(void* args);  //!< The callback func
    void* args;                                                //!< The args of callback func
} ov_callback_t;

/**
 * @struct ov_ProfilingInfo_t
 * @ingroup ov_infer_request_c_api
 * @brief Store profiling info data
 */
typedef struct {
    enum Status {           //!< Defines the general status of a node.
        NOT_RUN,            //!< A node is not executed.
        OPTIMIZED_OUT,      //!< A node is optimized out during graph optimization phase.
        EXECUTED            //!< A node is executed.
    } status;               //!< status
    int64_t real_time;      //!< The absolute time, in microseconds, that the node ran (in total).
    int64_t cpu_time;       //!< The net host CPU time that the node ran.
    const char* node_name;  //!< Name of a node.
    const char* exec_type;  //!< Execution type of a unit.
    const char* node_type;  //!< Node type.
} ov_profiling_info_t;

/**
 * @struct ov_profiling_info_list_t
 * @ingroup ov_infer_request_c_api
 * @brief A list of profiling info data
 */
typedef struct {
    ov_profiling_info_t* profiling_infos;  //!< The list of ov_profilling_info_t
    size_t size;                           //!< The list size
} ov_profiling_info_list_t;

/**
 * @brief Set an input/output tensor to infer on by the name of tensor.
 * @ingroup ov_infer_request_c_api
 * @param infer_request A pointer to the ov_infer_request_t.
 * @param tensor_name  Name of the input or output tensor.
 * @param tensor Reference to the tensor.
 * @return Status code of the operation: OK(0) for success.
 */
OPENVINO_C_API(ov_status_e)
ov_infer_request_set_tensor(ov_infer_request_t* infer_request, const char* tensor_name, const ov_tensor_t* tensor);

/**
 * @brief Set an input/output tensor to infer request for the port.
 * @ingroup ov_infer_request_c_api
 * @param infer_request A pointer to the ov_infer_request_t.
 * @param port Port of the input or output tensor, which can be got by calling ov_model_t/ov_compiled_model_t interface.
 * @param tensor Reference to the tensor.
 * @return Status code of the operation: OK(0) for success.
 */
OPENVINO_C_API(ov_status_e)
ov_infer_request_set_tensor_by_port(ov_infer_request_t* infer_request,
                                    const ov_output_port_t* port,
                                    const ov_tensor_t* tensor);

/**
 * @brief Set an input/output tensor to infer request for the port.
 * @ingroup ov_infer_request_c_api
 * @param infer_request A pointer to the ov_infer_request_t.
 * @param port Const port of the input or output tensor, which can be got by call interface from
 * ov_model_t/ov_compiled_model_t.
 * @param tensor Reference to the tensor.
 * @return Status code of the operation: OK(0) for success.
 */
OPENVINO_C_API(ov_status_e)
ov_infer_request_set_tensor_by_const_port(ov_infer_request_t* infer_request,
                                          const ov_output_const_port_t* port,
                                          const ov_tensor_t* tensor);

/**
 * @brief Set an input tensor to infer on by the index of tensor.
 * @ingroup ov_infer_request_c_api
 * @param infer_request A pointer to the ov_infer_request_t.
 * @param idx Index of the input port. If @p idx is greater than the number of model inputs, an error will return.
 * @param tensor Reference to the tensor.
 * @return Status code of the operation: OK(0) for success.
 */
OPENVINO_C_API(ov_status_e)
ov_infer_request_set_input_tensor_by_index(ov_infer_request_t* infer_request,
                                           const size_t idx,
                                           const ov_tensor_t* tensor);

/**
 * @brief Set an input tensor for the model with single input to infer on.
 * @note If model has several inputs, an error will return.
 * @ingroup ov_infer_request_c_api
 * @param infer_request A pointer to the ov_infer_request_t.
 * @param tensor Reference to the tensor.
 * @return Status code of the operation: OK(0) for success.
 */
OPENVINO_C_API(ov_status_e)
ov_infer_request_set_input_tensor(ov_infer_request_t* infer_request, const ov_tensor_t* tensor);

/**
 * @brief Set an output tensor to infer by the index of output tensor.
 * @note Index of the output preserved accross ov_model_t, ov_compiled_model_t.
 * @ingroup ov_infer_request_c_api
 * @param infer_request A pointer to the ov_infer_request_t.
 * @param idx Index of the output tensor.
 * @param tensor Reference to the tensor.
 * @return Status code of the operation: OK(0) for success.
 */
OPENVINO_C_API(ov_status_e)
ov_infer_request_set_output_tensor_by_index(ov_infer_request_t* infer_request,
                                            const size_t idx,
                                            const ov_tensor_t* tensor);

/**
 * @brief Set an output tensor to infer models with single output.
 * @ingroup ov_infer_request_c_api
 * @param infer_request A pointer to the ov_infer_request_t.
 * @param tensor Reference to the tensor.
 * @return Status code of the operation: OK(0) for success.
 */
OPENVINO_C_API(ov_status_e)
ov_infer_request_set_output_tensor(ov_infer_request_t* infer_request, const ov_tensor_t* tensor);

/**
 * @brief Get an input/output tensor by the name of tensor.
 * @note If model has several outputs, an error will return.
 * @ingroup ov_infer_request_c_api
 * @param infer_request A pointer to the ov_infer_request_t.
 * @param tensor_name Name of the input or output tensor to get.
 * @param tensor Reference to the tensor.
 * @return Status code of the operation: OK(0) for success.
 */
OPENVINO_C_API(ov_status_e)
ov_infer_request_get_tensor(const ov_infer_request_t* infer_request, const char* tensor_name, ov_tensor_t** tensor);

/**
 * @brief Get an input/output tensor by const port.
 * @ingroup ov_infer_request_c_api
 * @param infer_request A pointer to the ov_infer_request_t.
 * @param port Port of the tensor to get. @p port is not found, an error will return.
 * @param tensor Reference to the tensor.
 * @return Status code of the operation: OK(0) for success.
 */
OPENVINO_C_API(ov_status_e)
ov_infer_request_get_tensor_by_const_port(const ov_infer_request_t* infer_request,
                                          const ov_output_const_port_t* port,
                                          ov_tensor_t** tensor);

/**
 * @brief Get an input/output tensor by port.
 * @ingroup ov_infer_request_c_api
 * @param infer_request A pointer to the ov_infer_request_t.
 * @param port Port of the tensor to get. @p port is not found, an error will return.
 * @param tensor Reference to the tensor.
 * @return Status code of the operation: OK(0) for success.
 */
OPENVINO_C_API(ov_status_e)
ov_infer_request_get_tensor_by_port(const ov_infer_request_t* infer_request,
                                    const ov_output_port_t* port,
                                    ov_tensor_t** tensor);

/**
 * @brief Get an input tensor by the index of input tensor.
 * @ingroup ov_infer_request_c_api
 * @param infer_request A pointer to the ov_infer_request_t.
 * @param idx Index of the tensor to get. @p idx. If the tensor with the specified @p idx is not found, an error will
 * return.
 * @param tensor Reference to the tensor.
 * @return Status code of the operation: OK(0) for success.
 */
OPENVINO_C_API(ov_status_e)
ov_infer_request_get_input_tensor_by_index(const ov_infer_request_t* infer_request,
                                           const size_t idx,
                                           ov_tensor_t** tensor);

/**
 * @brief Get an input tensor from the model with only one input tensor.
 * @ingroup ov_infer_request_c_api
 * @param infer_request A pointer to the ov_infer_request_t.
 * @param tensor Reference to the tensor.
 * @return Status code of the operation: OK(0) for success.
 */
OPENVINO_C_API(ov_status_e)
ov_infer_request_get_input_tensor(const ov_infer_request_t* infer_request, ov_tensor_t** tensor);

/**
 * @brief Get an output tensor by the index of output tensor.
 * @ingroup ov_infer_request_c_api
 * @param infer_request A pointer to the ov_infer_request_t.
 * @param idx Index of the tensor to get. @p idx. If the tensor with the specified @p idx is not found, an error will
 * return.
 * @param tensor Reference to the tensor.
 * @return Status code of the operation: OK(0) for success.
 */
OPENVINO_C_API(ov_status_e)
ov_infer_request_get_output_tensor_by_index(const ov_infer_request_t* infer_request,
                                            const size_t idx,
                                            ov_tensor_t** tensor);

/**
 * @brief Get an output tensor from the model with only one output tensor.
 * @note If model has several outputs, an error will return.
 * @ingroup ov_infer_request_c_api
 * @param infer_request A pointer to the ov_infer_request_t.
 * @param tensor Reference to the tensor.
 * @return Status code of the operation: OK(0) for success.
 */
OPENVINO_C_API(ov_status_e)
ov_infer_request_get_output_tensor(const ov_infer_request_t* infer_request, ov_tensor_t** tensor);

/**
 * @brief Infer specified input(s) in synchronous mode.
 * @ingroup ov_infer_request_c_api
 * @param infer_request A pointer to the ov_infer_request_t.
 * @return Status code of the operation: OK(0) for success.
 */
OPENVINO_C_API(ov_status_e)
ov_infer_request_infer(ov_infer_request_t* infer_request);

/**
 * @brief Cancel inference request.
 * @ingroup ov_infer_request_c_api
 * @param infer_request A pointer to the ov_infer_request_t.
 * @return Status code of the operation: OK(0) for success.
 */
OPENVINO_C_API(ov_status_e)
ov_infer_request_cancel(ov_infer_request_t* infer_request);

/**
 * @brief Start inference of specified input(s) in asynchronous mode.
 * @ingroup ov_infer_request_c_api
 * @param infer_request A pointer to the ov_infer_request_t.
 * @return Status code of the operation: OK(0) for success.
 */
OPENVINO_C_API(ov_status_e)
ov_infer_request_start_async(ov_infer_request_t* infer_request);

/**
 * @brief Wait for the result to become available.
 * @ingroup ov_infer_request_c_api
 * @param infer_request A pointer to the ov_infer_request_t.
 * @return Status code of the operation: OK(0) for success.
 */
OPENVINO_C_API(ov_status_e)
ov_infer_request_wait(ov_infer_request_t* infer_request);

/**
 * @brief Waits for the result to become available. Blocks until the specified timeout has elapsed or the result
 * becomes available, whichever comes first.
 * @ingroup ov_infer_request_c_api
 * @param infer_request A pointer to the ov_infer_request_t.
 * @param timeout Maximum duration, in milliseconds, to block for.
 * @return Status code of the operation: OK(0) for success.
 */
OPENVINO_C_API(ov_status_e)
ov_infer_request_wait_for(ov_infer_request_t* infer_request, const int64_t timeout);

/**
 * @brief Set callback function, which will be called when inference is done.
 * @ingroup ov_infer_request_c_api
 * @param infer_request A pointer to the ov_infer_request_t.
 * @param callback  A function to be called.
 * @return Status code of the operation: OK(0) for success.
 */
OPENVINO_C_API(ov_status_e)
ov_infer_request_set_callback(ov_infer_request_t* infer_request, const ov_callback_t* callback);

/**
 * @brief Release the memory allocated by ov_infer_request_t.
 * @ingroup ov_infer_request_c_api
 * @param infer_request A pointer to the ov_infer_request_t to free memory.
 */
OPENVINO_C_API(void)
ov_infer_request_free(ov_infer_request_t* infer_request);

/**
 * @brief Query performance measures per layer to identify the most time consuming operation.
 * @ingroup ov_infer_request_c_api
 * @param infer_request A pointer to the ov_infer_request_t.
 * @param profiling_infos  Vector of profiling information for operations in a model.
 * @return Status code of the operation: OK(0) for success.
 */
OPENVINO_C_API(ov_status_e)
ov_infer_request_get_profiling_info(const ov_infer_request_t* infer_request, ov_profiling_info_list_t* profiling_infos);

/**
 * @brief Release the memory allocated by ov_profiling_info_list_t.
 * @ingroup ov_infer_request_c_api
 * @param profiling_infos A pointer to the ov_profiling_info_list_t to free memory.
 */
OPENVINO_C_API(void)
ov_profiling_info_list_free(ov_profiling_info_list_t* profiling_infos);
