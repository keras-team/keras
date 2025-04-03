// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @brief This is a header file for ov_model C API, which is a C wrapper for ov::Model class.
 * A user-defined model.
 * @file ov_model.h
 */

#pragma once

#include "openvino/c/ov_common.h"
#include "openvino/c/ov_node.h"
#include "openvino/c/ov_partial_shape.h"

/**
 * @struct ov_model_t
 * @ingroup ov_model_c_api
 * @brief type define ov_model_t from ov_model
 */
typedef struct ov_model ov_model_t;

/**
 * @brief Release the memory allocated by ov_model_t.
 * @ingroup ov_model_c_api
 * @param model A pointer to the ov_model_t to free memory.
 */
OPENVINO_C_API(void)
ov_model_free(ov_model_t* model);

/**
 * @brief Get a const input port of ov_model_t,which only support single input model.
 * @ingroup ov_model_c_api
 * @param model A pointer to the ov_model_t.
 * @param input_port A pointer to the ov_output_const_port_t.
 * @return Status code of the operation: OK(0) for success.
 */
OPENVINO_C_API(ov_status_e)
ov_model_const_input(const ov_model_t* model, ov_output_const_port_t** input_port);

/**
 * @brief Get a const input port of ov_model_t by name.
 * @ingroup ov_model_c_api
 * @param model A pointer to the ov_model_t.
 * @param tensor_name The name of input tensor.
 * @param input_port A pointer to the ov_output_const_port_t.
 * @return Status code of the operation: OK(0) for success.
 */
OPENVINO_C_API(ov_status_e)
ov_model_const_input_by_name(const ov_model_t* model, const char* tensor_name, ov_output_const_port_t** input_port);

/**
 * @brief Get a const input port of ov_model_t by port index.
 * @ingroup ov_model_c_api
 * @param model A pointer to the ov_model_t.
 * @param index input tensor index.
 * @param input_port A pointer to the ov_output_const_port_t.
 * @return Status code of the operation: OK(0) for success.
 */
OPENVINO_C_API(ov_status_e)
ov_model_const_input_by_index(const ov_model_t* model, const size_t index, ov_output_const_port_t** input_port);

/**
 * @brief Get single input port of ov_model_t, which only support single input model.
 * @ingroup ov_model_c_api
 * @param model A pointer to the ov_model_t.
 * @param input_port A pointer to the ov_output_port_t.
 * @return Status code of the operation: OK(0) for success.
 */
OPENVINO_C_API(ov_status_e)
ov_model_input(const ov_model_t* model, ov_output_port_t** input_port);

/**
 * @brief Get an input port of ov_model_t by name.
 * @ingroup ov_model_c_api
 * @param model A pointer to the ov_model_t.
 * @param tensor_name input tensor name (char *).
 * @param input_port A pointer to the ov_output_port_t.
 * @return Status code of the operation: OK(0) for success.
 */
OPENVINO_C_API(ov_status_e)
ov_model_input_by_name(const ov_model_t* model, const char* tensor_name, ov_output_port_t** input_port);

/**
 * @brief Get an input port of ov_model_t by port index.
 * @ingroup ov_model_c_api
 * @param model A pointer to the ov_model_t.
 * @param index input tensor index.
 * @param input_port A pointer to the ov_output_port_t.
 * @return Status code of the operation: OK(0) for success.
 */
OPENVINO_C_API(ov_status_e)
ov_model_input_by_index(const ov_model_t* model, const size_t index, ov_output_port_t** input_port);

/**
 * @brief Get a single const output port of ov_model_t, which only support single output model.
 * @ingroup ov_model_c_api
 * @param model A pointer to the ov_model_t.
 * @param output_port A pointer to the ov_output_const_port_t.
 * @return Status code of the operation: OK(0) for success.
 */
OPENVINO_C_API(ov_status_e)
ov_model_const_output(const ov_model_t* model, ov_output_const_port_t** output_port);

/**
 * @brief Get a const output port of ov_model_t by port index.
 * @ingroup ov_model_c_api
 * @param model A pointer to the ov_model_t.
 * @param index input tensor index.
 * @param output_port A pointer to the ov_output_const_port_t.
 * @return Status code of the operation: OK(0) for success.
 */
OPENVINO_C_API(ov_status_e)
ov_model_const_output_by_index(const ov_model_t* model, const size_t index, ov_output_const_port_t** output_port);

/**
 * @brief Get a const output port of ov_model_t by name.
 * @ingroup ov_model_c_api
 * @param model A pointer to the ov_model_t.
 * @param tensor_name input tensor name (char *).
 * @param output_port A pointer to the ov_output_const_port_t.
 * @return Status code of the operation: OK(0) for success.
 */
OPENVINO_C_API(ov_status_e)
ov_model_const_output_by_name(const ov_model_t* model, const char* tensor_name, ov_output_const_port_t** output_port);

/**
 * @brief Get a single output port of ov_model_t, which only support single output model.
 * @ingroup ov_model_c_api
 * @param model A pointer to the ov_model_t.
 * @param output_port A pointer to the ov_output_const_port_t.
 * @return Status code of the operation: OK(0) for success.
 */
OPENVINO_C_API(ov_status_e)
ov_model_output(const ov_model_t* model, ov_output_port_t** output_port);

/**
 * @brief Get an output port of ov_model_t by port index.
 * @ingroup ov_model_c_api
 * @param model A pointer to the ov_model_t.
 * @param index input tensor index.
 * @param output_port A pointer to the ov_output_port_t.
 * @return Status code of the operation: OK(0) for success.
 */
OPENVINO_C_API(ov_status_e)
ov_model_output_by_index(const ov_model_t* model, const size_t index, ov_output_port_t** output_port);

/**
 * @brief Get an output port of ov_model_t by name.
 * @ingroup ov_model_c_api
 * @param model A pointer to the ov_model_t.
 * @param tensor_name output tensor name (char *).
 * @param output_port A pointer to the ov_output_port_t.
 * @return Status code of the operation: OK(0) for success.
 */
OPENVINO_C_API(ov_status_e)
ov_model_output_by_name(const ov_model_t* model, const char* tensor_name, ov_output_port_t** output_port);

/**
 * @brief Get the input size of ov_model_t.
 * @ingroup ov_model_c_api
 * @param model A pointer to the ov_model_t.
 * @param input_size the model's input size.
 * @return Status code of the operation: OK(0) for success.
 */
OPENVINO_C_API(ov_status_e)
ov_model_inputs_size(const ov_model_t* model, size_t* input_size);

/**
 * @brief Get the output size of ov_model_t.
 * @ingroup ov_model_c_api
 * @param model A pointer to the ov_model_t.
 * @param output_size the model's output size.
 * @return Status code of the operation: OK(0) for success.
 */
OPENVINO_C_API(ov_status_e)
ov_model_outputs_size(const ov_model_t* model, size_t* output_size);

/**
 * @brief Returns true if any of the ops defined in the model is dynamic shape.
 * @param model A pointer to the ov_model_t.
 * @return true if model contains dynamic shapes
 */
OPENVINO_C_API(bool)
ov_model_is_dynamic(const ov_model_t* model);

/**
 * @brief Do reshape in model with a list of <name, partial shape>.
 * @ingroup ov_model_c_api
 * @param model A pointer to the ov_model_t.
 * @param tensor_names The list of input tensor names.
 * @param partialShape A PartialShape list.
 * @param size The item count in the list.
 * @return Status code of the operation: OK(0) for success.
 */
OPENVINO_C_API(ov_status_e)
ov_model_reshape(const ov_model_t* model,
                 const char** tensor_names,
                 const ov_partial_shape_t* partial_shapes,
                 size_t size);

/**
 * @brief Do reshape in model with partial shape for a specified name.
 * @ingroup ov_model_c_api
 * @param model A pointer to the ov_model_t.
 * @param tensor_name The tensor name of input tensor.
 * @param partialShape A PartialShape.
 * @return Status code of the operation: OK(0) for success.
 */
OPENVINO_C_API(ov_status_e)
ov_model_reshape_input_by_name(const ov_model_t* model,
                               const char* tensor_name,
                               const ov_partial_shape_t partial_shape);

/**
 * @brief Do reshape in model for one node(port 0).
 * @ingroup ov_model_c_api
 * @param model A pointer to the ov_model_t.
 * @param partialShape A PartialShape.
 * @return Status code of the operation: OK(0) for success.
 */
OPENVINO_C_API(ov_status_e)
ov_model_reshape_single_input(const ov_model_t* model, const ov_partial_shape_t partial_shape);

/**
 * @brief Do reshape in model with a list of <port id, partial shape>.
 * @ingroup ov_model_c_api
 * @param model A pointer to the ov_model_t.
 * @param port_indexes The array of port indexes.
 * @param partialShape A PartialShape list.
 * @param size The item count in the list.
 * @return Status code of the operation: OK(0) for success.
 */
OPENVINO_C_API(ov_status_e)
ov_model_reshape_by_port_indexes(const ov_model_t* model,
                                 const size_t* port_indexes,
                                 const ov_partial_shape_t* partial_shape,
                                 size_t size);

/**
 * @brief Do reshape in model with a list of <ov_output_port_t, partial shape>.
 * @ingroup ov_model_c_api
 * @param model A pointer to the ov_model_t.
 * @param output_ports The ov_output_port_t list.
 * @param partialShape A PartialShape list.
 * @param size The item count in the list.
 * @return Status code of the operation: OK(0) for success.
 */
OPENVINO_C_API(ov_status_e)
ov_model_reshape_by_ports(const ov_model_t* model,
                          const ov_output_port_t** output_ports,
                          const ov_partial_shape_t* partial_shapes,
                          size_t size);

/**
 * @brief Gets the friendly name for a model.
 * @ingroup ov_model_c_api
 * @param model A pointer to the ov_model_t.
 * @param friendly_name the model's friendly name.
 * @return Status code of the operation: OK(0) for success.
 */
OPENVINO_C_API(ov_status_e)
ov_model_get_friendly_name(const ov_model_t* model, char** friendly_name);
