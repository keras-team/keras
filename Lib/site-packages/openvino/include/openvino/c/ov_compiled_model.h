// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @brief This is a C header file for the ov_compiled_model API, which is a C wrapper for ov::CompiledModel class.
 * A compiled model is compiled by a specific device by applying multiple optimization
 * transformations, then mapping to compute kernels.
 * @file ov_compiled_model.h
 */

#pragma once

#include "openvino/c/ov_common.h"
#include "openvino/c/ov_infer_request.h"
#include "openvino/c/ov_model.h"
#include "openvino/c/ov_node.h"
#include "openvino/c/ov_property.h"
#include "openvino/c/ov_remote_context.h"

/**
 * @struct ov_compiled_model_t
 * @ingroup ov_compiled_model_c_api
 * @brief type define ov_compiled_model_t from ov_compiled_model
 */
typedef struct ov_compiled_model ov_compiled_model_t;

/**
 * @brief Get the input size of ov_compiled_model_t.
 * @ingroup ov_compiled_model_c_api
 * @param compiled_model A pointer to the ov_compiled_model_t.
 * @param input_size the compiled_model's input size.
 * @return Status code of the operation: OK(0) for success.
 */
OPENVINO_C_API(ov_status_e)
ov_compiled_model_inputs_size(const ov_compiled_model_t* compiled_model, size_t* size);

/**
 * @brief Get the single const input port of ov_compiled_model_t, which only support single input model.
 * @ingroup ov_compiled_model_c_api
 * @param compiled_model A pointer to the ov_compiled_model_t.
 * @param input_port A pointer to the ov_output_const_port_t.
 * @return Status code of the operation: OK(0) for success.
 */
OPENVINO_C_API(ov_status_e)
ov_compiled_model_input(const ov_compiled_model_t* compiled_model, ov_output_const_port_t** input_port);

/**
 * @brief Get a const input port of ov_compiled_model_t by port index.
 * @ingroup ov_compiled_model_c_api
 * @param compiled_model A pointer to the ov_compiled_model_t.
 * @param index input index.
 * @param input_port A pointer to the ov_output_const_port_t.
 * @return Status code of the operation: OK(0) for success.
 */
OPENVINO_C_API(ov_status_e)
ov_compiled_model_input_by_index(const ov_compiled_model_t* compiled_model,
                                 const size_t index,
                                 ov_output_const_port_t** input_port);

/**
 * @brief Get a const input port of ov_compiled_model_t by name.
 * @ingroup ov_compiled_model_c_api
 * @param compiled_model A pointer to the ov_compiled_model_t.
 * @param name input tensor name (char *).
 * @param input_port A pointer to the ov_output_const_port_t.
 * @return Status code of the operation: OK(0) for success.
 */
OPENVINO_C_API(ov_status_e)
ov_compiled_model_input_by_name(const ov_compiled_model_t* compiled_model,
                                const char* name,
                                ov_output_const_port_t** input_port);

/**
 * @brief Get the output size of ov_compiled_model_t.
 * @ingroup ov_compiled_model_c_api
 * @param compiled_model A pointer to the ov_compiled_model_t.
 * @param size the compiled_model's output size.
 * @return Status code of the operation: OK(0) for success.
 */
OPENVINO_C_API(ov_status_e)
ov_compiled_model_outputs_size(const ov_compiled_model_t* compiled_model, size_t* size);

/**
 * @brief Get the single const output port of ov_compiled_model_t, which only support single output model.
 * @ingroup ov_compiled_model_c_api
 * @param compiled_model A pointer to the ov_compiled_model_t.
 * @param output_port A pointer to the ov_output_const_port_t.
 * @return Status code of the operation: OK(0) for success.
 */
OPENVINO_C_API(ov_status_e)
ov_compiled_model_output(const ov_compiled_model_t* compiled_model, ov_output_const_port_t** output_port);

/**
 * @brief Get a const output port of ov_compiled_model_t by port index.
 * @ingroup ov_compiled_model_c_api
 * @param compiled_model A pointer to the ov_compiled_model_t.
 * @param index input index.
 * @param output_port A pointer to the ov_output_const_port_t.
 * @return Status code of the operation: OK(0) for success.
 */
OPENVINO_C_API(ov_status_e)
ov_compiled_model_output_by_index(const ov_compiled_model_t* compiled_model,
                                  const size_t index,
                                  ov_output_const_port_t** output_port);

/**
 * @brief Get a const output port of ov_compiled_model_t by name.
 * @ingroup ov_compiled_model_c_api
 * @param compiled_model A pointer to the ov_compiled_model_t.
 * @param name input tensor name (char *).
 * @param output_port A pointer to the ov_output_const_port_t.
 * @return Status code of the operation: OK(0) for success.
 */
OPENVINO_C_API(ov_status_e)
ov_compiled_model_output_by_name(const ov_compiled_model_t* compiled_model,
                                 const char* name,
                                 ov_output_const_port_t** output_port);

/**
 * @brief Gets runtime model information from a device.
 * @ingroup ov_compiled_model_c_api
 * @param compiled_model A pointer to the ov_compiled_model_t.
 * @param model A pointer to the ov_model_t.
 * @return Status code of the operation: OK(0) for success.
 */
OPENVINO_C_API(ov_status_e)
ov_compiled_model_get_runtime_model(const ov_compiled_model_t* compiled_model, ov_model_t** model);

/**
 * @brief Creates an inference request object used to infer the compiled model.
 * @ingroup ov_compiled_model_c_api
 * @param compiled_model A pointer to the ov_compiled_model_t.
 * @param infer_request A pointer to the ov_infer_request_t.
 * @return Status code of the operation: OK(0) for success.
 */
OPENVINO_C_API(ov_status_e)
ov_compiled_model_create_infer_request(const ov_compiled_model_t* compiled_model, ov_infer_request_t** infer_request);

/**
 * @brief Sets properties for a device, acceptable keys can be found in ov_property_key_xxx.
 * @ingroup ov_compiled_model_c_api
 * @param compiled_model A pointer to the ov_compiled_model_t.
 * @param ... variadic paramaters The format is <char *property_key, char* property_value>.
 * Supported property key please see ov_property.h.
 * @return Status code of the operation: OK(0) for success.
 */
OPENVINO_C_API(ov_status_e)
ov_compiled_model_set_property(const ov_compiled_model_t* compiled_model, ...);

/**
 * @brief Gets properties for current compiled model.
 * @ingroup ov_compiled_model_c_api
 * @param compiled_model A pointer to the ov_compiled_model_t.
 * @param property_key Property key.
 * @param property_value A pointer to property value.
 * @return Status code of the operation: OK(0) for success.
 */
OPENVINO_C_API(ov_status_e)
ov_compiled_model_get_property(const ov_compiled_model_t* compiled_model,
                               const char* property_key,
                               char** property_value);

/**
 * @brief Exports the current compiled model to an output stream `std::ostream`.
 * The exported model can also be imported via the ov::Core::import_model method.
 * @ingroup ov_compiled_model_c_api
 * @param compiled_model A pointer to the ov_compiled_model_t.
 * @param export_model_path Path to the file.
 * @return Status code of the operation: OK(0) for success.
 */
OPENVINO_C_API(ov_status_e)
ov_compiled_model_export_model(const ov_compiled_model_t* compiled_model, const char* export_model_path);

/**
 * @brief Release the memory allocated by ov_compiled_model_t.
 * @ingroup ov_compiled_model_c_api
 * @param compiled_model A pointer to the ov_compiled_model_t to free memory.
 */
OPENVINO_C_API(void)
ov_compiled_model_free(ov_compiled_model_t* compiled_model);

/**
 * @brief Returns pointer to device-specific shared context
 * on a remote accelerator device that was used to create this CompiledModel.
 * @ingroup ov_compiled_model_c_api
 * @param compiled_model A pointer to the ov_compiled_model_t.
 * @param context Return context.
 * @return Status code of the operation: OK(0) for success.
 *
 */
OPENVINO_C_API(ov_status_e)
ov_compiled_model_get_context(const ov_compiled_model_t* compiled_model, ov_remote_context_t** context);
