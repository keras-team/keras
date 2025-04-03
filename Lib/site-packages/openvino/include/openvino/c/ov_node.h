// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @brief This is a header file for ov_model C API, which is a C wrapper for ov::Node class.
 *
 * @file ov_node.h
 */

#pragma once

#include "openvino/c/ov_common.h"
#include "openvino/c/ov_partial_shape.h"
#include "openvino/c/ov_shape.h"

/**
 * @struct ov_output_const_port_t
 * @ingroup ov_node_c_api
 * @brief type define ov_output_const_port_t from ov_output_const_port
 */
typedef struct ov_output_const_port ov_output_const_port_t;

/**
 * @struct ov_output_port_t
 * @ingroup ov_node_c_api
 * @brief type define ov_output_port_t from ov_output_port
 */
typedef struct ov_output_port ov_output_port_t;

/**
 * @brief Get the shape of port object.
 * @ingroup ov_node_c_api
 * @param port A pointer to ov_output_const_port_t.
 * @param tensor_shape tensor shape.
 * @return Status code of the operation: OK(0) for success.
 */
OPENVINO_C_API(ov_status_e)
ov_const_port_get_shape(const ov_output_const_port_t* port, ov_shape_t* tensor_shape);

/**
 * @brief Get the shape of port object.
 * @ingroup ov_node_c_api
 * @param port A pointer to ov_output_port_t.
 * @param tensor_shape tensor shape.
 * @return Status code of the operation: OK(0) for success.
 */
OPENVINO_C_API(ov_status_e)
ov_port_get_shape(const ov_output_port_t* port, ov_shape_t* tensor_shape);

/**
 * @brief Get the tensor name of port.
 * @ingroup ov_node_c_api
 * @param port A pointer to the ov_output_const_port_t.
 * @param tensor_name A pointer to the tensor name.
 * @return Status code of the operation: OK(0) for success.
 */
OPENVINO_C_API(ov_status_e)
ov_port_get_any_name(const ov_output_const_port_t* port, char** tensor_name);

/**
 * @brief Get the partial shape of port.
 * @ingroup ov_node_c_api
 * @param port A pointer to the ov_output_const_port_t.
 * @param partial_shape Partial shape.
 * @return Status code of the operation: OK(0) for success.
 */
OPENVINO_C_API(ov_status_e)
ov_port_get_partial_shape(const ov_output_const_port_t* port, ov_partial_shape_t* partial_shape);

/**
 * @brief Get the tensor type of port.
 * @ingroup ov_node_c_api
 * @param port A pointer to the ov_output_const_port_t.
 * @param tensor_type tensor type.
 * @return Status code of the operation: OK(0) for success.
 */
OPENVINO_C_API(ov_status_e)
ov_port_get_element_type(const ov_output_const_port_t* port, ov_element_type_e* tensor_type);

/**
 * @brief free port object
 * @ingroup ov_node_c_api
 * @param port The pointer to the instance of the ov_output_port_t to free.
 */
OPENVINO_C_API(void)
ov_output_port_free(ov_output_port_t* port);

/**
 * @brief free const port
 * @ingroup ov_node_c_api
 * @param port The pointer to the instance of the ov_output_const_port_t to free.
 */
OPENVINO_C_API(void)
ov_output_const_port_free(ov_output_const_port_t* port);
