// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @brief This is a header file for ov_remote_context C API
 *
 * @file ov_remote_context.h
 */
#pragma once
#include "openvino/c/gpu/gpu_plugin_properties.h"
#include "openvino/c/ov_common.h"
#include "openvino/c/ov_shape.h"
#include "openvino/c/ov_tensor.h"

typedef struct ov_remote_context ov_remote_context_t;

/**
 * @brief Allocates memory tensor in device memory or wraps user-supplied memory handle
 * using the specified tensor description and low-level device-specific parameters.
 * Returns a pointer to the object that implements the RemoteTensor interface.
 * @ingroup ov_remote_context_c_api
 * @param context A pointer to the ov_remote_context_t instance.
 * @param type Defines the element type of the tensor.
 * @param shape Defines the shape of the tensor.
 * @param object_args_size Size of the low-level tensor object parameters.
 * @param remote_tensor Pointer to returned ov_tensor_t that contains remote tensor instance.
 * @param ... variadic params Contains low-level tensor object parameters.
 * @return Status code of the operation: OK(0) for success.
 */
OPENVINO_C_API(ov_status_e)
ov_remote_context_create_tensor(const ov_remote_context_t* context,
                                const ov_element_type_e type,
                                const ov_shape_t shape,
                                const size_t object_args_size,
                                ov_tensor_t** remote_tensor,
                                ...);

/**
 * @brief Returns name of a device on which underlying object is allocated.
 * @ingroup ov_remote_context_c_api
 * @param context A pointer to the ov_remote_context_t instance.
 * @param device_name Device name will be returned.
 * @return Status code of the operation: OK(0) for success.
 */
OPENVINO_C_API(ov_status_e)
ov_remote_context_get_device_name(const ov_remote_context_t* context, char** device_name);

/**
 * @brief Returns a string contains device-specific parameters required for low-level
 * operations with the underlying object.
 * Parameters include device/context handles, access flags,
 * etc. Content of the returned map depends on a remote execution context that is
 * currently set on the device (working scenario).
 * One actaul example: "CONTEXT_TYPE OCL OCL_CONTEXT 0x5583b2ec7b40 OCL_QUEUE 0x5583b2e98ff0"
 * @ingroup ov_remote_context_c_api
 * @param context A pointer to the ov_remote_context_t instance.
 * @param size The size of param pairs.
 * @param params Param name:value list.
 * @return Status code of the operation: OK(0) for success.
 */
OPENVINO_C_API(ov_status_e)
ov_remote_context_get_params(const ov_remote_context_t* context, size_t* size, char** params);

/**
 * @brief This method is used to create a host tensor object friendly for the device in current context.
 * For example, GPU context may allocate USM host memory (if corresponding extension is available),
 * which could be more efficient than regular host memory.
 * @ingroup ov_remote_context_c_api
 * @param context A pointer to the ov_remote_context_t instance.
 * @param type Defines the element type of the tensor.
 * @param shape Defines the shape of the tensor.
 * @param tensor Pointer to ov_tensor_t that contains host tensor.
 * @return Status code of the operation: OK(0) for success.
 */
OPENVINO_C_API(ov_status_e)
ov_remote_context_create_host_tensor(const ov_remote_context_t* context,
                                     const ov_element_type_e type,
                                     const ov_shape_t shape,
                                     ov_tensor_t** tensor);

/**
 * @brief Release the memory allocated by ov_remote_context_t.
 * @ingroup ov_remote_context_c_api
 * @param context A pointer to the ov_remote_context_t to free memory.
 * @return Status code of the operation: OK(0) for success.
 */
OPENVINO_C_API(void)
ov_remote_context_free(ov_remote_context_t* context);

/**
 * @brief Returns a string contains device-specific parameters required for low-level
 * operations with underlying object.
 * Parameters include device/context/surface/buffer handles, access flags,
 * etc. Content of the returned map depends on remote execution context that is
 * currently set on the device (working scenario).
 * One example: "MEM_HANDLE:0x559ff6904b00;OCL_CONTEXT:0x559ff71d62f0;SHARED_MEM_TYPE:OCL_BUFFER;"
 * @ingroup ov_remote_context_c_api
 * @param tensor Pointer to ov_tensor_t that contains host tensor.
 * @param size The size of param pairs.
 * @param params Param name:value list.
 * @return Status code of the operation: OK(0) for success.
 */
OPENVINO_C_API(ov_status_e)
ov_remote_tensor_get_params(ov_tensor_t* tensor, size_t* size, char** params);

/**
 * @brief Returns name of a device on which underlying object is allocated.
 * @ingroup ov_remote_context_c_api
 * @param remote_tensor A pointer to the remote tensor instance.
 * @param device_name Device name will be return.
 * @return Status code of the operation: OK(0) for success.
 */
OPENVINO_C_API(ov_status_e)
ov_remote_tensor_get_device_name(ov_tensor_t* remote_tensor, char** device_name);
