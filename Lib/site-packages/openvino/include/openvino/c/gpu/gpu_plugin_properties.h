// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @brief This is a specified header file for gpu plugin's properties
 *
 * @file gpu_plugin_properties.h
 */

#pragma once
#include "openvino/c/ov_common.h"

/**
 * @brief gpu plugin properties key for remote context/tensor
 */

//!< Read-write property: shared device context type, can be either pure OpenCL (OCL) or
//!< shared video decoder (VA_SHARED) context.
//!< Value is string, it can be one of below strings:
//!<    "OCL"       - Pure OpenCL context
//!<    "VA_SHARED" - Context shared with a video decoding device
OPENVINO_C_VAR(const char*) ov_property_key_intel_gpu_context_type;

//!< Read-write property<void *>: identifies OpenCL context handle in a shared context or shared memory blob
//!< parameter map.
OPENVINO_C_VAR(const char*) ov_property_key_intel_gpu_ocl_context;

//!< Read-write property<int string>: ID of device in OpenCL context if multiple devices are present in the context.
OPENVINO_C_VAR(const char*) ov_property_key_intel_gpu_ocl_context_device_id;

//!< Read-write property<int string>: In case of multi-tile system, this key identifies tile within given context.
OPENVINO_C_VAR(const char*) ov_property_key_intel_gpu_tile_id;

//!< Read-write property<void *>: OpenCL queue handle in a shared context
OPENVINO_C_VAR(const char*) ov_property_key_intel_gpu_ocl_queue;

//!< Read-write property<void *>: video acceleration device/display handle in a shared context or shared
//!< memory blob parameter map.
OPENVINO_C_VAR(const char*) ov_property_key_intel_gpu_va_device;

//!< Read-write property: type of internal shared memory in a shared memory blob
//!< parameter map.
//!< Value is string, it can be one of below strings:
//!<    "OCL_BUFFER"        - Shared OpenCL buffer blob
//!<    "OCL_IMAGE2D"       - Shared OpenCL 2D image blob
//!<    "USM_USER_BUFFER"   - Shared USM pointer allocated by user
//!<    "USM_HOST_BUFFER"   - Shared USM pointer type with host allocation type allocated by plugin
//!<    "USM_DEVICE_BUFFER" - Shared USM pointer type with device allocation type allocated by plugin
//!<    "VA_SURFACE"        - Shared video decoder surface or D3D 2D texture blob
//!<    "DX_BUFFER"         - Shared D3D buffer blob
OPENVINO_C_VAR(const char*) ov_property_key_intel_gpu_shared_mem_type;

//!< Read-write property<void *>: OpenCL memory handle in a shared memory blob parameter map.
OPENVINO_C_VAR(const char*) ov_property_key_intel_gpu_mem_handle;

//!< Read-write property<uint32_t string>: video decoder surface handle in a shared memory blob parameter map.
OPENVINO_C_VAR(const char*) ov_property_key_intel_gpu_dev_object_handle;

//!< Read-write property<uint32_t string>: video decoder surface plane in a shared memory blob parameter map.
OPENVINO_C_VAR(const char*) ov_property_key_intel_gpu_va_plane;
