// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @brief This is a specified header file for auto plugin's properties
 *
 * @file properties.h
 */

#pragma once
#include "openvino/c/ov_common.h"

/**
 * @brief Read-write property<string> for setting that enables performance improvement by binding
 * buffer to hw infer request
 * @ingroup ov_property_c_api
 */
OPENVINO_C_VAR(const char*)
ov_property_key_intel_auto_device_bind_buffer;

/**
 * @brief Read-write property<string> to enable/disable CPU as accelerator (or helper device) at the beginning
 * @ingroup ov_property_c_api
 */
OPENVINO_C_VAR(const char*)
ov_property_key_intel_auto_enable_startup_fallback;

/**
 * @brief Read-write property<string> to enable/disable runtime fallback to other devices when infer fails
 * @ingroup ov_property_c_api
 */
OPENVINO_C_VAR(const char*)
ov_property_key_intel_auto_enable_runtime_fallback;