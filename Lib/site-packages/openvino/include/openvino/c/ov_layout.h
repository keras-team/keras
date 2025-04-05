// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @brief This is a header file for ov_layout C API
 *
 * @file ov_layout.h
 */

#pragma once

#include "openvino/c/ov_common.h"

/**
 * @struct ov_layout_t
 * @ingroup ov_layout_c_api
 * @brief type define ov_layout_t from ov_layout
 */
typedef struct ov_layout ov_layout_t;

/**
 * @brief Create a layout object.
 * @ingroup ov_layout_c_api
 * @param layout The layout input pointer.
 * @param layout_desc The description of layout.
 * @return ov_status_e a status code, return OK if successful
 */
OPENVINO_C_API(ov_status_e)
ov_layout_create(const char* layout_desc, ov_layout_t** layout);

/**
 * @brief Free layout object.
 * @ingroup ov_layout_c_api
 * @param layout will be released.
 */
OPENVINO_C_API(void)
ov_layout_free(ov_layout_t* layout);

/**
 * @brief Convert layout object to a readable string.
 * @ingroup ov_layout_c_api
 * @param layout will be converted.
 * @return string that describes the layout content.
 */
OPENVINO_C_API(const char*)
ov_layout_to_string(const ov_layout_t* layout);
