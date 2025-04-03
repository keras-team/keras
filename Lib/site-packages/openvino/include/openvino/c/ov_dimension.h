// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @brief This is a header file for ov_dimension C API, which is a C wrapper for ov::Dimension class.
 *
 * @file ov_dimension.h
 */

#pragma once

#include "openvino/c/ov_common.h"

/**
 * @struct ov_dimension
 * @ingroup ov_dimension_c_api
 * @brief This is a structure interface equal to ov::Dimension
 */
typedef struct ov_dimension {
    int64_t min;  //!< The lower inclusive limit for the dimension.
    int64_t max;  //!< The upper inclusive limit for the dimension.
} ov_dimension_t;

/**
 * @brief Check this dimension whether is dynamic
 * @ingroup ov_dimension_c_api
 * @param dim The dimension pointer that will be checked.
 * @return Boolean, true is dynamic and false is static.
 */
OPENVINO_C_API(bool)
ov_dimension_is_dynamic(const ov_dimension_t dim);
