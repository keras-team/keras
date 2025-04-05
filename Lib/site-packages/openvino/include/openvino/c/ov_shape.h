// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @brief This is a header file for ov_shape C API
 *
 * @file ov_shape.h
 */

#pragma once

#include "openvino/c/ov_common.h"

/**
 * @struct ov_shape_t
 * @ingroup ov_shape_c_api
 * @brief Reprents a static shape.
 */
typedef struct {
    int64_t rank;   //!< the rank of shape
    int64_t* dims;  //!< the dims of shape
} ov_shape_t;

/**
 * @brief Initialize a fully shape object, allocate space for its dimensions and set its content id dims is not null.
 * @ingroup ov_shape_c_api
 * @param rank The rank value for this object, it should be more than 0(>0)
 * @param dims The dimensions data for this shape object, it's size should be equal to rank.
 * @param shape The input/output shape object pointer.
 * @return ov_status_e The return status code.
 */
OPENVINO_C_API(ov_status_e)
ov_shape_create(const int64_t rank, const int64_t* dims, ov_shape_t* shape);

/**
 * @brief Free a shape object's internal memory.
 * @ingroup ov_shape_c_api
 * @param shape The input shape object pointer.
 * @return ov_status_e The return status code.
 */
OPENVINO_C_API(ov_status_e)
ov_shape_free(ov_shape_t* shape);
