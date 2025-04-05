// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @brief This is a header file for ov_shape C API
 *
 * @file ov_rank.h
 */

#pragma once

#include "openvino/c/ov_dimension.h"

/**
 * @struct ov_rank_t
 * @ingroup ov_rank_c_api
 * @brief type define ov_rank_t from ov_dimension_t
 */
typedef ov_dimension_t ov_rank_t;

/**
 * @brief Check this rank whether is dynamic
 * @ingroup ov_rank_c_api
 * @param rank The rank pointer that will be checked.
 * @return bool The return value.
 */
OPENVINO_C_API(bool)
ov_rank_is_dynamic(const ov_rank_t rank);
