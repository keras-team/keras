// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @file openvino.h
 * C API of OpenVINO 2.0 bridge unlocks using of OpenVINO 2.0
 * library and all its plugins in native applications disabling usage
 * of C++ API. The scope of API covers significant part of C++ API and includes
 * an ability to read model from the disk, modify input and output information
 * to correspond their runtime representation like data types or memory layout,
 * load in-memory model to different devices including
 * heterogeneous and multi-device modes, manage memory where input and output
 * is allocated and manage inference flow.
 **/
#pragma once

#ifdef _WINDOWS_
#    pragma message( \
        "The BOOLEAN define in ov_element_type_e conflict with Windows.h BOOLEAN define. The BOOLEAN of ov_element_type_e redefine to OV_BOOLEAN here. If you want to use BOOLEAN of Windows.h, pls redefine befor include openvino/c/openvino.h, such as typedef BOOLEAN WIN_BOOLEAN")
#endif
#define BOOLEAN OV_BOOLEAN

#include "openvino/c/auto/properties.h"
#include "openvino/c/ov_common.h"
#include "openvino/c/ov_compiled_model.h"
#include "openvino/c/ov_core.h"
#include "openvino/c/ov_dimension.h"
#include "openvino/c/ov_infer_request.h"
#include "openvino/c/ov_layout.h"
#include "openvino/c/ov_model.h"
#include "openvino/c/ov_node.h"
#include "openvino/c/ov_partial_shape.h"
#include "openvino/c/ov_prepostprocess.h"
#include "openvino/c/ov_property.h"
#include "openvino/c/ov_rank.h"
#include "openvino/c/ov_remote_context.h"
#include "openvino/c/ov_shape.h"
#include "openvino/c/ov_tensor.h"
