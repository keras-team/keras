// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

//
// The public OpenVINO core API
//

#include "openvino/core/attribute_adapter.hpp"
#include "openvino/core/attribute_visitor.hpp"
#include "openvino/core/axis_set.hpp"
#include "openvino/core/axis_vector.hpp"
#include "openvino/core/coordinate.hpp"
#include "openvino/core/coordinate_diff.hpp"
#include "openvino/core/core_visibility.hpp"
#include "openvino/core/dimension.hpp"
#include "openvino/core/enum_mask.hpp"
#include "openvino/core/enum_names.hpp"
#include "openvino/core/except.hpp"
#include "openvino/core/graph_util.hpp"
#include "openvino/core/interval.hpp"
#include "openvino/core/layout.hpp"
#include "openvino/core/model.hpp"
#include "openvino/core/node.hpp"
#include "openvino/core/node_input.hpp"
#include "openvino/core/node_output.hpp"
#include "openvino/core/node_vector.hpp"
#include "openvino/core/partial_shape.hpp"
#include "openvino/core/rank.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/core/rtti.hpp"
#include "openvino/core/runtime_attribute.hpp"
#include "openvino/core/shape.hpp"
#include "openvino/core/strides.hpp"
#include "openvino/core/type.hpp"
#include "openvino/core/version.hpp"
#include "openvino/core/visibility.hpp"

// Descriptor
#include "openvino/core/descriptor/input.hpp"
#include "openvino/core/descriptor/output.hpp"
#include "openvino/core/descriptor/tensor.hpp"

// Preprocessing
#include "openvino/core/preprocess/color_format.hpp"
#include "openvino/core/preprocess/input_info.hpp"
#include "openvino/core/preprocess/input_model_info.hpp"
#include "openvino/core/preprocess/input_tensor_info.hpp"
#include "openvino/core/preprocess/output_info.hpp"
#include "openvino/core/preprocess/output_model_info.hpp"
#include "openvino/core/preprocess/output_tensor_info.hpp"
#include "openvino/core/preprocess/padding_mode.hpp"
#include "openvino/core/preprocess/postprocess_steps.hpp"
#include "openvino/core/preprocess/pre_post_process.hpp"
#include "openvino/core/preprocess/preprocess_steps.hpp"
#include "openvino/core/preprocess/resize_algorithm.hpp"

// type
#include "openvino/core/type/element_type.hpp"
#include "openvino/core/type/element_type_traits.hpp"
