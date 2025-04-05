/*******************************************************************************
* Copyright 2022-2024 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/

#ifndef GRAPH_INTERFACE_OP_DEF_CONSTRAINT_HPP
#define GRAPH_INTERFACE_OP_DEF_CONSTRAINT_HPP

#include "graph/interface/op.hpp"

#include "common/verbose.hpp"

namespace dnnl {
namespace impl {
namespace graph {
bool check_pads(const op_t *n);

bool check_bn_data_type(const op_t *n);

bool check_ln_data_type(const op_t *n);

bool check_typecast_data_type(const op_t *n);

bool check_avgpool_bwd_input_shape(const op_t *n);

bool check_conv_bwd_data_output_shape(const op_t *n);

bool check_conv_bwd_weights_weights_shape(const op_t *n);

bool check_interpolate_sizes_scales(const op_t *n);

bool check_ln_fwd_outputs_num(const op_t *n);

bool check_ln_bwd_use_affine(const op_t *n);

bool check_reduce_axes(const op_t *n);

bool check_quant_dequant_scales_zps(const op_t *n);

bool check_dyn_quant_dequant_scales_zps(const op_t *n);

bool check_maxpool_dilations(const op_t *n);
} // namespace graph
} // namespace impl
} // namespace dnnl

#endif
