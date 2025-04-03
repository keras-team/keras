/*******************************************************************************
* Copyright 2021-2023 Intel Corporation
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

#ifndef COMMON_SERIALIZATION_HPP
#define COMMON_SERIALIZATION_HPP

#include "common/c_types_map.hpp"
#include "common/primitive_attr.hpp"
#include "common/serialization_stream.hpp"
#include "common/type_helpers.hpp"
#include "oneapi/dnnl/dnnl.h"

namespace dnnl {
namespace impl {
namespace serialization {

void serialize_post_ops(
        serialization_stream_t &sstream, const post_ops_t &post_ops);
void serialize_attr(
        serialization_stream_t &sstream, const primitive_attr_t &attr);
void serialize_md(serialization_stream_t &sstream, const memory_desc_t &md);
void serialize_desc(serialization_stream_t &sstream, const concat_desc_t &desc);
void serialize_desc(serialization_stream_t &sstream,
        const batch_normalization_desc_t &desc);
void serialize_desc(serialization_stream_t &sstream, const binary_desc_t &desc);
void serialize_desc(
        serialization_stream_t &sstream, const convolution_desc_t &desc);
void serialize_desc(
        serialization_stream_t &sstream, const eltwise_desc_t &desc);
void serialize_desc(serialization_stream_t &sstream, const gemm_desc_t &desc);
void serialize_desc(serialization_stream_t &sstream,
        const group_normalization_desc_t &desc);
void serialize_desc(
        serialization_stream_t &sstream, const inner_product_desc_t &desc);
void serialize_desc(serialization_stream_t &sstream,
        const layer_normalization_desc_t &desc);
void serialize_desc(serialization_stream_t &sstream, const lrn_desc_t &desc);
void serialize_desc(serialization_stream_t &sstream, const matmul_desc_t &desc);
void serialize_desc(
        serialization_stream_t &sstream, const pooling_desc_t &desc);
void serialize_desc(serialization_stream_t &sstream, const prelu_desc_t &desc);
void serialize_desc(
        serialization_stream_t &sstream, const reduction_desc_t &desc);
void serialize_desc(
        serialization_stream_t &sstream, const reorder_desc_t &desc);
void serialize_desc(
        serialization_stream_t &sstream, const resampling_desc_t &desc);
void serialize_desc(serialization_stream_t &sstream, const rnn_desc_t &desc);
void serialize_desc(
        serialization_stream_t &sstream, const shuffle_desc_t &desc);
void serialize_desc(
        serialization_stream_t &sstream, const softmax_desc_t &desc);
void serialize_desc(serialization_stream_t &sstream, const sum_desc_t &desc);

status_t serialize_desc(
        serialization_stream_t &sstream, const op_desc_t *op_desc);

} // namespace serialization
} // namespace impl
} // namespace dnnl

#endif
