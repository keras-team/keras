/*******************************************************************************
* Copyright 2020-2023 Intel Corporation
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

#ifndef GRAPH_INTERFACE_SHAPE_INFER_HPP
#define GRAPH_INTERFACE_SHAPE_INFER_HPP

#include <algorithm>
#include <cmath>
#include <functional>
#include <string>
#include <utility>
#include <vector>

#include "graph/interface/logical_tensor.hpp"
#include "graph/interface/op.hpp"

#include "common/verbose.hpp"

namespace dnnl {
namespace impl {
namespace graph {

/// convert shape to ncx or oix
dims canonicalize(const dims &shape, const std::string &format);

inline dims ncx2nxc(const dims &shape);

/// make a dims according to the format. Only for data format ncx or nxc.
inline dims make_data_dims(
        const std::string &format, const dim_t n, const dim_t c, const dims &x);

/// make a dims according to the format. Only for filter format xio or oix.
inline dims make_filter_dims(
        const std::string &format, const dim_t i, const dim_t o, const dims &x);

/// validate the inferred shape with the expected one.
bool validate(const dims &inferred, const dims &expected);

/// get the dense strides of a given shape
/// eg. (3, 4, 5) -> (20, 5, 1)
inline dims get_dense_strides(const dims &shape);

/// shapes of the logical tensors in the vector are known
inline bool every_shape_is_known(const std::vector<logical_tensor_t *> &lts);

inline bool verify_shapes_in_range(const std::vector<logical_tensor_t *> &lts,
        const size_t begin, const size_t end,
        const std::function<bool(const dims)> &validator);

void set_shape_and_strides(logical_tensor_t &lt, const dims &shape);

inline void set_shapes_in_range(const std::vector<logical_tensor_t *> &lts,
        const size_t begin, const size_t end, const dims &shape);

/// infer the padding sizes according auto_pad type
status_t infer_auto_pad(const dim_t in_dim, const dim_t stride,
        const dim_t kernel, const dim_t dilation, const std::string &auto_pad,
        dim_t &pad_begin, dim_t &pad_end, bool is_deconv = false);

/// numpy broadcasting
/// TODO(xxx): 0-D broadcasting?
status_t broadcast(const dims &lhs, const dims &rhs, dims &broadcasted);

status_t one_way_broadcast(const dims &lhs, const dims &rhs);

/// This function assumes the size of all vectors are correct. Eg. size of
/// strides/dilations/pads should be the same as spatial size of src_dims and
/// fil_dims. Size of output_dims should be the same as size of src_dims.
inline void infer_conv_ncx_oix(const dims &src_dims, const dims &fil_dims,
        const dims &strides, const dims &dilations, const dims &pads_begin,
        const dims &pads_end, dims &output_dims);

/// Calculate convolution output shape according to the input shapes. If
/// auto_pad, the real size of pads_begin and pads_end will also be calculated.
/// The inferred output shape will be written to the logical tensor in outputs.
/// The inferred pads_begin and pads_end will be attached to the operator
/// directly. Hence the function will change the state of the input operator.
status_t infer_conv_output_shape(op_t *n,
        std::vector<logical_tensor_t *> &inputs,
        std::vector<logical_tensor_t *> &outputs);

status_t infer_conv_bprop_data_output_shape(op_t *n,
        std::vector<logical_tensor_t *> &inputs,
        std::vector<logical_tensor_t *> &outputs);

status_t infer_conv_bprop_filters_output_shape(op_t *n,
        std::vector<logical_tensor_t *> &inputs,
        std::vector<logical_tensor_t *> &outputs);

status_t infer_convtranspose_output_shape(op_t *n,
        std::vector<logical_tensor_t *> &inputs,
        std::vector<logical_tensor_t *> &outputs);

/// This function assumes the size of all vectors are correct. Eg. size of
/// strides/dilations/pads should be the same as spatial size of src_dims and
/// fil_dims. Size of output_dims should be the same as size of src_dims.
inline void infer_convtranspose_ncx_oix(const dims &src_dims,
        const dims &fil_dims, const dims &strides, const dims &dilations,
        const dims &pads_begin, const dims &pads_end, dims &output_dims);

status_t infer_convtranspose_bprop_data_output_shape(op_t *n,
        std::vector<logical_tensor_t *> &inputs,
        std::vector<logical_tensor_t *> &outputs);

status_t infer_convtranspose_bprop_filters_output_shape(op_t *n,
        std::vector<logical_tensor_t *> &inputs,
        std::vector<logical_tensor_t *> &outputs);

status_t infer_conv_bprop_filters_output_shape_common(op_t *n,
        std::vector<logical_tensor_t *> &inputs,
        std::vector<logical_tensor_t *> &outputs, const size_t in_num);

// check if output shape is already known
// if shape is unknown, infer output shape (change output lt)
// otherwise infer pad (change op attrs)
status_t infer_pool_output_shape(op_t *n,
        std::vector<logical_tensor_t *> &inputs,
        std::vector<logical_tensor_t *> &outputs);

status_t infer_pool_bwd_output_shape(op_t *n,
        std::vector<logical_tensor_t *> &inputs,
        std::vector<logical_tensor_t *> &outputs);

status_t infer_matmul_output_shape(op_t *n,
        std::vector<logical_tensor_t *> &inputs,
        std::vector<logical_tensor_t *> &outputs);

status_t infer_identity_output_shape(op_t *n,
        std::vector<logical_tensor_t *> &inputs,
        std::vector<logical_tensor_t *> &outputs);

status_t identity_output_shape_on_pos(op_t *n,
        std::vector<logical_tensor_t *> &inputs,
        std::vector<logical_tensor_t *> &outputs,
        std::vector<std::pair<uint32_t, uint32_t>> &positions);

status_t infer_bias_backprop_output_shape(op_t *n,
        std::vector<logical_tensor_t *> &inputs,
        std::vector<logical_tensor_t *> &outputs);

status_t infer_bias_add_output_shape(op_t *n,
        std::vector<logical_tensor_t *> &inputs,
        std::vector<logical_tensor_t *> &outputs);

status_t infer_norm_output_shape(op_t *n,
        std::vector<logical_tensor_t *> &inputs,
        std::vector<logical_tensor_t *> &outputs);

status_t infer_norm_bprop_output_shape(op_t *n,
        std::vector<logical_tensor_t *> &inputs,
        std::vector<logical_tensor_t *> &outputs);

status_t infer_elemwise_arithmetic_output_shape(op_t *n,
        std::vector<logical_tensor_t *> &inputs,
        std::vector<logical_tensor_t *> &outputs);

status_t infer_bn_fwd_train_output_shape(op_t *n,
        std::vector<logical_tensor_t *> &inputs,
        std::vector<logical_tensor_t *> &outputs);

status_t infer_bn_bwd_output_shape(op_t *n,
        std::vector<logical_tensor_t *> &inputs,
        std::vector<logical_tensor_t *> &outputs);

status_t infer_concat_output_shape(op_t *n,
        std::vector<logical_tensor_t *> &inputs,
        std::vector<logical_tensor_t *> &outputs);

status_t infer_unsupported_output_shape(op_t *n,
        std::vector<logical_tensor_t *> &inputs,
        std::vector<logical_tensor_t *> &outputs);

status_t infer_reduce_output_shape(op_t *n,
        std::vector<logical_tensor_t *> &inputs,
        std::vector<logical_tensor_t *> &outputs);

status_t infer_select_output_shape(op_t *n,
        std::vector<logical_tensor_t *> &inputs,
        std::vector<logical_tensor_t *> &outputs);

status_t infer_static_reshape_output_shape(op_t *n,
        std::vector<logical_tensor_t *> &inputs,
        std::vector<logical_tensor_t *> &outputs);
status_t infer_static_transpose_output_shape(op_t *n,
        std::vector<logical_tensor_t *> &inputs,
        std::vector<logical_tensor_t *> &outputs);

status_t infer_interpolate_output_shape(op_t *n,
        std::vector<logical_tensor_t *> &inputs,
        std::vector<logical_tensor_t *> &outputs);
status_t infer_prelu_bwd_output_shape(op_t *n,
        std::vector<logical_tensor_t *> &inputs,
        std::vector<logical_tensor_t *> &outputs);
} // namespace graph
} // namespace impl
} // namespace dnnl

#endif
