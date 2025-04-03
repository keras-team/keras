/*******************************************************************************
* Copyright 2021-2023 Arm Ltd. and affiliates
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

#ifndef CPU_AARCH64_ACL_UTILS_HPP
#define CPU_AARCH64_ACL_UTILS_HPP

#include <mutex>

#include "oneapi/dnnl/dnnl_types.h"

#include "common/dnnl_thread.hpp"
#include "common/memory_tracking.hpp"
#include "common/primitive.hpp"
#include "common/resource.hpp"
#include "common/utils.hpp"

#include "arm_compute/runtime/NEON/NEFunctions.h"
#include "arm_compute/runtime/Scheduler.h"

namespace dnnl {
namespace impl {
namespace cpu {
namespace aarch64 {

namespace acl_utils {

arm_compute::DataType get_acl_data_t(
        const dnnl_data_type_t dt, const bool is_quantized = false);

// Convert alg_kind_t, alpha and beta into an ACL ActivationLayerInfo. Will
// return unimplemented and a disabled ActivationLayerInfo if the conversion
// fails
status_t convert_to_acl_act(alg_kind_t eltwise_alg, float alpha, float beta,
        arm_compute::ActivationLayerInfo &act_info);

// Convert an eltwise_desc_t into an ACL ActivationLayerInfo. Will return
// unimplemented and a disabled ActivationLayerInfo if the conversion fails
status_t convert_to_acl_act(
        const eltwise_desc_t &ed, arm_compute::ActivationLayerInfo &act_info);

// Convert an eltwise post op into an ACL ActivationLayerInfo. Will return
// unimplemented and a disabled ActivationLayerInfo if the conversion fails
status_t convert_to_acl_act(const post_ops_t::entry_t::eltwise_t &elt,
        arm_compute::ActivationLayerInfo &act_info);

// Convert a memory desc to an arm_compute::TensorInfo. Note that memory desc
// must be blocking format, plain, dense and have no zero dimensions.
status_t tensor_info(arm_compute::TensorInfo &info, const memory_desc_t &md);
status_t tensor_info(
        arm_compute::TensorInfo &info, const memory_desc_wrapper &md);

// Insert a dimension of size 1 at the index dim_i of TensorInfo
status_t insert_singleton_dimension(arm_compute::TensorInfo &ti, size_t dim_i);

// Reorder the logical dimensions of the memory descriptors (mds) by stride so
// that accessing the tensor elements in the natural order is dense. Note, this
// does not reorder the data, it just reorders the logical indices. The
// permutation is common to all mds, so the function returns when it cannot find
// a dimension with a common smallest stride. Returns the number of dimensions
// that we managed to reorder to be dense.
int reorder_dimensions_by_stride(std::vector<memory_desc_t *> permuted_mds,
        std::vector<const memory_desc_t *> mds);

// Reorder a memory_desc_t and set the strides on a arm_compute::TensorInfo to
// match an arm_compute::WeightFormat. You are required to specify how various
// logical dimensions in oneDNN correspond to logical dimensions in arm_compute.
// info  TensorInfo where the strides will be changed to match the reordering
// md    memory descriptor where the stride and padded dimensions will be
//       changed or reordering
// wf    Describes the memory format/layout of the weights
// I_dim The logical dimension of md corresponding to the input channel of
//       a convolution or the K dimension in a matmul
// O_dim The logical dimension of md corresponding to the output channel of a
//       convolution or the N dimension in a matmul
// spatial_dims The logical dimensions of md corresponding to the spatial
//              dimensions of the weights (H, W, D for example). These will be
//              the next densest after the inner blocks and the input channel.
// batch_dims The logical dimensions of md related to the batch in a batched
//            matmul, ordered from innermost to outermost. ACL calls these
//            the multi_stride_b. These will become the outermost (least dense)
//            dimensions and will be collapsed.
void reorder_to_weight_format(arm_compute::TensorInfo &info, memory_desc_t &md,
        arm_compute::WeightFormat wf, dim_t I_dim, dim_t O_dim,
        std::vector<dim_t> spatial_dims, std::vector<dim_t> batch_dims = {});

// Logs a custom 'info' line describing an unsupported case
#define LOG_ACL_UNSUPPORTED(msg) \
    do { \
        if (get_verbose(verbose_t::create_dispatch)) \
            printf("onednn_verbose,cpu,acl,unsupported: %s\n", (msg)); \
    } while (0)

// Returns unimplemented if error code x is NOT OK
#define ACL_CHECK_VALID(x) \
    do { \
        arm_compute::Status s = x; \
        if (s.error_code() != arm_compute::ErrorCode::OK) { \
            LOG_ACL_UNSUPPORTED(s.error_description().c_str()); \
            return dnnl::impl::status::unimplemented; \
        } \
    } while (0)

// Returns unimplemented on condition x == true
#define ACL_CHECK_SUPPORT(x, msg) \
    do { \
        if (x) { \
            LOG_ACL_UNSUPPORTED(msg); \
            return dnnl::impl::status::unimplemented; \
        } \
    } while (0)

} // namespace acl_utils

} // namespace aarch64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif // CPU_AARCH64_ACL_UTILS_HPP
