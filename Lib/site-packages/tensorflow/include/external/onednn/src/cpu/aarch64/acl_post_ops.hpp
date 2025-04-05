/*******************************************************************************
* Copyright 2022-2024 Arm Ltd. and affiliates
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

#ifndef CPU_AARCH64_ACL_POST_OPS_HPP
#define CPU_AARCH64_ACL_POST_OPS_HPP

#include "cpu/aarch64/acl_binary.hpp"
#include "cpu/aarch64/acl_eltwise.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace aarch64 {

struct acl_post_ops_t {

    acl_post_ops_t() = default;

    // init the acl_post_ops_t. Note that this function modifies the passed in
    // post ops by setting the preferred memory formats
    status_t init(engine_t *engine, post_ops_t &post_ops,
            const memory_desc_t &dst_md, int post_op_start_index = 0) {

        post_op_start_index_ = post_op_start_index;

        CHECK(post_ops.set_default_formats(&dst_md));
        dst_data_type = dst_md.data_type;

        // Reset properties derived from post_ops
        sum_index = -1;
        post_op_primitives = {};

        for (int i = post_op_start_index; i < post_ops.len(); i++) {
            auto &po = post_ops.entry_[i];

            if (po.is_sum()) {
                ACL_CHECK_SUPPORT(po.sum.scale != 1.0f,
                        "sum post op scale must be 1 (no scale)");

                ACL_CHECK_SUPPORT(po.sum.zero_point != 0,
                        "sum post op zero point must be 0 (no shift)");

                // >= 0 means we had one already
                ACL_CHECK_SUPPORT(sum_index >= 0,
                        "there must not be more than 1 sum post op");

                sum_index = i;

                // Sum is an add primitive where dst = temp_dst + dst
                binary_desc_t po_desc;
                po_desc.primitive_kind = primitive_kind::binary;
                po_desc.alg_kind = alg_kind::binary_add;
                po_desc.src_desc[0] = dst_md;
                po_desc.src_desc[1] = dst_md;
                po_desc.dst_desc = dst_md;
                auto empty_attr = dnnl_primitive_attr();
                typename acl_binary_t::pd_t acl_binary_pd(
                        &po_desc, &empty_attr, nullptr);
                CHECK(acl_binary_pd.init(engine));

                auto acl_binary
                        = std::make_shared<acl_binary_t>(&acl_binary_pd);
                CHECK(acl_binary->init(engine));
                post_op_primitives.push_back(acl_binary);

            } else if (po.is_binary()) {
                binary_desc_t po_desc;
                po_desc.primitive_kind = primitive_kind::binary;
                po_desc.alg_kind = po.binary.alg;
                po_desc.src_desc[0] = dst_md;
                po_desc.src_desc[1] = po.binary.src1_desc;
                po_desc.dst_desc = dst_md;
                auto empty_attr = dnnl_primitive_attr();
                typename acl_binary_t::pd_t acl_binary_pd(
                        &po_desc, &empty_attr, nullptr);
                CHECK(acl_binary_pd.init(engine));

                auto acl_binary
                        = std::make_shared<acl_binary_t>(&acl_binary_pd);
                CHECK(acl_binary->init(engine));
                post_op_primitives.push_back(acl_binary);

            } else if (po.is_eltwise()) {
                ACL_CHECK_SUPPORT(po.eltwise.scale != 1.0f,
                        "eltwise post op scale must be 1 (no scale)");

                eltwise_desc_t eltwise_desc;
                eltwise_desc.primitive_kind = primitive_kind::eltwise;
                eltwise_desc.alg_kind = po.eltwise.alg;
                eltwise_desc.alpha = po.eltwise.alpha;
                eltwise_desc.beta = po.eltwise.beta;
                memory_desc_t temp_dst = dst_md;
                // pass eltwise a desc with f32 datatype to perform the operation in fp32 rather than fp16
                // since oneDNN requires all post-ops to run in fp32.
                // we don't need to do that to the other post-ops as executing them in fp16 yields the same result.
                if (dst_data_type == data_type::f16) {
                    temp_dst.data_type = data_type::f32;
                }
                eltwise_desc.src_desc = temp_dst;
                eltwise_desc.dst_desc = temp_dst;
                eltwise_desc.prop_kind = prop_kind_t::dnnl_forward;
                auto empty_attr = dnnl_primitive_attr();
                typename acl_eltwise_fwd_t::pd_t acl_eltwise_pd(
                        &eltwise_desc, &empty_attr, nullptr);
                CHECK(acl_eltwise_pd.init(engine));

                auto acl_eltwise
                        = std::make_shared<acl_eltwise_fwd_t>(&acl_eltwise_pd);
                CHECK(acl_eltwise->init(engine));
                post_op_primitives.push_back(acl_eltwise);

            } else {
                // Unsupported catchall
                return status::unimplemented;
            }
        }

        return status::success;
    }

    // init acl_post_ops_t, ignoring the first post op if it is an eltwise so
    // that it can be fused, placing it in act_info_to_fuse. Note that this
    // function modifies the passed in post ops by setting the preferred memory
    // formats
    status_t init(engine_t *engine, post_ops_t &base_post_ops,
            const memory_desc_t &dst_md,
            arm_compute::ActivationLayerInfo &act_info_to_fuse,
            int post_op_start_index = 0) {

        CHECK(base_post_ops.set_default_formats(&dst_md));
        dst_data_type = dst_md.data_type;
        // If the first entry is eltwise, we fuse it, except when the datatype
        // is fp16 because in this case we want to execute the eltwise in fp32.
        if (base_post_ops.len() >= 1 && base_post_ops.entry_[0].is_eltwise()
                && dst_data_type != data_type::f16) {

            const auto &first_po = base_post_ops.entry_[0].eltwise;
            ACL_CHECK_SUPPORT(first_po.scale != 1.0f,
                    "eltwise post op scale must be 1 (no scale)");
            CHECK(acl_utils::convert_to_acl_act(first_po, act_info_to_fuse));

            // post_op_start_index + 1 to skip the fused eltwise
            return init(engine, base_post_ops, dst_md, post_op_start_index + 1);
        } else {
            // Nothing to fuse, just copy all post ops
            return init(engine, base_post_ops, dst_md, post_op_start_index);
        }
    }

    bool has_sum() const { return sum_index >= 0; }

    status_t create_resource(
            engine_t *engine, resource_mapper_t &mapper) const {
        for (const auto &post_op : post_op_primitives) {
            CHECK(post_op->create_resource(engine, mapper));
        }
        return status::success;
    }

    status_t execute(const exec_ctx_t &ctx, void *src) const;

private:
    // Index of the sum post op if there is one, < 0 means no sum
    int sum_index = -1;
    // Index of the first post op this primitive executes. This is typically the
    // number of post ops which were fused.
    int post_op_start_index_ = 0;
    data_type_t dst_data_type;
    // Vector of primitives used to execute the post ops. They are constructed
    // in init to be either acl_binary_t (for sum, add, sub, div, mul, min and
    // max) or acl_eltwise_fwd_t (for relu, elu, tanh, square, abs etc)
    std::vector<std::shared_ptr<primitive_t>> post_op_primitives;
};

} // namespace aarch64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif
