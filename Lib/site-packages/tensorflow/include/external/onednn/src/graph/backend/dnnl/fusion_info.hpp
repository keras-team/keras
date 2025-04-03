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
#ifndef GRAPH_BACKEND_DNNL_FUSION_INFO_HPP
#define GRAPH_BACKEND_DNNL_FUSION_INFO_HPP

#include <algorithm>
#include <functional>
#include <map>
#include <memory>
#include <set>
#include <utility>
#include <vector>
#include <unordered_map>

#include "graph/interface/c_types_map.hpp"
#include "graph/interface/op.hpp"
#include "graph/interface/value.hpp"
#include "graph/utils/utils.hpp"

#include "graph/backend/dnnl/internal_attrs.hpp"
#include "graph/backend/dnnl/internal_ops.hpp"
#include "graph/backend/dnnl/utils.hpp"

#include "oneapi/dnnl/dnnl.hpp"

namespace dnnl {
namespace impl {
namespace graph {
namespace dnnl_impl {

// This class is used to represent an op's fusion information, such as the post
// ops, the zero points or scales.
class fusion_info_t {
    using op_ptr = std::shared_ptr<op_t>;

    // A meta op is used to represent the op that has been fused away, like the
    // scales op or post-ops op.
    class meta_op_t {
    public:
        // for scales and zps
        meta_op_t(const op_ptr &op) : op_(op) {};
        // for post-eltwise
        meta_op_t(const op_ptr &op, float scale) : op_(op), scale_(scale) {};
        // for post-sum and post_binary
        meta_op_t(const op_ptr &op,
                const std::vector<size_t> &extra_input_indices, float scale,
                int32_t zp)
            : op_(op)
            , scale_(scale)
            , zp_(zp)
            , unfused_input_indices_(extra_input_indices) {};
        // for post-conv
        meta_op_t(const op_ptr &op,
                const std::vector<size_t> &extra_input_indices)
            : op_(op), unfused_input_indices_(extra_input_indices) {};

        float get_scale() const { return scale_; }
        int32_t get_zp() const { return zp_; }
        const std::vector<size_t> &get_unfused_input_indices() const {
            return unfused_input_indices_;
        }

        const op_t *get_op() const { return op_.get(); }

        bool is_post_sum() const {
            return op_->get_kind() == op_kind::dnnl_binary && is_post_sum_;
        }

        bool is_post_binary() const {
            return op_->get_kind() == op_kind::dnnl_binary && !is_post_sum_;
        }

        void set_post_sum() { is_post_sum_ = true; }

    private:
        std::shared_ptr<op_t> op_;
        // used to represent post-eltwise and post-sum's scale
        float scale_ = 1.0f;
        // used to represent post-sum's zp
        int32_t zp_ = 0;
        // used to represent post-sum, post-binary and post-convolution's
        // unfused input index
        std::vector<size_t> unfused_input_indices_;
        // used to identify post-sum
        bool is_post_sum_ = false;
    };

public:
    friend dnnl::primitive_attr make_dnnl_primitive_attr(
            const op_ptr &op, const fusion_info_t &fusion_info);

    fusion_info_t() = default;

    // used to modify the fused arg scales, like modifying it's axis after
    // inserting reshape op
    op_t *get_mutable_scales(bool is_input, size_t index) {
        if (is_input) {
            if (input_scales_.find(index) == input_scales_.end())
                return nullptr;
            return const_cast<op_t *>(input_scales_.at(index)->get_op());
        } else {
            assertm(index == 0, "index for output scales must be 0");
            if (!dst_scales_) return nullptr;
            return const_cast<op_t *>(dst_scales_->get_op());
        }
    }

    void set_zero_points(const op_ptr &op, bool is_input, size_t index) {
        auto fused_zps = std::make_shared<meta_op_t>(op);
        if (is_input) {
            input_zps_[index] = std::move(fused_zps);
        } else {
            output_zps_ = std::move(fused_zps);
        }
    }

    void set_runtime_scales(const op_ptr &op, bool is_input, size_t index) {
        auto fused_scales = std::make_shared<meta_op_t>(op);
        if (is_input) {
            input_scales_[index] = std::move(fused_scales);
        } else {
            dst_scales_ = std::move(fused_scales);
        }
    }

    // used to modify the fused zps, like modifying it's axis after inserting
    // reshape op
    op_t *get_mutable_zero_points(bool is_input, size_t index) const {
        if (is_input) {
            if (input_zps_.find(index) == input_zps_.end()) return nullptr;
            return const_cast<op_t *>(input_zps_.at(index)->get_op());
        } else {
            assertm(index == 0, "index for output zps must be 0");
            if (!output_zps_) return nullptr;
            return const_cast<op_t *>(output_zps_->get_op());
        }
    }

    void append_post_eltwise(const op_ptr &op, float scale = 1.0f) {
        post_ops_.emplace_back(std::make_shared<meta_op_t>(op, scale));
    }

    // the extra input means the unfused input that has been added to the fused
    // op, like the following case, we fuse a binary mul into the conv, the src1
    // of mul op is unfused, and it becomes the 3rd input of conv. So the extra
    // input indices should be this input's index 2.
    //
    //   src   wei            src  wei  src1
    //     \   /                 \  |   /
    //      conv   src1   ->       conv
    //         \   /                 |
    //          mul
    //           |
    void append_post_binary(const op_ptr &op,
            const std::vector<size_t> &extra_input_indices, float scale = 1.0f,
            int32_t zp = 0) {
        post_ops_.emplace_back(std::make_shared<meta_op_t>(
                op, extra_input_indices, scale, zp));
    }

    // the meaning of extra input is same as that in append_post_binary function
    void append_post_dw_conv(
            const op_ptr &op, const std::vector<size_t> &extra_input_indices) {
        post_ops_.emplace_back(
                std::make_shared<meta_op_t>(op, extra_input_indices));
    }

    const std::vector<std::shared_ptr<meta_op_t>> &get_post_ops() const {
        return post_ops_;
    }

    bool has_post_dw_conv() const {
        auto pos = std::find_if(post_ops_.begin(), post_ops_.end(),
                [](const std::shared_ptr<meta_op_t> &mop) {
                    return mop->get_op()->get_kind()
                            == op_kind::dnnl_convolution;
                });
        return pos != post_ops_.end();
    }

    const std::shared_ptr<meta_op_t> &get_post_dw_conv() const {
        auto pos = std::find_if(post_ops_.begin(), post_ops_.end(),
                [](const std::shared_ptr<meta_op_t> &mop) {
                    return mop->get_op()->get_kind()
                            == op_kind::dnnl_convolution;
                });

        assertm(pos != post_ops_.end(), "cannot find post dw_conv");
        return *pos;
    }

    bool with_runtime_zero_points(bool is_input, size_t indice) const {
        if (is_input) {
            if (input_zps_.find(indice) == input_zps_.end()) return false;
            const op_t *zp_op
                    = const_cast<op_t *>(input_zps_.at(indice)->get_op());
            if (zp_op->has_attr(op_attr::with_runtime_zps)) {
                return zp_op->get_attr<bool>(op_attr::with_runtime_zps);
            } else {
                return false;
            }
        } else {
            if (!output_zps_) return false;
            const op_t *zp_op = const_cast<op_t *>(output_zps_->get_op());
            if (zp_op->has_attr(op_attr::with_runtime_zps)) {
                return zp_op->get_attr<bool>(op_attr::with_runtime_zps);
            } else {
                return false;
            }
        }
    }

    bool with_runtime_scales(bool is_input, size_t indice) const {
        if (is_input) {
            if (input_scales_.find(indice) == input_scales_.end()) return false;
            const op_t *zp_op
                    = const_cast<op_t *>(input_scales_.at(indice)->get_op());
            if (zp_op->has_attr(op_attr::with_runtime_scales)) {
                return zp_op->get_attr<bool>(op_attr::with_runtime_scales);
            } else {
                return false;
            }
        } else {
            if (!dst_scales_) return false;
            const op_t *zp_op = const_cast<op_t *>(dst_scales_->get_op());
            if (zp_op->has_attr(op_attr::with_runtime_scales)) {
                return zp_op->get_attr<bool>(op_attr::with_runtime_scales);
            } else {
                return false;
            }
        }
    }

private:
    std::unordered_map<size_t, std::shared_ptr<meta_op_t>> input_zps_;
    std::shared_ptr<meta_op_t> output_zps_;
    std::unordered_map<size_t, std::shared_ptr<meta_op_t>> input_scales_;
    std::shared_ptr<meta_op_t> dst_scales_;
    std::vector<std::shared_ptr<meta_op_t>> post_ops_;
};

// This class is used to manage all fusion infos in a subgraph. The
// fusion_info_t can't be directly stored in op's attribute system, so we store
// them in this manager class and then store the generated int64_t typed key in
// op's attribute system. When using an ops' fusion info, we can use the fusion
// info key to query it out from the manager.
class fusion_info_mgr_t {
public:
    fusion_info_mgr_t(fpmath_mode_t fpm_mode = fpmath_mode::strict,
            bool can_use_blocked_layout = false)
        : fpmath_mode_(fpm_mode)
        , can_use_blocked_layout_(can_use_blocked_layout) {}

    // Disable assignment and copy
    fusion_info_mgr_t(const fusion_info_mgr_t &) = delete;
    fusion_info_mgr_t(fusion_info_mgr_t &&) = delete;
    fusion_info_mgr_t &operator=(const fusion_info_mgr_t &) = delete;
    fusion_info_mgr_t &operator=(fusion_info_mgr_t &&) = delete;

    // Initialize an empty fusion info object and return its key
    int64_t init_info() {
        data_.emplace_back(fusion_info_t());
        return static_cast<int64_t>(data_.size() - 1);
    }

    // Get out a mutable fusion info reference according to the key
    fusion_info_t &get_mutable_info(int64_t key) {
        size_t k = static_cast<size_t>(key);
        assertm(k < data_.size(), "invalid key");
        return data_[k];
    }

    // Get out a constant fusion info reference according to the key
    const fusion_info_t &get_info(int64_t key) const {
        size_t k = static_cast<size_t>(key);
        assertm(k < data_.size(), "invalid key");
        return data_[k];
    }

    fpmath_mode_t get_fpmath_mode() const { return fpmath_mode_; }
    bool get_use_blocked_layout() const { return can_use_blocked_layout_; }

private:
    std::vector<fusion_info_t> data_;
    // specified floating-point math mode for all fusions
    fpmath_mode_t fpmath_mode_ {};
    bool can_use_blocked_layout_;
};

// This function is used to make a dnnl::primitive_attr from the fusion info.
// Note that the op and fusion_info arguments must be matched since a fusion
// info make sense only when it belongs to a specific op.
dnnl::primitive_attr make_dnnl_primitive_attr(
        const std::shared_ptr<op_t> &op, const fusion_info_t &fusion_info);

} // namespace dnnl_impl
} // namespace graph
} // namespace impl
} // namespace dnnl

#endif
