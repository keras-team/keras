// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/core/runtime_attribute.hpp"
#include "openvino/pass/pass.hpp"

namespace ov {
namespace pass {

/**
 * @brief Constant folding iterates over the function and tries to evaluate nodes
 *        with constant inputs. Such nodes are then replaced with new Constants containing
 *        the result of a folded operation.
 * @ingroup ov_pass_cpp_api
 */
class OPENVINO_API ConstantFolding : public ModelPass {
public:
    OPENVINO_MODEL_PASS_RTTI("ConstantFolding");
    bool run_on_model(const std::shared_ptr<ov::Model>& model) override;

protected:
    void copy_runtime_info_from_input_values(const std::shared_ptr<Node>& node);
    /// \brief Folds pre-calculated output tensor values to constants in case lower and
    /// upper estimations are equal. Traverses graph backwards starting from the results.
    bool pre_calculated_values_folding(const std::shared_ptr<ov::Model>& model);
};

/**
 * @brief this method disables constant folding for given node. Under constant folding we consider ConstantFolding
 *        transformation, so other type of constant folding like `get_constant_from_source` doesn't work with
 *        this attribute. Also before using this attribute please consider two corner cases:
 *        1. If for sub-graph like ShapeOf->ShapeOf we disable cf for first ShapeOf node, it doesn't spread
 *        to the second ShapeOf, so the entire sub-graph will be folded. (In case if first ShapeOf has exactly one
 *        consumer)
 *        2. If node with disable_constant_folding was replaced with another node, the attribute will be lost because
 *        it is not copyable.
 * @ingroup ov_pass_cpp_api
 */
OPENVINO_API void disable_constant_folding(const std::shared_ptr<Node>& node);

OPENVINO_API void enable_constant_folding(const std::shared_ptr<Node>& node);

/**
 * @brief Check if constant folding is disabled on @ref Node.
 *
 * @param node  Smart pointer to the node.
 *
 * @return true if attribute constant folding set otherwise false.
 * @ingroup ov_pass_cpp_api
 */
OPENVINO_API bool constant_folding_is_disabled(const std::shared_ptr<Node>& node);

/**
 * @brief Check if constant folding is disabled on @ref Node.
 *
 * @param node  Pointer to the node.
 *
 * @return true if attribute constant folding set otherwise false.
 * @ingroup ov_pass_cpp_api
 */
OPENVINO_API bool constant_folding_is_disabled(const Node* const node);

class OPENVINO_API DisableConstantFolding : public ov::RuntimeAttribute {
public:
    OPENVINO_RTTI("DisableConstantFolding", "0", ov::RuntimeAttribute);
    DisableConstantFolding() = default;
    bool is_copyable() const override {
        return false;
    }
};
}  // namespace pass
}  // namespace ov
