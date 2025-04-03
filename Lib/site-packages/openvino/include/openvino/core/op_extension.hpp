// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/core/attribute_visitor.hpp"
#include "openvino/core/extension.hpp"
#include "openvino/core/node.hpp"
#include "openvino/core/node_vector.hpp"

namespace ov {

/**
 * @brief The base interface for OpenVINO operation extensions
 */
class OPENVINO_API BaseOpExtension : public Extension {
public:
    using Ptr = std::shared_ptr<BaseOpExtension>;
    /**
     * @brief Returns the type info of operation
     *
     * @return ov::DiscreteTypeInfo
     */
    virtual const ov::DiscreteTypeInfo& get_type_info() const = 0;
    /**
     * @brief Method creates an OpenVINO operation
     *
     * @param inputs vector of input ports
     * @param visitor attribute visitor which allows to read necessaty arguments
     *
     * @return vector of output ports
     */
    virtual ov::OutputVector create(const ov::OutputVector& inputs, ov::AttributeVisitor& visitor) const = 0;

    /**
     * @brief Returns extensions that should be registered together with this extension class object.
     *
     * Attached extensions may include frontend extensions that OpenVINO op to framework ops or necessary
     * transformations that should be applied to the network which consist of target op.
     *
     * @return
     */
    virtual std::vector<ov::Extension::Ptr> get_attached_extensions() const = 0;

    /**
     * @brief Destructor
     */
    virtual ~BaseOpExtension() override;
};

namespace detail {
#define OV_COLLECT_ATTACHED_EXTENSIONS(FRAMEWORK)                                                         \
    template <class T>                                                                                    \
    static auto collect_attached_extensions_##FRAMEWORK(std::vector<ov::Extension::Ptr>& res)             \
        ->decltype(typename T::template __openvino_framework_map_helper_##FRAMEWORK<T>().get(), void()) { \
        res.emplace_back(typename T::template __openvino_framework_map_helper_##FRAMEWORK<T>().get());    \
    }                                                                                                     \
    template <class>                                                                                      \
    static auto collect_attached_extensions_##FRAMEWORK(ov::Any)->void {}

OV_COLLECT_ATTACHED_EXTENSIONS(onnx)
OV_COLLECT_ATTACHED_EXTENSIONS(paddle)
OV_COLLECT_ATTACHED_EXTENSIONS(tensorflow)
OV_COLLECT_ATTACHED_EXTENSIONS(pytorch)
}  // namespace detail

/**
 * @brief The default implementation of OpenVINO operation extensions
 */
template <class T>
class OpExtension : public BaseOpExtension {
public:
    /**
     * @brief Default constructor
     */
    OpExtension() {
        const auto& ext_type = get_type_info();
        OPENVINO_ASSERT(ext_type.name != nullptr && ext_type.version_id != nullptr,
                        "Extension type should have information about operation set and operation type.");
    }

    const ov::DiscreteTypeInfo& get_type_info() const override {
        return T::get_type_info_static();
    }

    ov::OutputVector create(const ov::OutputVector& inputs, ov::AttributeVisitor& visitor) const override {
        std::shared_ptr<ov::Node> node = std::make_shared<T>();

        node->set_arguments(inputs);
        if (node->visit_attributes(visitor)) {
            node->constructor_validate_and_infer_types();
        }
        return node->outputs();
    }

    std::vector<ov::Extension::Ptr> get_attached_extensions() const override {
        std::vector<ov::Extension::Ptr> res;
        detail::collect_attached_extensions_onnx<T>(res);
        detail::collect_attached_extensions_paddle<T>(res);
        detail::collect_attached_extensions_tensorflow<T>(res);
        detail::collect_attached_extensions_pytorch<T>(res);
        return res;
    }
};

}  // namespace ov
