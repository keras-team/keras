// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <locale>
#include <map>
#include <mutex>
#include <set>
#include <utility>

#include "openvino/core/node.hpp"

namespace ov {
/**
 * @brief Run-time opset information
 * @ingroup ov_opset_cpp_api
 */
class OPENVINO_API OpSet {
public:
    OpSet() = default;
    OpSet(const std::string& name);
    OpSet(const OpSet& opset);
    virtual ~OpSet();
    OpSet& operator=(const OpSet& opset);

    std::set<NodeTypeInfo>::size_type size() const;

    /// \brief Insert OP_TYPE into the opset with a special name and the default factory
    template <typename OP_TYPE>
    void insert(const std::string& name) {
        std::lock_guard<std::mutex> guard(opset_mutex);
        insert(name, OP_TYPE::get_type_info_static(), get_op_default_ctor<OP_TYPE>());
    }

    /// \brief Insert OP_TYPE into the opset with the default name and factory
    template <typename OP_TYPE>
    void insert() {
        insert<OP_TYPE>(OP_TYPE::get_type_info_static().name);
    }

    const std::set<NodeTypeInfo>& get_types_info() const;

    /// \brief Create the op named name using it's factory
    ov::Node* create(const std::string& name) const;

    /// \brief Create the op named name using it's factory
    ov::Node* create_insensitive(const std::string& name) const;

    /// \brief Return true if OP_TYPE is in the opset
    bool contains_type(const NodeTypeInfo& type_info) const;

    /// \brief Return true if OP_TYPE is in the opset
    template <typename OP_TYPE>
    bool contains_type() const {
        return contains_type(OP_TYPE::get_type_info_static());
    }

    /// \brief Return true if name is in the opset
    bool contains_type(const std::string& name) const;

    /// \brief Return true if name is in the opset
    bool contains_type_insensitive(const std::string& name) const;

    /// \brief Return true if node's type is in the opset
    bool contains_op_type(const Node* node) const;

    const std::set<NodeTypeInfo>& get_type_info_set() const;

protected:
    /// \brief Factory function which create object using default ctor.
    using DefaultOp = std::function<Node*()>;
    /// \brief Factory map hold object type_info as key and Factory function.
    using FactoryMap = std::unordered_map<typename Node::type_info_t, DefaultOp>;

    /// \brief Insert an op into the opset
    void insert(const std::string& name, const NodeTypeInfo& type_info, DefaultOp func);

    FactoryMap m_factory_registry;

private:
    /// \brief Get the default factory for OP_TYPE. Specialize as needed.
    template <typename OP_TYPE>
    static DefaultOp get_op_default_ctor() {
        return [] {
            return new OP_TYPE();
        };
    }

    static std::string to_upper_name(const std::string& name);

    std::string m_name;
    std::set<NodeTypeInfo> m_op_types;
    std::map<std::string, NodeTypeInfo> m_name_type_info_map;
    std::map<std::string, NodeTypeInfo> m_case_insensitive_type_info_map;
    mutable std::mutex opset_mutex;
};

/**
 * @brief Returns opset1
 * @ingroup ov_opset_cpp_api
 */
const OPENVINO_API OpSet& get_opset1();
/**
 * @brief Returns opset2
 * @ingroup ov_opset_cpp_api
 */
const OPENVINO_API OpSet& get_opset2();
/**
 * @brief Returns opset3
 * @ingroup ov_opset_cpp_api
 */
const OPENVINO_API OpSet& get_opset3();
/**
 * @brief Returns opset4
 * @ingroup ov_opset_cpp_api
 */
const OPENVINO_API OpSet& get_opset4();
/**
 * @brief Returns opset5
 * @ingroup ov_opset_cpp_api
 */
const OPENVINO_API OpSet& get_opset5();
/**
 * @brief Returns opset6
 * @ingroup ov_opset_cpp_api
 */
const OPENVINO_API OpSet& get_opset6();
/**
 * @brief Returns opset7
 * @ingroup ov_opset_cpp_api
 */
const OPENVINO_API OpSet& get_opset7();
/**
 * @brief Returns opset8
 * @ingroup ov_opset_cpp_api
 */
const OPENVINO_API OpSet& get_opset8();
/**
 * @brief Returns opset9
 * @ingroup ov_opset_cpp_api
 */
const OPENVINO_API OpSet& get_opset9();
/**
 * @brief Returns opset10
 * @ingroup ov_opset_cpp_api
 */
const OPENVINO_API OpSet& get_opset10();
/**
 * @brief Returns opset11
 * @ingroup ov_opset_cpp_api
 */
const OPENVINO_API OpSet& get_opset11();
/**
 * @brief Returns opset12
 * @ingroup ov_opset_cpp_api
 */
const OPENVINO_API OpSet& get_opset12();
/**
 * @brief Returns opset13
 * @ingroup ov_opset_cpp_api
 */
const OPENVINO_API OpSet& get_opset13();
/**
 * @brief Returns opset14
 * @ingroup ov_opset_cpp_api
 */
const OPENVINO_API OpSet& get_opset14();
/**
 * @brief Returns opset15
 * @ingroup ov_opset_cpp_api
 */
const OPENVINO_API OpSet& get_opset15();
/**
 * @brief Returns opset16
 * @ingroup ov_opset_cpp_api
 */
const OPENVINO_API OpSet& get_opset16();
/**
 * @brief Returns map of available opsets
 * @ingroup ov_opset_cpp_api
 */
const OPENVINO_API std::map<std::string, std::function<const ov::OpSet&()>>& get_available_opsets();
}  // namespace ov
