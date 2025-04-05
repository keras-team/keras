// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <string>
#include <unordered_map>
#include <utility>

#include "openvino/core/type.hpp"

namespace ov {
class Node;
class Model;
template <typename T>
class ValueAccessor;
template <typename T>
class AttributeAdapter;
class VisitorAdapter;

/// \brief Visits the attributes of a node, primarily for serialization-like tasks.
///
/// Attributes are the node parameters that are always compile-time constants.
/// Values computed from the graph topology and attributes during compilation are not
/// attributes.
///
/// Attributes have a wide variety of types, but serialization formats are more restricted.
/// We assume serialization easily supports scalar types of bool 64-bit signed, string, and double,
/// and has specialized ways to support numeric arrays and raw data+size. The visitor and
/// adapter convert between the limited serialization types and the unlimited attribute types.
///
/// A visitor is passed to an op's visit_attributes method. The visit_attributes method calls
/// the template method visitor.on_attribute<AT>(const std::string& name, AT& value) on each
/// attribute. The visitor can read or write the attribute's value. The on_attribute
/// method creates an AttributeAdapter<AT> for the value and passes it to one of the visitors
/// on_adapter methods. The on_adapter methods expect a reference to a ValueAccessor<VAT> or a
/// VisitorAdapter. A ValueAccessor<VAT> has get/set methods that can be used to read/write the
/// attribute value as type VAT. These methods are triggered by deriving AttributeAdapter<AT>
/// from ValueAccessor<VAT>. For more complex cases, such as structs, the on_adapter method for
/// VisitorAdapter passes the name and visitor to the adapter, so that the adapter can perform
/// additional work such as visiting struct members or sequence values.
///
/// When a node visits an attribute with structure, the node's on_attribute passes a name for
/// the entire attribute, but the struct will have its own methods to be visited. Similarly, a
/// vector will have a sequence of members to be visited. The adapter may use the visitor
/// methods start_struct/finish_struct and start_vector/next_vector/finish_vector to inidicate
/// nexted members.
///
/// The visitor method get_name_with_context creates a generic nested version of the name.
/// Visitors can override according to their serialization requirements.
///
/// Attributes that are shared_ptr<Node> are special. They must have been already been
/// registered with the visitor using register_node, which needs a shared pointer to a node and
/// a string ID. The ID string will be used to serialize the node or find the node during
/// deserialization.
class OPENVINO_API AttributeVisitor {
public:
    virtual ~AttributeVisitor();
    // Must implement these methods
    /// \brief handles all specialized on_adapter methods implemented by the visitor.
    ///
    /// The adapter implements get_type_info(), which can be used to determine the adapter
    /// directly
    /// or via is_type and as_type on any platform
    virtual void on_adapter(const std::string& name, ValueAccessor<void>& adapter) = 0;
    // The remaining adapter methods fall back on the void adapter if not implemented
    virtual void on_adapter(const std::string& name, ValueAccessor<void*>& adapter);
    virtual void on_adapter(const std::string& name, ValueAccessor<std::string>& adapter);
    virtual void on_adapter(const std::string& name, ValueAccessor<bool>& adapter);
    virtual void on_adapter(const std::string& name, ValueAccessor<int8_t>& adapter);
    virtual void on_adapter(const std::string& name, ValueAccessor<int16_t>& adapter);
    virtual void on_adapter(const std::string& name, ValueAccessor<int32_t>& adapter);
    virtual void on_adapter(const std::string& name, ValueAccessor<int64_t>& adapter);
    virtual void on_adapter(const std::string& name, ValueAccessor<uint8_t>& adapter);
    virtual void on_adapter(const std::string& name, ValueAccessor<uint16_t>& adapter);
    virtual void on_adapter(const std::string& name, ValueAccessor<uint32_t>& adapter);
    virtual void on_adapter(const std::string& name, ValueAccessor<uint64_t>& adapter);
    virtual void on_adapter(const std::string& name, ValueAccessor<float>& adapter);
    virtual void on_adapter(const std::string& name, ValueAccessor<double>& adapter);
    virtual void on_adapter(const std::string& name, ValueAccessor<std::vector<int8_t>>& adapter);
    virtual void on_adapter(const std::string& name, ValueAccessor<std::vector<int16_t>>& adapter);
    virtual void on_adapter(const std::string& name, ValueAccessor<std::vector<int32_t>>& adapter);
    virtual void on_adapter(const std::string& name, ValueAccessor<std::vector<int64_t>>& adapter);
    virtual void on_adapter(const std::string& name, ValueAccessor<std::vector<uint8_t>>& adapter);
    virtual void on_adapter(const std::string& name, ValueAccessor<std::vector<uint16_t>>& adapter);
    virtual void on_adapter(const std::string& name, ValueAccessor<std::vector<uint32_t>>& adapter);
    virtual void on_adapter(const std::string& name, ValueAccessor<std::vector<uint64_t>>& adapter);
    virtual void on_adapter(const std::string& name, ValueAccessor<std::vector<float>>& adapter);
    virtual void on_adapter(const std::string& name, ValueAccessor<std::vector<double>>& adapter);
    virtual void on_adapter(const std::string& name, ValueAccessor<std::vector<std::string>>& adapter);
    /// \brief Hook for adapters that need visitor access
    virtual void on_adapter(const std::string& name, VisitorAdapter& adapter);

    /// \brief Provides API to handle openvino Function attribute type, accessed as ValueAccessor
    /// \param name attribute name
    /// \param adapter reference to a Function ValueAccessor<VAT>
    virtual void on_adapter(const std::string& name, ValueAccessor<std::shared_ptr<ov::Model>>& adapter);

    /// The generic visitor. There must be a definition of AttributeAdapter<T> that can convert
    /// to a ValueAccessor<U> for one of the on_adpater methods.
    template <typename AT>
    void on_attribute(const std::string& name, AT& value) {
        AttributeAdapter<AT> adapter(value);
        start_structure(name);
        on_adapter(get_name_with_context(), adapter);
        finish_structure();
    }
    /// \returns The nested context of visits
    const std::vector<std::string>& get_context() const {
        return m_context;
    }
    /// \returns context prepended to names
    virtual std::string get_name_with_context();
    /// \brief Start visiting a nested structure
    virtual void start_structure(const std::string& name);
    /// \brief Finish visiting a nested structure
    virtual std::string finish_structure();
    using node_id_t = std::string;
    static constexpr const char* invalid_node_id = "";
    /// \brief Associate a node with an id.
    ///
    /// No node may be used as an attribute unless it has already been registered with an ID.
    /// References to nodes are visited with a ValueAccessor of their ID.
    virtual void register_node(const std::shared_ptr<Node>& node, node_id_t id = invalid_node_id);
    /// Returns the node with the given id, or nullptr if there is no registered node
    virtual std::shared_ptr<Node> get_registered_node(node_id_t id);
    /// Returns the id for the node, or -1 if the node is not registered
    virtual node_id_t get_registered_node_id(const std::shared_ptr<Node>& node);

protected:
    std::vector<std::string> m_context;
    std::unordered_map<std::shared_ptr<Node>, node_id_t> m_node_id_map;
    std::unordered_map<node_id_t, std::shared_ptr<Node>> m_id_node_map;
};
}  // namespace ov
