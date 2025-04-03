// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/core/node_output.hpp"
#include "openvino/core/type/element_type.hpp"
#include "openvino/frontend/variable.hpp"
#include "openvino/frontend/visibility.hpp"

namespace ov {
namespace frontend {

/// \brief HashTable is a special type of Variable that has a complex value including keys and values.
/// Keys and values are represented with two separate graph at each time step
class FRONTEND_API HashTable : public Variable {
public:
    using Ptr = std::shared_ptr<HashTable>;
    OPENVINO_OP("HashTable", "ov::frontend", Variable);

    HashTable(const std::string& name,
              const ov::element::Type& key_type,
              const ov::element::Type& value_type,
              const std::shared_ptr<DecoderBase>& decoder = nullptr)
        : Variable(name, decoder),
          m_key_type(key_type),
          m_value_type(value_type) {
        validate_and_infer_types();
    }

    HashTable(const HashTable& other, const ov::Output<ov::Node>& keys, const ov::Output<ov::Node>& values)
        : HashTable(other) {
        m_keys = keys;
        m_values = values;
        m_is_initialized = true;
        ++m_init_counter;
    }

    // it must be used only for cloning
    // other ways are illegal
    HashTable(const std::string& name,
              const ov::element::Type& key_type,
              const ov::element::Type& value_type,
              const ov::Output<ov::Node>& keys,
              const ov::Output<ov::Node>& values,
              bool is_initialized,
              uint64_t init_counter,
              const std::shared_ptr<DecoderBase>& decoder = nullptr)
        : Variable(name, decoder),
          m_key_type(key_type),
          m_value_type(value_type),
          m_keys(keys),
          m_values(values) {
        m_init_counter = init_counter;
        m_is_initialized = is_initialized;
        validate_and_infer_types();
    }

    void validate_and_infer_types() override {
        // this is a type of resource so its shape and type is not applicable
        // its output serves to store a reference to a resource
        set_output_type(0, ov::element::dynamic, ov::PartialShape::dynamic());
        // these two outputs serves to store keys and values of a resource
        // keys and values are 1D tensors
        set_output_type(1, m_key_type, ov::PartialShape::dynamic(1));
        set_output_type(2, m_value_type, ov::PartialShape::dynamic(1));
    }

    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& inputs) const override {
        auto hash_table_node = std::make_shared<HashTable>(m_name,
                                                           m_key_type,
                                                           m_value_type,
                                                           m_keys,
                                                           m_values,
                                                           m_is_initialized,
                                                           m_init_counter,
                                                           m_decoder);
        hash_table_node->set_attrs(get_attrs());
        return hash_table_node;
    }

    ov::Output<ov::Node> get_value() override {
        return output(0);
    }

    /// \brief Returns a value corresponding keys of hash table
    ov::Output<ov::Node> get_keys() {
        if (m_is_initialized) {
            return m_keys;
        } else if (m_other_keys.size() > 0) {
            return *(m_other_keys.begin());
        }

        return output(1);
    }

    /// \brief Returns a value corresponding values of hash table
    ov::Output<ov::Node> get_values() {
        if (m_is_initialized) {
            return m_values;
        } else if (m_other_values.size() > 0) {
            return *(m_other_values.begin());
        }

        return output(2);
    }

    ov::element::Type get_key_type() const {
        return m_key_type;
    }

    ov::element::Type get_value_type() const {
        return m_value_type;
    }

    void add_other_keys_values(const ov::Output<ov::Node>& other_key, const ov::Output<ov::Node>& other_value) {
        m_other_keys.insert(other_key);
        m_other_values.insert(other_value);
    }

    virtual ~HashTable();

private:
    ov::element::Type m_key_type;
    ov::element::Type m_value_type;
    ov::Output<ov::Node> m_keys;
    ov::Output<ov::Node> m_values;

    std::set<ov::Output<ov::Node>> m_other_keys;
    std::set<ov::Output<ov::Node>> m_other_values;
};

}  // namespace frontend
}  // namespace ov
