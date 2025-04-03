// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/frontend/decoder.hpp"
#include "openvino/frontend/visibility.hpp"
#include "openvino/op/util/framework_node.hpp"

namespace ov {
namespace frontend {

/// \brief Variable is a special node used in a conversion step
/// It can have several values (or states) during the conversion.
/// Variable value at some time step is represented with a graph.
class FRONTEND_API Variable : public ov::op::util::FrameworkNode {
public:
    using Ptr = std::shared_ptr<Variable>;
    OPENVINO_OP("Variable", "ov::frontend", ov::op::util::FrameworkNode);

    Variable(const std::string& name, const std::shared_ptr<DecoderBase>& decoder)
        : ov::op::util::FrameworkNode(ov::OutputVector{}, 1),
          m_name(name),
          m_shape(ov::Shape{}),
          m_type(ov::element::dynamic),
          m_decoder(decoder),
          m_is_initialized(false),
          m_init_counter(0) {
        validate_and_infer_types();
    }

    Variable(const std::string& name,
             const ov::Shape& shape,
             const ov::element::Type& type,
             const std::shared_ptr<DecoderBase>& decoder)
        : ov::op::util::FrameworkNode(ov::OutputVector{}, 1),
          m_name(name),
          m_shape(shape),
          m_type(type),
          m_decoder(decoder),
          m_is_initialized(false),
          m_init_counter(0) {
        validate_and_infer_types();
    }

    Variable(const std::string& name,
             const ov::Shape& shape,
             const ov::element::Type& type,
             const ov::Output<ov::Node>& value,
             const std::shared_ptr<DecoderBase>& decoder)
        : Variable(name, shape, type, decoder) {
        m_value = value;
        // reset names of tensor corresponding to variable value
        // that is because variable can have multiple values during inference
        m_value.set_names({});
        m_is_initialized = true;
        ++m_init_counter;
    }

    Variable(const Variable& other, const ov::Output<ov::Node>& value) : Variable(other) {
        m_value = value;
        // reset names of tensor corresponding to variable value
        // that is because variable can have multiple values during inference
        m_value.set_names({});
        m_is_initialized = true;
        ++m_init_counter;
    }

    void validate_and_infer_types() override {
        set_output_type(0, m_type, m_shape);
    }

    /// \brief Checks if variable is initialized with some value
    bool is_initialized() const {
        return m_is_initialized;
    }

    /// \brief Returns a value at the current step of conversion
    virtual ov::Output<ov::Node> get_value() {
        FRONT_END_GENERAL_CHECK(m_is_initialized, "internal error: get_value() is called for uninitialized variable");
        return m_value;
    }

    std::string get_name() const {
        return m_name;
    }

    /// \brief Returns a counter value (a number of values that have assigned to this variable)
    uint64_t get_init_counter() const {
        return m_init_counter;
    }

    std::shared_ptr<DecoderBase> get_decoder() const {
        return m_decoder;
    }

    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& inputs) const override {
        auto new_variable = std::make_shared<Variable>(*this);
        new_variable->set_attrs(get_attrs());
        return new_variable;
    }

    virtual ~Variable();

protected:
    std::string m_name;
    ov::Shape m_shape;
    ov::element::Type m_type;
    std::shared_ptr<DecoderBase> m_decoder;
    bool m_is_initialized;
    ov::Output<ov::Node> m_value;
    uint64_t m_init_counter;
};

}  // namespace frontend
}  // namespace ov
