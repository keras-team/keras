// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
#include "openvino/core/any.hpp"
#include "openvino/frontend/node_context.hpp"
#include "openvino/frontend/paddle/visibility.hpp"

namespace ov {
namespace frontend {
namespace paddle {
using InPortName = std::string;
using OutPortName = std::string;
using TensorName = std::string;
using NamedOutputs = std::map<OutPortName, OutputVector>;
using NamedInputs = std::map<InPortName, OutputVector>;

class DecoderBase {
public:
    /// \brief Get attribute value by name and requested type
    ///
    /// \param name Attribute name
    /// \return Shared pointer to appropriate value if it exists, 'nullptr' otherwise
    virtual ov::Any get_attribute(const std::string& name) const = 0;

    /// \brief Applies additional conversion rules to the data based on type_info
    ///
    /// \param data Data
    /// \param type_info Attribute type information
    /// \return Shared pointer to appropriate value if it exists, 'nullptr' otherwise
    virtual ov::Any convert_attribute(const ov::Any& data, const std::type_info& type_info) const = 0;

    /// \brief Get the output names
    virtual std::vector<OutPortName> get_output_names() const = 0;
    virtual std::vector<TensorName> get_output_var_names(const std::string& var_name) const = 0;
    virtual std::vector<TensorName> get_input_var_names(const std::string& var_name) const = 0;

    /// \brief Get the output size
    virtual size_t get_output_size() const = 0;
    virtual size_t get_output_size(const std::string& port_name) const = 0;

    /// \brief Get the version
    virtual int64_t get_version() const = 0;

    /// \brief Get output port type
    ///
    /// Current API assumes that output port has only one output type.
    /// If decoder supports multiple types for specified port, it shall throw general
    /// exception
    ///
    /// \param port_name Port name for the node
    ///
    /// \return Type of specified output port
    virtual ov::element::Type get_out_port_type(const std::string& port_name) const = 0;
    virtual std::vector<std::pair<ov::element::Type, ov::PartialShape>> get_output_port_infos(
        const std::string& port_name) const = 0;

    /// \brief Get the type of the operation
    virtual std::string get_op_type() const = 0;

    /// \brief Destructor
    virtual ~DecoderBase();
};
}  // namespace paddle
}  // namespace frontend
}  // namespace ov
