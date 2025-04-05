// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/core/any.hpp"
#include "openvino/frontend/exception.hpp"
#include "openvino/frontend/visibility.hpp"

namespace ov {
namespace frontend {

// Extendable type system which reflects Framework data types
// Type nestings are built with the help of ov::Any
namespace type {

struct Tensor {
    Tensor() = default;
    explicit Tensor(const Any& _element_type) : element_type(_element_type) {}
    Any element_type;
};

struct Tuple;

struct List {
    List() = default;

    // Specifies list of elements of element_type type, all elements have the same given type
    explicit List(const Any& _element_type) : element_type(_element_type) {}
    Any element_type;
};

struct Str {};

struct PyNone {};

struct PyScalar {
    PyScalar() = default;
    explicit PyScalar(const Any& _element_type) : element_type(_element_type) {}
    Any element_type;
};

struct Optional;
struct Dict;
struct NamedTuple;
struct Union;

}  // namespace type

/// Plays a role of node, block and module decoder
class FRONTEND_API IDecoder {
public:
    virtual ~IDecoder();
};

class FRONTEND_API DecoderBase : public IDecoder {
public:
    using OpTypeByName = std::unordered_map<std::string, std::string>;
    /// \brief Get attribute value by name
    ///
    /// \param name Attribute name
    /// \return Shared pointer to appropriate value converted to openvino data type if it exists, 'nullptr' otherwise
    virtual ov::Any get_attribute(const std::string& name) const = 0;

    /// \brief Get a number of inputs
    virtual size_t get_input_size() const = 0;

    /// \brief Get a producer name and its output port index
    ///
    /// \param input_port_idx              Input port index by which data is consumed
    /// \param producer_name               A producer name
    /// \param producer_output_port_name   Output port name if exists
    /// \param producer_output_port_index  Output port index from which data is generated
    virtual void get_input_node(size_t input_port_idx,
                                std::string& producer_name,
                                std::string& producer_output_port_name,
                                size_t& producer_output_port_index) const = 0;

    /// \brief Get operation type
    virtual const std::string& get_op_type() const = 0;

    /// \brief Get node name
    virtual const std::string& get_op_name() const = 0;

    /// \brief Destructor
    virtual ~DecoderBase();
};

}  // namespace frontend
}  // namespace ov
