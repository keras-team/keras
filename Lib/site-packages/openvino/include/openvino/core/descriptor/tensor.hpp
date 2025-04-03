// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <atomic>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_set>

#include "openvino/core/any.hpp"
#include "openvino/core/core_visibility.hpp"
#include "openvino/core/partial_shape.hpp"
#include "openvino/core/shape.hpp"
#include "openvino/core/type/element_type.hpp"
#include "openvino/runtime/tensor.hpp"

namespace ov {
class Node;
/// \brief Alias for symbol tensor.
using TensorSymbol = std::vector<std::shared_ptr<Symbol>>;

/// \brief Alias for vector of symbol tensors.
using TensorSymbolVector = std::vector<TensorSymbol>;

namespace descriptor {
class ITensorDescriptor;

/// \brief Compile-time descriptor of a first-class value that is a tensor.
class OPENVINO_API Tensor {
public:
    /// \brief Creates Tensor descriptor
    /// \param element_type Element type
    /// \param pshape       Partial shape of tensor
    /// \param names        Tensor names (optional default empty).
    Tensor(const element::Type& element_type,
           const PartialShape& pshape,
           const std::unordered_set<std::string>& names = {});

    OPENVINO_DEPRECATED("This constructor is deprecated. Will be removed in 2026.0")
    Tensor(const element::Type& element_type, const PartialShape& pshape, Node* node, size_t node_output_number);

    Tensor(const Tensor&) = delete;
    Tensor& operator=(const Tensor&) = delete;

    /// \brief Gets any tensor name.
    /// Throws if tensor has no names.
    const std::string& get_any_name() const;

    /// \brief Gets tensor names
    const std::unordered_set<std::string>& get_names() const;

    /// \brief Set new names.
    /// \param names Names to set.
    void set_names(const std::unordered_set<std::string>& names);

    /// \brief Adds new names to tensor.
    /// \param names new names to be added.
    void add_names(const std::unordered_set<std::string>& names);

    /// \brief sets lower bound value description
    void set_lower_value(const ov::Tensor& value);

    /// \brief sets upper bound value description
    void set_upper_value(const ov::Tensor& value);

    /// \brief sets value symbol description
    void set_value_symbol(const TensorSymbol& value_symbol);

    /// \brief unsets bound value descriptions
    void invalidate_values();

    /// \brief Gets element type.
    const element::Type& get_element_type() const;

    /// \brief Gets shape.
    /// Throw if Tensor's shape is not static.
    const Shape& get_shape() const;

    /// \brief Gets partial shape.
    const PartialShape& get_partial_shape() const;

    /// \brief gets lower bound value description
    const ov::Tensor& get_lower_value() const;

    /// \brief gets upper bound value description
    const ov::Tensor& get_upper_value() const;

    /// \brief gets symbol value description
    TensorSymbol get_value_symbol() const;

    /// \brief checks if lower and upper bound are set and point to the same Tensor
    bool has_and_set_bound() const;

    /// \brief Get Tensor size in bytes.
    /// \return Size in bytes.
    size_t size() const;

    /// \brief  Gets runtime informations.
    /// \return Runtime information map which can be modified.
    RTMap& get_rt_info();

    /// \brief  Gets runtime informations.
    /// \return Read only runtime information map.
    const RTMap& get_rt_info() const;

    /// \brief  Clones Tensor from the other.
    /// \param other  Tensor used to clone its properties.
    void clone_from(const Tensor& other);

protected:
    ov::Tensor m_lower_value, m_upper_value;
    TensorSymbol m_value_symbol;
    std::shared_ptr<ITensorDescriptor> m_impl;

private:
    // hidden extension API for Tensor descriptor
    friend struct TensorExtension;
};

OPENVINO_API
std::ostream& operator<<(std::ostream&, const ov::descriptor::Tensor&);
}  // namespace descriptor

}  // namespace ov
