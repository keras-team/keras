// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>

#include "openvino/core/core_visibility.hpp"

namespace ov {

class Symbol;

namespace symbol {
/// \brief If both symbols are valid, sets them as equal
void OPENVINO_API set_equal(const std::shared_ptr<Symbol>& lhs, const std::shared_ptr<Symbol>& rhs);
/// \brief Returns true if both symbols are valid and are equal otherwise returns false
bool OPENVINO_API are_equal(const std::shared_ptr<Symbol>& lhs, const std::shared_ptr<Symbol>& rhs);
/// \brief Returns a representative (the most distant parent) of an equality group of this symbol
std::shared_ptr<Symbol> OPENVINO_API ancestor_of(const std::shared_ptr<Symbol>& x);
}  // namespace symbol

/// \brief Class representing unique symbol for the purpose of symbolic shape inference. Equality of symbols is being
/// tracked by Disjoint-set data structure
/// \ingroup ov_model_cpp_api
class OPENVINO_API Symbol {
public:
    /// \brief Default constructs a unique symbol
    Symbol() = default;

private:
    friend std::shared_ptr<Symbol> ov::symbol::ancestor_of(const std::shared_ptr<Symbol>& x);
    friend void ov::symbol::set_equal(const std::shared_ptr<Symbol>& lhs, const std::shared_ptr<Symbol>& rhs);

    std::shared_ptr<Symbol> m_parent = nullptr;
};

}  // namespace ov
