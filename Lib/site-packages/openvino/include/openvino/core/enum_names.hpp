// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <algorithm>
#include <string>
#include <utility>
#include <vector>

#include "openvino/core/except.hpp"

namespace ov {
/// Uses a pairings defined by EnumTypes::get() to convert between strings
/// and enum values.
template <typename EnumType>
class EnumNames {
public:
    /// Converts strings to enum values
    static EnumType as_enum(const std::string& name) {
        auto to_lower = [](const std::string& s) {
            std::string rc = s;
            std::transform(rc.begin(), rc.end(), rc.begin(), [](char c) {
                return static_cast<char>(::tolower(static_cast<int>(c)));
            });
            return rc;
        };
        for (const auto& p : get().m_string_enums) {
            if (to_lower(p.first) == to_lower(name)) {
                return p.second;
            }
        }
        OPENVINO_ASSERT(false, "\"", name, "\"", " is not a member of enum ", get().m_enum_name);
    }

    /// Converts enum values to strings
    static const std::string& as_string(EnumType e) {
        for (const auto& p : get().m_string_enums) {
            if (p.second == e) {
                return p.first;
            }
        }
        OPENVINO_ASSERT(false, " invalid member of enum ", get().m_enum_name);
    }

private:
    /// Creates the mapping.
    EnumNames(const std::string& enum_name, const std::vector<std::pair<std::string, EnumType>> string_enums)
        : m_enum_name(enum_name),
          m_string_enums(string_enums) {}

    /// Must be defined to returns a singleton for each supported enum class
    static EnumNames<EnumType>& get();

    const std::string m_enum_name;
    std::vector<std::pair<std::string, EnumType>> m_string_enums;
};

/// Returns the enum value matching the string
template <typename Type, typename Value>
typename std::enable_if<std::is_convertible<Value, std::string>::value, Type>::type as_enum(const Value& value) {
    return EnumNames<Type>::as_enum(value);
}

/// Returns the string matching the enum value
template <typename Value>
const std::string& as_string(Value value) {
    return EnumNames<Value>::as_string(value);
}
}  // namespace ov
