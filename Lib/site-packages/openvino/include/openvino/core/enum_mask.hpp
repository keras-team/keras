// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <utility>

namespace ov {

//
// EnumMask is intended to work with a scoped enum type. It's used to store
// a combination of enum values and provides easy access and manipulation
// of these enum values as a mask.
//
// EnumMask does not provide a set_all() or invert() operator because they
// could do things unexpected by the user, i.e. for enum with 4 bit values,
// invert(001000...) != 110100..., due to the extra bits.
//
template <typename T>
class EnumMask {
public:
    /// Make sure the template type is an enum.
    static_assert(std::is_enum<T>::value, "EnumMask template type must be an enum");
    /// Extract the underlying type of the enum.
    using value_type = typename std::underlying_type<T>::type;
    /// Some bit operations are not safe for signed values, we require enum
    /// type to use unsigned underlying type.
    static_assert(std::is_unsigned<value_type>::value, "EnumMask enum must use unsigned type.");

    constexpr EnumMask() = default;
    constexpr EnumMask(const T& enum_value) : m_value{static_cast<value_type>(enum_value)} {}
    EnumMask(const EnumMask& other) : m_value{other.m_value} {}
    EnumMask(std::initializer_list<T> enum_values) {
        for (auto& v : enum_values) {
            m_value |= static_cast<value_type>(v);
        }
    }
    value_type value() const {
        return m_value;
    }
    /// Check if any of the input parameter enum bit mask match
    bool is_any_set(const EnumMask& p) const {
        return m_value & p.m_value;
    }
    /// Check if all of the input parameter enum bit mask match
    bool is_set(const EnumMask& p) const {
        return (m_value & p.m_value) == p.m_value;
    }
    /// Check if any of the input parameter enum bit mask does not match
    bool is_any_clear(const EnumMask& p) const {
        return !is_set(p);
    }
    /// Check if all of the input parameter enum bit mask do not match
    bool is_clear(const EnumMask& p) const {
        return !is_any_set(p);
    }
    void set(const EnumMask& p) {
        m_value |= p.m_value;
    }
    void clear(const EnumMask& p) {
        m_value &= ~p.m_value;
    }
    void clear_all() {
        m_value = 0;
    }
    bool operator[](const EnumMask& p) const {
        return is_set(p);
    }
    bool operator==(const EnumMask& other) const {
        return m_value == other.m_value;
    }
    bool operator!=(const EnumMask& other) const {
        return m_value != other.m_value;
    }
    EnumMask& operator=(const EnumMask& other) {
        m_value = other.m_value;
        return *this;
    }
    EnumMask& operator&=(const EnumMask& other) {
        m_value &= other.m_value;
        return *this;
    }

    EnumMask& operator|=(const EnumMask& other) {
        m_value |= other.m_value;
        return *this;
    }

    EnumMask operator&(const EnumMask& other) const {
        return EnumMask(m_value & other.m_value);
    }

    EnumMask operator|(const EnumMask& other) const {
        return EnumMask(m_value | other.m_value);
    }

    friend std::ostream& operator<<(std::ostream& os, const EnumMask& m) {
        os << m.m_value;
        return os;
    }

private:
    /// Only used internally
    explicit EnumMask(const value_type& value) : m_value{value} {}

    value_type m_value{};
};

}  // namespace ov
