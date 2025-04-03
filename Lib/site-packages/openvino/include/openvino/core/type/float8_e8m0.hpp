// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cmath>
#include <iostream>
#include <limits>
#include <memory>
#include <string>
#include <vector>

#include "openvino/core/core_visibility.hpp"

namespace ov {

/**
 * @brief Class to represent the f8e8m0 type.
 */
class OPENVINO_API float8_e8m0 {
public:
    float8_e8m0() = default;
    float8_e8m0(const float value);

    template <typename I>
    explicit float8_e8m0(I value) : float8_e8m0{static_cast<float>(value)} {}

    template <typename T>
    bool operator==(const T& other) const;
    template <typename T>
    bool operator!=(const T& other) const {
        return !(*this == other);
    }

    template <typename T>
    bool operator<(const T& other) const;
    template <typename T>
    bool operator<=(const T& other) const;
    template <typename T>
    bool operator>(const T& other) const;
    template <typename T>
    bool operator>=(const T& other) const;
    template <typename T>
    float8_e8m0 operator+(const T& other) const;
    template <typename T>
    float8_e8m0 operator+=(const T& other);
    template <typename T>
    float8_e8m0 operator-(const T& other) const;
    template <typename T>
    float8_e8m0 operator-=(const T& other);
    template <typename T>
    float8_e8m0 operator*(const T& other) const;
    template <typename T>
    float8_e8m0 operator*=(const T& other);
    template <typename T>
    float8_e8m0 operator/(const T& other) const;
    template <typename T>
    float8_e8m0 operator/=(const T& other);

    operator float() const;

    static constexpr float8_e8m0 from_bits(uint8_t bits) {
        return float8_e8m0(bits, true);
    }

    uint8_t to_bits() const;

    friend std::ostream& operator<<(std::ostream& out, const float8_e8m0& obj) {
        out << static_cast<float>(obj);
        return out;
    }

private:
    constexpr float8_e8m0(const uint8_t x, bool) : m_value{x} {}

    uint8_t m_value{};
};

#if defined(_MSC_VER)
#    pragma warning(push)
#    pragma warning(disable : 4756)
#endif
template <typename T>
bool float8_e8m0::operator==(const T& other) const {
#if defined(__GNUC__)
#    pragma GCC diagnostic push
#    pragma GCC diagnostic ignored "-Wfloat-equal"
#endif
    return (static_cast<float>(*this) == static_cast<float>(other));
#if defined(__GNUC__)
#    pragma GCC diagnostic pop
#endif
}

template <typename T>
bool float8_e8m0::operator<(const T& other) const {
    return (static_cast<float>(*this) < static_cast<float>(other));
}

template <typename T>
bool float8_e8m0::operator<=(const T& other) const {
    return (static_cast<float>(*this) <= static_cast<float>(other));
}

template <typename T>
bool float8_e8m0::operator>(const T& other) const {
    return (static_cast<float>(*this) > static_cast<float>(other));
}

template <typename T>
bool float8_e8m0::operator>=(const T& other) const {
    return (static_cast<float>(*this) >= static_cast<float>(other));
}

template <typename T>
float8_e8m0 float8_e8m0::operator+(const T& other) const {
    return {static_cast<float>(*this) + static_cast<float>(other)};
}

template <typename T>
float8_e8m0 float8_e8m0::operator+=(const T& other) {
    return *this = *this + other;
}

template <typename T>
float8_e8m0 float8_e8m0::operator-(const T& other) const {
    return {static_cast<float>(*this) - static_cast<float>(other)};
}

template <typename T>
float8_e8m0 float8_e8m0::operator-=(const T& other) {
    return *this = *this - other;
}

template <typename T>
float8_e8m0 float8_e8m0::operator*(const T& other) const {
    return {static_cast<float>(*this) * static_cast<float>(other)};
}

template <typename T>
float8_e8m0 float8_e8m0::operator*=(const T& other) {
    return *this = *this * other;
}

template <typename T>
float8_e8m0 float8_e8m0::operator/(const T& other) const {
    return {static_cast<float>(*this) / static_cast<float>(other)};
}

template <typename T>
float8_e8m0 float8_e8m0::operator/=(const T& other) {
    return *this = *this / other;
}
#if defined(_MSC_VER)
#    pragma warning(pop)
#endif
}  // namespace ov

namespace std {
template <>
class numeric_limits<ov::float8_e8m0> {
public:
    static constexpr bool is_specialized = true;
    static constexpr ov::float8_e8m0 min() noexcept {
        return ov::float8_e8m0::from_bits(0b00000000u);
    }
    static constexpr ov::float8_e8m0 max() noexcept {
        return ov::float8_e8m0::from_bits(0b11111110u);
    }
    static constexpr ov::float8_e8m0 lowest() noexcept {
        return ov::float8_e8m0::from_bits(0b00000000u);
    }
    static constexpr int digits = 1;
    static constexpr int digits10 = 0;

    static constexpr bool is_signed = false;
    static constexpr bool is_integer = false;
    static constexpr bool is_exact = false;

    static constexpr int radix = 2;

    static constexpr ov::float8_e8m0 epsilon() noexcept {
        return ov::float8_e8m0::from_bits(0b00000001u);
    }
    static constexpr ov::float8_e8m0 round_error() noexcept {
        return ov::float8_e8m0::from_bits(0b01111110u);
    }

    static constexpr int min_exponent = -126;
    static constexpr int min_exponent10 = -38;
    static constexpr int max_exponent = 128;
    static constexpr int max_exponent10 = 38;

    static constexpr bool has_infinity = false;
    static constexpr bool has_quiet_NaN = true;
    static constexpr bool has_signaling_NaN = false;

    static constexpr float_denorm_style has_denorm = denorm_absent;
    static constexpr bool has_denorm_loss = false;

    static constexpr ov::float8_e8m0 infinity() noexcept {
        return ov::float8_e8m0::from_bits(0);  // no infinity
    }
    static constexpr ov::float8_e8m0 quiet_NaN() noexcept {
        return ov::float8_e8m0::from_bits(0b11111111);
    }
    static constexpr ov::float8_e8m0 signaling_NaN() noexcept {
        return ov::float8_e8m0::from_bits(0);  // no signaling NaN
    }
    static constexpr bool is_iec559 = false;
    static constexpr bool is_bounded = false;
    static constexpr bool is_modulo = false;
    static constexpr bool traps = false;
    static constexpr bool tinyness_before = false;
    static constexpr float_round_style round_style = round_to_nearest;
};
}  // namespace std
