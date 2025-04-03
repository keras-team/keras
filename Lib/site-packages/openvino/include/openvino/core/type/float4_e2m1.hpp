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
 * @brief Class to represent the f4e2m1 type.
 */
class OPENVINO_API float4_e2m1 {
public:
    float4_e2m1() = default;
    float4_e2m1(float value);

    template <typename I>
    explicit float4_e2m1(I value) : float4_e2m1(static_cast<float>(value)) {}

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
    float4_e2m1 operator+(const T& other) const;
    template <typename T>
    float4_e2m1 operator+=(const T& other);
    template <typename T>
    float4_e2m1 operator-(const T& other) const;
    template <typename T>
    float4_e2m1 operator-=(const T& other);
    template <typename T>
    float4_e2m1 operator*(const T& other) const;
    template <typename T>
    float4_e2m1 operator*=(const T& other);
    template <typename T>
    float4_e2m1 operator/(const T& other) const;
    template <typename T>
    float4_e2m1 operator/=(const T& other);

    operator float() const;

    static constexpr float4_e2m1 from_bits(uint8_t bits) {
        return float4_e2m1(bits, true);
    }

    uint8_t to_bits() const;

    friend std::ostream& operator<<(std::ostream& out, const float4_e2m1& obj) {
        out << static_cast<float>(obj);
        return out;
    }

private:
    constexpr float4_e2m1(uint8_t x, bool) : m_value{x} {}

    uint8_t m_value;
};

#if defined(_MSC_VER)
#    pragma warning(push)
#    pragma warning(disable : 4756)
#endif
template <typename T>
bool float4_e2m1::operator==(const T& other) const {
    return (static_cast<float>(*this) == static_cast<float>(other));
}

template <typename T>
bool float4_e2m1::operator<(const T& other) const {
    return (static_cast<float>(*this) < static_cast<float>(other));
}

template <typename T>
bool float4_e2m1::operator<=(const T& other) const {
    return (static_cast<float>(*this) <= static_cast<float>(other));
}

template <typename T>
bool float4_e2m1::operator>(const T& other) const {
    return (static_cast<float>(*this) > static_cast<float>(other));
}

template <typename T>
bool float4_e2m1::operator>=(const T& other) const {
    return (static_cast<float>(*this) >= static_cast<float>(other));
}

template <typename T>
float4_e2m1 float4_e2m1::operator+(const T& other) const {
    return {static_cast<float>(*this) + static_cast<float>(other)};
}

template <typename T>
float4_e2m1 float4_e2m1::operator+=(const T& other) {
    return *this = *this + other;
}

template <typename T>
float4_e2m1 float4_e2m1::operator-(const T& other) const {
    return {static_cast<float>(*this) - static_cast<float>(other)};
}

template <typename T>
float4_e2m1 float4_e2m1::operator-=(const T& other) {
    return *this = *this - other;
}

template <typename T>
float4_e2m1 float4_e2m1::operator*(const T& other) const {
    return {static_cast<float>(*this) * static_cast<float>(other)};
}

template <typename T>
float4_e2m1 float4_e2m1::operator*=(const T& other) {
    return *this = *this * other;
}

template <typename T>
float4_e2m1 float4_e2m1::operator/(const T& other) const {
    return {static_cast<float>(*this) / static_cast<float>(other)};
}

template <typename T>
float4_e2m1 float4_e2m1::operator/=(const T& other) {
    return *this = *this / other;
}
#if defined(_MSC_VER)
#    pragma warning(pop)
#endif
}  // namespace ov

namespace std {
template <>
class numeric_limits<ov::float4_e2m1> {
public:
    static constexpr bool is_specialized = true;
    static constexpr ov::float4_e2m1 min() noexcept {
        return ov::float4_e2m1::from_bits(0b0010);  // minimum positive normalized value
    }
    static constexpr ov::float4_e2m1 max() noexcept {
        return ov::float4_e2m1::from_bits(0b0111);
    }
    static constexpr ov::float4_e2m1 lowest() noexcept {
        return ov::float4_e2m1::from_bits(0b1111);
    }
    static constexpr int digits = 2;
    static constexpr int digits10 = 0;

    static constexpr bool is_signed = true;
    static constexpr bool is_integer = false;
    static constexpr bool is_exact = false;

    static constexpr int radix = 2;

    static constexpr ov::float4_e2m1 epsilon() noexcept {
        return ov::float4_e2m1::from_bits(0b0001);
    }
    static constexpr ov::float4_e2m1 round_error() noexcept {
        return ov::float4_e2m1::from_bits(0b001);
    }

    static constexpr int min_exponent = 1;
    static constexpr int min_exponent10 = 0;
    static constexpr int max_exponent = 3;
    static constexpr int max_exponent10 = 0;

    static constexpr bool has_infinity = false;
    static constexpr bool has_quiet_NaN = false;
    static constexpr bool has_signaling_NaN = false;

    static constexpr float_denorm_style has_denorm = denorm_present;
    static constexpr bool has_denorm_loss = false;

    static constexpr ov::float4_e2m1 infinity() noexcept {
        return ov::float4_e2m1::from_bits(0);  // no infinity
    }
    static constexpr ov::float4_e2m1 quiet_NaN() noexcept {
        return ov::float4_e2m1::from_bits(0);  // no quiet NaN
    }
    static constexpr ov::float4_e2m1 signaling_NaN() noexcept {
        return ov::float4_e2m1::from_bits(0);  // no signaling NaN
    }
    static constexpr ov::float4_e2m1 denorm_min() noexcept {
        return ov::float4_e2m1::from_bits(0b0001);  // minimum positive denormalized value
    }
    static constexpr bool is_iec559 = false;
    static constexpr bool is_bounded = false;
    static constexpr bool is_modulo = false;
    static constexpr bool traps = false;
    static constexpr bool tinyness_before = false;
    static constexpr float_round_style round_style = round_to_nearest;
};
}  // namespace std
