/*******************************************************************************
* Copyright 2021-2024 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/

#ifndef COMMON_OPTIONAL_HPP
#define COMMON_OPTIONAL_HPP

#if __cplusplus >= 201606L
#include <optional>
#else
#include <cassert>
#include <memory>
#include <type_traits>
#endif

namespace dnnl {
namespace impl {
namespace utils {

// Use C++17 std::optional if available
#if __cplusplus >= 201606L

using nullopt_t = std::nullopt_t;
static constexpr nullopt_t nullopt = std::nullopt;

template <typename T>
using optional_t = std::optional<T>;

#else

// This is a simple version of the std::optional package
// When C++17 will be supported it is highly recommended
// to remove this and start using std::optional instead.
struct nullopt_t {
    nullopt_t() = default;
};
static constexpr nullopt_t nullopt {};

template <class T>
class optional_t {
public:
    static_assert(!std::is_lvalue_reference<T>::value, "");
    static_assert(!std::is_rvalue_reference<T>::value, "");
    static_assert(!std::is_const<T>::value, "");
    static_assert(!std::is_volatile<T>::value, "");

    optional_t(const nullopt_t nullopt_) : has_value_(false), dummy {} {}
    optional_t() : optional_t(nullopt) {}
    optional_t(T object) : has_value_(true), value_(object) {}
    optional_t(const optional_t &other)
        : has_value_(other.has_value_), dummy {} {
        if (has_value_) new (std::addressof(value_)) T(other.value_);
    }
    optional_t(optional_t<T> &&other) noexcept
        : has_value_(other.has_value_), dummy {} {
        if (has_value_) new (std::addressof(value_)) T(std::move(other.value_));
    }
    ~optional_t() {
        if (has_value_) value_.~T();
    }

    optional_t &operator=(const nullopt_t nullopt_) {
        if (has_value_) value_.~T();
        has_value_ = false;
    }
    optional_t &operator=(const optional_t &other) {
        if (this == &other) return *this;
        if (has_value_) value_.~T();
        has_value_ = other.has_value_;
        if (has_value_) value_ = other.value_;
        return *this;
    }
    optional_t &operator=(optional_t &&other) {
        if (this == &other) return *this;
        if (has_value_) value_.~T();
        has_value_ = other.has_value_;
        if (has_value_) value_ = std::move(other.value_);
        return *this;
    }

    const T *operator->() const {
        assert(has_value_);
        return &value_;
    }
    T *operator->() {
        assert(has_value_);
        return &value_;
    }
    const T &operator*() const {
        assert(has_value_);
        return value_;
    }
    T &operator*() {
        assert(has_value_);
        return value_;
    }
    operator bool() const { return has_value_; }

    T value_or(T &&returned_value) {
        return has_value_ ? value_ : returned_value;
    }
    T value_or(T &&returned_value) const {
        return has_value_ ? value_ : returned_value;
    }
    const T &value() const {
        assert(has_value_);
        return value_;
    }
    T &value() {
        assert(has_value_);
        return value_;
    }
    bool has_value() const { return has_value_; }
    void reset() {
        if (has_value_) value_.~T();
        has_value_ = false;
    }

private:
    bool has_value_;

    union {
        char dummy;
        T value_;
    };
};

#endif // __cplusplus >= 201606L

} // namespace utils
} // namespace impl
} // namespace dnnl

#endif
