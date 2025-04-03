/*******************************************************************************
 * Copyright 2021-2022 Intel Corporation
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

#ifndef GRAPH_UTILS_ATTRIBUTE_VALUE_HPP
#define GRAPH_UTILS_ATTRIBUTE_VALUE_HPP

#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "graph/interface/c_types_map.hpp"
#include "graph/utils/json.hpp"

namespace dnnl {
namespace impl {
namespace graph {
namespace utils {
// Add new attribute types by specializing on the C++ type and
// defining an enum value below
template <typename value_type>
struct attribute_value_traits;

template <>
struct attribute_value_traits<int64_t> {
    static attribute_kind_t constexpr kind = attribute_kind::i;
};

template <>
struct attribute_value_traits<std::vector<int64_t>> {
    static attribute_kind_t constexpr kind = attribute_kind::is;
};

template <>
struct attribute_value_traits<float> {
    static attribute_kind_t constexpr kind = attribute_kind::f;
};

template <>
struct attribute_value_traits<std::vector<float>> {
    static attribute_kind_t constexpr kind = attribute_kind::fs;
};

template <>
struct attribute_value_traits<std::string> {
    static attribute_kind_t constexpr kind = attribute_kind::s;
};

template <>
struct attribute_value_traits<bool> {
    static attribute_kind_t constexpr kind = attribute_kind::b;
};

template <typename value_type>
using base_value_type = typename std::remove_cv<
        typename std::remove_reference<value_type>::type>::type;

template <typename value_type>
constexpr attribute_kind_t get_attribute_kind() {
    static_assert(std::is_same<float, value_type>::value
                    || std::is_same<int64_t, value_type>::value
                    || std::is_same<std::vector<float>, value_type>::value
                    || std::is_same<std::vector<int64_t>, value_type>::value
                    || std::is_same<std::string, value_type>::value
                    || std::is_same<bool, value_type>::value,
            "value_type should be one of int64_t, float, string, "
            "bool, vector<float>, or vector<int64_t>.");

    return attribute_value_traits<base_value_type<value_type>>::kind;
}

template <typename value_type>
class attribute_value_cell_imp_t;

// Interface to cells the hold specific C++ types
class attribute_value_cell_t {
protected:
    template <typename value_type>
    using imp_class = attribute_value_cell_imp_t<base_value_type<value_type>>;

public:
    virtual ~attribute_value_cell_t() = default;
    virtual attribute_kind_t get_kind() const = 0;
    // Copy the underlying value
    virtual std::unique_ptr<attribute_value_cell_t> duplicate() const = 0;
    virtual bool operator==(const attribute_value_cell_t &other) const = 0;

    template <typename value_type>
    attribute_value_cell_t &operator=(value_type &&value) {
        if (get_kind() == get_attribute_kind<value_type>()) {
            static_cast<imp_class<value_type> &>(*this).set(
                    std::forward<value_type>(value));
            return *this;
        } else {
            throw std::runtime_error(
                    "Attempt to set attribute to invalid type.\n");
        }
    }

    template <typename value_type>
    value_type &get() {
        if (get_kind() == get_attribute_kind<value_type>()) {
            return static_cast<imp_class<value_type> &>(*this).get();
        } else {
            throw std::runtime_error(
                    "Attempt to get attribute using invalid type.\n");
        }
    }

    template <typename value_type>
    const value_type &get() const {
        if (get_kind() == get_attribute_kind<value_type>()) {
            return static_cast<imp_class<value_type> &>(*this).get();
        } else {
            throw std::runtime_error(
                    "Attempt to get attribute using invalid type.\n");
        }
    }
};

// Type-specific instantiations of attribute_value_cell_t API
template <typename value_type>
class attribute_value_cell_imp_t : public attribute_value_cell_t {
public:
    attribute_value_cell_imp_t(const value_type &value) : value_(value) {}
    attribute_value_cell_imp_t(value_type &&value) : value_(std::move(value)) {}
    attribute_kind_t get_kind() const override {
        return get_attribute_kind<value_type>();
    }
    std::unique_ptr<attribute_value_cell_t> duplicate() const override {
        return std::unique_ptr<attribute_value_cell_t>(
                new attribute_value_cell_imp_t<value_type>(value_));
    }
    bool operator==(const attribute_value_cell_t &other) const override {
        return (other.get_kind() == get_kind())
                && static_cast<const attribute_value_cell_imp_t<value_type> &>(
                           other)
                           .value_
                == value_;
    }
    void set(const value_type &value) { value_ = value; }
    void set(value_type &&value) { value_ = std::move(value); }
    value_type &get() { return value_; }
    const value_type &get() const { return value_; }

protected:
    value_type value_;
};

// Wrapper for a unique_ptr to an attribute_value_cell_t
class attribute_value_t {
public:
    attribute_value_t() = default;

    template <typename value_type>
    attribute_value_t(const value_type &value)
        : value_cell_(new attribute_value_cell_imp_t<value_type>(value)) {}

    attribute_value_t(attribute_value_t &&orig)
        : value_cell_(std::move(orig.value_cell_)) {}

    attribute_value_t(const attribute_value_t &orig) {
        if (orig.value_cell_) {
            this->value_cell_ = orig.value_cell_->duplicate();
        }
    }

    // Serialize
    status_t save(dnnl::impl::graph::utils::json::json_writer_t *writer) const {
        writer->begin_object();
        if (check_type<bool>()) {
            writer->write_keyvalue("type", std::string("bool"));
            writer->write_keyvalue("value", get<bool>());
        } else if (check_type<int64_t>()) {
            writer->write_keyvalue("type", std::string("s64"));
            writer->write_keyvalue("value", get<int64_t>());
        } else if (check_type<float>()) {
            writer->write_keyvalue("type", std::string("f32"));
            writer->write_keyvalue("value", get<float>());
        } else if (check_type<std::vector<float>>()) {
            writer->write_keyvalue("type", std::string("f32[]"));
            writer->write_keyvalue("value", get<std::vector<float>>());
        } else if (check_type<std::vector<int64_t>>()) {
            writer->write_keyvalue("type", std::string("s64[]"));
            writer->write_keyvalue("value", get<std::vector<int64_t>>());
        } else if (check_type<std::string>()) {
            writer->write_keyvalue("type", std::string("string"));
            writer->write_keyvalue("value", get<std::string>());
        } else {
        }
        writer->end_object();
        return graph::status::success;
    }

    template <typename value_type>
    attribute_value_t &operator=(value_type &&value) {
        *value_cell_ = std::forward<value_type>(value);
        return *this;
    }

    attribute_value_t &operator=(attribute_value_t &&orig) {
        value_cell_ = std::move(orig.value_cell_);
        return *this;
    }

    attribute_value_t &operator=(const attribute_value_t &orig) {
        if (orig.value_cell_) {
            this->value_cell_ = orig.value_cell_->duplicate();
        }
        return *this;
    }

    bool operator==(const attribute_value_t &value) const {
        return *value_cell_ == *value.value_cell_;
    }

    bool operator!=(const attribute_value_t &value) const {
        return !operator==(value);
    }

    attribute_kind_t get_kind() const { return value_cell_->get_kind(); }

    template <typename value_type>
    bool check_type() const {
        return value_cell_->get_kind() == get_attribute_kind<value_type>();
    }

    template <typename value_type>
    value_type &get() const {
        return value_cell_->get<value_type>();
    }

private:
    std::unique_ptr<attribute_value_cell_t> value_cell_;
};
} // namespace utils
} // namespace graph
} // namespace impl
} // namespace dnnl

#endif
