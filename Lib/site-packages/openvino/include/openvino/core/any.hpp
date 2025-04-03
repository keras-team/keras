// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @brief A header file for the Any class
 * @file openvino/runtime/any.hpp
 */
#pragma once

#include <map>
#include <memory>
#include <set>
#include <string>
#include <typeindex>
#include <typeinfo>
#include <unordered_map>
#include <utility>

#include "openvino/core/attribute_visitor.hpp"
#include "openvino/core/except.hpp"
#include "openvino/core/runtime_attribute.hpp"

namespace ov {
class Plugin;
/** @cond INTERNAL */
class Any;

using AnyMap = std::map<std::string, Any>;

namespace util {

OPENVINO_API bool equal(std::type_index lhs, std::type_index rhs);

template <typename T, typename = void>
struct Read;

template <class T>
struct Readable {
    template <class U>
    static auto test(U*)
        -> decltype(std::declval<Read<U>>()(std::declval<std::istream&>(), std::declval<U&>()), std::true_type()) {
        return {};
    }
    template <typename>
    static auto test(...) -> std::false_type {
        return {};
    }
    constexpr static const auto value = std::is_same<std::true_type, decltype(test<T>(nullptr))>::value;
};
template <class T>
struct Istreamable {
    template <class U>
    static auto test(U*) -> decltype(std::declval<std::istream&>() >> std::declval<U&>(), std::true_type()) {
        return {};
    }
    template <typename>
    static auto test(...) -> std::false_type {
        return {};
    }
    constexpr static const auto value = std::is_same<std::true_type, decltype(test<T>(nullptr))>::value;
};

template <>
struct OPENVINO_API Read<bool> {
    void operator()(std::istream& is, bool& value) const;
};

template <>
struct OPENVINO_API Read<Any> {
    void operator()(std::istream& is, Any& any) const;
};

template <>
struct OPENVINO_API Read<int> {
    void operator()(std::istream& is, int& value) const;
};

template <>
struct OPENVINO_API Read<long> {
    void operator()(std::istream& is, long& value) const;
};

template <>
struct OPENVINO_API Read<long long> {
    void operator()(std::istream& is, long long& value) const;
};

template <>
struct OPENVINO_API Read<unsigned> {
    void operator()(std::istream& is, unsigned& value) const;
};

template <>
struct OPENVINO_API Read<unsigned long> {
    void operator()(std::istream& is, unsigned long& value) const;
};

template <>
struct OPENVINO_API Read<unsigned long long> {
    void operator()(std::istream& is, unsigned long long& value) const;
};

template <>
struct OPENVINO_API Read<float> {
    void operator()(std::istream& is, float& value) const;
};

template <>
struct OPENVINO_API Read<double> {
    void operator()(std::istream& is, double& value) const;
};

template <>
struct OPENVINO_API Read<long double> {
    void operator()(std::istream& is, long double& value) const;
};

template <>
struct OPENVINO_API Read<std::tuple<unsigned int, unsigned int, unsigned int>> {
    void operator()(std::istream& is, std::tuple<unsigned int, unsigned int, unsigned int>& tuple) const;
};

template <>
struct OPENVINO_API Read<std::tuple<unsigned int, unsigned int>> {
    void operator()(std::istream& is, std::tuple<unsigned int, unsigned int>& tuple) const;
};

template <>
struct OPENVINO_API Read<AnyMap> {
    void operator()(std::istream& is, AnyMap& map) const;
};

template <typename T>
auto from_string(const std::string& str) -> const
    typename std::enable_if<std::is_same<T, std::string>::value, T>::type& {
    return str;
}

template <typename T>
auto from_string(const std::string& val) ->
    typename std::enable_if<Readable<T>::value && !std::is_same<T, std::string>::value, T>::type {
    std::stringstream ss(val);
    T value;
    Read<T>{}(ss, value);
    return value;
}

template <typename T>
auto from_string(const std::string& val) ->
    typename std::enable_if<!Readable<T>::value && Istreamable<T>::value && !std::is_same<T, std::string>::value,
                            T>::type {
    std::stringstream ss(val);
    T value;
    ss >> value;
    return value;
}

template <class T>
struct ValueTyped {
    template <class U>
    static auto test(U*) -> decltype(std::declval<typename U::value_type&>(), std::true_type()) {
        return {};
    }
    template <typename>
    static auto test(...) -> std::false_type {
        return {};
    }
    constexpr static const auto value = std::is_same<std::true_type, decltype(test<T>(nullptr))>::value;
};

template <typename T,
          typename std::enable_if<ValueTyped<T>::value && Readable<typename T::value_type>::value, bool>::type = true>
typename T::value_type from_string(const std::string& val, const T&) {
    std::stringstream ss(val);
    typename T::value_type value;
    Read<typename T::value_type, void>{}(ss, value);
    return value;
}

template <typename T,
          typename std::enable_if<ValueTyped<T>::value && !Readable<typename T::value_type>::value &&
                                      Istreamable<typename T::value_type>::value,
                                  bool>::type = true>
typename T::value_type from_string(const std::string& val, const T&) {
    std::stringstream ss(val);
    typename T::value_type value;
    ss >> value;
    return value;
}

template <typename T>
auto from_string(const std::string& val) ->
    typename std::enable_if<!Readable<T>::value && !Istreamable<T>::value && !std::is_same<T, std::string>::value,
                            T>::type {
    OPENVINO_THROW("Could read type without std::istream& operator>>(std::istream&, T)",
                   " defined or ov::util::Read<T> class specialization, T: ",
                   typeid(T).name());
}

template <typename T, typename A>
struct Read<std::vector<T, A>, typename std::enable_if<std::is_default_constructible<T>::value>::type> {
    void operator()(std::istream& is, std::vector<T, A>& vec) const {
        while (is.good()) {
            std::string str;
            is >> str;
            auto v = from_string<T>(str);
            vec.push_back(std::move(v));
        }
    }
};

template <typename K, typename C, typename A>
struct Read<std::set<K, C, A>, typename std::enable_if<std::is_default_constructible<K>::value>::type> {
    void operator()(std::istream& is, std::set<K, C, A>& set) const {
        while (is.good()) {
            std::string str;
            is >> str;
            auto v = from_string<K>(str);
            set.insert(std::move(v));
        }
    }
};

template <typename K, typename T, typename C, typename A>
struct Read<
    std::map<K, T, C, A>,
    typename std::enable_if<std::is_default_constructible<K>::value && std::is_default_constructible<T>::value>::type> {
    void operator()(std::istream& is, std::map<K, T, C, A>& map) const {
        char c;

        is >> c;
        OPENVINO_ASSERT(c == '{', "Failed to parse std::map<K, T>. Starting symbols is not '{', it's ", c);

        while (c != '}') {
            std::string key, value;
            std::getline(is, key, ':');
            size_t enclosed_container_level = 0;

            while (is.good()) {
                is >> c;
                if (c == ',') {                         // delimiter between map's pairs
                    if (enclosed_container_level == 0)  // we should interrupt after delimiter
                        break;
                }
                if (c == '{' || c == '[')  // case of enclosed maps / arrays
                    ++enclosed_container_level;
                if (c == '}' || c == ']') {
                    if (enclosed_container_level == 0)
                        break;  // end of map
                    --enclosed_container_level;
                }

                value += c;  // accumulate current value
            }
            map.emplace(from_string<K>(key), from_string<T>(value));
        }

        OPENVINO_ASSERT(c == '}', "Failed to parse std::map<K, T>. Ending symbols is not '}', it's ", c);
    }
};

template <typename T>
struct Write;

template <class T>
struct Ostreamable {
    template <class U>
    static auto test(U*) -> decltype(std::declval<std::ostream&>() << std::declval<U>(), std::true_type()) {
        return {};
    }
    template <typename>
    static auto test(...) -> std::false_type {
        return {};
    }
    constexpr static const auto value = std::is_same<std::true_type, decltype(test<T>(nullptr))>::value;
};

template <class T>
struct Writable {
    template <class U>
    static auto test(U*) -> decltype(std::declval<Write<U>>()(std::declval<std::ostream&>(), std::declval<const U&>()),
                                     std::true_type()) {
        return {};
    }
    template <typename>
    static auto test(...) -> std::false_type {
        return {};
    }
    constexpr static const auto value = std::is_same<std::true_type, decltype(test<T>(nullptr))>::value;
};

template <>
struct OPENVINO_API Write<bool> {
    void operator()(std::ostream& is, const bool& b) const;
};

template <>
struct OPENVINO_API Write<Any> {
    void operator()(std::ostream& is, const Any& any) const;
};

template <>
struct OPENVINO_API Write<std::tuple<unsigned int, unsigned int, unsigned int>> {
    void operator()(std::ostream& os, const std::tuple<unsigned int, unsigned int, unsigned int>& tuple) const;
};

template <>
struct OPENVINO_API Write<std::tuple<unsigned int, unsigned int>> {
    void operator()(std::ostream& os, const std::tuple<unsigned int, unsigned int>& tuple) const;
};

template <typename T>
auto to_string(const T& str) -> const typename std::enable_if<std::is_same<T, std::string>::value, T>::type& {
    return str;
}

template <typename T>
auto to_string(const T& value) ->
    typename std::enable_if<Writable<T>::value && !std::is_same<T, std::string>::value, std::string>::type {
    std::stringstream ss;
    Write<T>{}(ss, value);
    return ss.str();
}

template <typename T>
auto to_string(const T& value) ->
    typename std::enable_if<!Writable<T>::value && Ostreamable<T>::value && !std::is_same<T, std::string>::value,
                            std::string>::type {
    std::stringstream ss;
    ss << value;
    return ss.str();
}

template <typename T>
auto to_string(const T&) ->
    typename std::enable_if<!Writable<T>::value && !Ostreamable<T>::value && !std::is_same<T, std::string>::value,
                            std::string>::type {
    OPENVINO_THROW("Could convert to string from type without std::ostream& operator>>(std::ostream&, const T&)",
                   " defined or ov::util::Write<T> class specialization, T: ",
                   typeid(T).name());
}

template <typename T, typename A>
struct Write<std::vector<T, A>> {
    void operator()(std::ostream& os, const std::vector<T, A>& vec) const {
        if (!vec.empty()) {
            std::size_t i = 0;
            for (auto&& v : vec) {
                os << to_string(v);
                if (i < (vec.size() - 1))
                    os << ' ';
                ++i;
            }
        }
    }
};

template <typename K, typename C, typename A>
struct Write<std::set<K, C, A>> {
    void operator()(std::ostream& os, const std::set<K, C, A>& set) const {
        if (!set.empty()) {
            std::size_t i = 0;
            for (auto&& v : set) {
                os << to_string(v);
                if (i < (set.size() - 1))
                    os << ' ';
                ++i;
            }
        }
    }
};

template <typename K, typename T, typename C, typename A>
struct Write<std::map<K, T, C, A>> {
    void operator()(std::ostream& os, const std::map<K, T, C, A>& map) const {
        if (!map.empty()) {
            std::size_t i = 0;
            os << '{';
            for (auto&& v : map) {
                os << to_string(v.first) << ':' << to_string(v.second);
                if (i < (map.size() - 1))
                    os << ',';
                ++i;
            }
            os << '}';
        }
    }
};
}  // namespace util
/** @endcond */

class Node;
class RuntimeAttribute;

class CompiledModel;
namespace proxy {

class CompiledModel;

}
class RemoteContext;
class RemoteTensor;

/**
 * @brief This class represents an object to work with different types
 */
class OPENVINO_API Any {
    std::shared_ptr<void> _so;

    template <typename T>
    using decay_t = typename std::decay<T>::type;

    template <typename T>
    struct EqualityComparable {
        static void* conv(bool);
        template <typename U>
        static char test(decltype(conv(std::declval<U>() == std::declval<U>())));
        template <typename U>
        static long test(...);
        constexpr static const bool value = sizeof(test<T>(nullptr)) == sizeof(char);
    };

    template <typename... T>
    struct EqualityComparable<std::map<T...>> {
        static void* conv(bool);
        template <typename U>
        static char test(decltype(conv(std::declval<typename U::key_type>() == std::declval<typename U::key_type>() &&
                                       std::declval<typename U::mapped_type>() ==
                                           std::declval<typename U::mapped_type>())));
        template <typename U>
        static long test(...);
        constexpr static const bool value = sizeof(test<std::map<T...>>(nullptr)) == sizeof(char);
    };

    template <typename... T>
    struct EqualityComparable<std::vector<T...>> {
        static void* conv(bool);
        template <typename U>
        static char test(decltype(conv(std::declval<typename U::value_type>() ==
                                       std::declval<typename U::value_type>())));
        template <typename U>
        static long test(...);
        constexpr static const bool value = sizeof(test<std::vector<T...>>(nullptr)) == sizeof(char);
    };

    template <class U>
    static typename std::enable_if<EqualityComparable<U>::value, bool>::type equal_impl(const U& rhs, const U& lhs) {
        return rhs == lhs;
    }

    template <class U>
    [[noreturn]] static typename std::enable_if<!EqualityComparable<U>::value, bool>::type equal_impl(const U&,
                                                                                                      const U&) {
        OPENVINO_THROW("Could not compare types without equality operator");
    }

    template <typename T>
    struct HasBaseMemberType {
        template <class U>
        static auto test(U*) -> decltype(std::is_class<typename U::Base>::value, std::true_type()) {
            return {};
        }
        template <typename>
        static auto test(...) -> std::false_type {
            return {};
        }
        constexpr static const auto value = std::is_same<std::true_type, decltype(test<T>(nullptr))>::value;
    };

    template <typename>
    struct TupleToTypeIndex;

    template <typename... Args>
    struct TupleToTypeIndex<std::tuple<Args...>> {
        static std::vector<std::type_index> get() {
            return {typeid(Args)...};
        }
    };

    class OPENVINO_API Base : public std::enable_shared_from_this<Base> {
    public:
        void type_check(const std::type_info&) const;

        using Ptr = std::shared_ptr<Base>;
        virtual const std::type_info& type_info() const = 0;
        virtual std::vector<std::type_index> base_type_info() const = 0;
        bool is_base_type_info(const std::type_info& type_info) const;
        virtual const void* addressof() const = 0;
        void* addressof() {
            return const_cast<void*>(const_cast<const Base*>(this)->addressof());
        }
        virtual Base::Ptr copy() const = 0;
        virtual bool equal(const Base& rhs) const = 0;
        virtual void print(std::ostream& os) const = 0;
        virtual void read(std::istream& os) = 0;
        void read_to(Base& other) const;

        virtual const DiscreteTypeInfo& get_type_info() const = 0;
        virtual std::shared_ptr<RuntimeAttribute> as_runtime_attribute() const;
        virtual bool is_copyable() const;
        virtual Any init(const std::shared_ptr<Node>& node);
        virtual Any merge(const std::vector<std::shared_ptr<Node>>& nodes);
        virtual std::string to_string();
        virtual bool visit_attributes(AttributeVisitor&);
        bool visit_attributes(AttributeVisitor&) const;
        std::string to_string() const;

        bool is(const std::type_info& other) const;
        bool is_signed_integral() const;
        bool is_unsigned_integral() const;
        bool is_floating_point() const;

        template <class T>
        bool is() const {
            return is(typeid(decay_t<T>));
        }

        template <class T>
        T& as() & {
            return *static_cast<decay_t<T>*>(addressof());
        }

        template <class T>
        const T& as() const& {
            return *static_cast<const decay_t<T>*>(addressof());
        }

        template <class T>
        T convert() const;

    protected:
        template <class U>
        [[noreturn]] U convert_impl() const;

        template <class U, class T, class... Others>
        U convert_impl() const;

        virtual ~Base();
    };

    template <class T, typename = void>
    struct Impl;
    template <class T>
    struct Impl<T, typename std::enable_if<std::is_convertible<T, std::shared_ptr<RuntimeAttribute>>::value>::type>
        : public Base {
        const DiscreteTypeInfo& get_type_info() const override {
            return static_cast<RuntimeAttribute*>(runtime_attribute.get())->get_type_info();
        }
        std::shared_ptr<RuntimeAttribute> as_runtime_attribute() const override {
            return std::static_pointer_cast<RuntimeAttribute>(runtime_attribute);
        }
        bool is_copyable() const override {
            return static_cast<RuntimeAttribute*>(runtime_attribute.get())->is_copyable();
        }
        Any init(const std::shared_ptr<Node>& node) override {
            return static_cast<RuntimeAttribute*>(runtime_attribute.get())->init(node);
        }
        Any merge(const std::vector<std::shared_ptr<Node>>& nodes) override {
            return static_cast<RuntimeAttribute*>(runtime_attribute.get())->merge(nodes);
        }
        std::string to_string() override {
            return static_cast<RuntimeAttribute*>(runtime_attribute.get())->to_string();
        }

        bool visit_attributes(AttributeVisitor& visitor) override {
            return static_cast<RuntimeAttribute*>(runtime_attribute.get())->visit_attributes(visitor);
        }

        Impl(const T& runtime_attribute) : runtime_attribute{runtime_attribute} {}

        const std::type_info& type_info() const override {
            return typeid(T);
        }

        std::vector<std::type_index> base_type_info() const override {
            return {typeid(std::shared_ptr<RuntimeAttribute>)};
        }

        const void* addressof() const override {
            return std::addressof(runtime_attribute);
        }

        Base::Ptr copy() const override {
            return std::make_shared<Impl<T>>(this->runtime_attribute);
        }

        bool equal(const Base& rhs) const override {
            if (rhs.is<T>()) {
                return equal_impl(this->runtime_attribute, rhs.as<T>());
            }
            return false;
        }

        void print(std::ostream& os) const override {
            os << runtime_attribute->to_string();
        }

        void read(std::istream&) override {
            OPENVINO_THROW("Pointer to runtime attribute is not readable from std::istream");
        }

        T runtime_attribute;
    };

    template <class T>
    struct Impl<T, typename std::enable_if<!std::is_convertible<T, std::shared_ptr<RuntimeAttribute>>::value>::type>
        : public Base {
        OPENVINO_RTTI(typeid(T).name());

        template <typename... Args>
        Impl(Args&&... args) : value(std::forward<Args>(args)...) {}

        const std::type_info& type_info() const override {
            return typeid(T);
        }

        const void* addressof() const override {
            return std::addressof(value);
        }

        Base::Ptr copy() const override {
            return std::make_shared<Impl<T>>(this->value);
        }

        template <class U>
        static std::vector<std::type_index> base_type_info_impl(
            typename std::enable_if<HasBaseMemberType<U>::value, std::true_type>::type = {}) {
            return TupleToTypeIndex<typename T::Base>::get();
        }
        template <class U>
        static std::vector<std::type_index> base_type_info_impl(
            typename std::enable_if<!HasBaseMemberType<U>::value, std::false_type>::type = {}) {
            return {typeid(T)};
        }

        std::vector<std::type_index> base_type_info() const override {
            return base_type_info_impl<T>();
        }

        bool equal(const Base& rhs) const override {
            if (rhs.is<T>()) {
                return equal_impl(this->value, rhs.as<T>());
            }
            return false;
        }

        template <typename U>
        static typename std::enable_if<util::Writable<U>::value>::type print_impl(std::ostream& os, const U& value) {
            util::Write<U>{}(os, value);
        }

        template <typename U>
        static typename std::enable_if<!util::Writable<U>::value && util::Ostreamable<U>::value>::type print_impl(
            std::ostream& os,
            const U& value) {
            os << value;
        }

        template <typename U>
        static typename std::enable_if<!util::Writable<U>::value && !util::Ostreamable<U>::value>::type print_impl(
            std::ostream&,
            const U&) {}

        void print(std::ostream& os) const override {
            print_impl(os, value);
        }

        template <typename U>
        static typename std::enable_if<util::Readable<U>::value>::type read_impl(std::istream& is, U& value) {
            util::Read<U>{}(is, value);
        }

        template <typename U>
        static typename std::enable_if<!util::Readable<U>::value && util::Istreamable<U>::value>::type read_impl(
            std::istream& is,
            U& value) {
            is >> value;
        }

        template <typename U>
        static typename std::enable_if<!util::Readable<U>::value && !util::Istreamable<U>::value>::type read_impl(
            std::istream&,
            U&) {
            OPENVINO_THROW("Could read type without std::istream& operator>>(std::istream&, T)",
                           " defined or ov::util::Read<T> class specialization, T: ",
                           typeid(T).name());
        }

        void read(std::istream& is) override {
            read_impl(is, value);
        }

        T value;
    };

    // Generic if there is no specialization for T.
    template <class T>
    T& as_impl(...) {
        impl_check();
        if (is<T>()) {
            return _impl->as<T>();
        }

        OPENVINO_THROW("Bad as from: ", _impl->type_info().name(), " to: ", typeid(T).name());
    }

    template <class T, typename std::enable_if<std::is_same<T, std::string>::value>::type* = nullptr>
    T& as_impl(int) {
        if (_impl != nullptr) {
            if (_impl->is<T>()) {
                return _impl->as<T>();
            } else {
                _temp = std::make_shared<Impl<std::string>>();
                _impl->read_to(*_temp);
                return _temp->as<std::string>();
            }
        } else {
            _temp = std::make_shared<Impl<std::string>>();
            return _temp->as<std::string>();
        }
    }

    template <
        class T,
        typename std::enable_if<std::is_convertible<T, std::shared_ptr<RuntimeAttribute>>::value>::type* = nullptr>
    T& as_impl(int) {
        if (_impl == nullptr) {
            _temp = std::make_shared<Impl<decay_t<T>>>(T{});
            return _temp->as<T>();
        } else {
            if (_impl->is<T>()) {
                return _impl->as<T>();
            } else {
                auto runtime_attribute = _impl->as_runtime_attribute();
                if (runtime_attribute == nullptr) {
                    OPENVINO_THROW("Any does not contains pointer to runtime_attribute. It contains ",
                                   _impl->type_info().name());
                }
                auto vptr = ov::as_type_ptr<typename T::element_type>(runtime_attribute);
                if (vptr == nullptr && T::element_type::get_type_info_static() != runtime_attribute->get_type_info() &&
                    T::element_type::get_type_info_static() != RuntimeAttribute::get_type_info_static()) {
                    OPENVINO_THROW("Could not as Any runtime_attribute to ",
                                   typeid(T).name(),
                                   " from ",
                                   _impl->type_info().name(),
                                   "; from ",
                                   static_cast<std::string>(runtime_attribute->get_type_info()),
                                   " to ",
                                   static_cast<std::string>(T::element_type::get_type_info_static()));
                }
                _temp = std::make_shared<Impl<decay_t<T>>>(
                    std::static_pointer_cast<typename T::element_type>(runtime_attribute));
                return _temp->as<T>();
            }
        }
    }

    template <class T,
              typename std::enable_if<std::is_arithmetic<T>::value &&
                                      !std::is_same<typename std::decay<T>::type, bool>::value>::type* = nullptr>
    T& as_impl(int);

    template <class T,
              typename std::enable_if<
                  (util::Istreamable<T>::value || util::Readable<T>::value) && !std::is_same<T, std::string>::value &&
                  (!std::is_arithmetic<T>::value || std::is_same<typename std::decay<T>::type, bool>::value)>::type* =
                  nullptr>
    T& as_impl(int) {
        impl_check();

        if (is<T>()) {
            return _impl->as<T>();
        } else if (_impl->is<std::string>()) {
            _temp = std::make_shared<Impl<decay_t<T>>>();
            _impl->read_to(*_temp);
            return _temp->as<T>();
        }

        OPENVINO_THROW("Bad as from: ", _impl->type_info().name(), " to: ", typeid(T).name());
    }

    friend class ::ov::RuntimeAttribute;
    friend class ::ov::CompiledModel;
    friend class ::ov::proxy::CompiledModel;
    friend class ::ov::RemoteContext;
    friend class ::ov::RemoteTensor;
    friend class ::ov::Plugin;

    Any(const Any& other, const std::shared_ptr<void>& so);

    void impl_check() const;

    mutable Base::Ptr _temp;

    Base::Ptr _impl;

public:
    /// @brief Default constructor
    Any() = default;

    /// @brief Copy constructor
    /// @param other other Any object
    Any(const Any& other);

    /// @brief Copy assignment operator
    /// @param other other Any object
    /// @return reference to the current object
    Any& operator=(const Any& other);

    /// @brief Default move constructor
    /// @param other other Any object
    Any(Any&& other) = default;

    /// @brief Default move assignment operator
    /// @param other other Any object
    /// @return reference to the current object
    Any& operator=(Any&& other) = default;

    /**
     * @brief Destructor preserves unloading order of implementation object and reference to library
     */
    ~Any();

    /**
     * @brief Constructor creates any with object
     *
     * @tparam T Any type
     * @param value object
     */
    template <typename T,
              typename std::enable_if<!std::is_same<decay_t<T>, Any>::value && !std::is_abstract<decay_t<T>>::value &&
                                          !std::is_convertible<decay_t<T>, Base::Ptr>::value,
                                      bool>::type = true>
    Any(T&& value) : _impl{std::make_shared<Impl<decay_t<T>>>(std::forward<T>(value))} {}

    /**
     * @brief Constructor creates string any from char *
     *
     * @param str char array
     */
    Any(const char* str);

    /**
     * @brief Empty constructor
     *
     */
    Any(const std::nullptr_t);

    /**
     * @brief Inplace value construction function
     *
     * @tparam T Any type
     * @tparam Args pack of parameter types passed to T constructor
     * @param args pack of parameters passed to T constructor
     */
    template <typename T, typename... Args>
    static Any make(Args&&... args) {
        Any any;
        any._impl = std::make_shared<Impl<decay_t<T>>>(std::forward<Args>(args)...);
        return any;
    }

    /**
     * Returns type info
     * @return type info
     */
    const std::type_info& type_info() const;

    /**
     * Checks that any contains a value
     * @return false if any contains a value else false
     */
    bool empty() const;

    /**
     * @brief Check that stored type can be casted to specified type.
     * If internal type supports Base
     * @tparam T Type of value
     * @return true if type of value is correct. Return false if any is empty
     */
    template <class T>
    bool is() const {
        return _impl && (_impl->is<T>() || _impl->is_base_type_info(typeid(decay_t<T>)));
    }

    /**
     * Dynamic as to specified type
     * @tparam T type
     * @return reference to caster object
     */
    template <class T>
    T& as() {
        return as_impl<T>(int{});
    }

    /**
     * Dynamic as to specified type
     * @tparam T type
     * @return const reference to caster object
     */
    template <class T>
    const T& as() const {
        return const_cast<Any*>(this)->as<T>();
    }

    /**
     * @brief The comparison operator for the Any
     *
     * @param other object to compare
     * @return true if objects are equal
     */
    bool operator==(const Any& other) const;

    /**
     * @brief The comparison operator for the Any
     *
     * @param other object to compare
     * @return true if objects are equal
     */
    bool operator==(const std::nullptr_t&) const;

    /**
     * @brief The comparison operator for the Any
     *
     * @param other object to compare
     * @return true if objects aren't equal
     */
    bool operator!=(const Any& other) const;

    /**
     * @brief Prints underlying object to the given output stream.
     * Uses operator<< if it is defined, leaves stream unchanged otherwise.
     * In case of empty any or nullptr stream immediately returns.
     *
     * @param stream Output stream object will be printed to.
     */
    void print(std::ostream& stream) const;

    /**
     * @brief Read into underlying object from the given input stream.
     * Uses operator>> if it is defined, leaves stream unchanged otherwise.
     * In case of empty any or nullptr stream immediately returns.
     *
     * @param stream Output stream object will be printed to.
     */
    void read(std::istream& stream);

    /**
     * @brief Returns address to internal value if any is not empty and `nullptr` instead
     * @return address to internal stored value
     */
    void* addressof();

    /**
     * @brief Returns address to internal value if any is not empty and `nullptr` instead
     * @return address to internal stored value
     */
    const void* addressof() const;
};

using RTMap = AnyMap;

using AnyVector = std::vector<ov::Any>;

/** @cond INTERNAL */
inline static void PrintTo(const Any& any, std::ostream* os) {
    any.print(*os);
}
/** @endcond */

template <>
OPENVINO_API unsigned long long Any::Base::convert<unsigned long long>() const;

template <>
OPENVINO_API long long Any::Base::convert<long long>() const;

template <>
OPENVINO_API double Any::Base::convert<double>() const;

template <class T,
          typename std::enable_if<std::is_arithmetic<T>::value &&
                                  !std::is_same<typename std::decay<T>::type, bool>::value>::type*>
T& Any::as_impl(int) {
    impl_check();
    if (is<T>()) {
        return _impl->as<T>();
    } else if (util::Readable<T>::value && _impl->is<std::string>()) {
        _temp = std::make_shared<Impl<decay_t<T>>>();
        _impl->read_to(*_temp);
        return _temp->as<T>();
    } else if (_impl->is_signed_integral()) {
        auto value = _impl->convert<long long>();
        _temp = std::make_shared<Impl<decay_t<T>>>(static_cast<T>(value));
        return _temp->as<T>();
    } else if (_impl->is_unsigned_integral()) {
        auto value = _impl->convert<unsigned long long>();
        _temp = std::make_shared<Impl<decay_t<T>>>(static_cast<T>(value));
        return _temp->as<T>();
    } else if (_impl->is_floating_point()) {
        auto value = _impl->convert<double>();
        _temp = std::make_shared<Impl<decay_t<T>>>(static_cast<T>(value));
        return _temp->as<T>();
    }

    OPENVINO_THROW("Bad as from: ", _impl->type_info().name(), " to: ", typeid(T).name());
}
}  // namespace ov
