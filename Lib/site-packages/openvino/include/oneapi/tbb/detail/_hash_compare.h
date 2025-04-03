/*
    Copyright (c) 2005-2021 Intel Corporation

    Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at

        http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.
*/

#ifndef __TBB_detail__hash_compare_H
#define __TBB_detail__hash_compare_H

#include <functional>

#include "_containers_helpers.h"

namespace tbb {
namespace detail {
namespace d1 {

template <typename Key, typename Hash, typename KeyEqual>
class hash_compare {
    using is_transparent_hash = has_transparent_key_equal<Key, Hash, KeyEqual>;
public:
    using hasher = Hash;
    using key_equal = typename is_transparent_hash::type;

    hash_compare() = default;
    hash_compare( hasher hash, key_equal equal ) : my_hasher(hash), my_equal(equal) {}

    std::size_t operator()( const Key& key ) const {
        return std::size_t(my_hasher(key));
    }

    bool operator()( const Key& key1, const Key& key2 ) const {
        return my_equal(key1, key2);
    }

    template <typename K, typename = typename std::enable_if<is_transparent_hash::value, K>::type>
    std::size_t operator()( const K& key ) const {
        return std::size_t(my_hasher(key));
    }

    template <typename K1, typename K2, typename = typename std::enable_if<is_transparent_hash::value, K1>::type>
    bool operator()( const K1& key1, const K2& key2 ) const {
        return my_equal(key1, key2);
    }

    hasher hash_function() const {
        return my_hasher;
    }

    key_equal key_eq() const {
        return my_equal;
    }


private:
    hasher my_hasher;
    key_equal my_equal;
}; // class hash_compare

//! hash_compare that is default argument for concurrent_hash_map
template <typename Key>
class tbb_hash_compare {
public:
    std::size_t hash( const Key& a ) const { return my_hash_func(a); }
    bool equal( const Key& a, const Key& b ) const { return my_key_equal(a, b); }
private:
    std::hash<Key> my_hash_func;
    std::equal_to<Key> my_key_equal;
};

} // namespace d1
} // namespace detail
} // namespace tbb

#if TBB_DEFINE_STD_HASH_SPECIALIZATIONS

namespace std {

template <typename T, typename U>
struct hash<std::pair<T, U>> {
public:
    std::size_t operator()( const std::pair<T, U>& p ) const {
        return first_hash(p.first) ^ second_hash(p.second);
    }

private:
    std::hash<T> first_hash;
    std::hash<U> second_hash;
}; // struct hash<std::pair>

// Apple clang and MSVC defines their own specializations for std::hash<std::basic_string<T, Traits, Alloc>>
#if !(_LIBCPP_VERSION) && !(_CPPLIB_VER)

template <typename CharT, typename Traits, typename Allocator>
struct hash<std::basic_string<CharT, Traits, Allocator>> {
public:
    std::size_t operator()( const std::basic_string<CharT, Traits, Allocator>& s ) const {
        std::size_t h = 0;
        for ( const CharT* c = s.c_str(); *c; ++c ) {
            h = h * hash_multiplier ^ char_hash(*c);
        }
        return h;
    }

private:
    static constexpr std::size_t hash_multiplier = tbb::detail::select_size_t_constant<2654435769U, 11400714819323198485ULL>::value;

    std::hash<CharT> char_hash;
}; // struct hash<std::basic_string>

#endif // !(_LIBCPP_VERSION || _CPPLIB_VER)

} // namespace std

#endif // TBB_DEFINE_STD_HASH_SPECIALIZATIONS

#endif // __TBB_detail__hash_compare_H
