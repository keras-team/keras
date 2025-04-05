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

#ifndef __TBB_concurrent_unordered_set_H
#define __TBB_concurrent_unordered_set_H

#include "detail/_namespace_injection.h"
#include "detail/_concurrent_unordered_base.h"
#include "tbb_allocator.h"

namespace tbb {
namespace detail {
namespace d1 {

template <typename Key, typename Hash, typename KeyEqual, typename Allocator, bool AllowMultimapping>
struct concurrent_unordered_set_traits {
    using key_type = Key;
    using value_type = key_type;
    using allocator_type = Allocator;
    using hash_compare_type = hash_compare<key_type, Hash, KeyEqual>;
    static constexpr bool allow_multimapping = AllowMultimapping;

    static constexpr const key_type& get_key( const value_type& value ) {
        return value;
    }
}; // class concurrent_unordered_set_traits

template <typename Key, typename Hash, typename KeyEqual, typename Allocator>
class concurrent_unordered_multiset;

template <typename Key, typename Hash = std::hash<Key>, typename KeyEqual = std::equal_to<Key>,
          typename Allocator = tbb::tbb_allocator<Key>>
class concurrent_unordered_set
    : public concurrent_unordered_base<concurrent_unordered_set_traits<Key, Hash, KeyEqual, Allocator, false>>
{
    using traits_type = concurrent_unordered_set_traits<Key, Hash, KeyEqual, Allocator, false>;
    using base_type = concurrent_unordered_base<traits_type>;
public:
    using key_type = typename base_type::key_type;
    using value_type = typename base_type::value_type;
    using size_type = typename base_type::size_type;
    using difference_type = typename base_type::difference_type;
    using hasher = typename base_type::hasher;
    using key_equal = typename base_type::key_equal;
    using allocator_type = typename base_type::allocator_type;
    using reference = typename base_type::reference;
    using const_reference = typename base_type::const_reference;
    using pointer = typename base_type::pointer;
    using const_pointer = typename base_type::const_pointer;
    using iterator = typename base_type::iterator;
    using const_iterator = typename base_type::const_iterator;
    using local_iterator = typename base_type::local_iterator;
    using const_local_iterator = typename base_type::const_local_iterator;
    using node_type = typename base_type::node_type;

    // Include constructors of base_type;
    using base_type::base_type;
    using base_type::operator=;
    // Required for implicit deduction guides
    concurrent_unordered_set() = default;
    concurrent_unordered_set( const concurrent_unordered_set& ) = default;
    concurrent_unordered_set( const concurrent_unordered_set& other, const allocator_type& alloc ) : base_type(other, alloc) {}
    concurrent_unordered_set( concurrent_unordered_set&& ) = default;
    concurrent_unordered_set( concurrent_unordered_set&& other, const allocator_type& alloc ) : base_type(std::move(other), alloc) {}
    // Required to respect the rule of 5
    concurrent_unordered_set& operator=( const concurrent_unordered_set& ) = default;
    concurrent_unordered_set& operator=( concurrent_unordered_set&& ) = default;

    template <typename OtherHash, typename OtherKeyEqual>
    void merge( concurrent_unordered_set<key_type, OtherHash, OtherKeyEqual, allocator_type>& source ) {
        this->internal_merge(source);
    }

    template <typename OtherHash, typename OtherKeyEqual>
    void merge( concurrent_unordered_set<key_type, OtherHash, OtherKeyEqual, allocator_type>&& source ) {
        this->internal_merge(std::move(source));
    }

    template <typename OtherHash, typename OtherKeyEqual>
    void merge( concurrent_unordered_multiset<key_type, OtherHash, OtherKeyEqual, allocator_type>& source ) {
        this->internal_merge(source);
    }

    template <typename OtherHash, typename OtherKeyEqual>
    void merge( concurrent_unordered_multiset<key_type, OtherHash, OtherKeyEqual, allocator_type>&& source ) {
        this->internal_merge(std::move(source));
    }
}; // class concurrent_unordered_set

#if __TBB_CPP17_DEDUCTION_GUIDES_PRESENT

template <typename It,
          typename Hash = std::hash<iterator_value_t<It>>,
          typename KeyEq = std::equal_to<iterator_value_t<It>>,
          typename Alloc = tbb::tbb_allocator<iterator_value_t<It>>,
          typename = std::enable_if_t<is_input_iterator_v<It>>,
          typename = std::enable_if_t<is_allocator_v<Alloc>>,
          typename = std::enable_if_t<!is_allocator_v<Hash>>,
          typename = std::enable_if_t<!is_allocator_v<KeyEq>>,
          typename = std::enable_if_t<!std::is_integral_v<Hash>>>
concurrent_unordered_set( It, It, std::size_t = {}, Hash = Hash(), KeyEq = KeyEq(), Alloc = Alloc() )
-> concurrent_unordered_set<iterator_value_t<It>, Hash, KeyEq, Alloc>;

template <typename T,
          typename Hash = std::hash<T>,
          typename KeyEq = std::equal_to<T>,
          typename Alloc = tbb::tbb_allocator<T>,
          typename = std::enable_if_t<is_allocator_v<Alloc>>,
          typename = std::enable_if_t<!is_allocator_v<Hash>>,
          typename = std::enable_if_t<!is_allocator_v<KeyEq>>,
          typename = std::enable_if_t<!std::is_integral_v<Hash>>>
concurrent_unordered_set( std::initializer_list<T>, std::size_t = {},
                          Hash = Hash(), KeyEq = KeyEq(), Alloc = Alloc() )
-> concurrent_unordered_set<T, Hash, KeyEq, Alloc>;

template <typename It, typename Alloc,
          typename = std::enable_if_t<is_input_iterator_v<It>>,
          typename = std::enable_if_t<is_allocator_v<Alloc>>>
concurrent_unordered_set( It, It, std::size_t, Alloc )
-> concurrent_unordered_set<iterator_value_t<It>, std::hash<iterator_value_t<It>>,
                            std::equal_to<iterator_value_t<It>>, Alloc>;

template <typename It, typename Hash, typename Alloc,
          typename = std::enable_if_t<is_input_iterator_v<It>>,
          typename = std::enable_if_t<is_allocator_v<Alloc>>,
          typename = std::enable_if_t<!is_allocator_v<Hash>>,
          typename = std::enable_if_t<!std::is_integral_v<Hash>>>
concurrent_unordered_set( It, It, std::size_t, Hash, Alloc )
-> concurrent_unordered_set<iterator_value_t<It>, Hash, std::equal_to<iterator_value_t<It>>, Alloc>;

template <typename T, typename Alloc,
          typename = std::enable_if_t<is_allocator_v<Alloc>>>
concurrent_unordered_set( std::initializer_list<T>, std::size_t, Alloc )
-> concurrent_unordered_set<T, std::hash<T>, std::equal_to<T>, Alloc>;

template <typename T, typename Alloc,
          typename = std::enable_if_t<is_allocator_v<Alloc>>>
concurrent_unordered_set( std::initializer_list<T>, Alloc )
-> concurrent_unordered_set<T, std::hash<T>, std::equal_to<T>, Alloc>;

template <typename T, typename Hash, typename Alloc,
          typename = std::enable_if_t<is_allocator_v<Alloc>>,
          typename = std::enable_if_t<!is_allocator_v<Hash>>,
          typename = std::enable_if_t<!std::is_integral_v<Hash>>>
concurrent_unordered_set( std::initializer_list<T>, std::size_t, Hash, Alloc )
-> concurrent_unordered_set<T, Hash, std::equal_to<T>, Alloc>;

#endif // __TBB_CPP17_DEDUCTION_GUIDES_PRESENT

template <typename Key, typename Hash, typename KeyEqual, typename Allocator>
void swap( concurrent_unordered_set<Key, Hash, KeyEqual, Allocator>& lhs,
           concurrent_unordered_set<Key, Hash, KeyEqual, Allocator>& rhs ) {
    lhs.swap(rhs);
}

template <typename Key, typename Hash = std::hash<Key>, typename KeyEqual = std::equal_to<Key>,
          typename Allocator = tbb::tbb_allocator<Key>>
class concurrent_unordered_multiset
    : public concurrent_unordered_base<concurrent_unordered_set_traits<Key, Hash, KeyEqual, Allocator, true>>
{
    using traits_type = concurrent_unordered_set_traits<Key, Hash, KeyEqual, Allocator, true>;
    using base_type = concurrent_unordered_base<traits_type>;
public:
    using key_type = typename base_type::key_type;
    using value_type = typename base_type::value_type;
    using size_type = typename base_type::size_type;
    using difference_type = typename base_type::difference_type;
    using hasher = typename base_type::hasher;
    using key_equal = typename base_type::key_equal;
    using allocator_type = typename base_type::allocator_type;
    using reference = typename base_type::reference;
    using const_reference = typename base_type::const_reference;
    using pointer = typename base_type::pointer;
    using const_pointer = typename base_type::const_pointer;
    using iterator = typename base_type::iterator;
    using const_iterator = typename base_type::const_iterator;
    using local_iterator = typename base_type::local_iterator;
    using const_local_iterator = typename base_type::const_local_iterator;
    using node_type = typename base_type::node_type;

    // Include constructors of base_type;
    using base_type::base_type;
    using base_type::operator=;

    // Required for implicit deduction guides
    concurrent_unordered_multiset() = default;
    concurrent_unordered_multiset( const concurrent_unordered_multiset& ) = default;
    concurrent_unordered_multiset( const concurrent_unordered_multiset& other, const allocator_type& alloc ) : base_type(other, alloc) {}
    concurrent_unordered_multiset( concurrent_unordered_multiset&& ) = default;
    concurrent_unordered_multiset( concurrent_unordered_multiset&& other, const allocator_type& alloc ) : base_type(std::move(other), alloc) {}
    // Required to respect the rule of 5
    concurrent_unordered_multiset& operator=( const concurrent_unordered_multiset& ) = default;
    concurrent_unordered_multiset& operator=( concurrent_unordered_multiset&& ) = default;

    template <typename OtherHash, typename OtherKeyEqual>
    void merge( concurrent_unordered_set<key_type, OtherHash, OtherKeyEqual, allocator_type>& source ) {
        this->internal_merge(source);
    }

    template <typename OtherHash, typename OtherKeyEqual>
    void merge( concurrent_unordered_set<key_type, OtherHash, OtherKeyEqual, allocator_type>&& source ) {
        this->internal_merge(std::move(source));
    }

    template <typename OtherHash, typename OtherKeyEqual>
    void merge( concurrent_unordered_multiset<key_type, OtherHash, OtherKeyEqual, allocator_type>& source ) {
        this->internal_merge(source);
    }

    template <typename OtherHash, typename OtherKeyEqual>
    void merge( concurrent_unordered_multiset<key_type, OtherHash, OtherKeyEqual, allocator_type>&& source ) {
        this->internal_merge(std::move(source));
    }
}; // class concurrent_unordered_multiset

#if __TBB_CPP17_DEDUCTION_GUIDES_PRESENT
template <typename It,
          typename Hash = std::hash<iterator_value_t<It>>,
          typename KeyEq = std::equal_to<iterator_value_t<It>>,
          typename Alloc = tbb::tbb_allocator<iterator_value_t<It>>,
          typename = std::enable_if_t<is_input_iterator_v<It>>,
          typename = std::enable_if_t<is_allocator_v<Alloc>>,
          typename = std::enable_if_t<!is_allocator_v<Hash>>,
          typename = std::enable_if_t<!is_allocator_v<KeyEq>>,
          typename = std::enable_if_t<!std::is_integral_v<Hash>>>
concurrent_unordered_multiset( It, It, std::size_t = {}, Hash = Hash(), KeyEq = KeyEq(), Alloc = Alloc() )
-> concurrent_unordered_multiset<iterator_value_t<It>, Hash, KeyEq, Alloc>;

template <typename T,
          typename Hash = std::hash<T>,
          typename KeyEq = std::equal_to<T>,
          typename Alloc = tbb::tbb_allocator<T>,
          typename = std::enable_if_t<is_allocator_v<Alloc>>,
          typename = std::enable_if_t<!is_allocator_v<Hash>>,
          typename = std::enable_if_t<!is_allocator_v<KeyEq>>,
          typename = std::enable_if_t<!std::is_integral_v<Hash>>>
concurrent_unordered_multiset( std::initializer_list<T>, std::size_t = {},
                          Hash = Hash(), KeyEq = KeyEq(), Alloc = Alloc() )
-> concurrent_unordered_multiset<T, Hash, KeyEq, Alloc>;

template <typename It, typename Alloc,
          typename = std::enable_if_t<is_input_iterator_v<It>>,
          typename = std::enable_if_t<is_allocator_v<Alloc>>>
concurrent_unordered_multiset( It, It, std::size_t, Alloc )
-> concurrent_unordered_multiset<iterator_value_t<It>, std::hash<iterator_value_t<It>>,
                            std::equal_to<iterator_value_t<It>>, Alloc>;

template <typename It, typename Hash, typename Alloc,
          typename = std::enable_if_t<is_input_iterator_v<It>>,
          typename = std::enable_if_t<is_allocator_v<Alloc>>,
          typename = std::enable_if_t<!is_allocator_v<Hash>>,
          typename = std::enable_if_t<!std::is_integral_v<Hash>>>
concurrent_unordered_multiset( It, It, std::size_t, Hash, Alloc )
-> concurrent_unordered_multiset<iterator_value_t<It>, Hash, std::equal_to<iterator_value_t<It>>, Alloc>;

template <typename T, typename Alloc,
          typename = std::enable_if_t<is_allocator_v<Alloc>>>
concurrent_unordered_multiset( std::initializer_list<T>, std::size_t, Alloc )
-> concurrent_unordered_multiset<T, std::hash<T>, std::equal_to<T>, Alloc>;

template <typename T, typename Alloc,
          typename = std::enable_if_t<is_allocator_v<Alloc>>>
concurrent_unordered_multiset( std::initializer_list<T>, Alloc )
-> concurrent_unordered_multiset<T, std::hash<T>, std::equal_to<T>, Alloc>;

template <typename T, typename Hash, typename Alloc,
          typename = std::enable_if_t<is_allocator_v<Alloc>>,
          typename = std::enable_if_t<!is_allocator_v<Hash>>,
          typename = std::enable_if_t<!std::is_integral_v<Hash>>>
concurrent_unordered_multiset( std::initializer_list<T>, std::size_t, Hash, Alloc )
-> concurrent_unordered_multiset<T, Hash, std::equal_to<T>, Alloc>;

#endif // __TBB_CPP17_DEDUCTION_GUIDES_PRESENT

template <typename Key, typename Hash, typename KeyEqual, typename Allocator>
void swap( concurrent_unordered_multiset<Key, Hash, KeyEqual, Allocator>& lhs,
           concurrent_unordered_multiset<Key, Hash, KeyEqual, Allocator>& rhs ) {
    lhs.swap(rhs);
}

} // namespace d1
} // namespace detail

inline namespace v1 {

using detail::d1::concurrent_unordered_set;
using detail::d1::concurrent_unordered_multiset;
using detail::split;

} // inline namespace v1
} // namespace tbb

#endif // __TBB_concurrent_unordered_set_H
