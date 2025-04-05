/*
    Copyright (c) 2019-2021 Intel Corporation

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

#ifndef __TBB_concurrent_map_H
#define __TBB_concurrent_map_H

#include "detail/_namespace_injection.h"
#include "detail/_concurrent_skip_list.h"
#include "tbb_allocator.h"
#include <functional>
#include <tuple>
#include <utility>

namespace tbb {
namespace detail {
namespace d1 {

template<typename Key, typename Value, typename KeyCompare, typename RandomGenerator,
         typename Allocator, bool AllowMultimapping>
struct map_traits {
    static constexpr std::size_t max_level = RandomGenerator::max_level;
    using random_level_generator_type = RandomGenerator;
    using key_type = Key;
    using mapped_type = Value;
    using compare_type = KeyCompare;
    using value_type = std::pair<const key_type, mapped_type>;
    using reference = value_type&;
    using const_reference = const value_type&;
    using allocator_type = Allocator;

    static constexpr bool allow_multimapping = AllowMultimapping;

    class value_compare {
    public:
        bool operator()(const value_type& lhs, const value_type& rhs) const {
            return comp(lhs.first, rhs.first);
        }

    protected:
        value_compare(compare_type c) : comp(c) {}

        friend struct map_traits;

        compare_type comp;
    };

    static value_compare value_comp(compare_type comp) { return value_compare(comp); }

    static const key_type& get_key(const_reference val) {
        return val.first;
    }
}; // struct map_traits

template <typename Key, typename Value, typename Compare, typename Allocator>
class concurrent_multimap;

template <typename Key, typename Value, typename Compare = std::less<Key>, typename Allocator = tbb::tbb_allocator<std::pair<const Key, Value>>>
class concurrent_map : public concurrent_skip_list<map_traits<Key, Value, Compare, concurrent_geometric_level_generator<32>, Allocator, false>> {
    using base_type = concurrent_skip_list<map_traits<Key, Value, Compare, concurrent_geometric_level_generator<32>, Allocator, false>>;
public:
    using key_type = Key;
    using mapped_type = Value;
    using value_type = typename base_type::value_type;
    using size_type = typename base_type::size_type;
    using difference_type = typename base_type::difference_type;
    using key_compare = Compare;
    using value_compare = typename base_type::value_compare;
    using allocator_type = Allocator;

    using reference = typename base_type::reference;
    using const_reference = typename base_type::const_reference;
    using pointer = typename base_type::pointer;
    using const_pointer = typename base_type::const_pointer;

    using iterator = typename base_type::iterator;
    using const_iterator = typename base_type::const_iterator;

    using node_type = typename base_type::node_type;

    // Include constructors of base type
    using base_type::base_type;
    using base_type::operator=;

    // Required for implicit deduction guides
    concurrent_map() = default;
    concurrent_map( const concurrent_map& ) = default;
    concurrent_map( const concurrent_map& other, const allocator_type& alloc ) : base_type(other, alloc) {}
    concurrent_map( concurrent_map&& ) = default;
    concurrent_map( concurrent_map&& other, const allocator_type& alloc ) : base_type(std::move(other), alloc) {}
    // Required to respect the rule of 5
    concurrent_map& operator=( const concurrent_map& ) = default;
    concurrent_map& operator=( concurrent_map&& ) = default;

    // Observers
    mapped_type& at(const key_type& key) {
        iterator it = this->find(key);

        if (it == this->end()) {
            throw_exception(exception_id::invalid_key);
        }
        return it->second;
    }

    const mapped_type& at(const key_type& key) const {
        return const_cast<concurrent_map*>(this)->at(key);
    }

    mapped_type& operator[](const key_type& key) {
        iterator it = this->find(key);

        if (it == this->end()) {
            it = this->emplace(std::piecewise_construct, std::forward_as_tuple(key), std::tuple<>()).first;
        }
        return it->second;
    }

    mapped_type& operator[](key_type&& key) {
        iterator it = this->find(key);

        if (it == this->end()) {
            it = this->emplace(std::piecewise_construct, std::forward_as_tuple(std::move(key)), std::tuple<>()).first;
        }
        return it->second;
    }

    using base_type::insert;

    template <typename P>
    typename std::enable_if<std::is_constructible<value_type, P&&>::value,
                            std::pair<iterator, bool>>::type insert( P&& value )
    {
        return this->emplace(std::forward<P>(value));
    }

    template <typename P>
    typename std::enable_if<std::is_constructible<value_type, P&&>::value,
                            iterator>::type insert( const_iterator hint, P&& value )
    {
        return this->emplace_hint(hint, std::forward<P>(value));
    }

    template<typename OtherCompare>
    void merge(concurrent_map<key_type, mapped_type, OtherCompare, Allocator>& source) {
        this->internal_merge(source);
    }

    template<typename OtherCompare>
    void merge(concurrent_map<key_type, mapped_type, OtherCompare, Allocator>&& source) {
        this->internal_merge(std::move(source));
    }

    template<typename OtherCompare>
    void merge(concurrent_multimap<key_type, mapped_type, OtherCompare, Allocator>& source) {
        this->internal_merge(source);
    }

    template<typename OtherCompare>
    void merge(concurrent_multimap<key_type, mapped_type, OtherCompare, Allocator>&& source) {
        this->internal_merge(std::move(source));
    }
}; // class concurrent_map

#if __TBB_CPP17_DEDUCTION_GUIDES_PRESENT

template <typename It,
          typename Comp = std::less<iterator_key_t<It>>,
          typename Alloc = tbb::tbb_allocator<iterator_alloc_pair_t<It>>,
          typename = std::enable_if_t<is_input_iterator_v<It>>,
          typename = std::enable_if_t<is_allocator_v<Alloc>>,
          typename = std::enable_if_t<!is_allocator_v<Comp>>>
concurrent_map( It, It, Comp = Comp(), Alloc = Alloc() )
-> concurrent_map<iterator_key_t<It>, iterator_mapped_t<It>, Comp, Alloc>;

template <typename Key, typename T,
          typename Comp = std::less<std::remove_const_t<Key>>,
          typename Alloc = tbb::tbb_allocator<std::pair<const Key, T>>,
          typename = std::enable_if_t<is_allocator_v<Alloc>>,
          typename = std::enable_if_t<!is_allocator_v<Comp>>>
concurrent_map( std::initializer_list<std::pair<Key, T>>, Comp = Comp(), Alloc = Alloc() )
-> concurrent_map<std::remove_const_t<Key>, T, Comp, Alloc>;

template <typename It, typename Alloc,
          typename = std::enable_if_t<is_input_iterator_v<It>>,
          typename = std::enable_if_t<is_allocator_v<Alloc>>>
concurrent_map( It, It, Alloc )
-> concurrent_map<iterator_key_t<It>, iterator_mapped_t<It>,
                  std::less<iterator_key_t<It>>, Alloc>;

template <typename Key, typename T, typename Alloc,
          typename = std::enable_if_t<is_allocator_v<Alloc>>>
concurrent_map( std::initializer_list<std::pair<Key, T>>, Alloc )
-> concurrent_map<std::remove_const_t<Key>, T, std::less<std::remove_const_t<Key>>, Alloc>;

#endif // __TBB_CPP17_DEDUCTION_GUIDES_PRESENT

template <typename Key, typename Value, typename Compare, typename Allocator>
void swap( concurrent_map<Key, Value, Compare, Allocator>& lhs,
           concurrent_map<Key, Value, Compare, Allocator>& rhs )
{
    lhs.swap(rhs);
}

template <typename Key, typename Value, typename Compare = std::less<Key>, typename Allocator = tbb::tbb_allocator<std::pair<const Key, Value>>>
class concurrent_multimap : public concurrent_skip_list<map_traits<Key, Value, Compare, concurrent_geometric_level_generator<32>, Allocator, true>> {
    using base_type = concurrent_skip_list<map_traits<Key, Value, Compare, concurrent_geometric_level_generator<32>, Allocator, true>>;
public:
    using key_type = Key;
    using mapped_type = Value;
    using value_type = typename base_type::value_type;
    using size_type = typename base_type::size_type;
    using difference_type = typename base_type::difference_type;
    using key_compare = Compare;
    using value_compare = typename base_type::value_compare;
    using allocator_type = Allocator;

    using reference = typename base_type::reference;
    using const_reference = typename base_type::const_reference;
    using pointer = typename base_type::pointer;
    using const_pointer = typename base_type::const_pointer;

    using iterator = typename base_type::iterator;
    using const_iterator = typename base_type::const_iterator;

    using node_type = typename base_type::node_type;

    // Include constructors of base_type
    using base_type::base_type;
    using base_type::insert;
    using base_type::operator=;

    // Required for implicit deduction guides
    concurrent_multimap() = default;
    concurrent_multimap( const concurrent_multimap& ) = default;
    concurrent_multimap( const concurrent_multimap& other, const allocator_type& alloc ) : base_type(other, alloc) {}
    concurrent_multimap( concurrent_multimap&& ) = default;
    concurrent_multimap( concurrent_multimap&& other, const allocator_type& alloc ) : base_type(std::move(other), alloc) {}
    // Required to respect the rule of 5
    concurrent_multimap& operator=( const concurrent_multimap& ) = default;
    concurrent_multimap& operator=( concurrent_multimap&& ) = default;

    template <typename P>
    typename std::enable_if<std::is_constructible<value_type, P&&>::value,
                            std::pair<iterator, bool>>::type insert( P&& value )
    {
        return this->emplace(std::forward<P>(value));
    }

    template <typename P>
    typename std::enable_if<std::is_constructible<value_type, P&&>::value,
                            iterator>::type insert( const_iterator hint, P&& value )
    {
        return this->emplace_hint(hint, std::forward<P>(value));
    }

    template<typename OtherCompare>
    void merge(concurrent_multimap<key_type, mapped_type, OtherCompare, Allocator>& source) {
        this->internal_merge(source);
    }

    template<typename OtherCompare>
    void merge(concurrent_multimap<key_type, mapped_type, OtherCompare, Allocator>&& source) {
        this->internal_merge(std::move(source));
    }

    template<typename OtherCompare>
    void merge(concurrent_map<key_type, mapped_type, OtherCompare, Allocator>& source) {
        this->internal_merge(source);
    }

    template<typename OtherCompare>
    void merge(concurrent_map<key_type, mapped_type, OtherCompare, Allocator>&& source) {
        this->internal_merge(std::move(source));
    }
}; // class concurrent_multimap

#if __TBB_CPP17_DEDUCTION_GUIDES_PRESENT

template <typename It,
          typename Comp = std::less<iterator_key_t<It>>,
          typename Alloc = tbb::tbb_allocator<iterator_alloc_pair_t<It>>,
          typename = std::enable_if_t<is_input_iterator_v<It>>,
          typename = std::enable_if_t<is_allocator_v<Alloc>>,
          typename = std::enable_if_t<!is_allocator_v<Comp>>>
concurrent_multimap( It, It, Comp = Comp(), Alloc = Alloc() )
-> concurrent_multimap<iterator_key_t<It>, iterator_mapped_t<It>, Comp, Alloc>;

template <typename Key, typename T,
          typename Comp = std::less<std::remove_const_t<Key>>,
          typename Alloc = tbb::tbb_allocator<std::pair<const Key, T>>,
          typename = std::enable_if_t<is_allocator_v<Alloc>>,
          typename = std::enable_if_t<!is_allocator_v<Comp>>>
concurrent_multimap( std::initializer_list<std::pair<Key, T>>, Comp = Comp(), Alloc = Alloc() )
-> concurrent_multimap<std::remove_const_t<Key>, T, Comp, Alloc>;

template <typename It, typename Alloc,
          typename = std::enable_if_t<is_input_iterator_v<It>>,
          typename = std::enable_if_t<is_allocator_v<Alloc>>>
concurrent_multimap( It, It, Alloc )
-> concurrent_multimap<iterator_key_t<It>, iterator_mapped_t<It>,
                       std::less<iterator_key_t<It>>, Alloc>;

template <typename Key, typename T, typename Alloc,
          typename = std::enable_if_t<is_allocator_v<Alloc>>>
concurrent_multimap( std::initializer_list<std::pair<Key, T>>, Alloc )
-> concurrent_multimap<std::remove_const_t<Key>, T, std::less<std::remove_const_t<Key>>, Alloc>;


#endif // __TBB_CPP17_DEDUCTION_GUIDES_PRESENT

template <typename Key, typename Value, typename Compare, typename Allocator>
void swap( concurrent_multimap<Key, Value, Compare, Allocator>& lhs,
           concurrent_multimap<Key, Value, Compare, Allocator>& rhs )
{
    lhs.swap(rhs);
}

} // namespace d1
} // namespace detail

inline namespace v1 {

using detail::d1::concurrent_map;
using detail::d1::concurrent_multimap;
using detail::split;

} // inline namespace v1
} // namespace tbb

#endif // __TBB_concurrent_map_H
