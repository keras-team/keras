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

#ifndef __TBB_detail__concurrent_unordered_base_H
#define __TBB_detail__concurrent_unordered_base_H

#if !defined(__TBB_concurrent_unordered_map_H) && !defined(__TBB_concurrent_unordered_set_H)
#error Do not #include this internal file directly; use public TBB headers instead.
#endif

#include "_range_common.h"
#include "_containers_helpers.h"
#include "_segment_table.h"
#include "_hash_compare.h"
#include "_allocator_traits.h"
#include "_node_handle.h"
#include "_assert.h"
#include "_utils.h"
#include "_exception.h"
#include <iterator>
#include <utility>
#include <functional>
#include <initializer_list>
#include <atomic>
#include <type_traits>
#include <memory>
#include <algorithm>

#if defined(_MSC_VER) && !defined(__INTEL_COMPILER)
#pragma warning(push)
#pragma warning(disable: 4127) // warning C4127: conditional expression is constant
#endif

namespace tbb {
namespace detail {
namespace d1 {

template <typename Traits>
class concurrent_unordered_base;

template<typename Container, typename Value>
class solist_iterator {
private:
    using node_ptr = typename Container::value_node_ptr;
    template <typename T, typename Allocator>
    friend class split_ordered_list;
    template<typename M, typename V>
    friend class solist_iterator;
    template <typename Traits>
    friend class concurrent_unordered_base;
    template<typename M, typename T, typename U>
    friend bool operator==( const solist_iterator<M,T>& i, const solist_iterator<M,U>& j );
    template<typename M, typename T, typename U>
    friend bool operator!=( const solist_iterator<M,T>& i, const solist_iterator<M,U>& j );
public:
    using value_type = Value;
    using difference_type = typename Container::difference_type;
    using pointer = value_type*;
    using reference = value_type&;
    using iterator_category = std::forward_iterator_tag;

    solist_iterator() : my_node_ptr(nullptr) {}
    solist_iterator( const solist_iterator<Container, typename Container::value_type>& other )
        : my_node_ptr(other.my_node_ptr) {}

    solist_iterator& operator=( const solist_iterator<Container, typename Container::value_type>& other ) {
        my_node_ptr = other.my_node_ptr;
        return *this;
    }

    reference operator*() const {
        return my_node_ptr->value();
    }

    pointer operator->() const {
        return my_node_ptr->storage();
    }

    solist_iterator& operator++() {
        auto next_node = my_node_ptr->next();
        while(next_node && next_node->is_dummy()) {
            next_node = next_node->next();
        }
        my_node_ptr = static_cast<node_ptr>(next_node);
        return *this;
    }

    solist_iterator operator++(int) {
        solist_iterator tmp = *this;
        ++*this;
        return tmp;
    }

private:
    solist_iterator( node_ptr pnode ) : my_node_ptr(pnode) {}

    node_ptr get_node_ptr() const { return my_node_ptr; }

    node_ptr my_node_ptr;
};

template<typename Solist, typename T, typename U>
bool operator==( const solist_iterator<Solist, T>& i, const solist_iterator<Solist, U>& j ) {
    return i.my_node_ptr == j.my_node_ptr;
}

template<typename Solist, typename T, typename U>
bool operator!=( const solist_iterator<Solist, T>& i, const solist_iterator<Solist, U>& j ) {
    return i.my_node_ptr != j.my_node_ptr;
}

template <typename SokeyType>
class list_node {
public:
    using node_ptr = list_node*;
    using sokey_type = SokeyType;

    list_node(sokey_type key) : my_next(nullptr), my_order_key(key) {}

    void init( sokey_type key ) {
        my_order_key = key;
    }

    sokey_type order_key() const {
        return my_order_key;
    }

    bool is_dummy() {
        // The last bit of order key is unset for dummy nodes
        return (my_order_key & 0x1) == 0;
    }

    node_ptr next() const {
        return my_next.load(std::memory_order_acquire);
    }

    void set_next( node_ptr next_node ) {
        my_next.store(next_node, std::memory_order_release);
    }

    bool try_set_next( node_ptr expected_next, node_ptr new_next ) {
        return my_next.compare_exchange_strong(expected_next, new_next);
    }

private:
    std::atomic<node_ptr> my_next;
    sokey_type my_order_key;
}; // class list_node

template <typename ValueType, typename SokeyType>
class value_node : public list_node<SokeyType>
{
public:
    using base_type = list_node<SokeyType>;
    using sokey_type = typename base_type::sokey_type;
    using value_type = ValueType;

    value_node( sokey_type ord_key ) : base_type(ord_key) {}
    ~value_node() {}
    value_type* storage() {
        return reinterpret_cast<value_type*>(&my_value);
    }

    value_type& value() {
        return *storage();
    }

private:
    using aligned_storage_type = typename std::aligned_storage<sizeof(value_type)>::type;
    aligned_storage_type my_value;
}; // class value_node

template <typename Traits>
class concurrent_unordered_base {
    using self_type = concurrent_unordered_base<Traits>;
    using traits_type = Traits;
    using hash_compare_type = typename traits_type::hash_compare_type;
    class unordered_segment_table;
public:
    using value_type = typename traits_type::value_type;
    using key_type = typename traits_type::key_type;
    using allocator_type = typename traits_type::allocator_type;

private:
    using allocator_traits_type = tbb::detail::allocator_traits<allocator_type>;
    // TODO: check assert conditions for different C++ standards
    static_assert(std::is_same<typename allocator_traits_type::value_type, value_type>::value,
                  "value_type of the container must be the same as its allocator");
    using sokey_type = std::size_t;

public:
    using size_type = std::size_t;
    using difference_type = std::ptrdiff_t;

    using iterator = solist_iterator<self_type, value_type>;
    using const_iterator = solist_iterator<self_type, const value_type>;
    using local_iterator = iterator;
    using const_local_iterator = const_iterator;

    using reference = value_type&;
    using const_reference = const value_type&;
    using pointer = typename allocator_traits_type::pointer;
    using const_pointer = typename allocator_traits_type::const_pointer;

    using hasher = typename hash_compare_type::hasher;
    using key_equal = typename hash_compare_type::key_equal;

private:
    using list_node_type = list_node<sokey_type>;
    using value_node_type = value_node<value_type, sokey_type>;
    using node_ptr = list_node_type*;
    using value_node_ptr = value_node_type*;

    using value_node_allocator_type = typename allocator_traits_type::template rebind_alloc<value_node_type>;
    using node_allocator_type = typename allocator_traits_type::template rebind_alloc<list_node_type>;

    using node_allocator_traits = tbb::detail::allocator_traits<node_allocator_type>;
    using value_node_allocator_traits = tbb::detail::allocator_traits<value_node_allocator_type>;

    static constexpr size_type round_up_to_power_of_two( size_type bucket_count ) {
        return size_type(1) << size_type(tbb::detail::log2(uintptr_t(bucket_count == 0 ? 1 : bucket_count) * 2 - 1));
    }

    template <typename T>
    using is_transparent = dependent_bool<has_transparent_key_equal<key_type, hasher, key_equal>, T>;
public:
    using node_type = node_handle<key_type, value_type, value_node_type, allocator_type>;

    explicit concurrent_unordered_base( size_type bucket_count, const hasher& hash = hasher(),
                                        const key_equal& equal = key_equal(), const allocator_type& alloc = allocator_type() )
        : my_size(0),
          my_bucket_count(round_up_to_power_of_two(bucket_count)),
          my_max_load_factor(float(initial_max_load_factor)),
          my_hash_compare(hash, equal),
          my_head(sokey_type(0)),
          my_segments(alloc) {}

    concurrent_unordered_base() : concurrent_unordered_base(initial_bucket_count) {}

    concurrent_unordered_base( size_type bucket_count, const allocator_type& alloc )
        : concurrent_unordered_base(bucket_count, hasher(), key_equal(), alloc) {}

    concurrent_unordered_base( size_type bucket_count, const hasher& hash, const allocator_type& alloc )
        : concurrent_unordered_base(bucket_count, hash, key_equal(), alloc) {}

    explicit concurrent_unordered_base( const allocator_type& alloc )
        : concurrent_unordered_base(initial_bucket_count, hasher(), key_equal(), alloc) {}

    template <typename InputIterator>
    concurrent_unordered_base( InputIterator first, InputIterator last,
                               size_type bucket_count = initial_bucket_count, const hasher& hash = hasher(),
                               const key_equal& equal = key_equal(), const allocator_type& alloc = allocator_type() )
        : concurrent_unordered_base(bucket_count, hash, equal, alloc)
    {
        insert(first, last);
    }

    template <typename InputIterator>
    concurrent_unordered_base( InputIterator first, InputIterator last,
                               size_type bucket_count, const allocator_type& alloc )
        : concurrent_unordered_base(first, last, bucket_count, hasher(), key_equal(), alloc) {}

    template <typename InputIterator>
    concurrent_unordered_base( InputIterator first, InputIterator last,
                               size_type bucket_count, const hasher& hash, const allocator_type& alloc )
        : concurrent_unordered_base(first, last, bucket_count, hash, key_equal(), alloc) {}

    concurrent_unordered_base( const concurrent_unordered_base& other )
        : my_size(other.my_size.load(std::memory_order_relaxed)),
          my_bucket_count(other.my_bucket_count.load(std::memory_order_relaxed)),
          my_max_load_factor(other.my_max_load_factor),
          my_hash_compare(other.my_hash_compare),
          my_head(other.my_head.order_key()),
          my_segments(other.my_segments)
    {
        try_call( [&] {
            internal_copy(other);
        } ).on_exception( [&] {
            clear();
        });
    }

    concurrent_unordered_base( const concurrent_unordered_base& other, const allocator_type& alloc )
        : my_size(other.my_size.load(std::memory_order_relaxed)),
          my_bucket_count(other.my_bucket_count.load(std::memory_order_relaxed)),
          my_max_load_factor(other.my_max_load_factor),
          my_hash_compare(other.my_hash_compare),
          my_head(other.my_head.order_key()),
          my_segments(other.my_segments, alloc)
    {
        try_call( [&] {
            internal_copy(other);
        } ).on_exception( [&] {
            clear();
        });
    }

    concurrent_unordered_base( concurrent_unordered_base&& other )
        : my_size(other.my_size.load(std::memory_order_relaxed)),
          my_bucket_count(other.my_bucket_count.load(std::memory_order_relaxed)),
          my_max_load_factor(std::move(other.my_max_load_factor)),
          my_hash_compare(std::move(other.my_hash_compare)),
          my_head(other.my_head.order_key()),
          my_segments(std::move(other.my_segments))
    {
        move_content(std::move(other));
    }

    concurrent_unordered_base( concurrent_unordered_base&& other, const allocator_type& alloc )
        : my_size(other.my_size.load(std::memory_order_relaxed)),
          my_bucket_count(other.my_bucket_count.load(std::memory_order_relaxed)),
          my_max_load_factor(std::move(other.my_max_load_factor)),
          my_hash_compare(std::move(other.my_hash_compare)),
          my_head(other.my_head.order_key()),
          my_segments(std::move(other.my_segments), alloc)
    {
        using is_always_equal = typename allocator_traits_type::is_always_equal;
        internal_move_construct_with_allocator(std::move(other), alloc, is_always_equal());
    }

    concurrent_unordered_base( std::initializer_list<value_type> init,
                               size_type bucket_count = initial_bucket_count,
                               const hasher& hash = hasher(), const key_equal& equal = key_equal(),
                               const allocator_type& alloc = allocator_type() )
        : concurrent_unordered_base(init.begin(), init.end(), bucket_count, hash, equal, alloc) {}

    concurrent_unordered_base( std::initializer_list<value_type> init,
                               size_type bucket_count, const allocator_type& alloc )
        : concurrent_unordered_base(init, bucket_count, hasher(), key_equal(), alloc) {}

    concurrent_unordered_base( std::initializer_list<value_type> init,
                               size_type bucket_count, const hasher& hash, const allocator_type& alloc )
        : concurrent_unordered_base(init, bucket_count, hash, key_equal(), alloc) {}

    ~concurrent_unordered_base() {
        internal_clear();
    }

    concurrent_unordered_base& operator=( const concurrent_unordered_base& other ) {
        if (this != &other) {
            clear();
            my_size.store(other.my_size.load(std::memory_order_relaxed), std::memory_order_relaxed);
            my_bucket_count.store(other.my_bucket_count.load(std::memory_order_relaxed), std::memory_order_relaxed);
            my_max_load_factor = other.my_max_load_factor;
            my_hash_compare = other.my_hash_compare;
            my_segments = other.my_segments;
            internal_copy(other); // TODO: guards for exceptions?
        }
        return *this;
    }

    concurrent_unordered_base& operator=( concurrent_unordered_base&& other ) noexcept(unordered_segment_table::is_noexcept_assignment) {
        if (this != &other) {
            clear();
            my_size.store(other.my_size.load(std::memory_order_relaxed), std::memory_order_relaxed);
            my_bucket_count.store(other.my_bucket_count.load(std::memory_order_relaxed), std::memory_order_relaxed);
            my_max_load_factor = std::move(other.my_max_load_factor);
            my_hash_compare = std::move(other.my_hash_compare);
            my_segments = std::move(other.my_segments);

            using pocma_type = typename allocator_traits_type::propagate_on_container_move_assignment;
            using is_always_equal = typename allocator_traits_type::is_always_equal;
            internal_move_assign(std::move(other), tbb::detail::disjunction<pocma_type, is_always_equal>());
        }
        return *this;
    }

    concurrent_unordered_base& operator=( std::initializer_list<value_type> init ) {
        clear();
        insert(init);
        return *this;
    }

    void swap( concurrent_unordered_base& other ) noexcept(unordered_segment_table::is_noexcept_swap) {
        if (this != &other) {
            using pocs_type = typename allocator_traits_type::propagate_on_container_swap;
            using is_always_equal = typename allocator_traits_type::is_always_equal;
            internal_swap(other, tbb::detail::disjunction<pocs_type, is_always_equal>());
        }
    }

    allocator_type get_allocator() const noexcept { return my_segments.get_allocator(); }

    iterator begin() noexcept { return iterator(first_value_node(&my_head)); }
    const_iterator begin() const noexcept { return const_iterator(first_value_node(const_cast<node_ptr>(&my_head))); }
    const_iterator cbegin() const noexcept { return const_iterator(first_value_node(const_cast<node_ptr>(&my_head))); }

    iterator end() noexcept { return iterator(nullptr); }
    const_iterator end() const noexcept { return const_iterator(nullptr); }
    const_iterator cend() const noexcept { return const_iterator(nullptr); }

    __TBB_nodiscard bool empty() const noexcept { return size() == 0; }
    size_type size() const noexcept { return my_size.load(std::memory_order_relaxed); }
    size_type max_size() const noexcept { return allocator_traits_type::max_size(get_allocator()); }

    void clear() noexcept {
        internal_clear();
    }

    std::pair<iterator, bool> insert( const value_type& value ) {
        return internal_insert_value(value);
    }

    std::pair<iterator, bool> insert( value_type&& value ) {
        return internal_insert_value(std::move(value));
    }

    iterator insert( const_iterator, const value_type& value ) {
        // Ignore hint
        return insert(value).first;
    }

    iterator insert( const_iterator, value_type&& value ) {
        // Ignore hint
        return insert(std::move(value)).first;
    }

    template <typename InputIterator>
    void insert( InputIterator first, InputIterator last ) {
        for (; first != last; ++first) {
            insert(*first);
        }
    }

    void insert( std::initializer_list<value_type> init ) {
        insert(init.begin(), init.end());
    }

    std::pair<iterator, bool> insert( node_type&& nh ) {
        if (!nh.empty()) {
            value_node_ptr insert_node = node_handle_accessor::get_node_ptr(nh);
            auto init_node = [&insert_node]( sokey_type order_key )->value_node_ptr {
                insert_node->init(order_key);
                return insert_node;
            };
            auto insert_result = internal_insert(insert_node->value(), init_node);
            if (insert_result.inserted) {
                // If the insertion succeeded - set node handle to the empty state
                __TBB_ASSERT(insert_result.remaining_node == nullptr,
                            "internal_insert_node should not return the remaining node if the insertion succeeded");
                node_handle_accessor::deactivate(nh);
            }
            return { iterator(insert_result.node_with_equal_key), insert_result.inserted };
        }
        return {end(), false};
    }

    iterator insert( const_iterator, node_type&& nh ) {
        // Ignore hint
        return insert(std::move(nh)).first;
    }

    template <typename... Args>
    std::pair<iterator, bool> emplace( Args&&... args ) {
        // Create a node with temporary order_key 0, which will be reinitialize
        // in internal_insert after the hash calculation
        value_node_ptr insert_node = create_node(0, std::forward<Args>(args)...);

        auto init_node = [&insert_node]( sokey_type order_key )->value_node_ptr {
            insert_node->init(order_key);
            return insert_node;
        };

        auto insert_result = internal_insert(insert_node->value(), init_node);

        if (!insert_result.inserted) {
            // If the insertion failed - destroy the node which was created
            insert_node->init(split_order_key_regular(1));
            destroy_node(insert_node);
        }

        return { iterator(insert_result.node_with_equal_key), insert_result.inserted };
    }

    template <typename... Args>
    iterator emplace_hint( const_iterator, Args&&... args ) {
        // Ignore hint
        return emplace(std::forward<Args>(args)...).first;
    }

    iterator unsafe_erase( const_iterator pos ) {
        return iterator(first_value_node(internal_erase(pos.get_node_ptr())));
    }

    iterator unsafe_erase( iterator pos ) {
        return iterator(first_value_node(internal_erase(pos.get_node_ptr())));
    }

    iterator unsafe_erase( const_iterator first, const_iterator last ) {
        while(first != last) {
            first = unsafe_erase(first);
        }
        return iterator(first.get_node_ptr());
    }

    size_type unsafe_erase( const key_type& key ) {
        return internal_erase_by_key(key);
    }

    template <typename K>
    typename std::enable_if<is_transparent<K>::value
                            && !std::is_convertible<K, const_iterator>::value
                            && !std::is_convertible<K, iterator>::value,
                            size_type>::type unsafe_erase( const K& key )
    {
        return internal_erase_by_key(key);
    }

    node_type unsafe_extract( const_iterator pos ) {
        internal_extract(pos.get_node_ptr());
        return node_handle_accessor::construct<node_type>(pos.get_node_ptr());
    }

    node_type unsafe_extract( iterator pos ) {
        internal_extract(pos.get_node_ptr());
        return node_handle_accessor::construct<node_type>(pos.get_node_ptr());
    }

    node_type unsafe_extract( const key_type& key ) {
        iterator item = find(key);
        return item == end() ? node_type() : unsafe_extract(item);
    }

    template <typename K>
    typename std::enable_if<is_transparent<K>::value
                            && !std::is_convertible<K, const_iterator>::value
                            && !std::is_convertible<K, iterator>::value,
                            node_type>::type unsafe_extract( const K& key )
    {
        iterator item = find(key);
        return item == end() ? node_type() : unsafe_extract(item);
    }

    // Lookup functions
    iterator find( const key_type& key ) {
        value_node_ptr result = internal_find(key);
        return result == nullptr ? end() : iterator(result);
    }

    const_iterator find( const key_type& key ) const {
        value_node_ptr result = const_cast<self_type*>(this)->internal_find(key);
        return result == nullptr ? end() : const_iterator(result);
    }

    template <typename K>
    typename std::enable_if<is_transparent<K>::value, iterator>::type find( const K& key ) {
        value_node_ptr result = internal_find(key);
        return result == nullptr ? end() : iterator(result);
    }

    template <typename K>
    typename std::enable_if<is_transparent<K>::value, const_iterator>::type find( const K& key ) const {
        value_node_ptr result = const_cast<self_type*>(this)->internal_find(key);
        return result == nullptr ? end() : const_iterator(result);
    }

    std::pair<iterator, iterator> equal_range( const key_type& key ) {
        auto result = internal_equal_range(key);
        return std::make_pair(iterator(result.first), iterator(result.second));
    }

    std::pair<const_iterator, const_iterator> equal_range( const key_type& key ) const {
        auto result = const_cast<self_type*>(this)->internal_equal_range(key);
        return std::make_pair(const_iterator(result.first), const_iterator(result.second));
    }

    template <typename K>
    typename std::enable_if<is_transparent<K>::value, std::pair<iterator, iterator>>::type equal_range( const K& key ) {
        auto result = internal_equal_range(key);
        return std::make_pair(iterator(result.first), iterator(result.second));
    }

    template <typename K>
    typename std::enable_if<is_transparent<K>::value, std::pair<const_iterator, const_iterator>>::type equal_range( const K& key ) const {
        auto result = const_cast<self_type*>(this)->internal_equal_range(key);
        return std::make_pair(iterator(result.first), iterator(result.second));
    }

    size_type count( const key_type& key ) const {
        return internal_count(key);
    }

    template <typename K>
    typename std::enable_if<is_transparent<K>::value, size_type>::type count( const K& key ) const {
        return internal_count(key);
    }

    bool contains( const key_type& key ) const {
        return find(key) != end();
    }

    template <typename K>
    typename std::enable_if<is_transparent<K>::value, bool>::type contains( const K& key ) const {
        return find(key) != end();
    }

    // Bucket interface
    local_iterator unsafe_begin( size_type n ) {
        return local_iterator(first_value_node(get_bucket(n)));
    }

    const_local_iterator unsafe_begin( size_type n ) const {
        auto bucket_begin = first_value_node(const_cast<self_type*>(this)->get_bucket(n));
        return const_local_iterator(bucket_begin);
    }

    const_local_iterator unsafe_cbegin( size_type n ) const {
        auto bucket_begin = first_value_node(const_cast<self_type*>(this)->get_bucket(n));
        return const_local_iterator(bucket_begin);
    }

    local_iterator unsafe_end( size_type n ) {
        size_type bucket_count = my_bucket_count.load(std::memory_order_relaxed);
        return n != bucket_count - 1 ? unsafe_begin(get_next_bucket_index(n)) : local_iterator(nullptr);
    }

    const_local_iterator unsafe_end( size_type n ) const {
        size_type bucket_count = my_bucket_count.load(std::memory_order_relaxed);
        return n != bucket_count - 1 ? unsafe_begin(get_next_bucket_index(n)) : const_local_iterator(nullptr);
    }

    const_local_iterator unsafe_cend( size_type n ) const {
        size_type bucket_count = my_bucket_count.load(std::memory_order_relaxed);
        return n != bucket_count - 1 ? unsafe_begin(get_next_bucket_index(n)) : const_local_iterator(nullptr);
    }

    size_type unsafe_bucket_count() const { return my_bucket_count.load(std::memory_order_relaxed); }

    size_type unsafe_max_bucket_count() const {
        return max_size();
    }

    size_type unsafe_bucket_size( size_type n ) const {
        return size_type(std::distance(unsafe_begin(n), unsafe_end(n)));
    }

    size_type unsafe_bucket( const key_type& key ) const {
        return my_hash_compare(key) % my_bucket_count.load(std::memory_order_relaxed);
    }

    // Hash policy
    float load_factor() const {
        return float(size() / float(my_bucket_count.load(std::memory_order_acquire)));
    }

    float max_load_factor() const { return my_max_load_factor; }

    void max_load_factor( float mlf ) {
        if (mlf != mlf || mlf < 0) {
            tbb::detail::throw_exception(exception_id::invalid_load_factor);
        }
        my_max_load_factor = mlf;
    } // TODO: unsafe?

    void rehash( size_type bucket_count ) {
        size_type current_bucket_count = my_bucket_count.load(std::memory_order_acquire);
        if (current_bucket_count < bucket_count) {
            // TODO: do we need do-while here?
            my_bucket_count.compare_exchange_strong(current_bucket_count, round_up_to_power_of_two(bucket_count));
        }
    }

    void reserve( size_type elements_count ) {
        size_type current_bucket_count = my_bucket_count.load(std::memory_order_acquire);
        size_type necessary_bucket_count = current_bucket_count;

        do {
            // TODO: Log2 seems useful here
            while (necessary_bucket_count * max_load_factor() < elements_count) {
                necessary_bucket_count <<= 1;
            }
        } while (current_bucket_count >= necessary_bucket_count ||
                 !my_bucket_count.compare_exchange_strong(current_bucket_count, necessary_bucket_count));
    }

    // Observers
    hasher hash_function() const { return my_hash_compare.hash_function(); }
    key_equal key_eq() const { return my_hash_compare.key_eq(); }

    class const_range_type {
    private:
        const concurrent_unordered_base& my_instance;
        node_ptr my_begin_node; // may be node* const
        node_ptr my_end_node;
        mutable node_ptr my_midpoint_node;
    public:
        using size_type = typename concurrent_unordered_base::size_type;
        using value_type = typename concurrent_unordered_base::value_type;
        using reference = typename concurrent_unordered_base::reference;
        using difference_type = typename concurrent_unordered_base::difference_type;
        using iterator = typename concurrent_unordered_base::const_iterator;

        bool empty() const { return my_begin_node == my_end_node; }

        bool is_divisible() const {
            return my_midpoint_node != my_end_node;
        }

        size_type grainsize() const { return 1; }

        const_range_type( const_range_type& range, split )
            : my_instance(range.my_instance),
              my_begin_node(range.my_midpoint_node),
              my_end_node(range.my_end_node)
        {
            range.my_end_node = my_begin_node;
            __TBB_ASSERT(!empty(), "Splitting despite the range is not divisible");
            __TBB_ASSERT(!range.empty(), "Splitting despite the range is not divisible");
            set_midpoint();
            range.set_midpoint();
        }

        iterator begin() const { return iterator(my_instance.first_value_node(my_begin_node)); }
        iterator end() const { return iterator(my_instance.first_value_node(my_end_node)); }

        const_range_type( const concurrent_unordered_base& table )
            : my_instance(table), my_begin_node(const_cast<node_ptr>(&table.my_head)), my_end_node(nullptr)
        {
            set_midpoint();
        }
    private:
        void set_midpoint() const {
            if (my_begin_node == my_end_node) {
                my_midpoint_node = my_end_node;
            } else {
                sokey_type invalid_key = ~sokey_type(0);
                sokey_type begin_key = my_begin_node != nullptr ? my_begin_node->order_key() : invalid_key;
                sokey_type end_key = my_end_node != nullptr ? my_end_node->order_key() : invalid_key;

                size_type mid_bucket = reverse_bits(begin_key + (end_key - begin_key) / 2) %
                    my_instance.my_bucket_count.load(std::memory_order_relaxed);
                while( my_instance.my_segments[mid_bucket].load(std::memory_order_relaxed) == nullptr) {
                    mid_bucket = my_instance.get_parent(mid_bucket);
                }
                if (reverse_bits(mid_bucket) > begin_key) {
                    // Found a dummy node between begin and end
                    my_midpoint_node = my_instance.first_value_node(
                        my_instance.my_segments[mid_bucket].load(std::memory_order_relaxed));
                } else {
                    // Didn't find a dummy node between begin and end
                    my_midpoint_node = my_end_node;
                }
            }
        }
    }; // class const_range_type

    class range_type : public const_range_type {
    public:
        using iterator = typename concurrent_unordered_base::iterator;
        using const_range_type::const_range_type;

        iterator begin() const { return iterator(const_range_type::begin().get_node_ptr()); }
        iterator end() const { return iterator(const_range_type::end().get_node_ptr()); }
    }; // class range_type

    // Parallel iteration
    range_type range() {
        return range_type(*this);
    }

    const_range_type range() const {
        return const_range_type(*this);
    }
protected:
    static constexpr bool allow_multimapping = traits_type::allow_multimapping;

private:
    static constexpr size_type initial_bucket_count = 8;
    static constexpr float initial_max_load_factor = 4; // TODO: consider 1?
    static constexpr size_type pointers_per_embedded_table = sizeof(size_type) * 8 - 1;

    class unordered_segment_table
        : public segment_table<std::atomic<node_ptr>, allocator_type, unordered_segment_table, pointers_per_embedded_table>
    {
        using self_type = unordered_segment_table;
        using atomic_node_ptr = std::atomic<node_ptr>;
        using base_type = segment_table<std::atomic<node_ptr>, allocator_type, unordered_segment_table, pointers_per_embedded_table>;
        using segment_type = typename base_type::segment_type;
        using base_allocator_type = typename base_type::allocator_type;

        using segment_allocator_type = typename allocator_traits_type::template rebind_alloc<atomic_node_ptr>;
        using segment_allocator_traits = tbb::detail::allocator_traits<segment_allocator_type>;
    public:
        // Segment table for unordered containers should not be extended in the wait- free implementation
        static constexpr bool allow_table_extending = false;
        static constexpr bool is_noexcept_assignment = std::is_nothrow_move_assignable<hasher>::value &&
                                                       std::is_nothrow_move_assignable<key_equal>::value &&
                                                       segment_allocator_traits::is_always_equal::value;
        static constexpr bool is_noexcept_swap = tbb::detail::is_nothrow_swappable<hasher>::value &&
                                                 tbb::detail::is_nothrow_swappable<key_equal>::value &&
                                                 segment_allocator_traits::is_always_equal::value;

        // TODO: using base_type::base_type is not compiling on Windows and Intel Compiler - investigate
        unordered_segment_table( const base_allocator_type& alloc = base_allocator_type() )
            : base_type(alloc) {}

        unordered_segment_table( const unordered_segment_table& ) = default;

        unordered_segment_table( const unordered_segment_table& other, const base_allocator_type& alloc )
            : base_type(other, alloc) {}

        unordered_segment_table( unordered_segment_table&& ) = default;

        unordered_segment_table( unordered_segment_table&& other, const base_allocator_type& alloc )
            : base_type(std::move(other), alloc) {}

        unordered_segment_table& operator=( const unordered_segment_table& ) = default;

        unordered_segment_table& operator=( unordered_segment_table&& ) = default;

        segment_type create_segment( typename base_type::segment_table_type, typename base_type::segment_index_type segment_index, size_type ) {
            segment_allocator_type alloc(this->get_allocator());
            size_type seg_size = this->segment_size(segment_index);
            segment_type new_segment = segment_allocator_traits::allocate(alloc, seg_size);
            for (size_type i = 0; i != seg_size; ++i) {
                segment_allocator_traits::construct(alloc, new_segment + i, nullptr);
            }
            return new_segment;
        }

        // deallocate_segment is required by the segment_table base class, but
        // in unordered, it is also necessary to call the destructor during deallocation
        void deallocate_segment( segment_type address, size_type index ) {
            destroy_segment(address, index);
        }

        void destroy_segment( segment_type address, size_type index ) {
            segment_allocator_type alloc(this->get_allocator());
            for (size_type i = 0; i != this->segment_size(index); ++i) {
                segment_allocator_traits::destroy(alloc, address + i);
            }
            segment_allocator_traits::deallocate(alloc, address, this->segment_size(index));
        }


        void copy_segment( size_type index, segment_type, segment_type to ) {
            if (index == 0) {
                // The first element in the first segment is embedded into the table (my_head)
                // so the first pointer should not be stored here
                // It would be stored during move ctor/assignment operation
                to[1].store(nullptr, std::memory_order_relaxed);
            } else {
                for (size_type i = 0; i != this->segment_size(index); ++i) {
                    to[i].store(nullptr, std::memory_order_relaxed);
                }
            }
        }

        void move_segment( size_type index, segment_type from, segment_type to ) {
            if (index == 0) {
                // The first element in the first segment is embedded into the table (my_head)
                // so the first pointer should not be stored here
                // It would be stored during move ctor/assignment operation
                to[1].store(from[1].load(std::memory_order_relaxed), std::memory_order_relaxed);
            } else {
                for (size_type i = 0; i != this->segment_size(index); ++i) {
                    to[i].store(from[i].load(std::memory_order_relaxed), std::memory_order_relaxed);
                    from[i].store(nullptr, std::memory_order_relaxed);
                }
            }
        }

        // allocate_long_table is required by the segment_table base class, but unused for unordered containers
        typename base_type::segment_table_type allocate_long_table( const typename base_type::atomic_segment*, size_type ) {
            __TBB_ASSERT(false, "This method should never been called");
            // TableType is a pointer
            return nullptr;
        }

        // destroy_elements is required by the segment_table base class, but unused for unordered containers
        // this function call but do nothing
        void destroy_elements() {}
    }; // struct unordered_segment_table

    void internal_clear() {
        // TODO: consider usefulness of two versions of clear() - with dummy nodes deallocation and without it
        node_ptr next = my_head.next();
        node_ptr curr = next;

        my_head.set_next(nullptr);

        while (curr != nullptr) {
            next = curr->next();
            destroy_node(curr);
            curr = next;
        }

        my_size.store(0, std::memory_order_relaxed);
        my_segments.clear();
    }

    void destroy_node( node_ptr node ) {
        if (node->is_dummy()) {
            node_allocator_type dummy_node_allocator(my_segments.get_allocator());
            // Destroy the node
            node_allocator_traits::destroy(dummy_node_allocator, node);
            // Deallocate the memory
            node_allocator_traits::deallocate(dummy_node_allocator, node, 1);
        } else {
            value_node_ptr val_node = static_cast<value_node_ptr>(node);
            value_node_allocator_type value_node_allocator(my_segments.get_allocator());
            // Destroy the value
            value_node_allocator_traits::destroy(value_node_allocator, val_node->storage());
            // Destroy the node
            value_node_allocator_traits::destroy(value_node_allocator, val_node);
            // Deallocate the memory
            value_node_allocator_traits::deallocate(value_node_allocator, val_node, 1);
        }
    }

    struct internal_insert_return_type {
        // If the insertion failed - the remaining_node points to the node, which was failed to insert
        // This node can be allocated in process of insertion
        value_node_ptr remaining_node;
        // If the insertion failed - node_with_equal_key points to the node in the list with the
        // key, equivalent to the inserted, otherwise it points to the node, which was inserted.
        value_node_ptr node_with_equal_key;
        // Insertion status
        // NOTE: if it is true - remaining_node should be nullptr
        bool inserted;
    }; // struct internal_insert_return_type

    // Inserts the value into the split ordered list
    template <typename ValueType>
    std::pair<iterator, bool> internal_insert_value( ValueType&& value ) {

        auto create_value_node = [&value, this]( sokey_type order_key )->value_node_ptr {
            return create_node(order_key, std::forward<ValueType>(value));
        };

        auto insert_result = internal_insert(value, create_value_node);

        if (insert_result.remaining_node != nullptr) {
            // If the insertion fails - destroy the node which was failed to insert if it exist
            __TBB_ASSERT(!insert_result.inserted,
                         "remaining_node should be nullptr if the node was successfully inserted");
            destroy_node(insert_result.remaining_node);
        }

        return { iterator(insert_result.node_with_equal_key), insert_result.inserted };
    }

    // Inserts the node into the split ordered list
    // Creates a node using the specified callback after the place for insertion was found
    // Returns internal_insert_return_type object, where:
    //     - If the insertion succeeded:
    //         - remaining_node is nullptr
    //         - node_with_equal_key point to the inserted node
    //         - inserted is true
    //     - If the insertion failed:
    //         - remaining_node points to the node, that was failed to insert if it was created.
    //           nullptr if the node was not created, because the requested key was already
    //           presented in the list
    //         - node_with_equal_key point to the element in the list with the key, equivalent to
    //           to the requested key
    //         - inserted is false
    template <typename ValueType, typename CreateInsertNode>
    internal_insert_return_type internal_insert( ValueType&& value, CreateInsertNode create_insert_node ) {
        static_assert(std::is_same<typename std::decay<ValueType>::type, value_type>::value,
                      "Incorrect type in internal_insert");
        const key_type& key = traits_type::get_key(value);
        sokey_type hash_key = sokey_type(my_hash_compare(key));

        sokey_type order_key = split_order_key_regular(hash_key);
        node_ptr prev = prepare_bucket(hash_key);
        __TBB_ASSERT(prev != nullptr, "Invalid head node");

        auto search_result = search_after(prev, order_key, key);

        if (search_result.second) {
            return internal_insert_return_type{ nullptr, search_result.first, false };
        }

        value_node_ptr new_node = create_insert_node(order_key);
        node_ptr curr = search_result.first;

        while (!try_insert(prev, new_node, curr)) {
            search_result = search_after(prev, order_key, key);
            if (search_result.second) {
                return internal_insert_return_type{ new_node, search_result.first, false };
            }
            curr = search_result.first;
        }

        auto sz = my_size.fetch_add(1);
        adjust_table_size(sz + 1, my_bucket_count.load(std::memory_order_acquire));
        return internal_insert_return_type{ nullptr, static_cast<value_node_ptr>(new_node), true };
    }

    // Searches the node with the key, equivalent to key with requested order key after the node prev
    // Returns the existing node and true if the node is already in the list
    // Returns the first node with the order key, greater than requested and false if the node is not presented in the list
    std::pair<value_node_ptr, bool> search_after( node_ptr& prev, sokey_type order_key, const key_type& key ) {
        // NOTE: static_cast<value_node_ptr>(curr) should be done only after we would ensure
        // that the node is not a dummy node

        node_ptr curr = prev->next();

        while (curr != nullptr && (curr->order_key() < order_key ||
               (curr->order_key() == order_key && !my_hash_compare(traits_type::get_key(static_cast<value_node_ptr>(curr)->value()), key))))
        {
            prev = curr;
            curr = curr->next();
        }

        if (curr != nullptr && curr->order_key() == order_key && !allow_multimapping) {
            return { static_cast<value_node_ptr>(curr), true };
        }
        return { static_cast<value_node_ptr>(curr), false };
    }

    void adjust_table_size( size_type total_elements, size_type current_size ) {
        // Grow the table by a factor of 2 if possible and needed
        if ( (float(total_elements) / float(current_size)) > my_max_load_factor ) {
            // Double the size of the hash only if size hash not changed in between loads
            my_bucket_count.compare_exchange_strong(current_size, 2u * current_size);
        }
    }

    node_ptr insert_dummy_node( node_ptr parent_dummy_node, sokey_type order_key ) {
        node_ptr prev_node = parent_dummy_node;

        node_ptr dummy_node = create_dummy_node(order_key);
        node_ptr next_node;

        do {
            next_node = prev_node->next();
            // Move forward through the list while the order key is less than requested
            while (next_node != nullptr && next_node->order_key() < order_key) {
                prev_node = next_node;
                next_node = next_node->next();
            }

            if (next_node != nullptr && next_node->order_key() == order_key) {
                // Another dummy node with the same order key was inserted by another thread
                // Destroy the node and exit
                destroy_node(dummy_node);
                return next_node;
            }
        } while (!try_insert(prev_node, dummy_node, next_node));

        return dummy_node;
    }

    // Try to insert a node between prev_node and expected next
    // If the next is not equal to expected next - return false
    static bool try_insert( node_ptr prev_node, node_ptr new_node, node_ptr current_next_node ) {
        new_node->set_next(current_next_node);
        return prev_node->try_set_next(current_next_node, new_node);
    }

    // Returns the bucket, associated with the hash_key
    node_ptr prepare_bucket( sokey_type hash_key ) {
        size_type bucket = hash_key % my_bucket_count.load(std::memory_order_acquire);
        return get_bucket(bucket);
    }

    // Initialize the corresponding bucket if it is not initialized
    node_ptr get_bucket( size_type bucket_index ) {
        if (my_segments[bucket_index].load(std::memory_order_acquire) == nullptr) {
            init_bucket(bucket_index);
        }
        return my_segments[bucket_index].load(std::memory_order_acquire);
    }

    void init_bucket( size_type bucket ) {
        if (bucket == 0) {
            // Atomicaly store the first bucket into my_head
            node_ptr disabled = nullptr;
            my_segments[0].compare_exchange_strong(disabled, &my_head);
            return;
        }

        size_type parent_bucket = get_parent(bucket);

        while (my_segments[parent_bucket].load(std::memory_order_acquire) == nullptr) {
            // Initialize all of the parent buckets
            init_bucket(parent_bucket);
        }

        __TBB_ASSERT(my_segments[parent_bucket].load(std::memory_order_acquire) != nullptr, "Parent bucket should be initialized");
        node_ptr parent = my_segments[parent_bucket].load(std::memory_order_acquire);

        // Insert dummy node into the list
        node_ptr dummy_node = insert_dummy_node(parent, split_order_key_dummy(bucket));
        // TODO: consider returning pair<node_ptr, bool> to avoid store operation if the bucket was stored by an other thread
        // or move store to insert_dummy_node
        // Add dummy_node into the segment table
        my_segments[bucket].store(dummy_node, std::memory_order_release);
    }

    node_ptr create_dummy_node( sokey_type order_key ) {
        node_allocator_type dummy_node_allocator(my_segments.get_allocator());
        node_ptr dummy_node = node_allocator_traits::allocate(dummy_node_allocator, 1);
        node_allocator_traits::construct(dummy_node_allocator, dummy_node, order_key);
        return dummy_node;
    }

    template <typename... Args>
    value_node_ptr create_node( sokey_type order_key, Args&&... args ) {
        value_node_allocator_type value_node_allocator(my_segments.get_allocator());
        // Allocate memory for the value_node
        value_node_ptr new_node = value_node_allocator_traits::allocate(value_node_allocator, 1);
        // Construct the node
        value_node_allocator_traits::construct(value_node_allocator, new_node, order_key);

        // try_call API is not convenient here due to broken
        // variadic capture on GCC 4.8.5
        auto value_guard = make_raii_guard([&] {
            value_node_allocator_traits::destroy(value_node_allocator, new_node);
            value_node_allocator_traits::deallocate(value_node_allocator, new_node, 1);
        });

        // Construct the value in the node
        value_node_allocator_traits::construct(value_node_allocator, new_node->storage(), std::forward<Args>(args)...);
        value_guard.dismiss();
        return new_node;
    }

    value_node_ptr first_value_node( node_ptr first_node ) const {
        while (first_node != nullptr && first_node->is_dummy()) {
            first_node = first_node->next();
        }
        return static_cast<value_node_ptr>(first_node);
    }

    // Unsafe method, which removes the node from the list and returns the next node
    node_ptr internal_erase( value_node_ptr node_to_erase ) {
        __TBB_ASSERT(node_to_erase != nullptr, "Invalid iterator for erase");
        node_ptr next_node = node_to_erase->next();
        internal_extract(node_to_erase);
        destroy_node(node_to_erase);
        return next_node;
    }

    template <typename K>
    size_type internal_erase_by_key( const K& key ) {
        // TODO: consider reimplementation without equal_range - it is not effective to perform lookup over a bucket
        // for each unsafe_erase call
        auto eq_range = equal_range(key);
        size_type erased_count = 0;

        for (auto it = eq_range.first; it != eq_range.second;) {
            it = unsafe_erase(it);
            ++erased_count;
        }
        return erased_count;
    }

    // Unsafe method, which extracts the node from the list
    void internal_extract( value_node_ptr node_to_extract ) {
        const key_type& key = traits_type::get_key(node_to_extract->value());
        sokey_type hash_key = sokey_type(my_hash_compare(key));

        node_ptr prev_node = prepare_bucket(hash_key);

        for (node_ptr node = prev_node->next(); node != nullptr; prev_node = node, node = node->next()) {
            if (node == node_to_extract) {
                unlink_node(prev_node, node, node_to_extract->next());
                my_size.store(my_size.load(std::memory_order_relaxed) - 1, std::memory_order_relaxed);
                return;
            }
            __TBB_ASSERT(node->order_key() <= node_to_extract->order_key(),
                         "node, which is going to be extracted should be presented in the list");
        }
    }

protected:
    template <typename SourceType>
    void internal_merge( SourceType&& source ) {
        static_assert(std::is_same<node_type, typename std::decay<SourceType>::type::node_type>::value,
                      "Incompatible containers cannot be merged");

        for (node_ptr source_prev = &source.my_head; source_prev->next() != nullptr;) {
            if (!source_prev->next()->is_dummy()) {
                value_node_ptr curr = static_cast<value_node_ptr>(source_prev->next());
                // If the multimapping is allowed, or the key is not presented
                // in the *this container - extract the node from the list
                if (allow_multimapping || !contains(traits_type::get_key(curr->value()))) {
                    node_ptr next_node = curr->next();
                    source.unlink_node(source_prev, curr, next_node);

                    // Remember the old order key
                    sokey_type old_order_key = curr->order_key();

                    // Node handle with curr cannot be used directly in insert call, because
                    // the destructor of node_type will destroy curr
                    node_type curr_node = node_handle_accessor::construct<node_type>(curr);

                    // If the insertion fails - return ownership of the node to the source
                    if (!insert(std::move(curr_node)).second) {
                        __TBB_ASSERT(!allow_multimapping, "Insertion should succeed for multicontainer");
                        __TBB_ASSERT(source_prev->next() == next_node,
                                     "Concurrent operations with the source container in merge are prohibited");

                        // Initialize the node with the old order key, because the order key
                        // can change during the insertion
                        curr->init(old_order_key);
                        __TBB_ASSERT(old_order_key >= source_prev->order_key() &&
                                     (next_node == nullptr || old_order_key <= next_node->order_key()),
                                     "Wrong nodes order in the source container");
                        // Merge is unsafe for source container, so the insertion back can be done without compare_exchange
                        curr->set_next(next_node);
                        source_prev->set_next(curr);
                        source_prev = curr;
                        node_handle_accessor::deactivate(curr_node);
                    } else {
                        source.my_size.fetch_sub(1, std::memory_order_relaxed);
                    }
                } else {
                    source_prev = curr;
                }
            } else {
                source_prev = source_prev->next();
            }
        }
    }

private:
    // Unsafe method, which unlinks the node between prev and next
    void unlink_node( node_ptr prev_node, node_ptr node_to_unlink, node_ptr next_node ) {
        __TBB_ASSERT(prev_node->next() == node_to_unlink &&
                     node_to_unlink->next() == next_node,
                     "erasing and extracting nodes from the containers are unsafe in concurrent mode");
        prev_node->set_next(next_node);
        node_to_unlink->set_next(nullptr);
    }

    template <typename K>
    value_node_ptr internal_find( const K& key ) {
        sokey_type hash_key = sokey_type(my_hash_compare(key));
        sokey_type order_key = split_order_key_regular(hash_key);

        node_ptr curr = prepare_bucket(hash_key);

        while (curr != nullptr) {
            if (curr->order_key() > order_key) {
                // If the order key is greater than the requested order key,
                // the element is not in the hash table
                return nullptr;
            } else if (curr->order_key() == order_key &&
                       my_hash_compare(traits_type::get_key(static_cast<value_node_ptr>(curr)->value()), key)) {
                // The fact that order keys match does not mean that the element is found.
                // Key function comparison has to be performed to check whether this is the
                // right element. If not, keep searching while order key is the same.
                return static_cast<value_node_ptr>(curr);
            }
            curr = curr->next();
        }

        return nullptr;
    }

    template <typename K>
    std::pair<value_node_ptr, value_node_ptr> internal_equal_range( const K& key ) {
        sokey_type hash_key = sokey_type(my_hash_compare(key));
        sokey_type order_key = split_order_key_regular(hash_key);

        node_ptr curr = prepare_bucket(hash_key);

        while (curr != nullptr) {
            if (curr->order_key() > order_key) {
                // If the order key is greater than the requested order key,
                // the element is not in the hash table
                return std::make_pair(nullptr, nullptr);
            } else if (curr->order_key() == order_key &&
                       my_hash_compare(traits_type::get_key(static_cast<value_node_ptr>(curr)->value()), key)) {
                value_node_ptr first = static_cast<value_node_ptr>(curr);
                node_ptr last = first;
                do {
                    last = last->next();
                } while (allow_multimapping && last != nullptr && !last->is_dummy() &&
                        my_hash_compare(traits_type::get_key(static_cast<value_node_ptr>(last)->value()), key));
                return std::make_pair(first, first_value_node(last));
            }
            curr = curr->next();
        }
        return {nullptr, nullptr};
    }

    template <typename K>
    size_type internal_count( const K& key ) const {
        if (allow_multimapping) {
            // TODO: consider reimplementing the internal_equal_range with elements counting to avoid std::distance
            auto eq_range = equal_range(key);
            return std::distance(eq_range.first, eq_range.second);
        } else {
            return contains(key) ? 1 : 0;
        }
    }

    void internal_copy( const concurrent_unordered_base& other ) {
        node_ptr last_node = &my_head;
        my_segments[0].store(&my_head, std::memory_order_relaxed);

        for (node_ptr node = other.my_head.next(); node != nullptr; node = node->next()) {
            node_ptr new_node;
            if (!node->is_dummy()) {
                // The node in the right table contains a value
                new_node = create_node(node->order_key(), static_cast<value_node_ptr>(node)->value());
            } else {
                // The node in the right table is a dummy node
                new_node = create_dummy_node(node->order_key());
                my_segments[reverse_bits(node->order_key())].store(new_node, std::memory_order_relaxed);
            }

            last_node->set_next(new_node);
            last_node = new_node;
        }
    }

    void internal_move( concurrent_unordered_base&& other ) {
        node_ptr last_node = &my_head;
        my_segments[0].store(&my_head, std::memory_order_relaxed);

        for (node_ptr node = other.my_head.next(); node != nullptr; node = node->next()) {
            node_ptr new_node;
            if (!node->is_dummy()) {
                // The node in the right table contains a value
                new_node = create_node(node->order_key(), std::move(static_cast<value_node_ptr>(node)->value()));
            } else {
                // TODO: do we need to destroy a dummy node in the right container?
                // The node in the right table is a dummy_node
                new_node = create_dummy_node(node->order_key());
                my_segments[reverse_bits(node->order_key())].store(new_node, std::memory_order_relaxed);
            }

            last_node->set_next(new_node);
            last_node = new_node;
        }
    }

    void move_content( concurrent_unordered_base&& other ) {
        // NOTE: allocators should be equal
        my_head.set_next(other.my_head.next());
        other.my_head.set_next(nullptr);
        my_segments[0].store(&my_head, std::memory_order_relaxed);

        other.my_bucket_count.store(initial_bucket_count, std::memory_order_relaxed);
        other.my_max_load_factor = initial_max_load_factor;
        other.my_size.store(0, std::memory_order_relaxed);
    }

    void internal_move_construct_with_allocator( concurrent_unordered_base&& other, const allocator_type&,
                                                 /*is_always_equal = */std::true_type ) {
        // Allocators are always equal - no need to compare for equality
        move_content(std::move(other));
    }

    void internal_move_construct_with_allocator( concurrent_unordered_base&& other, const allocator_type& alloc,
                                                 /*is_always_equal = */std::false_type ) {
        // Allocators are not always equal
        if (alloc == other.my_segments.get_allocator()) {
            move_content(std::move(other));
        } else {
            try_call( [&] {
                internal_move(std::move(other));
            } ).on_exception( [&] {
                clear();
            });
        }
    }

    // Move assigns the hash table to other is any instances of allocator_type are always equal
    // or propagate_on_container_move_assignment is true
    void internal_move_assign( concurrent_unordered_base&& other, /*is_always_equal || POCMA = */std::true_type ) {
        move_content(std::move(other));
    }

    // Move assigns the hash table to other is any instances of allocator_type are not always equal
    // and propagate_on_container_move_assignment is false
    void internal_move_assign( concurrent_unordered_base&& other, /*is_always_equal || POCMA = */std::false_type ) {
        if (my_segments.get_allocator() == other.my_segments.get_allocator()) {
            move_content(std::move(other));
        } else {
            // TODO: guards for exceptions
            internal_move(std::move(other));
        }
    }

    void internal_swap( concurrent_unordered_base& other, /*is_always_equal || POCS = */std::true_type ) {
        internal_swap_fields(other);
    }

    void internal_swap( concurrent_unordered_base& other, /*is_always_equal || POCS = */std::false_type ) {
        __TBB_ASSERT(my_segments.get_allocator() == other.my_segments.get_allocator(),
                     "Swapping with unequal allocators is not allowed");
        internal_swap_fields(other);
    }

    void internal_swap_fields( concurrent_unordered_base& other ) {
        node_ptr first_node = my_head.next();
        my_head.set_next(other.my_head.next());
        other.my_head.set_next(first_node);

        size_type current_size = my_size.load(std::memory_order_relaxed);
        my_size.store(other.my_size.load(std::memory_order_relaxed), std::memory_order_relaxed);
        other.my_size.store(current_size, std::memory_order_relaxed);

        size_type bucket_count = my_bucket_count.load(std::memory_order_relaxed);
        my_bucket_count.store(other.my_bucket_count.load(std::memory_order_relaxed), std::memory_order_relaxed);
        other.my_bucket_count.store(bucket_count, std::memory_order_relaxed);

        using std::swap;
        swap(my_max_load_factor, other.my_max_load_factor);
        swap(my_hash_compare, other.my_hash_compare);
        my_segments.swap(other.my_segments);

        // swap() method from segment table swaps all of the segments including the first segment
        // We should restore it to my_head. Without it the first segment of the container will point
        // to other.my_head.
        my_segments[0].store(&my_head, std::memory_order_relaxed);
        other.my_segments[0].store(&other.my_head, std::memory_order_relaxed);
    }

    // A regular order key has its original hash value reversed and the last bit set
    static constexpr sokey_type split_order_key_regular( sokey_type hash ) {
        return reverse_bits(hash) | 0x1;
    }

    // A dummy order key has its original hash value reversed and the last bit unset
    static constexpr sokey_type split_order_key_dummy( sokey_type hash ) {
        return reverse_bits(hash) & ~sokey_type(0x1);
    }

    size_type get_parent( size_type bucket ) const {
        // Unset bucket's most significant turned-on bit
        __TBB_ASSERT(bucket != 0, "Unable to get_parent of the bucket 0");
        size_type msb = tbb::detail::log2(bucket);
        return bucket & ~(size_type(1) << msb);
    }

    size_type get_next_bucket_index( size_type bucket ) const {
        size_type bits = tbb::detail::log2(my_bucket_count.load(std::memory_order_relaxed));
        size_type reversed_next = reverse_n_bits(bucket, bits) + 1;
        return reverse_n_bits(reversed_next, bits);
    }

    std::atomic<size_type> my_size;
    std::atomic<size_type> my_bucket_count;
    float my_max_load_factor;
    hash_compare_type my_hash_compare;

    list_node_type my_head; // Head node for split ordered list
    unordered_segment_table my_segments; // Segment table of pointers to nodes

    template <typename Container, typename Value>
    friend class solist_iterator;

    template <typename OtherTraits>
    friend class concurrent_unordered_base;
}; // class concurrent_unordered_base

template <typename Traits>
bool operator==( const concurrent_unordered_base<Traits>& lhs,
                 const concurrent_unordered_base<Traits>& rhs ) {
    if (&lhs == &rhs) { return true; }
    if (lhs.size() != rhs.size()) { return false; }

#if _MSC_VER
    // Passing "unchecked" iterators to std::permutation with 3 parameters
    // causes compiler warnings.
    // The workaround is to use overload with 4 parameters, which is
    // available since C++14 - minimally supported version on MSVC
    return std::is_permutation(lhs.begin(), lhs.end(), rhs.begin(), rhs.end());
#else
    return std::is_permutation(lhs.begin(), lhs.end(), rhs.begin());
#endif
}

#if !__TBB_CPP20_COMPARISONS_PRESENT
template <typename Traits>
bool operator!=( const concurrent_unordered_base<Traits>& lhs,
                 const concurrent_unordered_base<Traits>& rhs ) {
    return !(lhs == rhs);
}
#endif

#if defined(_MSC_VER) && !defined(__INTEL_COMPILER)
#pragma warning(pop) // warning 4127 is back
#endif

} // namespace d1
} // namespace detail
} // namespace tbb

#endif // __TBB_detail__concurrent_unordered_base_H
