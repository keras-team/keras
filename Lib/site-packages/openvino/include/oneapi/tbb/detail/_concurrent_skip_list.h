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

#ifndef __TBB_detail__concurrent_skip_list_H
#define __TBB_detail__concurrent_skip_list_H

#if !defined(__TBB_concurrent_map_H) && !defined(__TBB_concurrent_set_H)
#error Do not #include this internal file directly; use public TBB headers instead.
#endif

#include "_config.h"
#include "_range_common.h"
#include "_allocator_traits.h"
#include "_template_helpers.h"
#include "_node_handle.h"
#include "_containers_helpers.h"
#include "_assert.h"
#include "_exception.h"
#include "../enumerable_thread_specific.h"
#include <utility>
#include <initializer_list>
#include <atomic>
#include <array>
#include <type_traits>
#include <random> // Need std::geometric_distribution
#include <algorithm> // Need std::equal and std::lexicographical_compare
#include <cstdint>
#if __TBB_CPP20_COMPARISONS_PRESENT
#include <compare>
#endif

#if defined(_MSC_VER) && !defined(__INTEL_COMPILER)
#pragma warning(push)
#pragma warning(disable: 4127) // warning C4127: conditional expression is constant
#endif

namespace tbb {
namespace detail {
namespace d1 {

template <typename Value, typename Allocator>
class skip_list_node {
    using node_ptr = skip_list_node*;
public:
    using value_type = Value;
    using atomic_node_ptr = std::atomic<node_ptr>;
    using size_type = std::size_t;
    using container_allocator_type = Allocator;

    using reference = value_type&;
    using const_reference = const value_type&;
private:
    using allocator_traits = tbb::detail::allocator_traits<container_allocator_type>;

    // Allocator is the same as the container allocator=> allocates unitptr_t
    // It is required to rebind it to value_type to get the correct pointer and const_pointer
    using value_allocator_traits = typename allocator_traits::template rebind_traits<value_type>;
public:
    using pointer = typename value_allocator_traits::pointer;
    using const_pointer = typename value_allocator_traits::const_pointer;

    skip_list_node( size_type levels, container_allocator_type& alloc )
        : my_container_allocator(alloc), my_height(levels), my_index_number(0)
    {
        for (size_type l = 0; l < my_height; ++l) {
            allocator_traits::construct(my_container_allocator, &get_atomic_next(l), nullptr);
        }
    }

    ~skip_list_node() {
        for (size_type l = 0; l < my_height; ++l) {
            allocator_traits::destroy(my_container_allocator, &get_atomic_next(l));
        }
    }

    skip_list_node( const skip_list_node& ) = delete;
    skip_list_node( skip_list_node&& ) = delete;
    skip_list_node& operator=( const skip_list_node& ) = delete;
    skip_list_node& operator=( skip_list_node&& ) = delete;

    pointer storage() {
        return &my_value;
    }

    reference value() {
        return *storage();
    }

    node_ptr next( size_type level ) const {
        node_ptr res = get_atomic_next(level).load(std::memory_order_acquire);
        __TBB_ASSERT(res == nullptr || res->height() > level, "Broken internal structure");
        return res;
    }

    atomic_node_ptr& atomic_next( size_type level ) {
        atomic_node_ptr& res = get_atomic_next(level);
#if TBB_USE_DEBUG
        node_ptr node = res.load(std::memory_order_acquire);
        __TBB_ASSERT(node == nullptr || node->height() > level, "Broken internal structure");
#endif
        return res;
    }

    void set_next( size_type level, node_ptr n ) {
        __TBB_ASSERT(n == nullptr || n->height() > level, "Broken internal structure");
        get_atomic_next(level).store(n, std::memory_order_relaxed);
    }

    size_type height() const {
        return my_height;
    }

    void set_index_number( size_type index_num ) {
        my_index_number = index_num;
    }

    size_type index_number() const {
        return my_index_number;
    }

private:
    atomic_node_ptr& get_atomic_next( size_type level ) {
        atomic_node_ptr* arr = reinterpret_cast<atomic_node_ptr*>(this + 1);
        return arr[level];
    }

    const atomic_node_ptr& get_atomic_next( size_type level ) const {
        const atomic_node_ptr* arr = reinterpret_cast<const atomic_node_ptr*>(this + 1);
        return arr[level];
    }

    container_allocator_type& my_container_allocator;
    union {
        value_type my_value;
    };
    size_type my_height;
    size_type my_index_number;
}; // class skip_list_node

template <typename NodeType, typename ValueType>
class skip_list_iterator {
    using node_type = NodeType;
    using node_ptr = node_type*;
public:
    using iterator_category = std::forward_iterator_tag;
    using value_type = ValueType;

    using difference_type = std::ptrdiff_t;
    using pointer = value_type*;
    using reference = value_type&;

    skip_list_iterator() : skip_list_iterator(nullptr) {}

    skip_list_iterator( const skip_list_iterator<node_type, typename node_type::value_type>& other )
        : my_node_ptr(other.my_node_ptr) {}

    skip_list_iterator& operator=( const skip_list_iterator<node_type, typename node_type::value_type>& other ) {
        my_node_ptr = other.my_node_ptr;
        return *this;
    }

    reference operator*() const { return my_node_ptr->value(); }
    pointer operator->() const { return my_node_ptr->storage(); }

    skip_list_iterator& operator++() {
        __TBB_ASSERT(my_node_ptr != nullptr, nullptr);
        my_node_ptr = my_node_ptr->next(0);
        return *this;
    }

    skip_list_iterator operator++(int) {
        skip_list_iterator tmp = *this;
        ++*this;
        return tmp;
    }

private:
    skip_list_iterator(node_type* n) : my_node_ptr(n) {}

    node_ptr my_node_ptr;

    template <typename Traits>
    friend class concurrent_skip_list;

    template <typename N, typename V>
    friend class skip_list_iterator;

    friend class const_range;
    friend class range;

    friend bool operator==( const skip_list_iterator& lhs, const skip_list_iterator& rhs ) {
        return lhs.my_node_ptr == rhs.my_node_ptr;
    }

    friend bool operator!=( const skip_list_iterator& lhs, const skip_list_iterator& rhs ) {
        return lhs.my_node_ptr != rhs.my_node_ptr;
    }
}; // class skip_list_iterator

template <typename Traits>
class concurrent_skip_list {
protected:
    using container_traits = Traits;
    using self_type = concurrent_skip_list<container_traits>;
    using allocator_type = typename container_traits::allocator_type;
    using allocator_traits_type = tbb::detail::allocator_traits<allocator_type>;
    using key_compare = typename container_traits::compare_type;
    using value_compare = typename container_traits::value_compare;
    using key_type = typename container_traits::key_type;
    using value_type = typename container_traits::value_type;
    static_assert(std::is_same<value_type, typename allocator_type::value_type>::value,
                  "value_type of the container should be the same as its allocator");

    using size_type = std::size_t;
    using difference_type = std::ptrdiff_t;

    static constexpr size_type max_level = container_traits::max_level;

    using node_allocator_type = typename allocator_traits_type::template rebind_alloc<std::uint8_t>;
    using node_allocator_traits = tbb::detail::allocator_traits<node_allocator_type>;

    using list_node_type = skip_list_node<value_type, node_allocator_type>;
    using node_type = node_handle<key_type, value_type, list_node_type, allocator_type>;

    using iterator = skip_list_iterator<list_node_type, value_type>;
    using const_iterator = skip_list_iterator<list_node_type, const value_type>;

    using reference = value_type&;
    using const_reference = const value_type&;
    using pointer = typename allocator_traits_type::pointer;
    using const_pointer = typename allocator_traits_type::const_pointer;

    using random_level_generator_type = typename container_traits::random_level_generator_type;

    using node_ptr = list_node_type*;

    using array_type = std::array<node_ptr, max_level>;
private:
    template <typename T>
    using is_transparent = dependent_bool<comp_is_transparent<key_compare>, T>;
public:
    static constexpr bool allow_multimapping = container_traits::allow_multimapping;

    concurrent_skip_list() : my_head_ptr(nullptr), my_size(0), my_max_height(0) {}

    explicit concurrent_skip_list( const key_compare& comp, const allocator_type& alloc = allocator_type() )
        : my_node_allocator(alloc), my_compare(comp), my_head_ptr(nullptr), my_size(0), my_max_height(0) {}

    explicit concurrent_skip_list( const allocator_type& alloc )
        : concurrent_skip_list(key_compare(), alloc) {}

    template<typename InputIterator>
    concurrent_skip_list( InputIterator first, InputIterator last, const key_compare& comp = key_compare(),
                          const allocator_type& alloc = allocator_type() )
        : concurrent_skip_list(comp, alloc)
    {
        internal_copy(first, last);
    }

    template <typename InputIterator>
    concurrent_skip_list( InputIterator first, InputIterator last, const allocator_type& alloc )
        : concurrent_skip_list(first, last, key_compare(), alloc) {}

    concurrent_skip_list( std::initializer_list<value_type> init, const key_compare& comp = key_compare(),
                          const allocator_type& alloc = allocator_type() )
        : concurrent_skip_list(init.begin(), init.end(), comp, alloc) {}

    concurrent_skip_list( std::initializer_list<value_type> init, const allocator_type& alloc )
        : concurrent_skip_list(init, key_compare(), alloc) {}

    concurrent_skip_list( const concurrent_skip_list& other )
        : my_node_allocator(node_allocator_traits::select_on_container_copy_construction(other.get_allocator())),
          my_compare(other.my_compare), my_rng(other.my_rng), my_head_ptr(nullptr),
          my_size(0), my_max_height(0)
    {
        internal_copy(other);
        __TBB_ASSERT(my_size == other.my_size, "Wrong size of copy-constructed container");
    }

    concurrent_skip_list( const concurrent_skip_list& other, const allocator_type& alloc )
        : my_node_allocator(alloc), my_compare(other.my_compare), my_rng(other.my_rng), my_head_ptr(nullptr),
          my_size(0), my_max_height(0)
    {
        internal_copy(other);
        __TBB_ASSERT(my_size == other.my_size, "Wrong size of copy-constructed container");
    }

    concurrent_skip_list( concurrent_skip_list&& other )
        : my_node_allocator(std::move(other.my_node_allocator)), my_compare(other.my_compare),
          my_rng(std::move(other.my_rng)), my_head_ptr(nullptr) // my_head_ptr would be stored in internal_move
    {
        internal_move(std::move(other));
    }

    concurrent_skip_list( concurrent_skip_list&& other, const allocator_type& alloc )
        : my_node_allocator(alloc), my_compare(other.my_compare),
          my_rng(std::move(other.my_rng)), my_head_ptr(nullptr)
    {
        using is_always_equal = typename allocator_traits_type::is_always_equal;
        internal_move_construct_with_allocator(std::move(other), is_always_equal());
    }

    ~concurrent_skip_list() {
        clear();
        node_ptr head = my_head_ptr.load(std::memory_order_relaxed);
        if (head != nullptr) {
            delete_node(head);
        }
    }

    concurrent_skip_list& operator=( const concurrent_skip_list& other ) {
        if (this != &other) {
            clear();
            copy_assign_allocators(my_node_allocator, other.my_node_allocator);
            my_compare = other.my_compare;
            my_rng = other.my_rng;
            internal_copy(other);
        }
        return *this;
    }

    concurrent_skip_list& operator=( concurrent_skip_list&& other ) {
        if (this != &other) {
            clear();
            my_compare = std::move(other.my_compare);
            my_rng = std::move(other.my_rng);

            move_assign_allocators(my_node_allocator, other.my_node_allocator);
            using pocma_type = typename node_allocator_traits::propagate_on_container_move_assignment;
            using is_always_equal = typename node_allocator_traits::is_always_equal;
            internal_move_assign(std::move(other), tbb::detail::disjunction<pocma_type, is_always_equal>());
        }
        return *this;
    }

    concurrent_skip_list& operator=( std::initializer_list<value_type> il )
    {
        clear();
        insert(il.begin(),il.end());
        return *this;
    }

    std::pair<iterator, bool> insert( const value_type& value ) {
        return internal_insert(value);
    }

    std::pair<iterator, bool> insert( value_type&& value ) {
        return internal_insert(std::move(value));
    }

    iterator insert( const_iterator, const_reference value ) {
        // Ignore hint
        return insert(value).first;
    }

    iterator insert( const_iterator, value_type&& value ) {
        // Ignore hint
        return insert(std::move(value)).first;
    }

    template<typename InputIterator>
    void insert( InputIterator first, InputIterator last ) {
        while (first != last) {
            insert(*first);
            ++first;
        }
    }

    void insert( std::initializer_list<value_type> init ) {
        insert(init.begin(), init.end());
    }

    std::pair<iterator, bool> insert( node_type&& nh ) {
        if (!nh.empty()) {
            auto insert_node = node_handle_accessor::get_node_ptr(nh);
            std::pair<iterator, bool> insert_result = internal_insert_node(insert_node);
            if (insert_result.second) {
                node_handle_accessor::deactivate(nh);
            }
            return insert_result;
        }
        return std::pair<iterator, bool>(end(), false);
    }

    iterator insert( const_iterator, node_type&& nh ) {
        // Ignore hint
        return insert(std::move(nh)).first;
    }

    template<typename... Args>
    std::pair<iterator, bool> emplace( Args&&... args ) {
        return internal_insert(std::forward<Args>(args)...);
    }

    template<typename... Args>
    iterator emplace_hint( const_iterator, Args&&... args ) {
        // Ignore hint
        return emplace(std::forward<Args>(args)...).first;
    }

    iterator unsafe_erase( iterator pos ) {
        std::pair<node_ptr, node_ptr> extract_result = internal_extract(pos);
        if (extract_result.first) { // node was extracted
            delete_value_node(extract_result.first);
            return extract_result.second;
        }
        return end();
    }

    iterator unsafe_erase( const_iterator pos ) {
        return unsafe_erase(get_iterator(pos));
    }

    iterator unsafe_erase( const_iterator first, const_iterator last ) {
        while (first != last) {
            // Unsafe erase returns the iterator which follows the erased one
            first = unsafe_erase(first);
        }
        return get_iterator(first);
    }

    size_type unsafe_erase( const key_type& key ) {
        return internal_erase(key);
    }

    template <typename K>
    typename std::enable_if<is_transparent<K>::value
                            && !std::is_convertible<K, const_iterator>::value
                            && !std::is_convertible<K, iterator>::value,
                            size_type>::type unsafe_erase( const K& key )
    {
        return internal_erase(key);
    }

    node_type unsafe_extract( const_iterator pos ) {
        std::pair<node_ptr, node_ptr> extract_result = internal_extract(pos);
        return extract_result.first ? node_handle_accessor::construct<node_type>(extract_result.first) : node_type();
    }

    node_type unsafe_extract( iterator pos ) {
        return unsafe_extract(const_iterator(pos));
    }

    node_type unsafe_extract( const key_type& key ) {
        return unsafe_extract(find(key));
    }

    template <typename K>
    typename std::enable_if<is_transparent<K>::value
                            && !std::is_convertible<K, const_iterator>::value
                            && !std::is_convertible<K, iterator>::value,
                            node_type>::type unsafe_extract( const K& key )
    {
        return unsafe_extract(find(key));
    }

    iterator lower_bound( const key_type& key ) {
        return iterator(internal_get_bound(key, my_compare));
    }

    const_iterator lower_bound( const key_type& key ) const {
        return const_iterator(internal_get_bound(key, my_compare));
    }

    template <typename K>
    typename std::enable_if<is_transparent<K>::value, iterator>::type lower_bound( const K& key ) {
        return iterator(internal_get_bound(key, my_compare));
    }

    template <typename K>
    typename std::enable_if<is_transparent<K>::value, const_iterator>::type lower_bound( const K& key ) const {
        return const_iterator(internal_get_bound(key, my_compare));
    }

    iterator upper_bound( const key_type& key ) {
        return iterator(internal_get_bound(key, not_greater_compare(my_compare)));
    }

    const_iterator upper_bound( const key_type& key ) const {
        return const_iterator(internal_get_bound(key, not_greater_compare(my_compare)));
    }

    template <typename K>
    typename std::enable_if<is_transparent<K>::value, iterator>::type upper_bound( const K& key ) {
        return iterator(internal_get_bound(key, not_greater_compare(my_compare)));
    }

    template <typename K>
    typename std::enable_if<is_transparent<K>::value, const_iterator>::type upper_bound( const K& key ) const {
        return const_iterator(internal_get_bound(key, not_greater_compare(my_compare)));
    }

    iterator find( const key_type& key ) {
        return iterator(internal_find(key));
    }

    const_iterator find( const key_type& key ) const {
        return const_iterator(internal_find(key));
    }

    template <typename K>
    typename std::enable_if<is_transparent<K>::value, iterator>::type find( const K& key ) {
        return iterator(internal_find(key));
    }

    template <typename K>
    typename std::enable_if<is_transparent<K>::value, const_iterator>::type find( const K& key ) const {
        return const_iterator(internal_find(key));
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

    void clear() noexcept {
        // clear is not thread safe - load can be relaxed
        node_ptr head = my_head_ptr.load(std::memory_order_relaxed);

        if (head == nullptr) return; // Head is not allocated => container is empty

        node_ptr current = head->next(0);

        // Delete all value nodes in the container
        while (current) {
            node_ptr next = current->next(0);
            delete_value_node(current);
            current = next;
        }

        for (size_type level = 0; level < head->height(); ++level) {
            head->set_next(level, nullptr);
        }

        my_size.store(0, std::memory_order_relaxed);
        my_max_height.store(0, std::memory_order_relaxed);
    }

    iterator begin() {
        return iterator(internal_begin());
    }

    const_iterator begin() const {
        return const_iterator(internal_begin());
    }

    const_iterator cbegin() const {
        return const_iterator(internal_begin());
    }

    iterator end() {
        return iterator(nullptr);
    }

    const_iterator end() const {
        return const_iterator(nullptr);
    }

    const_iterator cend() const {
        return const_iterator(nullptr);
    }

    size_type size() const {
        return my_size.load(std::memory_order_relaxed);
    }

    size_type max_size() const {
        return node_allocator_traits::max_size(my_node_allocator);
    }

    __TBB_nodiscard bool empty() const {
        return 0 == size();
    }

    allocator_type get_allocator() const {
        return my_node_allocator;
    }

    void swap(concurrent_skip_list& other) {
        if (this != &other) {
            using pocs_type = typename node_allocator_traits::propagate_on_container_swap;
            using is_always_equal = typename node_allocator_traits::is_always_equal;
            internal_swap(other, tbb::detail::disjunction<pocs_type, is_always_equal>());
        }
    }

    std::pair<iterator, iterator> equal_range(const key_type& key) {
        return internal_equal_range(key);
    }

    std::pair<const_iterator, const_iterator> equal_range(const key_type& key) const {
        return internal_equal_range(key);
    }

    template <typename K>
    typename std::enable_if<is_transparent<K>::value, std::pair<iterator, iterator>>::type equal_range( const K& key ) {
        return internal_equal_range(key);
    }

    template <typename K>
    typename std::enable_if<is_transparent<K>::value, std::pair<const_iterator, const_iterator>>::type equal_range( const K& key ) const {
        return internal_equal_range(key);
    }

    key_compare key_comp() const { return my_compare; }

    value_compare value_comp() const { return container_traits::value_comp(my_compare); }

    class const_range_type {
    public:
        using size_type = typename concurrent_skip_list::size_type;
        using value_type = typename concurrent_skip_list::value_type;
        using iterator = typename concurrent_skip_list::const_iterator;

        bool empty() const {
            return my_begin.my_node_ptr->next(0) == my_end.my_node_ptr;
        }

        bool is_divisible() const {
            return my_level != 0 ? my_begin.my_node_ptr->next(my_level - 1) != my_end.my_node_ptr : false;
        }

        size_type size() const { return std::distance(my_begin, my_end); }

        const_range_type( const_range_type& r, split)
            : my_end(r.my_end) {
            my_begin = iterator(r.my_begin.my_node_ptr->next(r.my_level - 1));
            my_level = my_begin.my_node_ptr->height();
            r.my_end = my_begin;
        }

        const_range_type( const concurrent_skip_list& l)
            : my_end(l.end()), my_begin(l.begin()), my_level(my_begin.my_node_ptr->height() ) {}

        iterator begin() const { return my_begin; }
        iterator end() const { return my_end; }
        size_type grainsize() const { return 1; }

    private:
        const_iterator my_end;
        const_iterator my_begin;
        size_type my_level;
    }; // class const_range_type

    class range_type : public const_range_type {
    public:
        using iterator = typename concurrent_skip_list::iterator;

        range_type(range_type& r, split) : const_range_type(r, split()) {}
        range_type(const concurrent_skip_list& l) : const_range_type(l) {}

        iterator begin() const {
            node_ptr node = const_range_type::begin().my_node_ptr;
            return iterator(node);
        }

        iterator end() const {
            node_ptr node = const_range_type::end().my_node_ptr;
            return iterator(node);
        }
    }; // class range_type

    range_type range() { return range_type(*this); }
    const_range_type range() const { return const_range_type(*this); }

private:
    node_ptr internal_begin() const {
        node_ptr head = get_head();
        return head == nullptr ? head : head->next(0);
    }

    void internal_move(concurrent_skip_list&& other) {
        my_head_ptr.store(other.my_head_ptr.load(std::memory_order_relaxed), std::memory_order_relaxed);
        other.my_head_ptr.store(nullptr, std::memory_order_relaxed);

        my_size.store(other.my_size.load(std::memory_order_relaxed), std::memory_order_relaxed);
        other.my_size.store(0, std::memory_order_relaxed);

        my_max_height.store(other.my_max_height.load(std::memory_order_relaxed), std::memory_order_relaxed);
        other.my_max_height.store(0, std::memory_order_relaxed);
    }

    void internal_move_construct_with_allocator(concurrent_skip_list&& other,
                                                /*is_always_equal = */std::true_type) {
        internal_move(std::move(other));
    }

    void internal_move_construct_with_allocator(concurrent_skip_list&& other,
                                                /*is_always_equal = */std::false_type) {
        if (my_node_allocator == other.get_allocator()) {
            internal_move(std::move(other));
        } else {
            my_size.store(0, std::memory_order_relaxed);
            my_max_height.store(other.my_max_height.load(std::memory_order_relaxed), std::memory_order_relaxed);
            internal_copy(std::make_move_iterator(other.begin()), std::make_move_iterator(other.end()));
        }
    }

    static const key_type& get_key( node_ptr n ) {
        __TBB_ASSERT(n, nullptr);
        return container_traits::get_key(static_cast<node_ptr>(n)->value());
    }

    template <typename K>
    bool found( node_ptr node, const K& key ) const {
        return node != nullptr && !my_compare(key, get_key(node));
    }

    template <typename K>
    node_ptr internal_find(const K& key) const {
        return allow_multimapping ? internal_find_multi(key) : internal_find_unique(key);
    }

    template <typename K>
    node_ptr internal_find_multi( const K& key ) const {
        node_ptr prev = get_head();
        if (prev == nullptr) return nullptr; // If the head node is not allocated - exit

        node_ptr curr = nullptr;
        node_ptr old_curr = curr;

        for (size_type h = my_max_height.load(std::memory_order_acquire); h > 0; --h) {
            curr = internal_find_position(h - 1, prev, key, my_compare);

            if (curr != old_curr && found(curr, key)) {
                return curr;
            }
            old_curr = curr;
        }
        return nullptr;
    }

    template <typename K>
    node_ptr internal_find_unique( const K& key ) const {
        const_iterator it = lower_bound(key);
        return (it == end() || my_compare(key, container_traits::get_key(*it))) ? nullptr : it.my_node_ptr;
    }

    template <typename K>
    size_type internal_count( const K& key ) const {
        if (allow_multimapping) {
            // TODO: reimplement without double traversal
            std::pair<const_iterator, const_iterator> r = equal_range(key);
            return std::distance(r.first, r.second);
        }
        return size_type(contains(key) ? 1 : 0);
    }

    template <typename K>
    std::pair<iterator, iterator> internal_equal_range(const K& key) const {
        iterator lb = get_iterator(lower_bound(key));
        auto result = std::make_pair(lb, lb);

        // If the lower bound points to the node with the requested key
        if (found(lb.my_node_ptr, key)) {

            if (!allow_multimapping) {
                // For unique containers - move the second iterator forward and exit
                ++result.second;
            } else {
                // For multi containers - find the upper bound starting from the lower bound
                node_ptr prev = lb.my_node_ptr;
                node_ptr curr = nullptr;
                not_greater_compare cmp(my_compare);

                // Start from the lower bound of the range
                for (size_type h = prev->height(); h > 0; --h) {
                    curr = prev->next(h - 1);
                    while (curr && cmp(get_key(curr), key)) {
                        prev = curr;
                        // If the height of the next node is greater than the current one - jump to its height
                        if (h < curr->height()) {
                            h = curr->height();
                        }
                        curr = prev->next(h - 1);
                    }
                }
                result.second = iterator(curr);
            }
        }

        return result;
    }

    // Finds position on the level using comparator cmp starting from the node prev
    template <typename K, typename Comparator>
    node_ptr internal_find_position( size_type level, node_ptr& prev, const K& key,
                                     const Comparator& cmp ) const {
        __TBB_ASSERT(level < prev->height(), "Wrong level to find position");
        node_ptr curr = prev->next(level);

        while (curr && cmp(get_key(curr), key)) {
            prev = curr;
            __TBB_ASSERT(level < prev->height(), nullptr);
            curr = prev->next(level);
        }

        return curr;
    }

    // The same as previous overload, but allows index_number comparison
    template <typename Comparator>
    node_ptr internal_find_position( size_type level, node_ptr& prev, node_ptr node,
                                     const Comparator& cmp ) const {
        __TBB_ASSERT(level < prev->height(), "Wrong level to find position");
        node_ptr curr = prev->next(level);

        while (curr && cmp(get_key(curr), get_key(node))) {
            if (allow_multimapping && cmp(get_key(node), get_key(curr)) && curr->index_number() > node->index_number()) {
                break;
            }

            prev = curr;
            __TBB_ASSERT(level < prev->height(), nullptr);
            curr = prev->next(level);
        }
        return curr;
    }

    template <typename Comparator>
    void fill_prev_curr_arrays(array_type& prev_nodes, array_type& curr_nodes, node_ptr node, const key_type& key,
                               const Comparator& cmp, node_ptr head ) {

        size_type curr_max_height = my_max_height.load(std::memory_order_acquire);
        size_type node_height = node->height();
        if (curr_max_height < node_height) {
            std::fill(prev_nodes.begin() + curr_max_height, prev_nodes.begin() + node_height, head);
            std::fill(curr_nodes.begin() + curr_max_height, curr_nodes.begin() + node_height, nullptr);
        }

        node_ptr prev = head;
        for (size_type level = curr_max_height; level > 0; --level) {
            node_ptr curr = internal_find_position(level - 1, prev, key, cmp);
            prev_nodes[level - 1] = prev;
            curr_nodes[level - 1] = curr;
        }
    }

    void fill_prev_array_for_existing_node( array_type& prev_nodes, node_ptr node ) {
        node_ptr head = create_head_if_necessary();
        prev_nodes.fill(head);

        node_ptr prev = head;
        for (size_type level = node->height(); level > 0; --level) {
            while (prev->next(level - 1) != node) {
                prev = prev->next(level - 1);
            }
            prev_nodes[level - 1] = prev;
        }
    }

    struct not_greater_compare {
        const key_compare& my_less_compare;

        not_greater_compare( const key_compare& less_compare ) : my_less_compare(less_compare) {}

        template <typename K1, typename K2>
        bool operator()( const K1& first, const K2& second ) const {
            return !my_less_compare(second, first);
        }
    };

    not_greater_compare select_comparator( /*allow_multimapping = */ std::true_type ) {
        return not_greater_compare(my_compare);
    }

    key_compare select_comparator( /*allow_multimapping = */ std::false_type ) {
        return my_compare;
    }

    template<typename... Args>
    std::pair<iterator, bool> internal_insert( Args&&... args ) {
        node_ptr new_node = create_value_node(std::forward<Args>(args)...);
        std::pair<iterator, bool> insert_result = internal_insert_node(new_node);
        if (!insert_result.second) {
            delete_value_node(new_node);
        }
        return insert_result;
    }

    std::pair<iterator, bool> internal_insert_node( node_ptr new_node ) {
        array_type prev_nodes;
        array_type curr_nodes;
        size_type new_height = new_node->height();
        auto compare = select_comparator(std::integral_constant<bool, allow_multimapping>{});

        node_ptr head_node = create_head_if_necessary();

        for (;;) {
            fill_prev_curr_arrays(prev_nodes, curr_nodes, new_node, get_key(new_node), compare, head_node);

            node_ptr prev = prev_nodes[0];
            node_ptr next = curr_nodes[0];

            if (allow_multimapping) {
                new_node->set_index_number(prev->index_number() + 1);
            } else {
                if (found(next, get_key(new_node))) {
                    return std::pair<iterator, bool>(iterator(next), false);
                }
            }

            new_node->set_next(0, next);
            if (!prev->atomic_next(0).compare_exchange_strong(next, new_node)) {
                continue;
            }

            // If the node was successfully linked on the first level - it will be linked on other levels
            // Insertion cannot fail starting from this point

            // If the height of inserted node is greater than maximum - increase maximum
            size_type max_height = my_max_height.load(std::memory_order_acquire);
            for (;;) {
                if (new_height <= max_height || my_max_height.compare_exchange_strong(max_height, new_height)) {
                    // If the maximum was successfully updated by current thread
                    // or by an other thread for the value, greater or equal to new_height
                    break;
                }
            }

            for (std::size_t level = 1; level < new_height; ++level) {
                // Link the node on upper levels
                for (;;) {
                    prev = prev_nodes[level];
                    next = static_cast<node_ptr>(curr_nodes[level]);

                    new_node->set_next(level, next);
                    __TBB_ASSERT(new_node->height() > level, "Internal structure break");
                    if (prev->atomic_next(level).compare_exchange_strong(next, new_node)) {
                        break;
                    }

                    for (size_type lev = level; lev != new_height; ++lev ) {
                        curr_nodes[lev] = internal_find_position(lev, prev_nodes[lev], new_node, compare);
                    }
                }
            }
            ++my_size;
            return std::pair<iterator, bool>(iterator(new_node), true);
        }
    }

    template <typename K, typename Comparator>
    node_ptr internal_get_bound( const K& key, const Comparator& cmp ) const {
        node_ptr prev = get_head();
        if (prev == nullptr) return nullptr; // If the head node is not allocated - exit

        node_ptr curr = nullptr;

        for (size_type h = my_max_height.load(std::memory_order_acquire); h > 0; --h) {
            curr = internal_find_position(h - 1, prev, key, cmp);
        }

        return curr;
    }

    template <typename K>
    size_type internal_erase( const K& key ) {
        auto eq = equal_range(key);
        size_type old_size = size();
        unsafe_erase(eq.first, eq.second);
        return old_size - size();
    }

    // Returns node_ptr to the extracted node and node_ptr to the next node after the extracted
    std::pair<node_ptr, node_ptr> internal_extract( const_iterator it ) {
        std::pair<node_ptr, node_ptr> result(nullptr, nullptr);
        if ( it != end() ) {
            array_type prev_nodes;

            node_ptr erase_node = it.my_node_ptr;
            node_ptr next_node = erase_node->next(0);
            fill_prev_array_for_existing_node(prev_nodes, erase_node);

            for (size_type level = 0; level < erase_node->height(); ++level) {
                prev_nodes[level]->set_next(level, erase_node->next(level));
                erase_node->set_next(level, nullptr);
            }
            my_size.fetch_sub(1, std::memory_order_relaxed);

            result.first = erase_node;
            result.second = next_node;
        }
        return result;
    }

protected:
    template<typename SourceType>
    void internal_merge( SourceType&& source ) {
        using source_type = typename std::decay<SourceType>::type;
        using source_iterator = typename source_type::iterator;
        static_assert((std::is_same<node_type, typename source_type::node_type>::value), "Incompatible containers cannot be merged");

        for (source_iterator it = source.begin(); it != source.end();) {
            source_iterator where = it++;
            if (allow_multimapping || !contains(container_traits::get_key(*where))) {
                node_type handle = source.unsafe_extract(where);
                __TBB_ASSERT(!handle.empty(), "Extracted handle in merge is empty");

                if (!insert(std::move(handle)).second) {
                    //If the insertion fails - return the node into source
                    source.insert(std::move(handle));
                }
                __TBB_ASSERT(handle.empty(), "Node handle should be empty after the insertion");
            }
        }
    }

private:
    void internal_copy( const concurrent_skip_list& other ) {
        internal_copy(other.begin(), other.end());
    }

    template<typename Iterator>
    void internal_copy( Iterator first, Iterator last ) {
        try_call([&] {
            for (auto it = first; it != last; ++it) {
                insert(*it);
            }
        }).on_exception([&] {
            clear();
            node_ptr head = my_head_ptr.load(std::memory_order_relaxed);
            if (head != nullptr) {
                delete_node(head);
            }
        });
    }

    static size_type calc_node_size( size_type height ) {
        static_assert(alignof(list_node_type) >= alignof(typename list_node_type::atomic_node_ptr), "Incorrect alignment");
        return sizeof(list_node_type) + height * sizeof(typename list_node_type::atomic_node_ptr);
    }

    node_ptr create_node( size_type height ) {
        size_type sz = calc_node_size(height);
        node_ptr node = reinterpret_cast<node_ptr>(node_allocator_traits::allocate(my_node_allocator, sz));
        node_allocator_traits::construct(my_node_allocator, node, height, my_node_allocator);
        return node;
    }

    template <typename... Args>
    node_ptr create_value_node( Args&&... args ) {
        node_ptr node = create_node(my_rng());

        // try_call API is not convenient here due to broken
        // variadic capture on GCC 4.8.5
        auto value_guard = make_raii_guard([&] {
            delete_node(node);
        });

        // Construct the value inside the node
        node_allocator_traits::construct(my_node_allocator, node->storage(), std::forward<Args>(args)...);
        value_guard.dismiss();
        return node;
    }

    node_ptr create_head_node() {
        return create_node(max_level);
    }

    void delete_node( node_ptr node ) {
        size_type sz = calc_node_size(node->height());

        // Destroy the node
        node_allocator_traits::destroy(my_node_allocator, node);
        // Deallocate the node
        node_allocator_traits::deallocate(my_node_allocator, reinterpret_cast<std::uint8_t*>(node), sz);
    }

    void delete_value_node( node_ptr node ) {
        // Destroy the value inside the node
        node_allocator_traits::destroy(my_node_allocator, node->storage());
        delete_node(node);
    }

    node_ptr get_head() const {
        return my_head_ptr.load(std::memory_order_acquire);
    }

    node_ptr create_head_if_necessary() {
        node_ptr current_head = get_head();
        if (current_head == nullptr) {
            // Head node was not created - create it
            node_ptr new_head = create_head_node();
            if (my_head_ptr.compare_exchange_strong(current_head, new_head)) {
                current_head = new_head;
            } else {
                // If an other thread has already created the head node - destroy new_head
                // current_head now points to the actual head node
                delete_node(new_head);
            }
        }
        __TBB_ASSERT(my_head_ptr.load(std::memory_order_relaxed) != nullptr, nullptr);
        __TBB_ASSERT(current_head != nullptr, nullptr);
        return current_head;
    }

    static iterator get_iterator( const_iterator it ) {
        return iterator(it.my_node_ptr);
    }

    void internal_move_assign( concurrent_skip_list&& other, /*POCMA || is_always_equal =*/std::true_type ) {
        internal_move(std::move(other));
    }

    void internal_move_assign( concurrent_skip_list&& other, /*POCMA || is_always_equal =*/std::false_type ) {
        if (my_node_allocator == other.my_node_allocator) {
            internal_move(std::move(other));
        } else {
            internal_copy(std::make_move_iterator(other.begin()), std::make_move_iterator(other.end()));
        }
    }

    void internal_swap_fields( concurrent_skip_list& other ) {
        using std::swap;
        swap_allocators(my_node_allocator, other.my_node_allocator);
        swap(my_compare, other.my_compare);
        swap(my_rng, other.my_rng);

        swap_atomics_relaxed(my_head_ptr, other.my_head_ptr);
        swap_atomics_relaxed(my_size, other.my_size);
        swap_atomics_relaxed(my_max_height, other.my_max_height);
    }

    void internal_swap( concurrent_skip_list& other, /*POCMA || is_always_equal =*/std::true_type ) {
        internal_swap_fields(other);
    }

    void internal_swap( concurrent_skip_list& other, /*POCMA || is_always_equal =*/std::false_type ) {
        __TBB_ASSERT(my_node_allocator == other.my_node_allocator, "Swapping with unequal allocators is not allowed");
        internal_swap_fields(other);
    }

    node_allocator_type my_node_allocator;
    key_compare my_compare;
    random_level_generator_type my_rng;
    std::atomic<list_node_type*> my_head_ptr;
    std::atomic<size_type> my_size;
    std::atomic<size_type> my_max_height;

    template<typename OtherTraits>
    friend class concurrent_skip_list;
}; // class concurrent_skip_list

template <typename Traits>
bool operator==( const concurrent_skip_list<Traits>& lhs, const concurrent_skip_list<Traits>& rhs ) {
    if (lhs.size() != rhs.size()) return false;
#if _MSC_VER
    // Passing "unchecked" iterators to std::equal with 3 parameters
    // causes compiler warnings.
    // The workaround is to use overload with 4 parameters, which is
    // available since C++14 - minimally supported version on MSVC
    return std::equal(lhs.begin(), lhs.end(), rhs.begin(), rhs.end());
#else
    return std::equal(lhs.begin(), lhs.end(), rhs.begin());
#endif
}

#if !__TBB_CPP20_COMPARISONS_PRESENT
template <typename Traits>
bool operator!=( const concurrent_skip_list<Traits>& lhs, const concurrent_skip_list<Traits>& rhs ) {
    return !(lhs == rhs);
}
#endif

#if __TBB_CPP20_COMPARISONS_PRESENT && __TBB_CPP20_CONCEPTS_PRESENT
template <typename Traits>
tbb::detail::synthesized_three_way_result<typename Traits::value_type>
operator<=>( const concurrent_skip_list<Traits>& lhs, const concurrent_skip_list<Traits>& rhs ) {
    return std::lexicographical_compare_three_way(lhs.begin(), lhs.end(),
                                                  rhs.begin(), rhs.end(),
                                                  tbb::detail::synthesized_three_way_comparator{});
}
#else
template <typename Traits>
bool operator<( const concurrent_skip_list<Traits>& lhs, const concurrent_skip_list<Traits>& rhs ) {
    return std::lexicographical_compare(lhs.begin(), lhs.end(), rhs.begin(), rhs.end());
}

template <typename Traits>
bool operator>( const concurrent_skip_list<Traits>& lhs, const concurrent_skip_list<Traits>& rhs ) {
    return rhs < lhs;
}

template <typename Traits>
bool operator<=( const concurrent_skip_list<Traits>& lhs, const concurrent_skip_list<Traits>& rhs ) {
    return !(rhs < lhs);
}

template <typename Traits>
bool operator>=( const concurrent_skip_list<Traits>& lhs, const concurrent_skip_list<Traits>& rhs ) {
    return !(lhs < rhs);
}
#endif // __TBB_CPP20_COMPARISONS_PRESENT && __TBB_CPP20_CONCEPTS_PRESENT

// Generates a number from the interval [0, MaxLevel).
template <std::size_t MaxLevel>
class concurrent_geometric_level_generator {
public:
    static constexpr std::size_t max_level = MaxLevel;
    // TODO: modify the algorithm to accept other values of max_level
    static_assert(max_level == 32, "Incompatible max_level for rng");

    concurrent_geometric_level_generator() : engines(std::minstd_rand::result_type(time(nullptr))) {}

    std::size_t operator()() {
        // +1 is required to pass at least 1 into log2 (log2(0) is undefined)
        // -1 is required to have an ability to return 0 from the generator (max_level - log2(2^31) - 1)
        std::size_t result = max_level - std::size_t(tbb::detail::log2(engines.local()() + 1)) - 1;
        __TBB_ASSERT(result <= max_level, nullptr);
        return result;
    }

private:
    tbb::enumerable_thread_specific<std::minstd_rand> engines;
};

} // namespace d1
} // namespace detail
} // namespace tbb

#if defined(_MSC_VER) && !defined(__INTEL_COMPILER)
#pragma warning(pop) // warning 4127 is back
#endif

#endif // __TBB_detail__concurrent_skip_list_H
