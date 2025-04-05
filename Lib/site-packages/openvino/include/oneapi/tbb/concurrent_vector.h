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

#ifndef __TBB_concurrent_vector_H
#define __TBB_concurrent_vector_H

#include "detail/_namespace_injection.h"
#include "detail/_utils.h"
#include "detail/_assert.h"
#include "detail/_allocator_traits.h"
#include "detail/_segment_table.h"
#include "detail/_containers_helpers.h"
#include "blocked_range.h"
#include "cache_aligned_allocator.h"

#include <algorithm>
#include <utility> // std::move_if_noexcept
#include <algorithm>
#if __TBB_CPP20_COMPARISONS_PRESENT
#include <compare>
#endif

namespace tbb {
namespace detail {
namespace d1 {

template <typename Vector, typename Value>
class vector_iterator {
    using vector_type = Vector;

public:
    using value_type = Value;
    using size_type = typename vector_type::size_type;
    using difference_type = typename vector_type::difference_type;
    using pointer = value_type*;
    using reference = value_type&;
    using iterator_category = std::random_access_iterator_tag;

    template <typename Vec, typename Val>
    friend vector_iterator<Vec, Val> operator+( typename vector_iterator<Vec, Val>::difference_type, const vector_iterator<Vec, Val>& );

    template <typename Vec, typename Val1, typename Val2>
    friend typename vector_iterator<Vec, Val1>::difference_type operator-( const vector_iterator<Vec, Val1>&, const vector_iterator<Vec, Val2>& );

    template <typename Vec, typename Val1, typename Val2>
    friend bool operator==( const vector_iterator<Vec, Val1>&, const vector_iterator<Vec, Val2>& );

    template <typename Vec, typename Val1, typename Val2>
    friend bool operator<( const vector_iterator<Vec, Val1>&, const vector_iterator<Vec, Val2>& );

    template <typename Vec, typename Val>
    friend class vector_iterator;

    template <typename T, typename Allocator>
    friend class concurrent_vector;

private:
    vector_iterator( const vector_type& vector, size_type index, value_type* item = nullptr )
        : my_vector(const_cast<vector_type*>(&vector)), my_index(index), my_item(item)
    {}

public:
    vector_iterator() : my_vector(nullptr), my_index(~size_type(0)), my_item(nullptr)
    {}

    vector_iterator( const vector_iterator<vector_type, typename vector_type::value_type>& other )
        : my_vector(other.my_vector), my_index(other.my_index), my_item(other.my_item)
    {}

    vector_iterator& operator=( const vector_iterator<vector_type, typename vector_type::value_type>& other ) {
        my_vector = other.my_vector;
        my_index = other.my_index;
        my_item = other.my_item;
        return *this;
    }

    vector_iterator operator+( difference_type offset ) const {
        return vector_iterator(*my_vector, my_index + offset);
    }

    vector_iterator& operator+=( difference_type offset ) {
        my_index += offset;
        my_item = nullptr;
        return *this;
    }

    vector_iterator operator-( difference_type offset ) const {
        return vector_iterator(*my_vector, my_index - offset);
    }

    vector_iterator& operator-=( difference_type offset ) {
        my_index -= offset;
        my_item = nullptr;
        return *this;
    }

    reference operator*() const {
        value_type *item = my_item;
        if (item == nullptr) {
            item = &my_vector->internal_subscript(my_index);
        } else {
            __TBB_ASSERT(item == &my_vector->internal_subscript(my_index), "corrupt cache");
        }
        return *item;
    }

    pointer operator->() const { return &(operator*()); }

    reference operator[]( difference_type k ) const {
        return my_vector->internal_subscript(my_index + k);
    }

    vector_iterator& operator++() {
        ++my_index;
        if (my_item != nullptr) {
            if (vector_type::is_first_element_in_segment(my_index)) {
                // If the iterator crosses a segment boundary, the pointer become invalid
                // as possibly next segment is in another memory location
                my_item = nullptr;
            } else {
                ++my_item;
            }
        }
        return *this;
    }

    vector_iterator operator++(int) {
        vector_iterator result = *this;
        ++(*this);
        return result;
    }

    vector_iterator& operator--() {
        __TBB_ASSERT(my_index > 0, "operator--() applied to iterator already at beginning of concurrent_vector");
        --my_index;
        if (my_item != nullptr) {
            if (vector_type::is_first_element_in_segment(my_index)) {
                // If the iterator crosses a segment boundary, the pointer become invalid
                // as possibly next segment is in another memory location
                my_item = nullptr;
            } else {
                --my_item;
            }
        }
        return *this;
    }

    vector_iterator operator--(int) {
        vector_iterator result = *this;
        --(*this);
        return result;
    }

private:
    // concurrent_vector over which we are iterating.
    vector_type* my_vector;

    // Index into the vector
    size_type my_index;

    // Caches my_vector *it;
    // If my_item == nullptr cached value is not available use internal_subscript(my_index)
    mutable value_type* my_item;
}; // class vector_iterator

template <typename Vector, typename T>
vector_iterator<Vector, T> operator+( typename vector_iterator<Vector, T>::difference_type offset,
                                      const vector_iterator<Vector, T>& v )
{
    return vector_iterator<Vector, T>(*v.my_vector, v.my_index + offset);
}

template <typename Vector, typename T, typename U>
typename vector_iterator<Vector, T>::difference_type operator-( const vector_iterator<Vector, T>& i,
                                                                const vector_iterator<Vector, U>& j )
{
    using difference_type = typename vector_iterator<Vector, T>::difference_type;
    return static_cast<difference_type>(i.my_index) - static_cast<difference_type>(j.my_index);
}

template <typename Vector, typename T, typename U>
bool operator==( const vector_iterator<Vector, T>& i, const vector_iterator<Vector, U>& j ) {
    return i.my_vector == j.my_vector && i.my_index == j.my_index;
}

template <typename Vector, typename T, typename U>
bool operator!=( const vector_iterator<Vector, T>& i, const vector_iterator<Vector, U>& j ) {
    return !(i == j);
}

template <typename Vector, typename T, typename U>
bool operator<( const vector_iterator<Vector, T>& i, const vector_iterator<Vector, U>& j ) {
    return i.my_index < j.my_index;
}

template <typename Vector, typename T, typename U>
bool operator>( const vector_iterator<Vector, T>& i, const vector_iterator<Vector, U>& j ) {
    return j < i;
}

template <typename Vector, typename T, typename U>
bool operator>=( const vector_iterator<Vector, T>& i, const vector_iterator<Vector, U>& j ) {
    return !(i < j);
}

template <typename Vector, typename T, typename U>
bool operator<=( const vector_iterator<Vector, T>& i, const vector_iterator<Vector, U>& j ) {
    return !(j < i);
}

static constexpr std::size_t embedded_table_num_segments = 3;

template <typename T, typename Allocator = tbb::cache_aligned_allocator<T>>
class concurrent_vector
    : private segment_table<T, Allocator, concurrent_vector<T, Allocator>, embedded_table_num_segments>
{
    using self_type = concurrent_vector<T, Allocator>;
    using base_type = segment_table<T, Allocator, self_type, embedded_table_num_segments>;

    friend class segment_table<T, Allocator, self_type, embedded_table_num_segments>;

    template <typename Iterator>
    class generic_range_type : public tbb::blocked_range<Iterator> {
        using base_type = tbb::blocked_range<Iterator>;
    public:
        using value_type = T;
        using reference = T&;
        using const_reference = const T&;
        using iterator = Iterator;
        using difference_type = std::ptrdiff_t;

        using base_type::base_type;

        template<typename U>
        generic_range_type( const generic_range_type<U>& r) : blocked_range<Iterator>(r.begin(), r.end(), r.grainsize()) {}
        generic_range_type( generic_range_type& r, split ) : blocked_range<Iterator>(r, split()) {}
    }; // class generic_range_type

    static_assert(std::is_same<T, typename Allocator::value_type>::value,
                  "value_type of the container must be the same as its allocator's");
    using allocator_traits_type = tbb::detail::allocator_traits<Allocator>;
    // Segment table for concurrent_vector can be extended
    static constexpr bool allow_table_extending = true;
    static constexpr bool is_noexcept_assignment = allocator_traits_type::propagate_on_container_move_assignment::value ||
                                                   allocator_traits_type::is_always_equal::value;
    static constexpr bool is_noexcept_swap = allocator_traits_type::propagate_on_container_swap::value ||
                                             allocator_traits_type::is_always_equal::value;

public:
    using value_type = T;
    using allocator_type = Allocator;
    using size_type = std::size_t;
    using difference_type = std::ptrdiff_t;
    using reference = value_type&;
    using const_reference = const value_type&;

    using pointer = typename allocator_traits_type::pointer;
    using const_pointer = typename allocator_traits_type::const_pointer;

    using iterator = vector_iterator<concurrent_vector, value_type>;
    using const_iterator = vector_iterator<concurrent_vector, const value_type>;
    using reverse_iterator = std::reverse_iterator<iterator>;
    using const_reverse_iterator = std::reverse_iterator<const_iterator>;

    using range_type = generic_range_type<iterator>;
    using const_range_type = generic_range_type<const_iterator>;

    concurrent_vector() : concurrent_vector(allocator_type()) {}

    explicit concurrent_vector( const allocator_type& alloc ) noexcept
        : base_type(alloc)
    {}

    explicit concurrent_vector( size_type count, const value_type& value,
                                const allocator_type& alloc = allocator_type() )
        : concurrent_vector(alloc)
    {
        try_call( [&] {
            grow_by(count, value);
        } ).on_exception( [&] {
            base_type::clear();
        });
    }

    explicit concurrent_vector( size_type count, const allocator_type& alloc = allocator_type() )
        : concurrent_vector(alloc)
    {
        try_call( [&] {
            grow_by(count);
        } ).on_exception( [&] {
            base_type::clear();
        });
    }

    template <typename InputIterator>
    concurrent_vector( InputIterator first, InputIterator last, const allocator_type& alloc = allocator_type() )
        : concurrent_vector(alloc)
    {
        try_call( [&] {
            grow_by(first, last);
        } ).on_exception( [&] {
            base_type::clear();
        });
    }

    concurrent_vector( const concurrent_vector& other )
        : base_type(segment_table_allocator_traits::select_on_container_copy_construction(other.get_allocator()))
    {
        try_call( [&] {
            grow_by(other.begin(), other.end());
        } ).on_exception( [&] {
            base_type::clear();
        });
    }

    concurrent_vector( const concurrent_vector& other, const allocator_type& alloc )
        : base_type(other, alloc) {}

    concurrent_vector(concurrent_vector&& other) noexcept
        : base_type(std::move(other))
    {}

    concurrent_vector( concurrent_vector&& other, const allocator_type& alloc )
        : base_type(std::move(other), alloc)
    {}

    concurrent_vector( std::initializer_list<value_type> init,
                       const allocator_type& alloc = allocator_type() )
        : concurrent_vector(init.begin(), init.end(), alloc)
    {}

    ~concurrent_vector() {}

    // Assignment
    concurrent_vector& operator=( const concurrent_vector& other ) {
        base_type::operator=(other);
        return *this;
    }

    concurrent_vector& operator=( concurrent_vector&& other ) noexcept(is_noexcept_assignment) {
        base_type::operator=(std::move(other));
        return *this;
    }

    concurrent_vector& operator=( std::initializer_list<value_type> init ) {
        assign(init);
        return *this;
    }

    void assign( size_type count, const value_type& value ) {
        destroy_elements();
        grow_by(count, value);
    }

    template <typename InputIterator>
    typename std::enable_if<is_input_iterator<InputIterator>::value, void>::type
    assign( InputIterator first, InputIterator last ) {
        destroy_elements();
        grow_by(first, last);
    }

    void assign( std::initializer_list<value_type> init ) {
        destroy_elements();
        assign(init.begin(), init.end());
    }

    // Concurrent growth
    iterator grow_by( size_type delta ) {
        return internal_grow_by_delta(delta);
    }

    iterator grow_by( size_type delta, const value_type& value ) {
        return internal_grow_by_delta(delta, value);
    }

    template <typename ForwardIterator>
    typename std::enable_if<is_input_iterator<ForwardIterator>::value, iterator>::type
    grow_by( ForwardIterator first, ForwardIterator last ) {
        auto delta = std::distance(first, last);
        return internal_grow_by_delta(delta, first, last);
    }

    iterator grow_by( std::initializer_list<value_type> init ) {
        return grow_by(init.begin(), init.end());
    }

    iterator grow_to_at_least( size_type n ) {
        return internal_grow_to_at_least(n);
    }
    iterator grow_to_at_least( size_type n, const value_type& value ) {
        return internal_grow_to_at_least(n, value);
    }

    iterator push_back( const value_type& item ) {
        return internal_emplace_back(item);
    }

    iterator push_back( value_type&& item ) {
        return internal_emplace_back(std::move(item));
    }

    template <typename... Args>
    iterator emplace_back( Args&&... args ) {
        return internal_emplace_back(std::forward<Args>(args)...);
    }

    // Items access
    reference operator[]( size_type index ) {
        return internal_subscript(index);
    }
    const_reference operator[]( size_type index ) const {
        return internal_subscript(index);
    }

    reference at( size_type index ) {
        return internal_subscript_with_exceptions(index);
    }
    const_reference at( size_type index ) const {
        return internal_subscript_with_exceptions(index);
    }

    // Get range for iterating with parallel algorithms
    range_type range( size_t grainsize = 1 ) {
        return range_type(begin(), end(), grainsize);
    }

    // Get const range for iterating with parallel algorithms
    const_range_type range( size_t grainsize = 1 ) const {
        return const_range_type(begin(), end(), grainsize);
    }

    reference front() {
        return internal_subscript(0);
    }

    const_reference front() const {
        return internal_subscript(0);
    }

    reference back() {
        return internal_subscript(size() - 1);
    }

    const_reference back() const {
        return internal_subscript(size() - 1);
    }

    // Iterators
    iterator begin() { return iterator(*this, 0); }
    const_iterator begin() const { return const_iterator(*this, 0); }
    const_iterator cbegin() const { return const_iterator(*this, 0); }

    iterator end() { return iterator(*this, size()); }
    const_iterator end() const { return const_iterator(*this, size()); }
    const_iterator cend() const { return const_iterator(*this, size()); }

    reverse_iterator rbegin() { return reverse_iterator(end()); }
    const_reverse_iterator rbegin() const { return const_reverse_iterator(end()); }
    const_reverse_iterator crbegin() const { return const_reverse_iterator(cend()); }

    reverse_iterator rend() { return reverse_iterator(begin()); }
    const_reverse_iterator rend() const { return const_reverse_iterator(begin()); }
    const_reverse_iterator crend() const { return const_reverse_iterator(cbegin()); }

    allocator_type get_allocator() const {
        return base_type::get_allocator();
    }

    // Storage
    bool empty() const noexcept {
        return 0 == size();
    }

    size_type size() const noexcept {
        return std::min(this->my_size.load(std::memory_order_acquire), capacity());
    }

    size_type max_size() const noexcept {
        return allocator_traits_type::max_size(base_type::get_allocator());
    }

    size_type capacity() const noexcept {
        return base_type::capacity();
    }

    void reserve( size_type n ) {
        if (n == 0) return;

        if (n > max_size()) {
            tbb::detail::throw_exception(exception_id::reservation_length_error);
        }

        this->assign_first_block_if_necessary(this->segment_index_of(n - 1) + 1);
        base_type::reserve(n);
    }

    void resize( size_type n ) {
        internal_resize(n);
    }

    void resize( size_type n, const value_type& val ) {
        internal_resize(n, val);
    }

    void shrink_to_fit() {
        internal_compact();
    }

    void swap(concurrent_vector& other) noexcept(is_noexcept_swap) {
        base_type::swap(other);
    }

    void clear() {
        destroy_elements();
    }

private:
    using segment_type = typename base_type::segment_type;
    using segment_table_type = typename base_type::segment_table_type;
    using segment_table_allocator_traits = typename base_type::segment_table_allocator_traits;
    using segment_index_type = typename base_type::segment_index_type;

    using segment_element_type = typename base_type::value_type;
    using segment_element_allocator_type = typename allocator_traits_type::template rebind_alloc<segment_element_type>;
    using segment_element_allocator_traits = tbb::detail::allocator_traits<segment_element_allocator_type>;

    segment_table_type allocate_long_table( const typename base_type::atomic_segment* embedded_table, size_type start_index ) {
        __TBB_ASSERT(start_index <= this->embedded_table_size, "Start index out of embedded table");

        // If other threads are trying to set pointers in the short segment, wait for them to finish their
        // assignments before we copy the short segment to the long segment. Note: grow_to_at_least depends on it
        for (segment_index_type i = 0; this->segment_base(i) < start_index; ++i) {
            spin_wait_while_eq(embedded_table[i], segment_type(nullptr));
        }

        // It is possible that the table was extend by a thread allocating first_block, need to check this.
        if (this->get_table() != embedded_table) {
            return nullptr;
        }

        // Allocate long segment table and fill with null pointers
        segment_table_type new_segment_table = segment_table_allocator_traits::allocate(base_type::get_allocator(), this->pointers_per_long_table);
        // Copy segment pointers from the embedded table
        for (size_type segment_index = 0; segment_index < this->pointers_per_embedded_table; ++segment_index) {
            segment_table_allocator_traits::construct(base_type::get_allocator(), &new_segment_table[segment_index],
                embedded_table[segment_index].load(std::memory_order_relaxed));
        }
        for (size_type segment_index = this->pointers_per_embedded_table; segment_index < this->pointers_per_long_table; ++segment_index) {
            segment_table_allocator_traits::construct(base_type::get_allocator(), &new_segment_table[segment_index], nullptr);
        }

        return new_segment_table;
    }

    // create_segment function is required by the segment_table base class
    segment_type create_segment( segment_table_type table, segment_index_type seg_index, size_type index ) {
        size_type first_block = this->my_first_block.load(std::memory_order_relaxed);
        // First block allocation
        if (seg_index < first_block) {
            // If 0 segment is already allocated, then it remains to wait until the segments are filled to requested
            if (table[0].load(std::memory_order_acquire) != nullptr) {
                spin_wait_while_eq(table[seg_index], segment_type(nullptr));
                return nullptr;
            }

            segment_element_allocator_type segment_allocator(base_type::get_allocator());
            segment_type new_segment = nullptr;
            size_type first_block_size = this->segment_size(first_block);
            try_call( [&] {
                new_segment = segment_element_allocator_traits::allocate(segment_allocator, first_block_size);
            } ).on_exception( [&] {
                segment_type disabled_segment = nullptr;
                if (table[0].compare_exchange_strong(disabled_segment, this->segment_allocation_failure_tag)) {
                    size_type end_segment = table == this->my_embedded_table ? this->pointers_per_embedded_table : first_block;
                    for (size_type i = 1; i < end_segment; ++i) {
                        table[i].store(this->segment_allocation_failure_tag, std::memory_order_release);
                    }
                }
            });

            segment_type disabled_segment = nullptr;
            if (table[0].compare_exchange_strong(disabled_segment, new_segment)) {
                this->extend_table_if_necessary(table, 0, first_block_size);
                for (size_type i = 1; i < first_block; ++i) {
                    table[i].store(new_segment, std::memory_order_release);
                }

                // Other threads can wait on a snapshot of an embedded table, need to fill it.
                for (size_type i = 1; i < first_block && i < this->pointers_per_embedded_table; ++i) {
                    this->my_embedded_table[i].store(new_segment, std::memory_order_release);
                }
            } else if (new_segment != this->segment_allocation_failure_tag) {
                // Deallocate the memory
                segment_element_allocator_traits::deallocate(segment_allocator, new_segment, first_block_size);
                // 0 segment is already allocated, then it remains to wait until the segments are filled to requested
                spin_wait_while_eq(table[seg_index], segment_type(nullptr));
            }
        } else {
            size_type offset = this->segment_base(seg_index);
            if (index == offset) {
                __TBB_ASSERT(table[seg_index].load(std::memory_order_relaxed) == nullptr, "Only this thread can enable this segment");
                segment_element_allocator_type segment_allocator(base_type::get_allocator());
                segment_type new_segment = this->segment_allocation_failure_tag;
                try_call( [&] {
                    new_segment = segment_element_allocator_traits::allocate(segment_allocator,this->segment_size(seg_index));
                    // Shift base address to simplify access by index
                    new_segment -= this->segment_base(seg_index);
                } ).on_completion( [&] {
                    table[seg_index].store(new_segment, std::memory_order_release);
                });
            } else {
                spin_wait_while_eq(table[seg_index], segment_type(nullptr));
            }
        }
        return nullptr;
    }

    // Returns the number of elements in the segment to be destroy
    size_type number_of_elements_in_segment( segment_index_type seg_index ) {
        size_type curr_vector_size = this->my_size.load(std::memory_order_relaxed);
        size_type curr_segment_base = this->segment_base(seg_index);

        if (seg_index == 0) {
            return std::min(curr_vector_size, this->segment_size(seg_index));
        } else {
            // Perhaps the segment is allocated, but there are no elements in it.
            if (curr_vector_size < curr_segment_base) {
                return 0;
            }
            return curr_segment_base * 2 > curr_vector_size ? curr_vector_size - curr_segment_base : curr_segment_base;
        }
    }

    void deallocate_segment( segment_type address, segment_index_type seg_index ) {
        segment_element_allocator_type segment_allocator(base_type::get_allocator());
        size_type first_block = this->my_first_block.load(std::memory_order_relaxed);
        if (seg_index >= first_block) {
            segment_element_allocator_traits::deallocate(segment_allocator, address, this->segment_size(seg_index));
        }
        else if (seg_index == 0) {
            size_type elements_to_deallocate = first_block > 0 ? this->segment_size(first_block) : this->segment_size(0);
            segment_element_allocator_traits::deallocate(segment_allocator, address, elements_to_deallocate);
        }
    }

    // destroy_segment function is required by the segment_table base class
    void destroy_segment( segment_type address, segment_index_type seg_index ) {
        size_type elements_to_destroy = number_of_elements_in_segment(seg_index);
        segment_element_allocator_type segment_allocator(base_type::get_allocator());

        for (size_type i = 0; i < elements_to_destroy; ++i) {
            segment_element_allocator_traits::destroy(segment_allocator, address + i);
        }

        deallocate_segment(address, seg_index);
    }

    // copy_segment function is required by the segment_table base class
    void copy_segment( segment_index_type seg_index, segment_type from, segment_type to ) {
        size_type i = 0;
        try_call( [&] {
            for (; i != number_of_elements_in_segment(seg_index); ++i) {
                segment_table_allocator_traits::construct(base_type::get_allocator(), to + i, from[i]);
            }
        } ).on_exception( [&] {
            // Zero-initialize items left not constructed after the exception
            zero_unconstructed_elements(this->get_segment(seg_index) + i, this->segment_size(seg_index) - i);

            segment_index_type last_segment = this->segment_index_of(this->my_size.load(std::memory_order_relaxed));
            auto table = this->get_table();
            for (segment_index_type j = seg_index + 1; j != last_segment; ++j) {
                auto curr_segment = table[j].load(std::memory_order_relaxed);
                if (curr_segment) {
                    zero_unconstructed_elements(curr_segment + this->segment_base(j), this->segment_size(j));
                }
            }
            this->my_size.store(this->segment_size(seg_index) + i, std::memory_order_relaxed);
        });
    }

    // move_segment function is required by the segment_table base class
    void move_segment( segment_index_type seg_index, segment_type from, segment_type to ) {
        size_type i = 0;
        try_call( [&] {
            for (; i != number_of_elements_in_segment(seg_index); ++i) {
                segment_table_allocator_traits::construct(base_type::get_allocator(), to + i, std::move(from[i]));
            }
        } ).on_exception( [&] {
            // Zero-initialize items left not constructed after the exception
            zero_unconstructed_elements(this->get_segment(seg_index) + i, this->segment_size(seg_index) - i);

            segment_index_type last_segment = this->segment_index_of(this->my_size.load(std::memory_order_relaxed));
            auto table = this->get_table();
            for (segment_index_type j = seg_index + 1; j != last_segment; ++j) {
                auto curr_segment = table[j].load(std::memory_order_relaxed);
                if (curr_segment) {
                    zero_unconstructed_elements(curr_segment + this->segment_base(j), this->segment_size(j));
                }
            }
            this->my_size.store(this->segment_size(seg_index) + i, std::memory_order_relaxed);
        });
    }

    static constexpr bool is_first_element_in_segment( size_type index ) {
        // An element is the first in a segment if its index is equal to a power of two
        return is_power_of_two_at_least(index, 2);
    }

    const_reference internal_subscript( size_type index ) const {
        return const_cast<self_type*>(this)->internal_subscript(index);
    }

    reference internal_subscript( size_type index ) {
        __TBB_ASSERT(index < this->my_size.load(std::memory_order_relaxed), "Invalid subscript index");
        return base_type::template internal_subscript</*allow_out_of_range_access=*/false>(index);
    }

    const_reference internal_subscript_with_exceptions( size_type index ) const {
        return const_cast<self_type*>(this)->internal_subscript_with_exceptions(index);
    }

    reference internal_subscript_with_exceptions( size_type index ) {
        if (index >= this->my_size.load(std::memory_order_acquire)) {
            tbb::detail::throw_exception(exception_id::out_of_range);
        }

        segment_table_type table = this->my_segment_table.load(std::memory_order_acquire);

        size_type seg_index = this->segment_index_of(index);
        if (base_type::number_of_segments(table) < seg_index) {
            tbb::detail::throw_exception(exception_id::out_of_range);
        }

        if (table[seg_index] <= this->segment_allocation_failure_tag) {
            tbb::detail::throw_exception(exception_id::out_of_range);
        }

        return base_type::template internal_subscript</*allow_out_of_range_access=*/false>(index);
    }

    static void zero_unconstructed_elements( pointer start, size_type count ) {
        std::memset(static_cast<void *>(start), 0, count * sizeof(value_type));
    }

    template <typename... Args>
    iterator internal_emplace_back( Args&&... args ) {
        size_type old_size = this->my_size++;
        this->assign_first_block_if_necessary(default_first_block_size);
        auto element_address = &base_type::template internal_subscript</*allow_out_of_range_access=*/true>(old_size);

        // try_call API is not convenient here due to broken
        // variadic capture on GCC 4.8.5
        auto value_guard = make_raii_guard([&] {
            zero_unconstructed_elements(element_address, /*count =*/1);
        });

        segment_table_allocator_traits::construct(base_type::get_allocator(), element_address, std::forward<Args>(args)...);
        value_guard.dismiss();
        return iterator(*this, old_size, element_address);
    }

    template <typename... Args>
    void internal_loop_construct( segment_table_type table, size_type start_idx, size_type end_idx, const Args&... args ) {
        static_assert(sizeof...(Args) < 2, "Too many parameters");
        for (size_type idx = start_idx; idx < end_idx; ++idx) {
            auto element_address = &base_type::template internal_subscript</*allow_out_of_range_access=*/true>(idx);
            // try_call API is not convenient here due to broken
            // variadic capture on GCC 4.8.5
            auto value_guard = make_raii_guard( [&] {
                segment_index_type last_allocated_segment = this->find_last_allocated_segment(table);
                size_type segment_size = this->segment_size(last_allocated_segment);
                end_idx = end_idx < segment_size ? end_idx : segment_size;
                for (size_type i = idx; i < end_idx; ++i) {
                    zero_unconstructed_elements(&this->internal_subscript(i), /*count =*/1);
                }
            });
            segment_table_allocator_traits::construct(base_type::get_allocator(), element_address, args...);
            value_guard.dismiss();
        }
    }

    template <typename ForwardIterator>
    void internal_loop_construct( segment_table_type table, size_type start_idx, size_type end_idx, ForwardIterator first, ForwardIterator ) {
        for (size_type idx = start_idx; idx < end_idx; ++idx) {
            auto element_address = &base_type::template internal_subscript</*allow_out_of_range_access=*/true>(idx);
            try_call( [&] {
                segment_table_allocator_traits::construct(base_type::get_allocator(), element_address, *first++);
            } ).on_exception( [&] {
                segment_index_type last_allocated_segment = this->find_last_allocated_segment(table);
                size_type segment_size = this->segment_size(last_allocated_segment);
                end_idx = end_idx < segment_size ? end_idx : segment_size;
                for (size_type i = idx; i < end_idx; ++i) {
                    zero_unconstructed_elements(&this->internal_subscript(i), /*count =*/1);
                }
            });
        }
    }

    template <typename... Args>
    iterator internal_grow( size_type start_idx, size_type end_idx, const Args&... args ) {
        this->assign_first_block_if_necessary(this->segment_index_of(end_idx - 1) + 1);
        size_type seg_index = this->segment_index_of(end_idx - 1);
        segment_table_type table = this->get_table();
        this->extend_table_if_necessary(table, start_idx, end_idx);

        if (seg_index > this->my_first_block.load(std::memory_order_relaxed)) {
            // So that other threads be able to work with the last segment of grow_by, allocate it immediately.
            // If the last segment is not less than the first block
            if (table[seg_index].load(std::memory_order_relaxed) == nullptr) {
                size_type first_element = this->segment_base(seg_index);
                if (first_element >= start_idx && first_element < end_idx) {
                    segment_type segment = table[seg_index].load(std::memory_order_relaxed);
                    base_type::enable_segment(segment, table, seg_index, first_element);
                }
            }
        }

        internal_loop_construct(table, start_idx, end_idx, args...);

        return iterator(*this, start_idx, &base_type::template internal_subscript</*allow_out_of_range_access=*/false>(start_idx));
    }


    template <typename... Args>
    iterator internal_grow_by_delta( size_type delta, const Args&... args ) {
        if (delta == size_type(0)) {
            return end();
        }
        size_type start_idx = this->my_size.fetch_add(delta);
        size_type end_idx = start_idx + delta;

        return internal_grow(start_idx, end_idx, args...);
    }

    template <typename... Args>
    iterator internal_grow_to_at_least( size_type new_size, const Args&... args ) {
        size_type old_size = this->my_size.load(std::memory_order_relaxed);
        if (new_size == size_type(0)) return iterator(*this, 0);
        while (old_size < new_size && !this->my_size.compare_exchange_weak(old_size, new_size))
        {}

        int delta = static_cast<int>(new_size) - static_cast<int>(old_size);
        if (delta > 0) {
            return internal_grow(old_size, new_size, args...);
        }

        size_type end_segment = this->segment_index_of(new_size - 1);

        // Check/wait for segments allocation completes
        if (end_segment >= this->pointers_per_embedded_table &&
            this->get_table() == this->my_embedded_table)
        {
            spin_wait_while_eq(this->my_segment_table, this->my_embedded_table);
        }

        for (segment_index_type seg_idx = 0; seg_idx <= end_segment; ++seg_idx) {
            if (this->get_table()[seg_idx].load(std::memory_order_relaxed) == nullptr) {
                atomic_backoff backoff(true);
                while (this->get_table()[seg_idx].load(std::memory_order_relaxed) == nullptr) {
                    backoff.pause();
                }
            }
        }

    #if TBB_USE_DEBUG
        size_type cap = capacity();
        __TBB_ASSERT( cap >= new_size, NULL);
    #endif
        return iterator(*this, size());
    }

    template <typename... Args>
    void internal_resize( size_type n, const Args&... args ) {
        if (n == 0) {
            clear();
            return;
        }

        size_type old_size = this->my_size.load(std::memory_order_acquire);
        if (n > old_size) {
            reserve(n);
            grow_to_at_least(n, args...);
        } else {
            if (old_size == n) {
                return;
            }
            size_type last_segment = this->segment_index_of(old_size - 1);
            // Delete segments
            for (size_type seg_idx = this->segment_index_of(n - 1) + 1; seg_idx <= last_segment; ++seg_idx) {
                this->delete_segment(seg_idx);
            }

            // If n > segment_size(n) => we need to destroy all of the items in the first segment
            // Otherwise, we need to destroy only items with the index < n
            size_type n_segment = this->segment_index_of(n - 1);
            size_type last_index_to_destroy = std::min(this->segment_base(n_segment) + this->segment_size(n_segment), old_size);
            // Destroy elements in curr segment
            for (size_type idx = n; idx < last_index_to_destroy; ++idx) {
                segment_table_allocator_traits::destroy(base_type::get_allocator(), &base_type::template internal_subscript</*allow_out_of_range_access=*/false>(idx));
            }
            this->my_size.store(n, std::memory_order_release);
        }
    }

    void destroy_elements() {
        allocator_type alloc(base_type::get_allocator());
        for (size_type i = 0; i < this->my_size.load(std::memory_order_relaxed); ++i) {
            allocator_traits_type::destroy(alloc, &base_type::template internal_subscript</*allow_out_of_range_access=*/false>(i));
        }
        this->my_size.store(0, std::memory_order_relaxed);
    }

    static bool incompact_predicate( size_type size ) {
        // memory page size
        const size_type page_size = 4096;
        return size < page_size || ((size - 1) % page_size < page_size / 2 && size < page_size * 128);
    }

    void internal_compact() {
        const size_type curr_size = this->my_size.load(std::memory_order_relaxed);
        segment_table_type table = this->get_table();
        const segment_index_type k_end = this->find_last_allocated_segment(table);                   // allocated segments
        const segment_index_type k_stop = curr_size ? this->segment_index_of(curr_size - 1) + 1 : 0; // number of segments to store existing items: 0=>0; 1,2=>1; 3,4=>2; [5-8]=>3;..
        const segment_index_type first_block = this->my_first_block;                                 // number of merged segments, getting values from atomics

        segment_index_type k = first_block;
        if (k_stop < first_block) {
            k = k_stop;
        }
        else {
            while (k < k_stop && incompact_predicate(this->segment_size(k) * sizeof(value_type))) k++;
        }

        if (k_stop == k_end && k == first_block) {
            return;
        }

        // First segment optimization
        if (k != first_block && k) {
            size_type max_block = std::max(first_block, k);

            auto buffer_table = segment_table_allocator_traits::allocate(base_type::get_allocator(), max_block);

            for (size_type seg_idx = 0; seg_idx < max_block; ++seg_idx) {
                segment_table_allocator_traits::construct(base_type::get_allocator(), &buffer_table[seg_idx],
                    table[seg_idx].load(std::memory_order_relaxed));
                table[seg_idx].store(nullptr, std::memory_order_relaxed);
            }

            this->my_first_block.store(k, std::memory_order_relaxed);
            size_type index = 0;
            try_call( [&] {
                for (; index < std::min(this->segment_size(max_block), curr_size); ++index) {
                    auto element_address = &static_cast<base_type*>(this)->operator[](index);
                    segment_index_type seg_idx = this->segment_index_of(index);
                    segment_table_allocator_traits::construct(base_type::get_allocator(), element_address,
                    std::move_if_noexcept(buffer_table[seg_idx].load(std::memory_order_relaxed)[index]));
                }
            } ).on_exception( [&] {
                segment_element_allocator_type allocator(base_type::get_allocator());
                for (size_type i = 0; i < index; ++i) {
                    auto element_adress = &this->operator[](i);
                    segment_element_allocator_traits::destroy(allocator, element_adress);
                }
                segment_element_allocator_traits::deallocate(allocator,
                    table[0].load(std::memory_order_relaxed), this->segment_size(max_block));

                for (size_type seg_idx = 0; seg_idx < max_block; ++seg_idx) {
                    table[seg_idx].store(buffer_table[seg_idx].load(std::memory_order_relaxed),
                        std::memory_order_relaxed);
                    buffer_table[seg_idx].store(nullptr, std::memory_order_relaxed);
                }
                segment_table_allocator_traits::deallocate(base_type::get_allocator(),
                    buffer_table, max_block);
                this->my_first_block.store(first_block, std::memory_order_relaxed);
            });

            // Need to correct deallocate old segments
            // Method destroy_segment respect active first_block, therefore,
            // in order for the segment deletion to work correctly, set the first_block size that was earlier,
            // destroy the unnecessary segments.
            this->my_first_block.store(first_block, std::memory_order_relaxed);
            for (size_type seg_idx = max_block; seg_idx > 0 ; --seg_idx) {
                auto curr_segment = buffer_table[seg_idx - 1].load(std::memory_order_relaxed);
                if (curr_segment != nullptr) {
                    destroy_segment(buffer_table[seg_idx - 1].load(std::memory_order_relaxed) + this->segment_base(seg_idx - 1),
                        seg_idx - 1);
                }
            }

            this->my_first_block.store(k, std::memory_order_relaxed);

            for (size_type seg_idx = 0; seg_idx < max_block; ++seg_idx) {
                segment_table_allocator_traits::destroy(base_type::get_allocator(), &buffer_table[seg_idx]);
            }

            segment_table_allocator_traits::deallocate(base_type::get_allocator(), buffer_table, max_block);
        }
        // free unnecessary segments allocated by reserve() call
        if (k_stop < k_end) {
            for (size_type seg_idx = k_end; seg_idx != k_stop; --seg_idx) {
                if (table[seg_idx - 1].load(std::memory_order_relaxed) != nullptr) {
                    this->delete_segment(seg_idx - 1);
                }
            }
            if (!k) this->my_first_block.store(0, std::memory_order_relaxed);;
        }
    }

    // Lever for adjusting the size of first_block at the very first insertion.
    // TODO: consider >1 value, check performance
    static constexpr size_type default_first_block_size = 1;

    template <typename Vector, typename Value>
    friend class vector_iterator;
}; // class concurrent_vector

#if __TBB_CPP17_DEDUCTION_GUIDES_PRESENT
// Deduction guide for the constructor from two iterators
template <typename It, typename Alloc = tbb::cache_aligned_allocator<iterator_value_t<It>>,
          typename = std::enable_if_t<is_input_iterator_v<It>>,
          typename = std::enable_if_t<is_allocator_v<Alloc>>>
concurrent_vector( It, It, Alloc = Alloc() )
-> concurrent_vector<iterator_value_t<It>, Alloc>;
#endif

template <typename T, typename Allocator>
void swap(concurrent_vector<T, Allocator> &lhs,
          concurrent_vector<T, Allocator> &rhs)
{
    lhs.swap(rhs);
}

template <typename T, typename Allocator>
bool operator==(const concurrent_vector<T, Allocator> &lhs,
                const concurrent_vector<T, Allocator> &rhs)
{
    return lhs.size() == rhs.size() && std::equal(lhs.begin(), lhs.end(), rhs.begin());
}

#if !__TBB_CPP20_COMPARISONS_PRESENT
template <typename T, typename Allocator>
bool operator!=(const concurrent_vector<T, Allocator> &lhs,
                const concurrent_vector<T, Allocator> &rhs)
{
    return !(lhs == rhs);
}
#endif // !__TBB_CPP20_COMPARISONS_PRESENT

#if __TBB_CPP20_COMPARISONS_PRESENT && __TBB_CPP20_CONCEPTS_PRESENT
template <typename T, typename Allocator>
tbb::detail::synthesized_three_way_result<typename concurrent_vector<T, Allocator>::value_type>
operator<=>(const concurrent_vector<T, Allocator> &lhs,
            const concurrent_vector<T, Allocator> &rhs)
{
    return std::lexicographical_compare_three_way(lhs.begin(), lhs.end(),
                                                  rhs.begin(), rhs.end(),
                                                  tbb::detail::synthesized_three_way_comparator{});
}

#else

template <typename T, typename Allocator>
bool operator<(const concurrent_vector<T, Allocator> &lhs,
               const concurrent_vector<T, Allocator> &rhs)
{
    return std::lexicographical_compare(lhs.begin(), lhs.end(), rhs.begin(), rhs.end());
}

template <typename T, typename Allocator>
bool operator<=(const concurrent_vector<T, Allocator> &lhs,
                const concurrent_vector<T, Allocator> &rhs)
{
    return !(rhs < lhs);
}

template <typename T, typename Allocator>
bool operator>(const concurrent_vector<T, Allocator> &lhs,
               const concurrent_vector<T, Allocator> &rhs)
{
    return rhs < lhs;
}

template <typename T, typename Allocator>
bool operator>=(const concurrent_vector<T, Allocator> &lhs,
                const concurrent_vector<T, Allocator> &rhs)
{
    return !(lhs < rhs);
}
#endif // __TBB_CPP20_COMPARISONS_PRESENT && __TBB_CPP20_CONCEPTS_PRESENT

} // namespace d1
} // namespace detail

inline namespace v1 {
    using detail::d1::concurrent_vector;
} // namespace v1

} // namespace tbb

#endif // __TBB_concurrent_vector_H
