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

#ifndef __TBB_detail__segment_table_H
#define __TBB_detail__segment_table_H

#include "_config.h"
#include "_allocator_traits.h"
#include "_template_helpers.h"
#include "_utils.h"
#include "_assert.h"
#include "_exception.h"
#include <atomic>
#include <type_traits>
#include <memory>
#include <cstring>

#if defined(_MSC_VER) && !defined(__INTEL_COMPILER)
#pragma warning(push)
#pragma warning(disable: 4127) // warning C4127: conditional expression is constant
#endif

namespace tbb {
namespace detail {
namespace d1 {

template <typename T, typename Allocator, typename DerivedType, std::size_t PointersPerEmbeddedTable>
class segment_table {
public:
    using value_type = T;
    using segment_type = T*;
    using atomic_segment = std::atomic<segment_type>;
    using segment_table_type = atomic_segment*;

    using size_type = std::size_t;
    using segment_index_type = std::size_t;

    using allocator_type = Allocator;

    using allocator_traits_type = tbb::detail::allocator_traits<allocator_type>;
    using segment_table_allocator_type = typename allocator_traits_type::template rebind_alloc<atomic_segment>;
protected:
    using segment_table_allocator_traits = tbb::detail::allocator_traits<segment_table_allocator_type>;
    using derived_type = DerivedType;

    static constexpr size_type pointers_per_embedded_table = PointersPerEmbeddedTable;
    static constexpr size_type pointers_per_long_table = sizeof(size_type) * 8;
public:
    segment_table( const allocator_type& alloc = allocator_type() )
        : my_segment_table_allocator(alloc), my_segment_table(my_embedded_table)
        , my_first_block{}, my_size{}, my_segment_table_allocation_failed{}
    {
        zero_table(my_embedded_table, pointers_per_embedded_table);
    }

    segment_table( const segment_table& other )
        : my_segment_table_allocator(segment_table_allocator_traits::
                                     select_on_container_copy_construction(other.my_segment_table_allocator))
        , my_segment_table(my_embedded_table), my_first_block{}, my_size{}, my_segment_table_allocation_failed{}
    {
        zero_table(my_embedded_table, pointers_per_embedded_table);
        try_call( [&] {
            internal_transfer(other, copy_segment_body_type{*this});
        } ).on_exception( [&] {
            clear();
        });
    }

    segment_table( const segment_table& other, const allocator_type& alloc )
        : my_segment_table_allocator(alloc), my_segment_table(my_embedded_table)
        , my_first_block{}, my_size{}, my_segment_table_allocation_failed{}
    {
        zero_table(my_embedded_table, pointers_per_embedded_table);
        try_call( [&] {
            internal_transfer(other, copy_segment_body_type{*this});
        } ).on_exception( [&] {
            clear();
        });
    }

    segment_table( segment_table&& other )
        : my_segment_table_allocator(std::move(other.my_segment_table_allocator)), my_segment_table(my_embedded_table)
        , my_first_block{}, my_size{}, my_segment_table_allocation_failed{}
    {
        zero_table(my_embedded_table, pointers_per_embedded_table);
        internal_move(std::move(other));
    }

    segment_table( segment_table&& other, const allocator_type& alloc )
        : my_segment_table_allocator(alloc), my_segment_table(my_embedded_table), my_first_block{}
        , my_size{}, my_segment_table_allocation_failed{}
    {
        zero_table(my_embedded_table, pointers_per_embedded_table);
        using is_equal_type = typename segment_table_allocator_traits::is_always_equal;
        internal_move_construct_with_allocator(std::move(other), alloc, is_equal_type());
    }

    ~segment_table() {
        clear();
    }

    segment_table& operator=( const segment_table& other ) {
        if (this != &other) {
            copy_assign_allocators(my_segment_table_allocator, other.my_segment_table_allocator);
            internal_transfer(other, copy_segment_body_type{*this});
        }
        return *this;
    }

    segment_table& operator=( segment_table&& other ) 
        noexcept(derived_type::is_noexcept_assignment)
    {
        using pocma_type = typename segment_table_allocator_traits::propagate_on_container_move_assignment;
        using is_equal_type = typename segment_table_allocator_traits::is_always_equal;

        if (this != &other) {
            move_assign_allocators(my_segment_table_allocator, other.my_segment_table_allocator);
            internal_move_assign(std::move(other), tbb::detail::disjunction<is_equal_type, pocma_type>());
        }
        return *this;
    }

    void swap( segment_table& other ) 
        noexcept(derived_type::is_noexcept_swap)
    {
        using is_equal_type = typename segment_table_allocator_traits::is_always_equal;
        using pocs_type = typename segment_table_allocator_traits::propagate_on_container_swap;

        if (this != &other) {
            swap_allocators(my_segment_table_allocator, other.my_segment_table_allocator);
            internal_swap(other, tbb::detail::disjunction<is_equal_type, pocs_type>());
        }
    }

    segment_type get_segment( segment_index_type index ) const {
        return get_table()[index] + segment_base(index);
    }

    value_type& operator[]( size_type index ) {
        return internal_subscript<true>(index);
    }

    const value_type& operator[]( size_type index ) const {
        return const_cast<segment_table*>(this)->internal_subscript<true>(index);
    }

    const segment_table_allocator_type& get_allocator() const {
        return my_segment_table_allocator;
    }

    segment_table_allocator_type& get_allocator() {
        return my_segment_table_allocator;
    }

    void enable_segment( segment_type& segment, segment_table_type table, segment_index_type seg_index, size_type index ) {
        // Allocate new segment
        segment_type new_segment = self()->create_segment(table, seg_index, index);
        if (new_segment != nullptr) {
            // Store (new_segment - segment_base) into the segment table to allow access to the table by index via
            // my_segment_table[segment_index_of(index)][index]
            segment_type disabled_segment = nullptr;
            if (!table[seg_index].compare_exchange_strong(disabled_segment, new_segment - segment_base(seg_index))) {
                // compare_exchange failed => some other thread has already enabled this segment
                // Deallocate the memory
                self()->deallocate_segment(new_segment, seg_index);
            }
        }

        segment = table[seg_index].load(std::memory_order_acquire);
        __TBB_ASSERT(segment != nullptr, "If create_segment returned nullptr, the element should be stored in the table");
    }

    void delete_segment( segment_index_type seg_index ) {
        segment_type disabled_segment = nullptr;
        // Set the pointer to the segment to NULL in the table
        segment_type segment_to_delete = get_table()[seg_index].exchange(disabled_segment);
        if (segment_to_delete == segment_allocation_failure_tag) {
            return;
        }

        segment_to_delete += segment_base(seg_index);

        // Deallocate the segment
        self()->destroy_segment(segment_to_delete, seg_index);
    }

    size_type number_of_segments( segment_table_type table ) const {
        // Check for an active table, if it is embedded table - return the number of embedded segments
        // Otherwise - return the maximum number of segments
        return table == my_embedded_table ? pointers_per_embedded_table : pointers_per_long_table;
    }

    size_type capacity() const noexcept {
        segment_table_type table = get_table();
        size_type num_segments = number_of_segments(table);
        for (size_type seg_index = 0; seg_index < num_segments; ++seg_index) {
            // Check if the pointer is valid (allocated)
            if (table[seg_index].load(std::memory_order_relaxed) <= segment_allocation_failure_tag) {
                return segment_base(seg_index);
            }
        }
        return segment_base(num_segments);
    }

    size_type find_last_allocated_segment( segment_table_type table ) const noexcept {
        size_type end = 0;
        size_type num_segments = number_of_segments(table);
        for (size_type seg_index = 0; seg_index < num_segments; ++seg_index) {
            // Check if the pointer is valid (allocated)
            if (table[seg_index].load(std::memory_order_relaxed) > segment_allocation_failure_tag) {
                end = seg_index + 1;
            }
        }
        return end;
    }

    void reserve( size_type n ) {
        if (n > allocator_traits_type::max_size(my_segment_table_allocator)) {
            throw_exception(exception_id::reservation_length_error);
        }

        size_type size = my_size.load(std::memory_order_relaxed);
        segment_index_type start_seg_idx = size == 0 ? 0 : segment_index_of(size - 1) + 1;
        for (segment_index_type seg_idx = start_seg_idx; segment_base(seg_idx) < n; ++seg_idx) {
                size_type first_index = segment_base(seg_idx);
                internal_subscript<true>(first_index);
        }
    }

    void clear() {
        clear_segments();
        clear_table();
        my_size.store(0, std::memory_order_relaxed);
        my_first_block.store(0, std::memory_order_relaxed);
    }

    void clear_segments() {
        segment_table_type current_segment_table = get_table();
        for (size_type i = number_of_segments(current_segment_table); i != 0; --i) {
            if (current_segment_table[i - 1].load(std::memory_order_relaxed) != nullptr) {
                // If the segment was enabled - disable and deallocate it
                delete_segment(i - 1);
            }
        }
    }

    void clear_table() {
        segment_table_type current_segment_table = get_table();
        if (current_segment_table != my_embedded_table) {
            // If the active table is not the embedded one - deallocate the active table
            for (size_type i = 0; i != pointers_per_long_table; ++i) {
                segment_table_allocator_traits::destroy(my_segment_table_allocator, &current_segment_table[i]);
            }

            segment_table_allocator_traits::deallocate(my_segment_table_allocator, current_segment_table, pointers_per_long_table);
            my_segment_table.store(my_embedded_table, std::memory_order_relaxed);
            zero_table(my_embedded_table, pointers_per_embedded_table);
        }
    }

    void extend_table_if_necessary(segment_table_type& table, size_type start_index, size_type end_index) {
        // extend_segment_table if an active table is an embedded table
        // and the requested index is not in the embedded table
        if (table == my_embedded_table && end_index > embedded_table_size) {
            if (start_index <= embedded_table_size) {
                try_call([&] {
                    table = self()->allocate_long_table(my_embedded_table, start_index);
                    // It is possible that the table was extended by the thread that allocated first_block.
                    // In this case it is necessary to re-read the current table.

                    if (table) {
                        my_segment_table.store(table, std::memory_order_release);
                    } else {
                        table = my_segment_table.load(std::memory_order_acquire);
                    }
                }).on_exception([&] {
                    my_segment_table_allocation_failed.store(true, std::memory_order_relaxed);
                });
            } else {
                atomic_backoff backoff;
                do {
                    if (my_segment_table_allocation_failed.load(std::memory_order_relaxed)) {
                        throw_exception(exception_id::bad_alloc);
                    }
                    backoff.pause();
                    table = my_segment_table.load(std::memory_order_acquire); 
                } while (table == my_embedded_table);
            }
        }
    }

    // Return the segment where index is stored
    static constexpr segment_index_type segment_index_of( size_type index ) {
        return size_type(tbb::detail::log2(uintptr_t(index|1)));
    }

    // Needed to calculate the offset in segment
    static constexpr size_type segment_base( size_type index ) {
        return size_type(1) << index & ~size_type(1);
    }

    // Return size of the segment
    static constexpr size_type segment_size( size_type index ) {
        return index == 0 ? 2 : size_type(1) << index;
    }

private:

    derived_type* self() {
        return static_cast<derived_type*>(this);
    }

    struct copy_segment_body_type {
        void operator()( segment_index_type index, segment_type from, segment_type to ) const {
            my_instance.self()->copy_segment(index, from, to);
        }
        segment_table& my_instance;
    };

    struct move_segment_body_type {
        void operator()( segment_index_type index, segment_type from, segment_type to ) const {
            my_instance.self()->move_segment(index, from, to);
        }
        segment_table& my_instance;
    };

    // Transgers all segments from the other table
    template <typename TransferBody>
    void internal_transfer( const segment_table& other, TransferBody transfer_segment ) {
        static_cast<derived_type*>(this)->destroy_elements();

        assign_first_block_if_necessary(other.my_first_block.load(std::memory_order_relaxed));
        my_size.store(other.my_size.load(std::memory_order_relaxed), std::memory_order_relaxed);

        segment_table_type other_table = other.get_table();
        size_type end_segment_size = segment_size(other.find_last_allocated_segment(other_table));

        // If an exception occurred in other, then the size may be greater than the size of the end segment.
        size_type other_size = end_segment_size < other.my_size.load(std::memory_order_relaxed) ?
            other.my_size.load(std::memory_order_relaxed) : end_segment_size;
        other_size = my_segment_table_allocation_failed ? embedded_table_size : other_size;

        for (segment_index_type i = 0; segment_base(i) < other_size; ++i) {
            // If the segment in other table is enabled - transfer it
            if (other_table[i].load(std::memory_order_relaxed) == segment_allocation_failure_tag)
            {
                    my_size = segment_base(i);
                    break;
            } else if (other_table[i].load(std::memory_order_relaxed) != nullptr) {
                internal_subscript<true>(segment_base(i));
                transfer_segment(i, other.get_table()[i].load(std::memory_order_relaxed) + segment_base(i),
                                get_table()[i].load(std::memory_order_relaxed) + segment_base(i));
            }
        }
    }

    // Moves the other segment table
    // Only equal allocators are allowed
    void internal_move( segment_table&& other ) {
        // NOTE: allocators should be equal
        clear();
        my_first_block.store(other.my_first_block.load(std::memory_order_relaxed), std::memory_order_relaxed);
        my_size.store(other.my_size.load(std::memory_order_relaxed), std::memory_order_relaxed);
        // If an active table in other is embedded - restore all of the embedded segments
        if (other.get_table() == other.my_embedded_table) {
            for ( size_type i = 0; i != pointers_per_embedded_table; ++i ) {
                segment_type other_segment = other.my_embedded_table[i].load(std::memory_order_relaxed);
                my_embedded_table[i].store(other_segment, std::memory_order_relaxed);
                other.my_embedded_table[i].store(nullptr, std::memory_order_relaxed);
            }
            my_segment_table.store(my_embedded_table, std::memory_order_relaxed);
        } else {
            my_segment_table.store(other.my_segment_table, std::memory_order_relaxed);
            other.my_segment_table.store(other.my_embedded_table, std::memory_order_relaxed);
            zero_table(other.my_embedded_table, pointers_per_embedded_table);
        }
        other.my_size.store(0, std::memory_order_relaxed);
    }

    // Move construct the segment table with the allocator object
    // if any instances of allocator_type are always equal
    void internal_move_construct_with_allocator( segment_table&& other, const allocator_type&,
                                                 /*is_always_equal = */ std::true_type ) {
        internal_move(std::move(other));
    }

    // Move construct the segment table with the allocator object
    // if any instances of allocator_type are always equal
    void internal_move_construct_with_allocator( segment_table&& other, const allocator_type& alloc,
                                                 /*is_always_equal = */ std::false_type ) {
        if (other.my_segment_table_allocator == alloc) {
            // If allocators are equal - restore pointers
            internal_move(std::move(other));
        } else {
            // If allocators are not equal - perform per element move with reallocation
            try_call( [&] {
                internal_transfer(other, move_segment_body_type{*this});
            } ).on_exception( [&] {
                clear();
            });
        }
    }

    // Move assigns the segment table to other is any instances of allocator_type are always equal
    // or propagate_on_container_move_assignment is true
    void internal_move_assign( segment_table&& other, /*is_always_equal || POCMA = */ std::true_type ) {
        internal_move(std::move(other));
    }

    // Move assigns the segment table to other is any instances of allocator_type are not always equal
    // and propagate_on_container_move_assignment is false
    void internal_move_assign( segment_table&& other, /*is_always_equal || POCMA = */ std::false_type ) {
        if (my_segment_table_allocator == other.my_segment_table_allocator) {
            // If allocators are equal - restore pointers
            internal_move(std::move(other));
        } else {
            // If allocators are not equal - perform per element move with reallocation
            internal_transfer(other, move_segment_body_type{*this});
        }
    }

    // Swaps two segment tables if any instances of allocator_type are always equal
    // or propagate_on_container_swap is true
    void internal_swap( segment_table& other, /*is_always_equal || POCS = */ std::true_type ) {
        internal_swap_fields(other);
    }

    // Swaps two segment tables if any instances of allocator_type are not always equal
    // and propagate_on_container_swap is false
    // According to the C++ standard, swapping of two containers with unequal allocators
    // is an undefined behavior scenario
    void internal_swap( segment_table& other, /*is_always_equal || POCS = */ std::false_type ) {
        __TBB_ASSERT(my_segment_table_allocator == other.my_segment_table_allocator,
                     "Swapping with unequal allocators is not allowed");
        internal_swap_fields(other);
    }

    void internal_swap_fields( segment_table& other ) {
        // If an active table in either *this segment table or other is an embedded one - swaps the embedded tables
        if (get_table() == my_embedded_table ||
            other.get_table() == other.my_embedded_table) {

            for (size_type i = 0; i != pointers_per_embedded_table; ++i) {
                segment_type current_segment = my_embedded_table[i].load(std::memory_order_relaxed);
                segment_type other_segment = other.my_embedded_table[i].load(std::memory_order_relaxed);

                my_embedded_table[i].store(other_segment, std::memory_order_relaxed);
                other.my_embedded_table[i].store(current_segment, std::memory_order_relaxed);
            }
        }

        segment_table_type current_segment_table = get_table();
        segment_table_type other_segment_table = other.get_table();

        // If an active table is an embedded one -
        // store an active table in other to the embedded one from other
        if (current_segment_table == my_embedded_table) {
            other.my_segment_table.store(other.my_embedded_table, std::memory_order_relaxed);
        } else {
            // Otherwise - store it to the active segment table
            other.my_segment_table.store(current_segment_table, std::memory_order_relaxed);
        }

        // If an active table in other segment table is an embedded one -
        // store an active table in other to the embedded one from *this
        if (other_segment_table == other.my_embedded_table) {
            my_segment_table.store(my_embedded_table, std::memory_order_relaxed);
        } else {
            // Otherwise - store it to the active segment table in other
            my_segment_table.store(other_segment_table, std::memory_order_relaxed);
        }
        auto first_block = other.my_first_block.load(std::memory_order_relaxed);
        other.my_first_block.store(my_first_block.load(std::memory_order_relaxed), std::memory_order_relaxed);
        my_first_block.store(first_block, std::memory_order_relaxed);

        auto size = other.my_size.load(std::memory_order_relaxed);
        other.my_size.store(my_size.load(std::memory_order_relaxed), std::memory_order_relaxed);
        my_size.store(size, std::memory_order_relaxed);
    }

protected:
    // A flag indicates that an exception was throws during segment allocations
    const segment_type segment_allocation_failure_tag = reinterpret_cast<segment_type>(1);
    static constexpr size_type embedded_table_size = segment_size(pointers_per_embedded_table);

    template <bool allow_out_of_range_access>
    value_type& internal_subscript( size_type index ) {
        segment_index_type seg_index = segment_index_of(index);
        segment_table_type table = my_segment_table.load(std::memory_order_acquire);
        segment_type segment = nullptr;

        if (allow_out_of_range_access) {
            if (derived_type::allow_table_extending) {
                extend_table_if_necessary(table, index, index + 1);
            }

            segment = table[seg_index].load(std::memory_order_acquire);
            // If the required segment is disabled - enable it
            if (segment == nullptr) {
                enable_segment(segment, table, seg_index, index);
            }
            // Check if an exception was thrown during segment allocation
            if (segment == segment_allocation_failure_tag) {
                throw_exception(exception_id::bad_alloc);
            }
        } else {
            segment = table[seg_index].load(std::memory_order_acquire);
        }
        __TBB_ASSERT(segment != nullptr, nullptr);

        return segment[index];
    }

    void assign_first_block_if_necessary(segment_index_type index) {
        size_type zero = 0;
        if (this->my_first_block.load(std::memory_order_relaxed) == zero) {
            this->my_first_block.compare_exchange_strong(zero, index);
        }
    }

    void zero_table( segment_table_type table, size_type count ) {
        for (size_type i = 0; i != count; ++i) {
            table[i].store(nullptr, std::memory_order_relaxed);
        }
    }

    segment_table_type get_table() const {
        return my_segment_table.load(std::memory_order_acquire);
    }

    segment_table_allocator_type my_segment_table_allocator;
    std::atomic<segment_table_type> my_segment_table;
    atomic_segment my_embedded_table[pointers_per_embedded_table];
    // Number of segments in first block
    std::atomic<size_type> my_first_block;
    // Number of elements in table
    std::atomic<size_type> my_size;
    // Flag to indicate failed extend table
    std::atomic<bool> my_segment_table_allocation_failed;
}; // class segment_table

} // namespace d1
} // namespace detail
} // namespace tbb

#if defined(_MSC_VER) && !defined(__INTEL_COMPILER)
#pragma warning(pop) // warning 4127 is back
#endif

#endif // __TBB_detail__segment_table_H
