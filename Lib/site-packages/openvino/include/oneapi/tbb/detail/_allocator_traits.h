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

#ifndef __TBB_detail__allocator_traits_H
#define __TBB_detail__allocator_traits_H

#include "_config.h"
#include "_template_helpers.h"
#include <memory>
#include <type_traits>

namespace tbb {
namespace detail {
inline namespace d0 {

#if !__TBB_CPP17_ALLOCATOR_IS_ALWAYS_EQUAL_PRESENT
// Struct is_always_equal_detector provides the member type "type" which is
// Allocator::is_always_equal if it is present, std::false_type otherwise
template <typename Allocator, typename = void>
struct is_always_equal_detector {
    using type = std::false_type;
};

template <typename Allocator>
struct is_always_equal_detector<Allocator, tbb::detail::void_t<typename Allocator::is_always_equal>>
{
    using type = typename Allocator::is_always_equal;
};
#endif // !__TBB_CPP17_ALLOCATOR_IS_ALWAYS_EQUAL_PRESENT

template <typename Allocator>
class allocator_traits : public std::allocator_traits<Allocator>
{
    using base_type = std::allocator_traits<Allocator>;
public:
#if !__TBB_CPP17_ALLOCATOR_IS_ALWAYS_EQUAL_PRESENT
    using is_always_equal = typename is_always_equal_detector<Allocator>::type;
#endif

    template <typename T>
    using rebind_traits = typename tbb::detail::allocator_traits<typename base_type::template rebind_alloc<T>>;
}; // struct allocator_traits

template <typename Allocator>
void copy_assign_allocators_impl( Allocator& lhs, const Allocator& rhs, /*pocca = */std::true_type ) {
    lhs = rhs;
}

template <typename Allocator>
void copy_assign_allocators_impl( Allocator&, const Allocator&, /*pocca = */ std::false_type ) {}

// Copy assigns allocators only if propagate_on_container_copy_assignment is true
template <typename Allocator>
void copy_assign_allocators( Allocator& lhs, const Allocator& rhs ) {
    using pocca_type = typename allocator_traits<Allocator>::propagate_on_container_copy_assignment;
    copy_assign_allocators_impl(lhs, rhs, pocca_type());
}

template <typename Allocator>
void move_assign_allocators_impl( Allocator& lhs, Allocator& rhs, /*pocma = */ std::true_type ) {
    lhs = std::move(rhs);
}

template <typename Allocator>
void move_assign_allocators_impl( Allocator&, Allocator&, /*pocma = */ std::false_type ) {}

// Move assigns allocators only if propagate_on_container_move_assignment is true
template <typename Allocator>
void move_assign_allocators( Allocator& lhs, Allocator& rhs ) {
    using pocma_type = typename allocator_traits<Allocator>::propagate_on_container_move_assignment;
    move_assign_allocators_impl(lhs, rhs, pocma_type());
}

template <typename Allocator>
void swap_allocators_impl( Allocator& lhs, Allocator& rhs, /*pocs = */ std::true_type ) {
    using std::swap;
    swap(lhs, rhs);
}

template <typename Allocator>
void swap_allocators_impl( Allocator&, Allocator&, /*pocs = */ std::false_type ) {}

// Swaps allocators only if propagate_on_container_swap is true
template <typename Allocator>
void swap_allocators( Allocator& lhs, Allocator& rhs ) {
    using pocs_type = typename allocator_traits<Allocator>::propagate_on_container_swap;
    swap_allocators_impl(lhs, rhs, pocs_type());
}

} // inline namespace d0
} // namespace detail
} // namespace tbb

#endif // __TBB_detail__allocator_traits_H
