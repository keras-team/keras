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

#ifndef __TBB_cache_aligned_allocator_H
#define __TBB_cache_aligned_allocator_H

#include "detail/_utils.h"
#include "detail/_namespace_injection.h"
#include <cstdlib>
#include <utility>

#if __TBB_CPP17_MEMORY_RESOURCE_PRESENT
#include <memory_resource>
#endif

namespace tbb {
namespace detail {

namespace r1 {
void*       __TBB_EXPORTED_FUNC cache_aligned_allocate(std::size_t size);
void        __TBB_EXPORTED_FUNC cache_aligned_deallocate(void* p);
std::size_t __TBB_EXPORTED_FUNC cache_line_size();
}

namespace d1 {

template<typename T>
class cache_aligned_allocator {
public:
    using value_type = T;
    using propagate_on_container_move_assignment = std::true_type;

    //! Always defined for TBB containers (supported since C++17 for std containers)
    using is_always_equal = std::true_type;

    cache_aligned_allocator() = default;
    template<typename U> cache_aligned_allocator(const cache_aligned_allocator<U>&) noexcept {}

    //! Allocate space for n objects, starting on a cache/sector line.
    __TBB_nodiscard T* allocate(std::size_t n) {
        return static_cast<T*>(r1::cache_aligned_allocate(n * sizeof(value_type)));
    }

    //! Free block of memory that starts on a cache line
    void deallocate(T* p, std::size_t) {
        r1::cache_aligned_deallocate(p);
    }

    //! Largest value for which method allocate might succeed.
    std::size_t max_size() const noexcept {
        return (~std::size_t(0) - r1::cache_line_size()) / sizeof(value_type);
    }

#if TBB_ALLOCATOR_TRAITS_BROKEN
    using pointer = value_type*;
    using const_pointer = const value_type*;
    using reference = value_type&;
    using const_reference = const value_type&;
    using difference_type = std::ptrdiff_t;
    using size_type = std::size_t;
    template<typename U> struct rebind {
        using other = cache_aligned_allocator<U>;
    };
    template<typename U, typename... Args>
    void construct(U *p, Args&&... args)
        { ::new (p) U(std::forward<Args>(args)...); }
    void destroy(pointer p) { p->~value_type(); }
    pointer address(reference x) const { return &x; }
    const_pointer address(const_reference x) const { return &x; }
#endif // TBB_ALLOCATOR_TRAITS_BROKEN
};

#if TBB_ALLOCATOR_TRAITS_BROKEN
    template<>
    class cache_aligned_allocator<void> {
    public:
        using pointer = void*;
        using const_pointer = const void*;
        using value_type = void;
        template<typename U> struct rebind {
            using other = cache_aligned_allocator<U>;
        };
    };
#endif

template<typename T, typename U>
bool operator==(const cache_aligned_allocator<T>&, const cache_aligned_allocator<U>&) noexcept { return true; }

#if !__TBB_CPP20_COMPARISONS_PRESENT
template<typename T, typename U>
bool operator!=(const cache_aligned_allocator<T>&, const cache_aligned_allocator<U>&) noexcept { return false; }
#endif

#if __TBB_CPP17_MEMORY_RESOURCE_PRESENT

//! C++17 memory resource wrapper to ensure cache line size alignment
class cache_aligned_resource : public std::pmr::memory_resource {
public:
    cache_aligned_resource() : cache_aligned_resource(std::pmr::get_default_resource()) {}
    explicit cache_aligned_resource(std::pmr::memory_resource* upstream) : m_upstream(upstream) {}

    std::pmr::memory_resource* upstream_resource() const {
        return m_upstream;
    }

private:
    //! We don't know what memory resource set. Use padding to guarantee alignment
    void* do_allocate(std::size_t bytes, std::size_t alignment) override {
        // TODO: make it common with tbb_allocator.cpp
        std::size_t cache_line_alignment = correct_alignment(alignment);
        std::size_t space = correct_size(bytes) + cache_line_alignment;
        std::uintptr_t base = reinterpret_cast<std::uintptr_t>(m_upstream->allocate(space));
        __TBB_ASSERT(base != 0, "Upstream resource returned NULL.");

        // Round up to the next cache line (align the base address)
        std::uintptr_t result = (base + cache_line_alignment) & ~(cache_line_alignment - 1);
        __TBB_ASSERT((result - base) >= sizeof(std::uintptr_t), "Can`t store a base pointer to the header");
        __TBB_ASSERT(space - (result - base) >= bytes, "Not enough space for the storage");

        // Record where block actually starts.
        (reinterpret_cast<std::uintptr_t*>(result))[-1] = base;
        return reinterpret_cast<void*>(result);
    }

    void do_deallocate(void* ptr, std::size_t bytes, std::size_t alignment) override {
        if (ptr) {
            // Recover where block actually starts
            std::uintptr_t base = (reinterpret_cast<std::uintptr_t*>(ptr))[-1];
            m_upstream->deallocate(reinterpret_cast<void*>(base), correct_size(bytes) + correct_alignment(alignment));
        }
    }

    bool do_is_equal(const std::pmr::memory_resource& other) const noexcept override {
        if (this == &other) { return true; }
#if __TBB_USE_OPTIONAL_RTTI
        const cache_aligned_resource* other_res = dynamic_cast<const cache_aligned_resource*>(&other);
        return other_res && (upstream_resource() == other_res->upstream_resource());
#else
        return false;
#endif
    }

    std::size_t correct_alignment(std::size_t alignment) {
        __TBB_ASSERT(tbb::detail::is_power_of_two(alignment), "Alignment is not a power of 2");
#if __TBB_CPP17_HW_INTERFERENCE_SIZE_PRESENT
        std::size_t cache_line_size = std::hardware_destructive_interference_size;
#else
        std::size_t cache_line_size = r1::cache_line_size();
#endif
        return alignment < cache_line_size ? cache_line_size : alignment;
    }

    std::size_t correct_size(std::size_t bytes) {
        // To handle the case, when small size requested. There could be not
        // enough space to store the original pointer.
        return bytes < sizeof(std::uintptr_t) ? sizeof(std::uintptr_t) : bytes;
    }

    std::pmr::memory_resource* m_upstream;
};

#endif // __TBB_CPP17_MEMORY_RESOURCE_PRESENT

} // namespace d1
} // namespace detail

inline namespace v1 {
using detail::d1::cache_aligned_allocator;
#if __TBB_CPP17_MEMORY_RESOURCE_PRESENT
using detail::d1::cache_aligned_resource;
#endif
} // namespace v1
} // namespace tbb

#endif /* __TBB_cache_aligned_allocator_H */

