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

#ifndef __TBB_tbb_allocator_H
#define __TBB_tbb_allocator_H

#include "oneapi/tbb/detail/_utils.h"
#include "detail/_namespace_injection.h"
#include <cstdlib>
#include <utility>

#if __TBB_CPP17_MEMORY_RESOURCE_PRESENT
#include <memory_resource>
#endif

namespace tbb {
namespace detail {

namespace r1 {
void* __TBB_EXPORTED_FUNC allocate_memory(std::size_t size);
void  __TBB_EXPORTED_FUNC deallocate_memory(void* p);
bool  __TBB_EXPORTED_FUNC is_tbbmalloc_used();
}

namespace d1 {

template<typename T>
class tbb_allocator {
public:
    using value_type = T;
    using propagate_on_container_move_assignment = std::true_type;

    //! Always defined for TBB containers (supported since C++17 for std containers)
    using is_always_equal = std::true_type;

    //! Specifies current allocator
    enum malloc_type {
        scalable,
        standard
    };

    tbb_allocator() = default;
    template<typename U> tbb_allocator(const tbb_allocator<U>&) noexcept {}

    //! Allocate space for n objects.
    __TBB_nodiscard T* allocate(std::size_t n) {
        return static_cast<T*>(r1::allocate_memory(n * sizeof(value_type)));
    }

    //! Free previously allocated block of memory.
    void deallocate(T* p, std::size_t) {
        r1::deallocate_memory(p);
    }

    //! Returns current allocator
    static malloc_type allocator_type() {
        return r1::is_tbbmalloc_used() ? standard : scalable;
    }

#if TBB_ALLOCATOR_TRAITS_BROKEN
    using pointer = value_type*;
    using const_pointer = const value_type*;
    using reference = value_type&;
    using const_reference = const value_type&;
    using difference_type = std::ptrdiff_t;
    using size_type = std::size_t;
    template<typename U> struct rebind {
        using other = tbb_allocator<U>;
    };
    //! Largest value for which method allocate might succeed.
    size_type max_size() const noexcept {
        size_type max = ~(std::size_t(0)) / sizeof(value_type);
        return (max > 0 ? max : 1);
    }
    template<typename U, typename... Args>
    void construct(U *p, Args&&... args)
        { ::new (p) U(std::forward<Args>(args)...); }
    void destroy( pointer p ) { p->~value_type(); }
    pointer address(reference x) const { return &x; }
    const_pointer address(const_reference x) const { return &x; }
#endif // TBB_ALLOCATOR_TRAITS_BROKEN
};

#if TBB_ALLOCATOR_TRAITS_BROKEN
    template<>
    class tbb_allocator<void> {
    public:
        using pointer = void*;
        using const_pointer = const void*;
        using value_type = void;
        template<typename U> struct rebind {
            using other = tbb_allocator<U>;
        };
    };
#endif

template<typename T, typename U>
inline bool operator==(const tbb_allocator<T>&, const tbb_allocator<U>&) noexcept { return true; }

#if !__TBB_CPP20_COMPARISONS_PRESENT
template<typename T, typename U>
inline bool operator!=(const tbb_allocator<T>&, const tbb_allocator<U>&) noexcept { return false; }
#endif

} // namespace d1
} // namespace detail

inline namespace v1 {
using detail::d1::tbb_allocator;
} // namespace v1
} // namespace tbb

#endif /* __TBB_tbb_allocator_H */
