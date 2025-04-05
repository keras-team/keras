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

#ifndef __TBB_scalable_allocator_H
#define __TBB_scalable_allocator_H

#ifdef __cplusplus
#include "oneapi/tbb/detail/_config.h"
#include "oneapi/tbb/detail/_utils.h"
#include <cstdlib>
#include <utility>
#else
#include <stddef.h> /* Need ptrdiff_t and size_t from here. */
#if !_MSC_VER
#include <stdint.h> /* Need intptr_t from here. */
#endif
#endif

#if __TBB_CPP17_MEMORY_RESOURCE_PRESENT
#include <memory_resource>
#endif

#ifdef __cplusplus
extern "C" {
#endif /* __cplusplus */

#if _MSC_VER
    #define __TBB_EXPORTED_FUNC __cdecl
#else
    #define __TBB_EXPORTED_FUNC
#endif

/** The "malloc" analogue to allocate block of memory of size bytes.
  * @ingroup memory_allocation */
void* __TBB_EXPORTED_FUNC scalable_malloc(size_t size);

/** The "free" analogue to discard a previously allocated piece of memory.
    @ingroup memory_allocation */
void   __TBB_EXPORTED_FUNC scalable_free(void* ptr);

/** The "realloc" analogue complementing scalable_malloc.
    @ingroup memory_allocation */
void* __TBB_EXPORTED_FUNC scalable_realloc(void* ptr, size_t size);

/** The "calloc" analogue complementing scalable_malloc.
    @ingroup memory_allocation */
void* __TBB_EXPORTED_FUNC scalable_calloc(size_t nobj, size_t size);

/** The "posix_memalign" analogue.
    @ingroup memory_allocation */
int __TBB_EXPORTED_FUNC scalable_posix_memalign(void** memptr, size_t alignment, size_t size);

/** The "_aligned_malloc" analogue.
    @ingroup memory_allocation */
void* __TBB_EXPORTED_FUNC scalable_aligned_malloc(size_t size, size_t alignment);

/** The "_aligned_realloc" analogue.
    @ingroup memory_allocation */
void* __TBB_EXPORTED_FUNC scalable_aligned_realloc(void* ptr, size_t size, size_t alignment);

/** The "_aligned_free" analogue.
    @ingroup memory_allocation */
void __TBB_EXPORTED_FUNC scalable_aligned_free(void* ptr);

/** The analogue of _msize/malloc_size/malloc_usable_size.
    Returns the usable size of a memory block previously allocated by scalable_*,
    or 0 (zero) if ptr does not point to such a block.
    @ingroup memory_allocation */
size_t __TBB_EXPORTED_FUNC scalable_msize(void* ptr);

/* Results for scalable_allocation_* functions */
typedef enum {
    TBBMALLOC_OK,
    TBBMALLOC_INVALID_PARAM,
    TBBMALLOC_UNSUPPORTED,
    TBBMALLOC_NO_MEMORY,
    TBBMALLOC_NO_EFFECT
} ScalableAllocationResult;

/* Setting TBB_MALLOC_USE_HUGE_PAGES environment variable to 1 enables huge pages.
   scalable_allocation_mode call has priority over environment variable. */
typedef enum {
    TBBMALLOC_USE_HUGE_PAGES,  /* value turns using huge pages on and off */
    /* deprecated, kept for backward compatibility only */
    USE_HUGE_PAGES = TBBMALLOC_USE_HUGE_PAGES,
    /* try to limit memory consumption value (Bytes), clean internal buffers
       if limit is exceeded, but not prevents from requesting memory from OS */
    TBBMALLOC_SET_SOFT_HEAP_LIMIT,
    /* Lower bound for the size (Bytes), that is interpreted as huge
     * and not released during regular cleanup operations. */
    TBBMALLOC_SET_HUGE_SIZE_THRESHOLD
} AllocationModeParam;

/** Set TBB allocator-specific allocation modes.
    @ingroup memory_allocation */
int __TBB_EXPORTED_FUNC scalable_allocation_mode(int param, intptr_t value);

typedef enum {
    /* Clean internal allocator buffers for all threads.
       Returns TBBMALLOC_NO_EFFECT if no buffers cleaned,
       TBBMALLOC_OK if some memory released from buffers. */
    TBBMALLOC_CLEAN_ALL_BUFFERS,
    /* Clean internal allocator buffer for current thread only.
       Return values same as for TBBMALLOC_CLEAN_ALL_BUFFERS. */
    TBBMALLOC_CLEAN_THREAD_BUFFERS
} ScalableAllocationCmd;

/** Call TBB allocator-specific commands.
    @ingroup memory_allocation */
int __TBB_EXPORTED_FUNC scalable_allocation_command(int cmd, void *param);

#ifdef __cplusplus
} /* extern "C" */
#endif /* __cplusplus */

#ifdef __cplusplus

//! The namespace rml contains components of low-level memory pool interface.
namespace rml {
class MemoryPool;

typedef void *(*rawAllocType)(std::intptr_t pool_id, std::size_t &bytes);
// returns non-zero in case of error
typedef int   (*rawFreeType)(std::intptr_t pool_id, void* raw_ptr, std::size_t raw_bytes);

struct MemPoolPolicy {
    enum {
        TBBMALLOC_POOL_VERSION = 1
    };

    rawAllocType pAlloc;
    rawFreeType  pFree;
                 // granularity of pAlloc allocations. 0 means default used.
    std::size_t  granularity;
    int          version;
                 // all memory consumed at 1st pAlloc call and never returned,
                 // no more pAlloc calls after 1st
    unsigned     fixedPool : 1,
                 // memory consumed but returned only at pool termination
                 keepAllMemory : 1,
                 reserved : 30;

    MemPoolPolicy(rawAllocType pAlloc_, rawFreeType pFree_,
                  std::size_t granularity_ = 0, bool fixedPool_ = false,
                  bool keepAllMemory_ = false) :
        pAlloc(pAlloc_), pFree(pFree_), granularity(granularity_), version(TBBMALLOC_POOL_VERSION),
        fixedPool(fixedPool_), keepAllMemory(keepAllMemory_),
        reserved(0) {}
};

// enums have same values as appropriate enums from ScalableAllocationResult
// TODO: use ScalableAllocationResult in pool_create directly
enum MemPoolError {
    // pool created successfully
    POOL_OK = TBBMALLOC_OK,
    // invalid policy parameters found
    INVALID_POLICY = TBBMALLOC_INVALID_PARAM,
     // requested pool policy is not supported by allocator library
    UNSUPPORTED_POLICY = TBBMALLOC_UNSUPPORTED,
    // lack of memory during pool creation
    NO_MEMORY = TBBMALLOC_NO_MEMORY,
    // action takes no effect
    NO_EFFECT = TBBMALLOC_NO_EFFECT
};

MemPoolError pool_create_v1(std::intptr_t pool_id, const MemPoolPolicy *policy,
                            rml::MemoryPool **pool);

bool  pool_destroy(MemoryPool* memPool);
void *pool_malloc(MemoryPool* memPool, std::size_t size);
void *pool_realloc(MemoryPool* memPool, void *object, std::size_t size);
void *pool_aligned_malloc(MemoryPool* mPool, std::size_t size, std::size_t alignment);
void *pool_aligned_realloc(MemoryPool* mPool, void *ptr, std::size_t size, std::size_t alignment);
bool  pool_reset(MemoryPool* memPool);
bool  pool_free(MemoryPool *memPool, void *object);
MemoryPool *pool_identify(void *object);
std::size_t pool_msize(MemoryPool *memPool, void *object);

} // namespace rml

namespace tbb {
namespace detail {
namespace d1 {

// keep throw in a separate function to prevent code bloat
template<typename E>
void throw_exception(const E &e) {
#if TBB_USE_EXCEPTIONS
    throw e;
#else
    suppress_unused_warning(e);
#endif
}

template<typename T>
class scalable_allocator {
public:
    using value_type = T;
    using propagate_on_container_move_assignment = std::true_type;

    //! Always defined for TBB containers
    using is_always_equal = std::true_type;

    scalable_allocator() = default;
    template<typename U> scalable_allocator(const scalable_allocator<U>&) noexcept {}

    //! Allocate space for n objects.
    __TBB_nodiscard T* allocate(std::size_t n) {
        T* p = static_cast<T*>(scalable_malloc(n * sizeof(value_type)));
        if (!p) {
            throw_exception(std::bad_alloc());
        }
        return p;
    }

    //! Free previously allocated block of memory
    void deallocate(T* p, std::size_t) {
        scalable_free(p);
    }

#if TBB_ALLOCATOR_TRAITS_BROKEN
    using pointer = value_type*;
    using const_pointer = const value_type*;
    using reference = value_type&;
    using const_reference = const value_type&;
    using difference_type = std::ptrdiff_t;
    using size_type = std::size_t;
    template<typename U> struct rebind {
        using other = scalable_allocator<U>;
    };
    //! Largest value for which method allocate might succeed.
    size_type max_size() const noexcept {
        size_type absolutemax = static_cast<size_type>(-1) / sizeof (value_type);
        return (absolutemax > 0 ? absolutemax : 1);
    }
    template<typename U, typename... Args>
    void construct(U *p, Args&&... args)
        { ::new((void *)p) U(std::forward<Args>(args)...); }
    void destroy(pointer p) { p->~value_type(); }
    pointer address(reference x) const { return &x; }
    const_pointer address(const_reference x) const { return &x; }
#endif // TBB_ALLOCATOR_TRAITS_BROKEN

};

#if TBB_ALLOCATOR_TRAITS_BROKEN
    template<>
    class scalable_allocator<void> {
    public:
        using pointer = void*;
        using const_pointer = const void*;
        using value_type = void;
        template<typename U> struct rebind {
            using other = scalable_allocator<U>;
        };
    };
#endif

template<typename T, typename U>
inline bool operator==(const scalable_allocator<T>&, const scalable_allocator<U>&) noexcept { return true; }

#if !__TBB_CPP20_COMPARISONS_PRESENT
template<typename T, typename U>
inline bool operator!=(const scalable_allocator<T>&, const scalable_allocator<U>&) noexcept { return false; }
#endif

#if __TBB_CPP17_MEMORY_RESOURCE_PRESENT

//! C++17 memory resource implementation for scalable allocator
//! ISO C++ Section 23.12.2
class scalable_resource_impl : public std::pmr::memory_resource {
private:
    void* do_allocate(std::size_t bytes, std::size_t alignment) override {
        void* p = scalable_aligned_malloc(bytes, alignment);
        if (!p) {
            throw_exception(std::bad_alloc());
        }
        return p;
    }

    void do_deallocate(void* ptr, std::size_t /*bytes*/, std::size_t /*alignment*/) override {
        scalable_free(ptr);
    }

    //! Memory allocated by one instance of scalable_resource_impl could be deallocated by any
    //! other instance of this class
    bool do_is_equal(const std::pmr::memory_resource& other) const noexcept override {
        return this == &other ||
#if __TBB_USE_OPTIONAL_RTTI
            dynamic_cast<const scalable_resource_impl*>(&other) != nullptr;
#else
            false;
#endif
    }
};

//! Global scalable allocator memory resource provider
inline std::pmr::memory_resource* scalable_memory_resource() noexcept {
    static tbb::detail::d1::scalable_resource_impl scalable_res;
    return &scalable_res;
}

#endif // __TBB_CPP17_MEMORY_RESOURCE_PRESENT

} // namespace d1
} // namespace detail

inline namespace v1 {
using detail::d1::scalable_allocator;
#if __TBB_CPP17_MEMORY_RESOURCE_PRESENT
using detail::d1::scalable_memory_resource;
#endif
} // namespace v1

} // namespace tbb

#endif /* __cplusplus */

#endif /* __TBB_scalable_allocator_H */
