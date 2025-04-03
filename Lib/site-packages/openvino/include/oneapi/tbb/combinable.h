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

#ifndef __TBB_combinable_H
#define __TBB_combinable_H

#include "detail/_namespace_injection.h"

#include "enumerable_thread_specific.h"
#include "cache_aligned_allocator.h"

namespace tbb {
namespace detail {
namespace d1 {
/** \name combinable **/
//@{
//! Thread-local storage with optional reduction
/** @ingroup containers */
template <typename T>
class combinable {
    using my_alloc = typename tbb::cache_aligned_allocator<T>;
    using my_ets_type = typename tbb::enumerable_thread_specific<T, my_alloc, ets_no_key>;
    my_ets_type my_ets;

public:
    combinable() = default;

    template <typename Finit>
    explicit combinable(Finit _finit) : my_ets(_finit) { }

    void clear() { my_ets.clear(); }

    T& local() { return my_ets.local(); }

    T& local(bool& exists) { return my_ets.local(exists); }

    // combine_func_t has signature T(T,T) or T(const T&, const T&)
    template <typename CombineFunc>
    T combine(CombineFunc f_combine) { return my_ets.combine(f_combine); }

    // combine_func_t has signature void(T) or void(const T&)
    template <typename CombineFunc>
    void combine_each(CombineFunc f_combine) { my_ets.combine_each(f_combine); }
};

} // namespace d1
} // namespace detail

inline namespace v1 {
using detail::d1::combinable;
} // inline namespace v1

} // namespace tbb

#endif /* __TBB_combinable_H */

