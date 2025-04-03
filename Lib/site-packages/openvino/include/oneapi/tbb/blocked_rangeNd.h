/*
    Copyright (c) 2017-2021 Intel Corporation

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

#ifndef __TBB_blocked_rangeNd_H
#define __TBB_blocked_rangeNd_H

#if !TBB_PREVIEW_BLOCKED_RANGE_ND
    #error Set TBB_PREVIEW_BLOCKED_RANGE_ND to include blocked_rangeNd.h
#endif

#include <algorithm>    // std::any_of
#include <array>
#include <cstddef>
#include <type_traits>  // std::is_same, std::enable_if

#include "detail/_config.h"
#include "detail/_template_helpers.h" // index_sequence, make_index_sequence

#include "blocked_range.h"

namespace tbb {
namespace detail {
namespace d1 {

/*
    The blocked_rangeNd_impl uses make_index_sequence<N> to automatically generate a ctor with
    exactly N arguments of the type tbb::blocked_range<Value>. Such ctor provides an opportunity
    to use braced-init-list parameters to initialize each dimension.
    Use of parameters, whose representation is a braced-init-list, but they're not
    std::initializer_list or a reference to one, produces a non-deduced context
    within template argument deduction.

    NOTE: blocked_rangeNd must be exactly a templated alias to the blocked_rangeNd_impl
    (and not e.g. a derived class), otherwise it would need to declare its own ctor
    facing the same problem that the impl class solves.
*/

template<typename Value, unsigned int N, typename = detail::make_index_sequence<N>>
class blocked_rangeNd_impl;

template<typename Value, unsigned int N, std::size_t... Is>
class blocked_rangeNd_impl<Value, N, detail::index_sequence<Is...>> {
public:
    //! Type of a value.
    using value_type = Value;

private:
    //! Helper type to construct range with N tbb::blocked_range<value_type> objects.
    template<std::size_t>
    using dim_type_helper = tbb::blocked_range<value_type>;

public:
    blocked_rangeNd_impl() = delete;

    //! Constructs N-dimensional range over N half-open intervals each represented as tbb::blocked_range<Value>.
    blocked_rangeNd_impl(const dim_type_helper<Is>&... args) : my_dims{ {args...} } {}

    //! Dimensionality of a range.
    static constexpr unsigned int ndims() { return N; }

    //! Range in certain dimension.
    const tbb::blocked_range<value_type>& dim(unsigned int dimension) const {
        __TBB_ASSERT(dimension < N, "out of bound");
        return my_dims[dimension];
    }

    //------------------------------------------------------------------------
    // Methods that implement Range concept
    //------------------------------------------------------------------------

    //! True if at least one dimension is empty.
    bool empty() const {
        return std::any_of(my_dims.begin(), my_dims.end(), [](const tbb::blocked_range<value_type>& d) {
            return d.empty();
        });
    }

    //! True if at least one dimension is divisible.
    bool is_divisible() const {
        return std::any_of(my_dims.begin(), my_dims.end(), [](const tbb::blocked_range<value_type>& d) {
            return d.is_divisible();
        });
    }

    blocked_rangeNd_impl(blocked_rangeNd_impl& r, proportional_split proportion) : my_dims(r.my_dims) {
        do_split(r, proportion);
    }

    blocked_rangeNd_impl(blocked_rangeNd_impl& r, split proportion) : my_dims(r.my_dims) {
        do_split(r, proportion);
    }

private:
    static_assert(N != 0, "zero dimensional blocked_rangeNd can't be constructed");

    //! Ranges in each dimension.
    std::array<tbb::blocked_range<value_type>, N> my_dims;

    template<typename split_type>
    void do_split(blocked_rangeNd_impl& r, split_type proportion) {
        static_assert((std::is_same<split_type, split>::value || std::is_same<split_type, proportional_split>::value), "type of split object is incorrect");
        __TBB_ASSERT(r.is_divisible(), "can't split not divisible range");

        auto my_it = std::max_element(my_dims.begin(), my_dims.end(), [](const tbb::blocked_range<value_type>& first, const tbb::blocked_range<value_type>& second) {
            return (first.size() * second.grainsize() < second.size() * first.grainsize());
        });

        auto r_it = r.my_dims.begin() + (my_it - my_dims.begin());

        my_it->my_begin = tbb::blocked_range<value_type>::do_split(*r_it, proportion);

        // (!(my_it->my_begin < r_it->my_end) && !(r_it->my_end < my_it->my_begin)) equals to
        // (my_it->my_begin == r_it->my_end), but we can't use operator== due to Value concept
        __TBB_ASSERT(!(my_it->my_begin < r_it->my_end) && !(r_it->my_end < my_it->my_begin),
                     "blocked_range has been split incorrectly");
    }
};

template<typename Value, unsigned int N>
using blocked_rangeNd = blocked_rangeNd_impl<Value, N>;

} // namespace d1
} // namespace detail

inline namespace v1 {
using detail::d1::blocked_rangeNd;
} // namespace v1
} // namespace tbb

#endif /* __TBB_blocked_rangeNd_H */

