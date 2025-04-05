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

#ifndef __TBB_detail__range_common_H
#define __TBB_detail__range_common_H

#include "_config.h"
#include "_utils.h"

namespace tbb {
namespace detail {
inline namespace d0 {

//! Dummy type that distinguishes splitting constructor from copy constructor.
/**
 * See description of parallel_for and parallel_reduce for example usages.
 * @ingroup algorithms
 */
class split {};

//! Type enables transmission of splitting proportion from partitioners to range objects
/**
 * In order to make use of such facility Range objects must implement
 * splitting constructor with this type passed.
 */
class proportional_split : no_assign {
public:
    proportional_split(size_t _left = 1, size_t _right = 1) : my_left(_left), my_right(_right) { }

    size_t left() const { return my_left; }
    size_t right() const { return my_right; }

    // used when range does not support proportional split
    explicit operator split() const { return split(); }

private:
    size_t my_left, my_right;
};

template <typename Range, typename = void>
struct range_split_object_provider {
    template <typename PartitionerSplitType>
    static split get( PartitionerSplitType& ) { return split(); }
};

template <typename Range>
struct range_split_object_provider<Range,
                                   typename std::enable_if<std::is_constructible<Range, Range&, proportional_split&>::value>::type> {
    template <typename PartitionerSplitType>
    static PartitionerSplitType& get( PartitionerSplitType& split_obj ) { return split_obj; }
};

template <typename Range, typename PartitionerSplitType>
auto get_range_split_object( PartitionerSplitType& split_obj )
-> decltype(range_split_object_provider<Range>::get(split_obj)) {
    return range_split_object_provider<Range>::get(split_obj);
}

} // namespace d0
} // namespace detail
} // namespace tbb

#endif // __TBB_detail__range_common_H
