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

#ifndef __TBB_parallel_pipeline_H
#define __TBB_parallel_pipeline_H

#include "detail/_pipeline_filters.h"
#include "detail/_config.h"
#include "detail/_namespace_injection.h"
#include "task_group.h"

#include <cstddef>
#include <atomic>
#include <type_traits>

namespace tbb {
namespace detail {

namespace r1 {
void __TBB_EXPORTED_FUNC parallel_pipeline(task_group_context&, std::size_t, const d1::filter_node&);
}

namespace d1 {

enum class filter_mode : unsigned int
{
    //! processes multiple items in parallel and in no particular order
    parallel = base_filter::filter_is_out_of_order,
    //! processes items one at a time; all such filters process items in the same order
    serial_in_order =  base_filter::filter_is_serial,
    //! processes items one at a time and in no particular order
    serial_out_of_order = base_filter::filter_is_serial | base_filter::filter_is_out_of_order
};
//! Class representing a chain of type-safe pipeline filters
/** @ingroup algorithms */
template<typename InputType, typename OutputType>
class filter {
    filter_node_ptr my_root;
    filter( filter_node_ptr root ) : my_root(root) {}
    friend void parallel_pipeline( size_t, const filter<void,void>&, task_group_context& );
    template<typename T_, typename U_, typename Body>
    friend filter<T_,U_> make_filter( filter_mode, const Body& );
    template<typename T_, typename V_, typename U_>
    friend filter<T_,U_> operator&( const filter<T_,V_>&, const filter<V_,U_>& );
public:
    filter() = default;
    filter( const filter& rhs ) : my_root(rhs.my_root) {}
    filter( filter&& rhs ) : my_root(std::move(rhs.my_root)) {}

    void operator=(const filter& rhs) {
        my_root = rhs.my_root;
    }
    void operator=( filter&& rhs ) {
        my_root = std::move(rhs.my_root);
    }

    template<typename Body>
    filter( filter_mode mode, const Body& body ) :
        my_root( new(r1::allocate_memory(sizeof(filter_node_leaf<InputType, OutputType, Body>)))
                    filter_node_leaf<InputType, OutputType, Body>(static_cast<unsigned int>(mode), body) ) {
    }

    filter& operator&=( const filter<OutputType,OutputType>& right ) {
        *this = *this & right;
        return *this;
    }

    void clear() {
        // Like operator= with filter() on right side.
        my_root = nullptr;
    }
};

//! Create a filter to participate in parallel_pipeline
/** @ingroup algorithms */
template<typename InputType, typename OutputType, typename Body>
filter<InputType, OutputType> make_filter( filter_mode mode, const Body& body ) {
    return filter_node_ptr( new(r1::allocate_memory(sizeof(filter_node_leaf<InputType, OutputType, Body>)))
                                filter_node_leaf<InputType, OutputType, Body>(static_cast<unsigned int>(mode), body) );
}

//! Create a filter to participate in parallel_pipeline
/** @ingroup algorithms */
template<typename Body>
filter<filter_input<Body>, filter_output<Body>> make_filter( filter_mode mode, const Body& body ) {
    return make_filter<filter_input<Body>, filter_output<Body>>(mode, body);
}

//! Composition of filters left and right.
/** @ingroup algorithms */
template<typename T, typename V, typename U>
filter<T,U> operator&( const filter<T,V>& left, const filter<V,U>& right ) {
    __TBB_ASSERT(left.my_root,"cannot use default-constructed filter as left argument of '&'");
    __TBB_ASSERT(right.my_root,"cannot use default-constructed filter as right argument of '&'");
    return filter_node_ptr( new (r1::allocate_memory(sizeof(filter_node))) filter_node(left.my_root,right.my_root) );
}

#if __TBB_CPP17_DEDUCTION_GUIDES_PRESENT
template<typename Body>
filter(filter_mode, Body)
->filter<filter_input<Body>, filter_output<Body>>;
#endif // __TBB_CPP17_DEDUCTION_GUIDES_PRESENT

//! Parallel pipeline over chain of filters with user-supplied context.
/** @ingroup algorithms **/
inline void parallel_pipeline(size_t max_number_of_live_tokens, const filter<void,void>& filter_chain, task_group_context& context) {
    r1::parallel_pipeline(context, max_number_of_live_tokens, *filter_chain.my_root);
}

//! Parallel pipeline over chain of filters.
/** @ingroup algorithms **/
inline void parallel_pipeline(size_t max_number_of_live_tokens, const filter<void,void>& filter_chain) {
    task_group_context context;
    parallel_pipeline(max_number_of_live_tokens, filter_chain, context);
}

//! Parallel pipeline over sequence of filters.
/** @ingroup algorithms **/
template<typename F1, typename F2, typename... FiltersContext>
void parallel_pipeline(size_t max_number_of_live_tokens,
                              const F1& filter1,
                              const F2& filter2,
                              FiltersContext&&... filters) {
    parallel_pipeline(max_number_of_live_tokens, filter1 & filter2, std::forward<FiltersContext>(filters)...);
}

} // namespace d1
} // namespace detail

inline namespace v1
{
using detail::d1::parallel_pipeline;
using detail::d1::filter;
using detail::d1::make_filter;
using detail::d1::filter_mode;
using detail::d1::flow_control;
}
} // tbb

#endif /* __TBB_parallel_pipeline_H */
