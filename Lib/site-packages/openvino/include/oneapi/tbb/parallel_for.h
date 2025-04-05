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

#ifndef __TBB_parallel_for_H
#define __TBB_parallel_for_H

#include "detail/_config.h"
#include "detail/_namespace_injection.h"
#include "detail/_exception.h"
#include "detail/_task.h"
#include "detail/_small_object_pool.h"
#include "profiling.h"

#include "partitioner.h"
#include "blocked_range.h"
#include "task_group.h"

#include <cstddef>
#include <new>

namespace tbb {
namespace detail {
namespace d1 {

//! Task type used in parallel_for
/** @ingroup algorithms */
template<typename Range, typename Body, typename Partitioner>
struct start_for : public task {
    Range my_range;
    const Body my_body;
    node* my_parent;

    typename Partitioner::task_partition_type my_partition;
    small_object_allocator my_allocator;

    task* execute(execution_data&) override;
    task* cancel(execution_data&) override;
    void finalize(const execution_data&);

    //! Constructor for root task.
    start_for( const Range& range, const Body& body, Partitioner& partitioner, small_object_allocator& alloc ) :
        my_range(range),
        my_body(body),
        my_partition(partitioner),
        my_allocator(alloc) {}
    //! Splitting constructor used to generate children.
    /** parent_ becomes left child.  Newly constructed object is right child. */
    start_for( start_for& parent_, typename Partitioner::split_type& split_obj, small_object_allocator& alloc ) :
        my_range(parent_.my_range, get_range_split_object<Range>(split_obj)),
        my_body(parent_.my_body),
        my_partition(parent_.my_partition, split_obj),
        my_allocator(alloc) {}
    //! Construct right child from the given range as response to the demand.
    /** parent_ remains left child.  Newly constructed object is right child. */
    start_for( start_for& parent_, const Range& r, depth_t d, small_object_allocator& alloc ) :
        my_range(r),
        my_body(parent_.my_body),
        my_partition(parent_.my_partition, split()),
        my_allocator(alloc)
    {
        my_partition.align_depth( d );
    }
    static void run(const Range& range, const Body& body, Partitioner& partitioner) {
        task_group_context context(PARALLEL_FOR);
        run(range, body, partitioner, context);
    }

    static void run(const Range& range, const Body& body, Partitioner& partitioner, task_group_context& context) {
        if ( !range.empty() ) {
            small_object_allocator alloc{};
            start_for& for_task = *alloc.new_object<start_for>(range, body, partitioner, alloc);

            // defer creation of the wait node until task allocation succeeds
            wait_node wn;
            for_task.my_parent = &wn;
            execute_and_wait(for_task, context, wn.m_wait, context);
        }
    }
    //! Run body for range, serves as callback for partitioner
    void run_body( Range &r ) {
        my_body( r );
    }

    //! spawn right task, serves as callback for partitioner
    void offer_work(typename Partitioner::split_type& split_obj, execution_data& ed) {
       offer_work_impl(ed, *this, split_obj);
    }

    //! spawn right task, serves as callback for partitioner
    void offer_work(const Range& r, depth_t d, execution_data& ed) {
        offer_work_impl(ed, *this, r, d);
    }

private:
    template <typename... Args>
    void offer_work_impl(execution_data& ed, Args&&... constructor_args) {
        // New right child
        small_object_allocator alloc{};
        start_for& right_child = *alloc.new_object<start_for>(ed, std::forward<Args>(constructor_args)..., alloc);

        // New root node as a continuation and ref count. Left and right child attach to the new parent.
        right_child.my_parent = my_parent = alloc.new_object<tree_node>(ed, my_parent, 2, alloc);
        // Spawn the right sibling
        right_child.spawn_self(ed);
    }

    void spawn_self(execution_data& ed) {
        my_partition.spawn_task(*this, *context(ed));
    }
};

//! fold the tree and deallocate the task
template<typename Range, typename Body, typename Partitioner>
void start_for<Range, Body, Partitioner>::finalize(const execution_data& ed) {
    // Get the current parent and allocator an object destruction
    node* parent = my_parent;
    auto allocator = my_allocator;
    // Task execution finished - destroy it
    this->~start_for();
    // Unwind the tree decrementing the parent`s reference count

    fold_tree<tree_node>(parent, ed);
    allocator.deallocate(this, ed);

}

//! execute task for parallel_for
template<typename Range, typename Body, typename Partitioner>
task* start_for<Range, Body, Partitioner>::execute(execution_data& ed) {
    if (!is_same_affinity(ed)) {
        my_partition.note_affinity(execution_slot(ed));
    }
    my_partition.check_being_stolen(*this, ed);
    my_partition.execute(*this, my_range, ed);
    finalize(ed);
    return nullptr;
}

//! cancel task for parallel_for
template<typename Range, typename Body, typename Partitioner>
task* start_for<Range, Body, Partitioner>::cancel(execution_data& ed) {
    finalize(ed);
    return nullptr;
}

//! Calls the function with values from range [begin, end) with a step provided
template<typename Function, typename Index>
class parallel_for_body : detail::no_assign {
    const Function &my_func;
    const Index my_begin;
    const Index my_step;
public:
    parallel_for_body( const Function& _func, Index& _begin, Index& _step )
        : my_func(_func), my_begin(_begin), my_step(_step) {}

    void operator()( const blocked_range<Index>& r ) const {
        // A set of local variables to help the compiler with vectorization of the following loop.
        Index b = r.begin();
        Index e = r.end();
        Index ms = my_step;
        Index k = my_begin + b*ms;

#if __INTEL_COMPILER
#pragma ivdep
#if __TBB_ASSERT_ON_VECTORIZATION_FAILURE
#pragma vector always assert
#endif
#endif
        for ( Index i = b; i < e; ++i, k += ms ) {
            my_func( k );
        }
    }
};

// Requirements on Range concept are documented in blocked_range.h

/** \page parallel_for_body_req Requirements on parallel_for body
    Class \c Body implementing the concept of parallel_for body must define:
    - \code Body::Body( const Body& ); \endcode                 Copy constructor
    - \code Body::~Body(); \endcode                             Destructor
    - \code void Body::operator()( Range& r ) const; \endcode   Function call operator applying the body to range \c r.
**/

/** \name parallel_for
    See also requirements on \ref range_req "Range" and \ref parallel_for_body_req "parallel_for Body". **/
//@{

//! Parallel iteration over range with default partitioner.
/** @ingroup algorithms **/
template<typename Range, typename Body>
void parallel_for( const Range& range, const Body& body ) {
    start_for<Range,Body,const __TBB_DEFAULT_PARTITIONER>::run(range,body,__TBB_DEFAULT_PARTITIONER());
}

//! Parallel iteration over range with simple partitioner.
/** @ingroup algorithms **/
template<typename Range, typename Body>
void parallel_for( const Range& range, const Body& body, const simple_partitioner& partitioner ) {
    start_for<Range,Body,const simple_partitioner>::run(range,body,partitioner);
}

//! Parallel iteration over range with auto_partitioner.
/** @ingroup algorithms **/
template<typename Range, typename Body>
void parallel_for( const Range& range, const Body& body, const auto_partitioner& partitioner ) {
    start_for<Range,Body,const auto_partitioner>::run(range,body,partitioner);
}

//! Parallel iteration over range with static_partitioner.
/** @ingroup algorithms **/
template<typename Range, typename Body>
void parallel_for( const Range& range, const Body& body, const static_partitioner& partitioner ) {
    start_for<Range,Body,const static_partitioner>::run(range,body,partitioner);
}

//! Parallel iteration over range with affinity_partitioner.
/** @ingroup algorithms **/
template<typename Range, typename Body>
void parallel_for( const Range& range, const Body& body, affinity_partitioner& partitioner ) {
    start_for<Range,Body,affinity_partitioner>::run(range,body,partitioner);
}

//! Parallel iteration over range with default partitioner and user-supplied context.
/** @ingroup algorithms **/
template<typename Range, typename Body>
void parallel_for( const Range& range, const Body& body, task_group_context& context ) {
    start_for<Range,Body,const __TBB_DEFAULT_PARTITIONER>::run(range, body, __TBB_DEFAULT_PARTITIONER(), context);
}

//! Parallel iteration over range with simple partitioner and user-supplied context.
/** @ingroup algorithms **/
template<typename Range, typename Body>
void parallel_for( const Range& range, const Body& body, const simple_partitioner& partitioner, task_group_context& context ) {
    start_for<Range,Body,const simple_partitioner>::run(range, body, partitioner, context);
}

//! Parallel iteration over range with auto_partitioner and user-supplied context.
/** @ingroup algorithms **/
template<typename Range, typename Body>
void parallel_for( const Range& range, const Body& body, const auto_partitioner& partitioner, task_group_context& context ) {
    start_for<Range,Body,const auto_partitioner>::run(range, body, partitioner, context);
}

//! Parallel iteration over range with static_partitioner and user-supplied context.
/** @ingroup algorithms **/
template<typename Range, typename Body>
void parallel_for( const Range& range, const Body& body, const static_partitioner& partitioner, task_group_context& context ) {
    start_for<Range,Body,const static_partitioner>::run(range, body, partitioner, context);
}

//! Parallel iteration over range with affinity_partitioner and user-supplied context.
/** @ingroup algorithms **/
template<typename Range, typename Body>
void parallel_for( const Range& range, const Body& body, affinity_partitioner& partitioner, task_group_context& context ) {
    start_for<Range,Body,affinity_partitioner>::run(range,body,partitioner, context);
}

//! Implementation of parallel iteration over stepped range of integers with explicit step and partitioner
template <typename Index, typename Function, typename Partitioner>
void parallel_for_impl(Index first, Index last, Index step, const Function& f, Partitioner& partitioner) {
    if (step <= 0 )
        throw_exception(exception_id::nonpositive_step); // throws std::invalid_argument
    else if (last > first) {
        // Above "else" avoids "potential divide by zero" warning on some platforms
        Index end = (last - first - Index(1)) / step + Index(1);
        blocked_range<Index> range(static_cast<Index>(0), end);
        parallel_for_body<Function, Index> body(f, first, step);
        parallel_for(range, body, partitioner);
    }
}

//! Parallel iteration over a range of integers with a step provided and default partitioner
template <typename Index, typename Function>
void parallel_for(Index first, Index last, Index step, const Function& f) {
    parallel_for_impl<Index,Function,const auto_partitioner>(first, last, step, f, auto_partitioner());
}
//! Parallel iteration over a range of integers with a step provided and simple partitioner
template <typename Index, typename Function>
void parallel_for(Index first, Index last, Index step, const Function& f, const simple_partitioner& partitioner) {
    parallel_for_impl<Index,Function,const simple_partitioner>(first, last, step, f, partitioner);
}
//! Parallel iteration over a range of integers with a step provided and auto partitioner
template <typename Index, typename Function>
void parallel_for(Index first, Index last, Index step, const Function& f, const auto_partitioner& partitioner) {
    parallel_for_impl<Index,Function,const auto_partitioner>(first, last, step, f, partitioner);
}
//! Parallel iteration over a range of integers with a step provided and static partitioner
template <typename Index, typename Function>
void parallel_for(Index first, Index last, Index step, const Function& f, const static_partitioner& partitioner) {
    parallel_for_impl<Index,Function,const static_partitioner>(first, last, step, f, partitioner);
}
//! Parallel iteration over a range of integers with a step provided and affinity partitioner
template <typename Index, typename Function>
void parallel_for(Index first, Index last, Index step, const Function& f, affinity_partitioner& partitioner) {
    parallel_for_impl(first, last, step, f, partitioner);
}

//! Parallel iteration over a range of integers with a default step value and default partitioner
template <typename Index, typename Function>
void parallel_for(Index first, Index last, const Function& f) {
    parallel_for_impl<Index,Function,const auto_partitioner>(first, last, static_cast<Index>(1), f, auto_partitioner());
}
//! Parallel iteration over a range of integers with a default step value and simple partitioner
template <typename Index, typename Function>
void parallel_for(Index first, Index last, const Function& f, const simple_partitioner& partitioner) {
    parallel_for_impl<Index,Function,const simple_partitioner>(first, last, static_cast<Index>(1), f, partitioner);
}
//! Parallel iteration over a range of integers with a default step value and auto partitioner
template <typename Index, typename Function>
void parallel_for(Index first, Index last, const Function& f, const auto_partitioner& partitioner) {
    parallel_for_impl<Index,Function,const auto_partitioner>(first, last, static_cast<Index>(1), f, partitioner);
}
//! Parallel iteration over a range of integers with a default step value and static partitioner
template <typename Index, typename Function>
void parallel_for(Index first, Index last, const Function& f, const static_partitioner& partitioner) {
    parallel_for_impl<Index,Function,const static_partitioner>(first, last, static_cast<Index>(1), f, partitioner);
}
//! Parallel iteration over a range of integers with a default step value and affinity partitioner
template <typename Index, typename Function>
void parallel_for(Index first, Index last, const Function& f, affinity_partitioner& partitioner) {
    parallel_for_impl(first, last, static_cast<Index>(1), f, partitioner);
}

//! Implementation of parallel iteration over stepped range of integers with explicit step, task group context, and partitioner
template <typename Index, typename Function, typename Partitioner>
void parallel_for_impl(Index first, Index last, Index step, const Function& f, Partitioner& partitioner, task_group_context &context) {
    if (step <= 0 )
        throw_exception(exception_id::nonpositive_step); // throws std::invalid_argument
    else if (last > first) {
        // Above "else" avoids "potential divide by zero" warning on some platforms
        Index end = (last - first - Index(1)) / step + Index(1);
        blocked_range<Index> range(static_cast<Index>(0), end);
        parallel_for_body<Function, Index> body(f, first, step);
        parallel_for(range, body, partitioner, context);
    }
}

//! Parallel iteration over a range of integers with explicit step, task group context, and default partitioner
template <typename Index, typename Function>
void parallel_for(Index first, Index last, Index step, const Function& f, task_group_context &context) {
    parallel_for_impl<Index,Function,const auto_partitioner>(first, last, step, f, auto_partitioner(), context);
}
//! Parallel iteration over a range of integers with explicit step, task group context, and simple partitioner
template <typename Index, typename Function>
void parallel_for(Index first, Index last, Index step, const Function& f, const simple_partitioner& partitioner, task_group_context &context) {
    parallel_for_impl<Index,Function,const simple_partitioner>(first, last, step, f, partitioner, context);
}
//! Parallel iteration over a range of integers with explicit step, task group context, and auto partitioner
template <typename Index, typename Function>
void parallel_for(Index first, Index last, Index step, const Function& f, const auto_partitioner& partitioner, task_group_context &context) {
    parallel_for_impl<Index,Function,const auto_partitioner>(first, last, step, f, partitioner, context);
}
//! Parallel iteration over a range of integers with explicit step, task group context, and static partitioner
template <typename Index, typename Function>
void parallel_for(Index first, Index last, Index step, const Function& f, const static_partitioner& partitioner, task_group_context &context) {
    parallel_for_impl<Index,Function,const static_partitioner>(first, last, step, f, partitioner, context);
}
//! Parallel iteration over a range of integers with explicit step, task group context, and affinity partitioner
template <typename Index, typename Function>
void parallel_for(Index first, Index last, Index step, const Function& f, affinity_partitioner& partitioner, task_group_context &context) {
    parallel_for_impl(first, last, step, f, partitioner, context);
}

//! Parallel iteration over a range of integers with a default step value, explicit task group context, and default partitioner
template <typename Index, typename Function>
void parallel_for(Index first, Index last, const Function& f, task_group_context &context) {
    parallel_for_impl<Index,Function,const auto_partitioner>(first, last, static_cast<Index>(1), f, auto_partitioner(), context);
}
//! Parallel iteration over a range of integers with a default step value, explicit task group context, and simple partitioner
template <typename Index, typename Function>
void parallel_for(Index first, Index last, const Function& f, const simple_partitioner& partitioner, task_group_context &context) {
    parallel_for_impl<Index,Function,const simple_partitioner>(first, last, static_cast<Index>(1), f, partitioner, context);
}
//! Parallel iteration over a range of integers with a default step value, explicit task group context, and auto partitioner
template <typename Index, typename Function>
void parallel_for(Index first, Index last, const Function& f, const auto_partitioner& partitioner, task_group_context &context) {
    parallel_for_impl<Index,Function,const auto_partitioner>(first, last, static_cast<Index>(1), f, partitioner, context);
}
//! Parallel iteration over a range of integers with a default step value, explicit task group context, and static partitioner
template <typename Index, typename Function>
void parallel_for(Index first, Index last, const Function& f, const static_partitioner& partitioner, task_group_context &context) {
    parallel_for_impl<Index,Function,const static_partitioner>(first, last, static_cast<Index>(1), f, partitioner, context);
}
//! Parallel iteration over a range of integers with a default step value, explicit task group context, and affinity_partitioner
template <typename Index, typename Function>
void parallel_for(Index first, Index last, const Function& f, affinity_partitioner& partitioner, task_group_context &context) {
    parallel_for_impl(first, last, static_cast<Index>(1), f, partitioner, context);
}
// @}

} // namespace d1
} // namespace detail

inline namespace v1 {
using detail::d1::parallel_for;
// Split types
using detail::split;
using detail::proportional_split;
} // namespace v1

} // namespace tbb

#endif /* __TBB_parallel_for_H */
