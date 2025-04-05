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

#ifndef __TBB_parallel_reduce_H
#define __TBB_parallel_reduce_H

#include <new>
#include "detail/_namespace_injection.h"
#include "detail/_task.h"
#include "detail/_aligned_space.h"
#include "detail/_small_object_pool.h"

#include "task_group.h" // task_group_context
#include "partitioner.h"
#include "profiling.h"

namespace tbb {
namespace detail {
namespace d1 {

//! Tree node type for parallel_reduce.
/** @ingroup algorithms */
//TODO: consider folding tree via bypass execution(instead of manual folding)
// for better cancellation and critical tasks handling (performance measurements required).
template<typename Body>
struct reduction_tree_node : public tree_node {
    tbb::detail::aligned_space<Body> zombie_space;
    Body& left_body;
    bool has_right_zombie{false};

    reduction_tree_node(node* parent, int ref_count, Body& input_left_body, small_object_allocator& alloc) :
        tree_node{parent, ref_count, alloc},
        left_body(input_left_body) /* gcc4.8 bug - braced-initialization doesn't work for class members of reference type */
    {}

    void join(task_group_context* context) {
        if (has_right_zombie && !context->is_group_execution_cancelled())
            left_body.join(*zombie_space.begin());
    }

    ~reduction_tree_node() {
        if( has_right_zombie ) zombie_space.begin()->~Body();
    }
};

//! Task type used to split the work of parallel_reduce.
/** @ingroup algorithms */
template<typename Range, typename Body, typename Partitioner>
struct start_reduce : public task {
    Range my_range;
    Body* my_body;
    node* my_parent;

    typename Partitioner::task_partition_type my_partition;
    small_object_allocator my_allocator;
    bool is_right_child;

    task* execute(execution_data&) override;
    task* cancel(execution_data&) override;
    void finalize(const execution_data&);

    using tree_node_type = reduction_tree_node<Body>;

    //! Constructor reduce root task.
    start_reduce( const Range& range, Body& body, Partitioner& partitioner, small_object_allocator& alloc ) :
        my_range(range),
        my_body(&body),
        my_partition(partitioner),
        my_allocator(alloc),
        is_right_child(false) {}
    //! Splitting constructor used to generate children.
    /** parent_ becomes left child. Newly constructed object is right child. */
    start_reduce( start_reduce& parent_, typename Partitioner::split_type& split_obj, small_object_allocator& alloc ) :
        my_range(parent_.my_range, get_range_split_object<Range>(split_obj)),
        my_body(parent_.my_body),
        my_partition(parent_.my_partition, split_obj),
        my_allocator(alloc),
        is_right_child(true)
    {
        parent_.is_right_child = false;
    }
    //! Construct right child from the given range as response to the demand.
    /** parent_ remains left child. Newly constructed object is right child. */
    start_reduce( start_reduce& parent_, const Range& r, depth_t d, small_object_allocator& alloc ) :
        my_range(r),
        my_body(parent_.my_body),
        my_partition(parent_.my_partition, split()),
        my_allocator(alloc),
        is_right_child(true)
    {
        my_partition.align_depth( d );
        parent_.is_right_child = false;
    }
    static void run(const Range& range, Body& body, Partitioner& partitioner, task_group_context& context) {
        if ( !range.empty() ) {
            wait_node wn;
            small_object_allocator alloc{};
            auto reduce_task = alloc.new_object<start_reduce>(range, body, partitioner, alloc);
            reduce_task->my_parent = &wn;
            execute_and_wait(*reduce_task, context, wn.m_wait, context);
        }
    }
    static void run(const Range& range, Body& body, Partitioner& partitioner) {
        // Bound context prevents exceptions from body to affect nesting or sibling algorithms,
        // and allows users to handle exceptions safely by wrapping parallel_reduce in the try-block.
        task_group_context context(PARALLEL_REDUCE);
        run(range, body, partitioner, context);
    }
    //! Run body for range, serves as callback for partitioner
    void run_body( Range &r ) {
        (*my_body)(r);
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
    void offer_work_impl(execution_data& ed, Args&&... args) {
        small_object_allocator alloc{};
        // New right child
        auto right_child = alloc.new_object<start_reduce>(ed, std::forward<Args>(args)..., alloc);

        // New root node as a continuation and ref count. Left and right child attach to the new parent.
        right_child->my_parent = my_parent = alloc.new_object<tree_node_type>(ed, my_parent, 2, *my_body, alloc);

        // Spawn the right sibling
        right_child->spawn_self(ed);
    }

    void spawn_self(execution_data& ed) {
        my_partition.spawn_task(*this, *context(ed));
    }
};

//! fold the tree and deallocate the task
template<typename Range, typename Body, typename Partitioner>
void start_reduce<Range, Body, Partitioner>::finalize(const execution_data& ed) {
    // Get the current parent and wait object before an object destruction
    node* parent = my_parent;
    auto allocator = my_allocator;
    // Task execution finished - destroy it
    this->~start_reduce();
    // Unwind the tree decrementing the parent`s reference count
    fold_tree<tree_node_type>(parent, ed);
    allocator.deallocate(this, ed);
}

//! Execute parallel_reduce task
template<typename Range, typename Body, typename Partitioner>
task* start_reduce<Range,Body,Partitioner>::execute(execution_data& ed) {
    if (!is_same_affinity(ed)) {
        my_partition.note_affinity(execution_slot(ed));
    }
    my_partition.check_being_stolen(*this, ed);

    // The acquire barrier synchronizes the data pointed with my_body if the left
    // task has already finished.
    if( is_right_child && my_parent->m_ref_count.load(std::memory_order_acquire) == 2 ) {
        tree_node_type* parent_ptr = static_cast<tree_node_type*>(my_parent);
        my_body = (Body*) new( parent_ptr->zombie_space.begin() ) Body(*my_body, split());
        parent_ptr->has_right_zombie = true;
    }
    __TBB_ASSERT(my_body != nullptr, "Incorrect body value");

    my_partition.execute(*this, my_range, ed);

    finalize(ed);
    return nullptr;
}

//! Cancel parallel_reduce task
template<typename Range, typename Body, typename Partitioner>
task* start_reduce<Range, Body, Partitioner>::cancel(execution_data& ed) {
    finalize(ed);
    return nullptr;
}

//! Tree node type for parallel_deterministic_reduce.
/** @ingroup algorithms */
template<typename Body>
struct deterministic_reduction_tree_node : public tree_node {
    Body right_body;
    Body& left_body;

    deterministic_reduction_tree_node(node* parent, int ref_count, Body& input_left_body, small_object_allocator& alloc) :
        tree_node{parent, ref_count, alloc},
        right_body{input_left_body, detail::split()},
        left_body(input_left_body)
    {}

    void join(task_group_context* context) {
        if (!context->is_group_execution_cancelled())
            left_body.join(right_body);
    }
};

//! Task type used to split the work of parallel_deterministic_reduce.
/** @ingroup algorithms */
template<typename Range, typename Body, typename Partitioner>
struct start_deterministic_reduce : public task {
    Range my_range;
    Body& my_body;
    node* my_parent;

    typename Partitioner::task_partition_type my_partition;
    small_object_allocator my_allocator;

    task* execute(execution_data&) override;
    task* cancel(execution_data&) override;
    void finalize(const execution_data&);

    using tree_node_type = deterministic_reduction_tree_node<Body>;

    //! Constructor deterministic_reduce root task.
    start_deterministic_reduce( const Range& range, Partitioner& partitioner, Body& body, small_object_allocator& alloc ) :
        my_range(range),
        my_body(body),
        my_partition(partitioner),
        my_allocator(alloc) {}
    //! Splitting constructor used to generate children.
    /** parent_ becomes left child.  Newly constructed object is right child. */
    start_deterministic_reduce( start_deterministic_reduce& parent_, typename Partitioner::split_type& split_obj, Body& body,
                                small_object_allocator& alloc ) :
        my_range(parent_.my_range, get_range_split_object<Range>(split_obj)),
        my_body(body),
        my_partition(parent_.my_partition, split_obj),
        my_allocator(alloc) {}
    static void run(const Range& range, Body& body, Partitioner& partitioner, task_group_context& context) {
        if ( !range.empty() ) {
            wait_node wn;
            small_object_allocator alloc{};
            auto deterministic_reduce_task =
                alloc.new_object<start_deterministic_reduce>(range, partitioner, body, alloc);
            deterministic_reduce_task->my_parent = &wn;
            execute_and_wait(*deterministic_reduce_task, context, wn.m_wait, context);
        }
    }
    static void run(const Range& range, Body& body, Partitioner& partitioner) {
        // Bound context prevents exceptions from body to affect nesting or sibling algorithms,
        // and allows users to handle exceptions safely by wrapping parallel_deterministic_reduce
        // in the try-block.
        task_group_context context(PARALLEL_REDUCE);
        run(range, body, partitioner, context);
    }
    //! Run body for range, serves as callback for partitioner
    void run_body( Range &r ) {
        my_body( r );
    }
    //! Spawn right task, serves as callback for partitioner
    void offer_work(typename Partitioner::split_type& split_obj, execution_data& ed) {
        offer_work_impl(ed, *this, split_obj);
    }
private:
    template <typename... Args>
    void offer_work_impl(execution_data& ed, Args&&... args) {
        small_object_allocator alloc{};
        // New root node as a continuation and ref count. Left and right child attach to the new parent. Split the body.
        auto new_tree_node = alloc.new_object<tree_node_type>(ed, my_parent, 2, my_body, alloc);

        // New right child
        auto right_child = alloc.new_object<start_deterministic_reduce>(ed, std::forward<Args>(args)..., new_tree_node->right_body, alloc);

        right_child->my_parent = my_parent = new_tree_node;

        // Spawn the right sibling
        right_child->spawn_self(ed);
    }

    void spawn_self(execution_data& ed) {
        my_partition.spawn_task(*this, *context(ed));
    }
};

//! Fold the tree and deallocate the task
template<typename Range, typename Body, typename Partitioner>
void start_deterministic_reduce<Range, Body, Partitioner>::finalize(const execution_data& ed) {
    // Get the current parent and wait object before an object destruction
    node* parent = my_parent;

    auto allocator = my_allocator;
    // Task execution finished - destroy it
    this->~start_deterministic_reduce();
    // Unwind the tree decrementing the parent`s reference count
    fold_tree<tree_node_type>(parent, ed);
    allocator.deallocate(this, ed);
}

//! Execute parallel_deterministic_reduce task
template<typename Range, typename Body, typename Partitioner>
task* start_deterministic_reduce<Range,Body,Partitioner>::execute(execution_data& ed) {
    if (!is_same_affinity(ed)) {
        my_partition.note_affinity(execution_slot(ed));
    }
    my_partition.check_being_stolen(*this, ed);

    my_partition.execute(*this, my_range, ed);

    finalize(ed);
    return NULL;
}

//! Cancel parallel_deterministic_reduce task
template<typename Range, typename Body, typename Partitioner>
task* start_deterministic_reduce<Range, Body, Partitioner>::cancel(execution_data& ed) {
    finalize(ed);
    return NULL;
}


//! Auxiliary class for parallel_reduce; for internal use only.
/** The adaptor class that implements \ref parallel_reduce_body_req "parallel_reduce Body"
    using given \ref parallel_reduce_lambda_req "anonymous function objects".
 **/
/** @ingroup algorithms */
template<typename Range, typename Value, typename RealBody, typename Reduction>
class lambda_reduce_body {
//TODO: decide if my_real_body, my_reduction, and my_identity_element should be copied or referenced
//       (might require some performance measurements)

    const Value&     my_identity_element;
    const RealBody&  my_real_body;
    const Reduction& my_reduction;
    Value            my_value;
    lambda_reduce_body& operator= ( const lambda_reduce_body& other );
public:
    lambda_reduce_body( const Value& identity, const RealBody& body, const Reduction& reduction )
        : my_identity_element(identity)
        , my_real_body(body)
        , my_reduction(reduction)
        , my_value(identity)
    { }
    lambda_reduce_body( const lambda_reduce_body& other ) = default;
    lambda_reduce_body( lambda_reduce_body& other, tbb::split )
        : my_identity_element(other.my_identity_element)
        , my_real_body(other.my_real_body)
        , my_reduction(other.my_reduction)
        , my_value(other.my_identity_element)
    { }
    void operator()(Range& range) {
        my_value = my_real_body(range, const_cast<const Value&>(my_value));
    }
    void join( lambda_reduce_body& rhs ) {
        my_value = my_reduction(const_cast<const Value&>(my_value), const_cast<const Value&>(rhs.my_value));
    }
    Value result() const {
        return my_value;
    }
};


// Requirements on Range concept are documented in blocked_range.h

/** \page parallel_reduce_body_req Requirements on parallel_reduce body
    Class \c Body implementing the concept of parallel_reduce body must define:
    - \code Body::Body( Body&, split ); \endcode        Splitting constructor.
                                                        Must be able to run concurrently with operator() and method \c join
    - \code Body::~Body(); \endcode                     Destructor
    - \code void Body::operator()( Range& r ); \endcode Function call operator applying body to range \c r
                                                        and accumulating the result
    - \code void Body::join( Body& b ); \endcode        Join results.
                                                        The result in \c b should be merged into the result of \c this
**/

/** \page parallel_reduce_lambda_req Requirements on parallel_reduce anonymous function objects (lambda functions)
    TO BE DOCUMENTED
**/

/** \name parallel_reduce
    See also requirements on \ref range_req "Range" and \ref parallel_reduce_body_req "parallel_reduce Body". **/
//@{

//! Parallel iteration with reduction and default partitioner.
/** @ingroup algorithms **/
template<typename Range, typename Body>
void parallel_reduce( const Range& range, Body& body ) {
    start_reduce<Range,Body, const __TBB_DEFAULT_PARTITIONER>::run( range, body, __TBB_DEFAULT_PARTITIONER() );
}

//! Parallel iteration with reduction and simple_partitioner
/** @ingroup algorithms **/
template<typename Range, typename Body>
void parallel_reduce( const Range& range, Body& body, const simple_partitioner& partitioner ) {
    start_reduce<Range,Body,const simple_partitioner>::run( range, body, partitioner );
}

//! Parallel iteration with reduction and auto_partitioner
/** @ingroup algorithms **/
template<typename Range, typename Body>
void parallel_reduce( const Range& range, Body& body, const auto_partitioner& partitioner ) {
    start_reduce<Range,Body,const auto_partitioner>::run( range, body, partitioner );
}

//! Parallel iteration with reduction and static_partitioner
/** @ingroup algorithms **/
template<typename Range, typename Body>
void parallel_reduce( const Range& range, Body& body, const static_partitioner& partitioner ) {
    start_reduce<Range,Body,const static_partitioner>::run( range, body, partitioner );
}

//! Parallel iteration with reduction and affinity_partitioner
/** @ingroup algorithms **/
template<typename Range, typename Body>
void parallel_reduce( const Range& range, Body& body, affinity_partitioner& partitioner ) {
    start_reduce<Range,Body,affinity_partitioner>::run( range, body, partitioner );
}

//! Parallel iteration with reduction, default partitioner and user-supplied context.
/** @ingroup algorithms **/
template<typename Range, typename Body>
void parallel_reduce( const Range& range, Body& body, task_group_context& context ) {
    start_reduce<Range,Body,const __TBB_DEFAULT_PARTITIONER>::run( range, body, __TBB_DEFAULT_PARTITIONER(), context );
}

//! Parallel iteration with reduction, simple partitioner and user-supplied context.
/** @ingroup algorithms **/
template<typename Range, typename Body>
void parallel_reduce( const Range& range, Body& body, const simple_partitioner& partitioner, task_group_context& context ) {
    start_reduce<Range,Body,const simple_partitioner>::run( range, body, partitioner, context );
}

//! Parallel iteration with reduction, auto_partitioner and user-supplied context
/** @ingroup algorithms **/
template<typename Range, typename Body>
void parallel_reduce( const Range& range, Body& body, const auto_partitioner& partitioner, task_group_context& context ) {
    start_reduce<Range,Body,const auto_partitioner>::run( range, body, partitioner, context );
}

//! Parallel iteration with reduction, static_partitioner and user-supplied context
/** @ingroup algorithms **/
template<typename Range, typename Body>
void parallel_reduce( const Range& range, Body& body, const static_partitioner& partitioner, task_group_context& context ) {
    start_reduce<Range,Body,const static_partitioner>::run( range, body, partitioner, context );
}

//! Parallel iteration with reduction, affinity_partitioner and user-supplied context
/** @ingroup algorithms **/
template<typename Range, typename Body>
void parallel_reduce( const Range& range, Body& body, affinity_partitioner& partitioner, task_group_context& context ) {
    start_reduce<Range,Body,affinity_partitioner>::run( range, body, partitioner, context );
}
/** parallel_reduce overloads that work with anonymous function objects
    (see also \ref parallel_reduce_lambda_req "requirements on parallel_reduce anonymous function objects"). **/

//! Parallel iteration with reduction and default partitioner.
/** @ingroup algorithms **/
template<typename Range, typename Value, typename RealBody, typename Reduction>
Value parallel_reduce( const Range& range, const Value& identity, const RealBody& real_body, const Reduction& reduction ) {
    lambda_reduce_body<Range,Value,RealBody,Reduction> body(identity, real_body, reduction);
    start_reduce<Range,lambda_reduce_body<Range,Value,RealBody,Reduction>,const __TBB_DEFAULT_PARTITIONER>
                          ::run(range, body, __TBB_DEFAULT_PARTITIONER() );
    return body.result();
}

//! Parallel iteration with reduction and simple_partitioner.
/** @ingroup algorithms **/
template<typename Range, typename Value, typename RealBody, typename Reduction>
Value parallel_reduce( const Range& range, const Value& identity, const RealBody& real_body, const Reduction& reduction,
                       const simple_partitioner& partitioner ) {
    lambda_reduce_body<Range,Value,RealBody,Reduction> body(identity, real_body, reduction);
    start_reduce<Range,lambda_reduce_body<Range,Value,RealBody,Reduction>,const simple_partitioner>
                          ::run(range, body, partitioner );
    return body.result();
}

//! Parallel iteration with reduction and auto_partitioner
/** @ingroup algorithms **/
template<typename Range, typename Value, typename RealBody, typename Reduction>
Value parallel_reduce( const Range& range, const Value& identity, const RealBody& real_body, const Reduction& reduction,
                       const auto_partitioner& partitioner ) {
    lambda_reduce_body<Range,Value,RealBody,Reduction> body(identity, real_body, reduction);
    start_reduce<Range,lambda_reduce_body<Range,Value,RealBody,Reduction>,const auto_partitioner>
                          ::run( range, body, partitioner );
    return body.result();
}

//! Parallel iteration with reduction and static_partitioner
/** @ingroup algorithms **/
template<typename Range, typename Value, typename RealBody, typename Reduction>
Value parallel_reduce( const Range& range, const Value& identity, const RealBody& real_body, const Reduction& reduction,
                       const static_partitioner& partitioner ) {
    lambda_reduce_body<Range,Value,RealBody,Reduction> body(identity, real_body, reduction);
    start_reduce<Range,lambda_reduce_body<Range,Value,RealBody,Reduction>,const static_partitioner>
                                        ::run( range, body, partitioner );
    return body.result();
}

//! Parallel iteration with reduction and affinity_partitioner
/** @ingroup algorithms **/
template<typename Range, typename Value, typename RealBody, typename Reduction>
Value parallel_reduce( const Range& range, const Value& identity, const RealBody& real_body, const Reduction& reduction,
                       affinity_partitioner& partitioner ) {
    lambda_reduce_body<Range,Value,RealBody,Reduction> body(identity, real_body, reduction);
    start_reduce<Range,lambda_reduce_body<Range,Value,RealBody,Reduction>,affinity_partitioner>
                                        ::run( range, body, partitioner );
    return body.result();
}

//! Parallel iteration with reduction, default partitioner and user-supplied context.
/** @ingroup algorithms **/
template<typename Range, typename Value, typename RealBody, typename Reduction>
Value parallel_reduce( const Range& range, const Value& identity, const RealBody& real_body, const Reduction& reduction,
                       task_group_context& context ) {
    lambda_reduce_body<Range,Value,RealBody,Reduction> body(identity, real_body, reduction);
    start_reduce<Range,lambda_reduce_body<Range,Value,RealBody,Reduction>,const __TBB_DEFAULT_PARTITIONER>
                          ::run( range, body, __TBB_DEFAULT_PARTITIONER(), context );
    return body.result();
}

//! Parallel iteration with reduction, simple partitioner and user-supplied context.
/** @ingroup algorithms **/
template<typename Range, typename Value, typename RealBody, typename Reduction>
Value parallel_reduce( const Range& range, const Value& identity, const RealBody& real_body, const Reduction& reduction,
                       const simple_partitioner& partitioner, task_group_context& context ) {
    lambda_reduce_body<Range,Value,RealBody,Reduction> body(identity, real_body, reduction);
    start_reduce<Range,lambda_reduce_body<Range,Value,RealBody,Reduction>,const simple_partitioner>
                          ::run( range, body, partitioner, context );
    return body.result();
}

//! Parallel iteration with reduction, auto_partitioner and user-supplied context
/** @ingroup algorithms **/
template<typename Range, typename Value, typename RealBody, typename Reduction>
Value parallel_reduce( const Range& range, const Value& identity, const RealBody& real_body, const Reduction& reduction,
                       const auto_partitioner& partitioner, task_group_context& context ) {
    lambda_reduce_body<Range,Value,RealBody,Reduction> body(identity, real_body, reduction);
    start_reduce<Range,lambda_reduce_body<Range,Value,RealBody,Reduction>,const auto_partitioner>
                          ::run( range, body, partitioner, context );
    return body.result();
}

//! Parallel iteration with reduction, static_partitioner and user-supplied context
/** @ingroup algorithms **/
template<typename Range, typename Value, typename RealBody, typename Reduction>
Value parallel_reduce( const Range& range, const Value& identity, const RealBody& real_body, const Reduction& reduction,
                       const static_partitioner& partitioner, task_group_context& context ) {
    lambda_reduce_body<Range,Value,RealBody,Reduction> body(identity, real_body, reduction);
    start_reduce<Range,lambda_reduce_body<Range,Value,RealBody,Reduction>,const static_partitioner>
                                        ::run( range, body, partitioner, context );
    return body.result();
}

//! Parallel iteration with reduction, affinity_partitioner and user-supplied context
/** @ingroup algorithms **/
template<typename Range, typename Value, typename RealBody, typename Reduction>
Value parallel_reduce( const Range& range, const Value& identity, const RealBody& real_body, const Reduction& reduction,
                       affinity_partitioner& partitioner, task_group_context& context ) {
    lambda_reduce_body<Range,Value,RealBody,Reduction> body(identity, real_body, reduction);
    start_reduce<Range,lambda_reduce_body<Range,Value,RealBody,Reduction>,affinity_partitioner>
                                        ::run( range, body, partitioner, context );
    return body.result();
}

//! Parallel iteration with deterministic reduction and default simple partitioner.
/** @ingroup algorithms **/
template<typename Range, typename Body>
void parallel_deterministic_reduce( const Range& range, Body& body ) {
    start_deterministic_reduce<Range, Body, const simple_partitioner>::run(range, body, simple_partitioner());
}

//! Parallel iteration with deterministic reduction and simple partitioner.
/** @ingroup algorithms **/
template<typename Range, typename Body>
void parallel_deterministic_reduce( const Range& range, Body& body, const simple_partitioner& partitioner ) {
    start_deterministic_reduce<Range, Body, const simple_partitioner>::run(range, body, partitioner);
}

//! Parallel iteration with deterministic reduction and static partitioner.
/** @ingroup algorithms **/
template<typename Range, typename Body>
void parallel_deterministic_reduce( const Range& range, Body& body, const static_partitioner& partitioner ) {
    start_deterministic_reduce<Range, Body, const static_partitioner>::run(range, body, partitioner);
}

//! Parallel iteration with deterministic reduction, default simple partitioner and user-supplied context.
/** @ingroup algorithms **/
template<typename Range, typename Body>
void parallel_deterministic_reduce( const Range& range, Body& body, task_group_context& context ) {
    start_deterministic_reduce<Range,Body, const simple_partitioner>::run( range, body, simple_partitioner(), context );
}

//! Parallel iteration with deterministic reduction, simple partitioner and user-supplied context.
/** @ingroup algorithms **/
template<typename Range, typename Body>
void parallel_deterministic_reduce( const Range& range, Body& body, const simple_partitioner& partitioner, task_group_context& context ) {
    start_deterministic_reduce<Range, Body, const simple_partitioner>::run(range, body, partitioner, context);
}

//! Parallel iteration with deterministic reduction, static partitioner and user-supplied context.
/** @ingroup algorithms **/
template<typename Range, typename Body>
void parallel_deterministic_reduce( const Range& range, Body& body, const static_partitioner& partitioner, task_group_context& context ) {
    start_deterministic_reduce<Range, Body, const static_partitioner>::run(range, body, partitioner, context);
}

/** parallel_reduce overloads that work with anonymous function objects
    (see also \ref parallel_reduce_lambda_req "requirements on parallel_reduce anonymous function objects"). **/

//! Parallel iteration with deterministic reduction and default simple partitioner.
// TODO: consider making static_partitioner the default
/** @ingroup algorithms **/
template<typename Range, typename Value, typename RealBody, typename Reduction>
Value parallel_deterministic_reduce( const Range& range, const Value& identity, const RealBody& real_body, const Reduction& reduction ) {
    return parallel_deterministic_reduce(range, identity, real_body, reduction, simple_partitioner());
}

//! Parallel iteration with deterministic reduction and simple partitioner.
/** @ingroup algorithms **/
template<typename Range, typename Value, typename RealBody, typename Reduction>
Value parallel_deterministic_reduce( const Range& range, const Value& identity, const RealBody& real_body, const Reduction& reduction, const simple_partitioner& partitioner ) {
    lambda_reduce_body<Range,Value,RealBody,Reduction> body(identity, real_body, reduction);
    start_deterministic_reduce<Range,lambda_reduce_body<Range,Value,RealBody,Reduction>, const simple_partitioner>
                          ::run(range, body, partitioner);
    return body.result();
}

//! Parallel iteration with deterministic reduction and static partitioner.
/** @ingroup algorithms **/
template<typename Range, typename Value, typename RealBody, typename Reduction>
Value parallel_deterministic_reduce( const Range& range, const Value& identity, const RealBody& real_body, const Reduction& reduction, const static_partitioner& partitioner ) {
    lambda_reduce_body<Range, Value, RealBody, Reduction> body(identity, real_body, reduction);
    start_deterministic_reduce<Range, lambda_reduce_body<Range, Value, RealBody, Reduction>, const static_partitioner>
        ::run(range, body, partitioner);
    return body.result();
}

//! Parallel iteration with deterministic reduction, default simple partitioner and user-supplied context.
/** @ingroup algorithms **/
template<typename Range, typename Value, typename RealBody, typename Reduction>
Value parallel_deterministic_reduce( const Range& range, const Value& identity, const RealBody& real_body, const Reduction& reduction,
    task_group_context& context ) {
    return parallel_deterministic_reduce(range, identity, real_body, reduction, simple_partitioner(), context);
}

//! Parallel iteration with deterministic reduction, simple partitioner and user-supplied context.
/** @ingroup algorithms **/
template<typename Range, typename Value, typename RealBody, typename Reduction>
Value parallel_deterministic_reduce( const Range& range, const Value& identity, const RealBody& real_body, const Reduction& reduction,
    const simple_partitioner& partitioner, task_group_context& context ) {
    lambda_reduce_body<Range, Value, RealBody, Reduction> body(identity, real_body, reduction);
    start_deterministic_reduce<Range, lambda_reduce_body<Range, Value, RealBody, Reduction>, const simple_partitioner>
        ::run(range, body, partitioner, context);
    return body.result();
}

//! Parallel iteration with deterministic reduction, static partitioner and user-supplied context.
/** @ingroup algorithms **/
template<typename Range, typename Value, typename RealBody, typename Reduction>
Value parallel_deterministic_reduce( const Range& range, const Value& identity, const RealBody& real_body, const Reduction& reduction,
    const static_partitioner& partitioner, task_group_context& context ) {
    lambda_reduce_body<Range, Value, RealBody, Reduction> body(identity, real_body, reduction);
    start_deterministic_reduce<Range, lambda_reduce_body<Range, Value, RealBody, Reduction>, const static_partitioner>
        ::run(range, body, partitioner, context);
    return body.result();
}
//@}

} // namespace d1
} // namespace detail

inline namespace v1 {
using detail::d1::parallel_reduce;
using detail::d1::parallel_deterministic_reduce;
// Split types
using detail::split;
using detail::proportional_split;
} // namespace v1

} // namespace tbb
#endif /* __TBB_parallel_reduce_H */
