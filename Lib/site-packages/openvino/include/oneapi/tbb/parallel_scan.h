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

#ifndef __TBB_parallel_scan_H
#define __TBB_parallel_scan_H

#include <functional>

#include "detail/_config.h"
#include "detail/_namespace_injection.h"
#include "detail/_exception.h"
#include "detail/_task.h"

#include "profiling.h"
#include "partitioner.h"
#include "blocked_range.h"
#include "task_group.h"

namespace tbb {
namespace detail {
namespace d1 {

//! Used to indicate that the initial scan is being performed.
/** @ingroup algorithms */
struct pre_scan_tag {
    static bool is_final_scan() {return false;}
    operator bool() {return is_final_scan();}
};

//! Used to indicate that the final scan is being performed.
/** @ingroup algorithms */
struct final_scan_tag {
    static bool is_final_scan() {return true;}
    operator bool() {return is_final_scan();}
};

template<typename Range, typename Body>
struct sum_node;

//! Performs final scan for a leaf
/** @ingroup algorithms */
template<typename Range, typename Body>
struct final_sum : public task {
private:
    using sum_node_type = sum_node<Range, Body>;
    Body m_body;
    aligned_space<Range> m_range;
    //! Where to put result of last subrange, or nullptr if not last subrange.
    Body* m_stuff_last;

    wait_context& m_wait_context;
    sum_node_type* m_parent = nullptr;
public:
    small_object_allocator m_allocator;
    final_sum( Body& body, wait_context& w_o, small_object_allocator& alloc ) :
        m_body(body, split()), m_wait_context(w_o), m_allocator(alloc) {
        poison_pointer(m_stuff_last);
    }

    final_sum( final_sum& sum, small_object_allocator& alloc ) :
        m_body(sum.m_body, split()), m_wait_context(sum.m_wait_context), m_allocator(alloc) {
        poison_pointer(m_stuff_last);
    }

    ~final_sum() {
        m_range.begin()->~Range();
    }
    void finish_construction( sum_node_type* parent, const Range& range, Body* stuff_last ) {
        __TBB_ASSERT( m_parent == nullptr, nullptr );
        m_parent = parent;
        new( m_range.begin() ) Range(range);
        m_stuff_last = stuff_last;
    }
private:
    sum_node_type* release_parent() {
        call_itt_task_notify(releasing, m_parent);
        if (m_parent) {
            auto parent = m_parent;
            m_parent = nullptr;
            if (parent->ref_count.fetch_sub(1, std::memory_order_relaxed) == 1) {
                return parent;
            }
        }
        else
            m_wait_context.release();
        return nullptr;
    }
    sum_node_type* finalize(const execution_data& ed){
        sum_node_type* next_task = release_parent();
        m_allocator.delete_object<final_sum>(this, ed);
        return next_task;
    }

public:
    task* execute(execution_data& ed) override {
        m_body( *m_range.begin(), final_scan_tag() );
        if( m_stuff_last )
            m_stuff_last->assign(m_body);

        return finalize(ed);
    }
    task* cancel(execution_data& ed) override {
        return finalize(ed);
    }
    template<typename Tag>
    void operator()( const Range& r, Tag tag ) {
        m_body( r, tag );
    }
    void reverse_join( final_sum& a ) {
        m_body.reverse_join(a.m_body);
    }
    void reverse_join( Body& body ) {
        m_body.reverse_join(body);
    }
    void assign_to( Body& body ) {
        body.assign(m_body);
    }
    void self_destroy(const execution_data& ed) {
        m_allocator.delete_object<final_sum>(this, ed);
    }
};

//! Split work to be done in the scan.
/** @ingroup algorithms */
template<typename Range, typename Body>
struct sum_node : public task {
private:
    using final_sum_type = final_sum<Range,Body>;
public:
    final_sum_type *m_incoming;
    final_sum_type *m_body;
    Body *m_stuff_last;
private:
    final_sum_type *m_left_sum;
    sum_node *m_left;
    sum_node *m_right;
    bool m_left_is_final;
    Range m_range;
    wait_context& m_wait_context;
    sum_node* m_parent;
    small_object_allocator m_allocator;
public:
    std::atomic<unsigned int> ref_count{0};
    sum_node( const Range range, bool left_is_final_, sum_node* parent, wait_context& w_o, small_object_allocator& alloc ) :
        m_stuff_last(nullptr),
        m_left_sum(nullptr),
        m_left(nullptr),
        m_right(nullptr),
        m_left_is_final(left_is_final_),
        m_range(range),
        m_wait_context(w_o),
        m_parent(parent),
        m_allocator(alloc)
    {
        if( m_parent )
            m_parent->ref_count.fetch_add(1, std::memory_order_relaxed);
        // Poison fields that will be set by second pass.
        poison_pointer(m_body);
        poison_pointer(m_incoming);
    }

    ~sum_node() {
        if (m_parent)
            m_parent->ref_count.fetch_sub(1, std::memory_order_relaxed);
    }
private:
    sum_node* release_parent() {
        call_itt_task_notify(releasing, m_parent);
        if (m_parent) {
            auto parent = m_parent;
            m_parent = nullptr;
            if (parent->ref_count.fetch_sub(1, std::memory_order_relaxed) == 1) {
                return parent;
            }
        }
        else
            m_wait_context.release();
        return nullptr;
    }
    task* create_child( const Range& range, final_sum_type& body, sum_node* child, final_sum_type* incoming, Body* stuff_last ) {
        if( child ) {
            __TBB_ASSERT( is_poisoned(child->m_body) && is_poisoned(child->m_incoming), nullptr );
            child->prepare_for_execution(body, incoming, stuff_last);
            return child;
        } else {
            body.finish_construction(this, range, stuff_last);
            return &body;
        }
    }

    sum_node* finalize(const execution_data& ed) {
        sum_node* next_task = release_parent();
        m_allocator.delete_object<sum_node>(this, ed);
        return next_task;
    }

public:
    void prepare_for_execution(final_sum_type& body, final_sum_type* incoming, Body *stuff_last) {
        this->m_body = &body;
        this->m_incoming = incoming;
        this->m_stuff_last = stuff_last;
    }
    task* execute(execution_data& ed) override {
        if( m_body ) {
            if( m_incoming )
                m_left_sum->reverse_join( *m_incoming );
            task* right_child = this->create_child(Range(m_range,split()), *m_left_sum, m_right, m_left_sum, m_stuff_last);
            task* left_child = m_left_is_final ? nullptr : this->create_child(m_range, *m_body, m_left, m_incoming, nullptr);
            ref_count = (left_child != nullptr) + (right_child != nullptr);
            m_body = nullptr;
            if( left_child ) {
                spawn(*right_child, *ed.context);
                return left_child;
            } else {
                return right_child;
            }
        } else {
            return finalize(ed);
        }
    }
    task* cancel(execution_data& ed) override {
        return finalize(ed);
    }
    void self_destroy(const execution_data& ed) {
        m_allocator.delete_object<sum_node>(this, ed);
    }
    template<typename range,typename body,typename partitioner>
    friend struct start_scan;

    template<typename range,typename body>
    friend struct finish_scan;
};

//! Combine partial results
/** @ingroup algorithms */
template<typename Range, typename Body>
struct finish_scan : public task {
private:
    using sum_node_type = sum_node<Range,Body>;
    using final_sum_type = final_sum<Range,Body>;
    final_sum_type** const m_sum_slot;
    sum_node_type*& m_return_slot;
    small_object_allocator m_allocator;
public:
    final_sum_type* m_right_zombie;
    sum_node_type& m_result;
    std::atomic<unsigned int> ref_count{2};
    finish_scan*  m_parent;
    wait_context& m_wait_context;
    task* execute(execution_data& ed) override {
        __TBB_ASSERT( m_result.ref_count.load() == static_cast<unsigned int>((m_result.m_left!=nullptr)+(m_result.m_right!=nullptr)), nullptr );
        if( m_result.m_left )
            m_result.m_left_is_final = false;
        if( m_right_zombie && m_sum_slot )
            (*m_sum_slot)->reverse_join(*m_result.m_left_sum);
        __TBB_ASSERT( !m_return_slot, nullptr );
        if( m_right_zombie || m_result.m_right ) {
            m_return_slot = &m_result;
        } else {
            m_result.self_destroy(ed);
        }
        if( m_right_zombie && !m_sum_slot && !m_result.m_right ) {
            m_right_zombie->self_destroy(ed);
            m_right_zombie = nullptr;
        }
        return finalize(ed);
    }
    task* cancel(execution_data& ed) override {
        return finalize(ed);
    }
    finish_scan(sum_node_type*& return_slot, final_sum_type** sum, sum_node_type& result_, finish_scan* parent, wait_context& w_o, small_object_allocator& alloc) :
        m_sum_slot(sum),
        m_return_slot(return_slot),
        m_allocator(alloc),
        m_right_zombie(nullptr),
        m_result(result_),
        m_parent(parent),
        m_wait_context(w_o)
    {
        __TBB_ASSERT( !m_return_slot, nullptr );
    }
private:
    finish_scan* release_parent() {
        call_itt_task_notify(releasing, m_parent);
        if (m_parent) {
            auto parent = m_parent;
            m_parent = nullptr;
            if (parent->ref_count.fetch_sub(1, std::memory_order_relaxed) == 1) {
                return parent;
            }
        }
        else
            m_wait_context.release();
        return nullptr;
    }
    finish_scan* finalize(const execution_data& ed) {
        finish_scan* next_task = release_parent();
        m_allocator.delete_object<finish_scan>(this, ed);
        return next_task;
    }
};

//! Initial task to split the work
/** @ingroup algorithms */
template<typename Range, typename Body, typename Partitioner>
struct start_scan : public task {
private:
    using sum_node_type = sum_node<Range,Body>;
    using final_sum_type = final_sum<Range,Body>;
    using finish_pass1_type = finish_scan<Range,Body>;
    std::reference_wrapper<sum_node_type*> m_return_slot;
    Range m_range;
    std::reference_wrapper<final_sum_type> m_body;
    typename Partitioner::partition_type m_partition;
    /** Non-null if caller is requesting total. */
    final_sum_type** m_sum_slot;
    bool m_is_final;
    bool m_is_right_child;

    finish_pass1_type*  m_parent;
    small_object_allocator m_allocator;
    wait_context& m_wait_context;

    finish_pass1_type* release_parent() {
        call_itt_task_notify(releasing, m_parent);
        if (m_parent) {
            auto parent = m_parent;
            m_parent = nullptr;
            if (parent->ref_count.fetch_sub(1, std::memory_order_relaxed) == 1) {
                return parent;
            }
        }
        else
            m_wait_context.release();
        return nullptr;
    }

    finish_pass1_type* finalize( const execution_data& ed ) {
        finish_pass1_type* next_task = release_parent();
        m_allocator.delete_object<start_scan>(this, ed);
        return next_task;
    }

public:
    task* execute( execution_data& ) override;
    task* cancel( execution_data& ed ) override {
        return finalize(ed);
    }
    start_scan( sum_node_type*& return_slot, start_scan& parent, small_object_allocator& alloc ) :
        m_return_slot(return_slot),
        m_range(parent.m_range,split()),
        m_body(parent.m_body),
        m_partition(parent.m_partition,split()),
        m_sum_slot(parent.m_sum_slot),
        m_is_final(parent.m_is_final),
        m_is_right_child(true),
        m_parent(parent.m_parent),
        m_allocator(alloc),
        m_wait_context(parent.m_wait_context)
    {
        __TBB_ASSERT( !m_return_slot, nullptr );
        parent.m_is_right_child = false;
    }

    start_scan( sum_node_type*& return_slot, const Range& range, final_sum_type& body, const Partitioner& partitioner, wait_context& w_o, small_object_allocator& alloc ) :
        m_return_slot(return_slot),
        m_range(range),
        m_body(body),
        m_partition(partitioner),
        m_sum_slot(nullptr),
        m_is_final(true),
        m_is_right_child(false),
        m_parent(nullptr),
        m_allocator(alloc),
        m_wait_context(w_o)
    {
        __TBB_ASSERT( !m_return_slot, nullptr );
    }

    static void run( const Range& range, Body& body, const Partitioner& partitioner ) {
        if( !range.empty() ) {
            task_group_context context(PARALLEL_SCAN);

            using start_pass1_type = start_scan<Range,Body,Partitioner>;
            sum_node_type* root = nullptr;
            wait_context w_ctx{1};
            small_object_allocator alloc{};

            auto& temp_body = *alloc.new_object<final_sum_type>(body, w_ctx, alloc);
            temp_body.reverse_join(body);

            auto& pass1 = *alloc.new_object<start_pass1_type>(/*m_return_slot=*/root, range, temp_body, partitioner, w_ctx, alloc);

            execute_and_wait(pass1, context, w_ctx, context);
            if( root ) {
                root->prepare_for_execution(temp_body, nullptr, &body);
                w_ctx.reserve();
                execute_and_wait(*root, context, w_ctx, context);
            } else {
                temp_body.assign_to(body);
                temp_body.finish_construction(nullptr, range, nullptr);
                alloc.delete_object<final_sum_type>(&temp_body);
            }
        }
    }
};

template<typename Range, typename Body, typename Partitioner>
task* start_scan<Range,Body,Partitioner>::execute( execution_data& ed ) {
    // Inspecting m_parent->result.left_sum would ordinarily be a race condition.
    // But we inspect it only if we are not a stolen task, in which case we
    // know that task assigning to m_parent->result.left_sum has completed.
    __TBB_ASSERT(!m_is_right_child || m_parent, "right child is never an orphan");
    bool treat_as_stolen = m_is_right_child && (is_stolen(ed) || &m_body.get()!=m_parent->m_result.m_left_sum);
    if( treat_as_stolen ) {
        // Invocation is for right child that has been really stolen or needs to be virtually stolen
        small_object_allocator alloc{};
        m_parent->m_right_zombie = alloc.new_object<final_sum_type>(m_body, alloc);
        m_body = *m_parent->m_right_zombie;
        m_is_final = false;
    }
    task* next_task = nullptr;
    if( (m_is_right_child && !treat_as_stolen) || !m_range.is_divisible() || m_partition.should_execute_range(ed) ) {
        if( m_is_final )
            m_body(m_range, final_scan_tag());
        else if( m_sum_slot )
            m_body(m_range, pre_scan_tag());
        if( m_sum_slot )
            *m_sum_slot = &m_body.get();
        __TBB_ASSERT( !m_return_slot, nullptr );

        next_task = finalize(ed);
    } else {
        small_object_allocator alloc{};
        auto result = alloc.new_object<sum_node_type>(m_range,/*m_left_is_final=*/m_is_final, m_parent? &m_parent->m_result: nullptr, m_wait_context, alloc);

        auto new_parent = alloc.new_object<finish_pass1_type>(m_return_slot, m_sum_slot, *result, m_parent, m_wait_context, alloc);
        m_parent = new_parent;

        // Split off right child
        auto& right_child = *alloc.new_object<start_scan>(/*m_return_slot=*/result->m_right, *this, alloc);

        spawn(right_child, *ed.context);

        m_sum_slot = &result->m_left_sum;
        m_return_slot = result->m_left;

        __TBB_ASSERT( !m_return_slot, nullptr );
        next_task = this;
    }
    return next_task;
}

template<typename Range, typename Value, typename Scan, typename ReverseJoin>
class lambda_scan_body {
    Value               m_sum_slot;
    const Value&        identity_element;
    const Scan&         m_scan;
    const ReverseJoin&  m_reverse_join;
public:
    void operator=(const lambda_scan_body&) = delete;
    lambda_scan_body(const lambda_scan_body&) = default;

    lambda_scan_body( const Value& identity, const Scan& scan, const ReverseJoin& rev_join )
        : m_sum_slot(identity)
        , identity_element(identity)
        , m_scan(scan)
        , m_reverse_join(rev_join) {}

    lambda_scan_body( lambda_scan_body& b, split )
        : m_sum_slot(b.identity_element)
        , identity_element(b.identity_element)
        , m_scan(b.m_scan)
        , m_reverse_join(b.m_reverse_join) {}

    template<typename Tag>
    void operator()( const Range& r, Tag tag ) {
        m_sum_slot = m_scan(r, m_sum_slot, tag);
    }

    void reverse_join( lambda_scan_body& a ) {
        m_sum_slot = m_reverse_join(a.m_sum_slot, m_sum_slot);
    }

    void assign( lambda_scan_body& b ) {
        m_sum_slot = b.m_sum_slot;
    }

    Value result() const {
        return m_sum_slot;
    }
};

// Requirements on Range concept are documented in blocked_range.h

/** \page parallel_scan_body_req Requirements on parallel_scan body
    Class \c Body implementing the concept of parallel_scan body must define:
    - \code Body::Body( Body&, split ); \endcode    Splitting constructor.
                                                    Split \c b so that \c this and \c b can accumulate separately
    - \code Body::~Body(); \endcode                 Destructor
    - \code void Body::operator()( const Range& r, pre_scan_tag ); \endcode
                                                    Preprocess iterations for range \c r
    - \code void Body::operator()( const Range& r, final_scan_tag ); \endcode
                                                    Do final processing for iterations of range \c r
    - \code void Body::reverse_join( Body& a ); \endcode
                                                    Merge preprocessing state of \c a into \c this, where \c a was
                                                    created earlier from \c b by b's splitting constructor
**/

/** \name parallel_scan
    See also requirements on \ref range_req "Range" and \ref parallel_scan_body_req "parallel_scan Body". **/
//@{

//! Parallel prefix with default partitioner
/** @ingroup algorithms **/
template<typename Range, typename Body>
void parallel_scan( const Range& range, Body& body ) {
    start_scan<Range, Body, auto_partitioner>::run(range,body,__TBB_DEFAULT_PARTITIONER());
}

//! Parallel prefix with simple_partitioner
/** @ingroup algorithms **/
template<typename Range, typename Body>
void parallel_scan( const Range& range, Body& body, const simple_partitioner& partitioner ) {
    start_scan<Range, Body, simple_partitioner>::run(range, body, partitioner);
}

//! Parallel prefix with auto_partitioner
/** @ingroup algorithms **/
template<typename Range, typename Body>
void parallel_scan( const Range& range, Body& body, const auto_partitioner& partitioner ) {
    start_scan<Range,Body,auto_partitioner>::run(range, body, partitioner);
}

//! Parallel prefix with default partitioner
/** @ingroup algorithms **/
template<typename Range, typename Value, typename Scan, typename ReverseJoin>
Value parallel_scan( const Range& range, const Value& identity, const Scan& scan, const ReverseJoin& reverse_join ) {
    lambda_scan_body<Range, Value, Scan, ReverseJoin> body(identity, scan, reverse_join);
    parallel_scan(range, body, __TBB_DEFAULT_PARTITIONER());
    return body.result();
}

//! Parallel prefix with simple_partitioner
/** @ingroup algorithms **/
template<typename Range, typename Value, typename Scan, typename ReverseJoin>
Value parallel_scan( const Range& range, const Value& identity, const Scan& scan, const ReverseJoin& reverse_join,
                     const simple_partitioner& partitioner ) {
    lambda_scan_body<Range, Value, Scan, ReverseJoin> body(identity, scan, reverse_join);
    parallel_scan(range, body, partitioner);
    return body.result();
}

//! Parallel prefix with auto_partitioner
/** @ingroup algorithms **/
template<typename Range, typename Value, typename Scan, typename ReverseJoin>
Value parallel_scan( const Range& range, const Value& identity, const Scan& scan, const ReverseJoin& reverse_join,
                     const auto_partitioner& partitioner ) {
    lambda_scan_body<Range, Value, Scan, ReverseJoin> body(identity, scan, reverse_join);
    parallel_scan(range, body, partitioner);
    return body.result();
}

} // namespace d1
} // namespace detail

inline namespace v1 {
    using detail::d1::parallel_scan;
    using detail::d1::pre_scan_tag;
    using detail::d1::final_scan_tag;

} // namespace v1

} // namespace tbb

#endif /* __TBB_parallel_scan_H */

