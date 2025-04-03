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

#ifndef __TBB_flow_graph_H
#define __TBB_flow_graph_H

#include <atomic>
#include <memory>
#include <type_traits>

#include "detail/_config.h"
#include "detail/_namespace_injection.h"
#include "spin_mutex.h"
#include "null_mutex.h"
#include "spin_rw_mutex.h"
#include "null_rw_mutex.h"
#include "detail/_pipeline_filters.h"
#include "detail/_task.h"
#include "detail/_small_object_pool.h"
#include "cache_aligned_allocator.h"
#include "detail/_exception.h"
#include "detail/_template_helpers.h"
#include "detail/_aggregator.h"
#include "detail/_allocator_traits.h"
#include "profiling.h"
#include "task_arena.h"

#if TBB_USE_PROFILING_TOOLS && ( __linux__ || __APPLE__ )
   #if __INTEL_COMPILER
       // Disabled warning "routine is both inline and noinline"
       #pragma warning (push)
       #pragma warning( disable: 2196 )
   #endif
   #define __TBB_NOINLINE_SYM __attribute__((noinline))
#else
   #define __TBB_NOINLINE_SYM
#endif

#include <tuple>
#include <list>
#include <queue>

/** @file
  \brief The graph related classes and functions

  There are some applications that best express dependencies as messages
  passed between nodes in a graph.  These messages may contain data or
  simply act as signals that a predecessors has completed. The graph
  class and its associated node classes can be used to express such
  applications.
*/

namespace tbb {
namespace detail {

namespace d1 {

//! An enumeration the provides the two most common concurrency levels: unlimited and serial
enum concurrency { unlimited = 0, serial = 1 };

//! A generic null type
struct null_type {};

//! An empty class used for messages that mean "I'm done"
class continue_msg {};

//! Forward declaration section
template< typename T > class sender;
template< typename T > class receiver;
class continue_receiver;

template< typename T, typename U > class limiter_node;  // needed for resetting decrementer

template<typename T, typename M> class successor_cache;
template<typename T, typename M> class broadcast_cache;
template<typename T, typename M> class round_robin_cache;
template<typename T, typename M> class predecessor_cache;
template<typename T, typename M> class reservable_predecessor_cache;

#if __TBB_PREVIEW_FLOW_GRAPH_NODE_SET
namespace order {
struct following;
struct preceding;
}
template<typename Order, typename... Args> struct node_set;
#endif


} // namespace d1
} // namespace detail
} // namespace tbb

//! The graph class
#include "detail/_flow_graph_impl.h"

namespace tbb {
namespace detail {
namespace d1 {

static inline std::pair<graph_task*, graph_task*> order_tasks(graph_task* first, graph_task* second) {
    if (second->priority > first->priority)
        return std::make_pair(second, first);
    return std::make_pair(first, second);
}

// submit task if necessary. Returns the non-enqueued task if there is one.
static inline graph_task* combine_tasks(graph& g, graph_task* left, graph_task* right) {
    // if no RHS task, don't change left.
    if (right == NULL) return left;
    // right != NULL
    if (left == NULL) return right;
    if (left == SUCCESSFULLY_ENQUEUED) return right;
    // left contains a task
    if (right != SUCCESSFULLY_ENQUEUED) {
        // both are valid tasks
        auto tasks_pair = order_tasks(left, right);
        spawn_in_graph_arena(g, *tasks_pair.first);
        return tasks_pair.second;
    }
    return left;
}

//! Pure virtual template class that defines a sender of messages of type T
template< typename T >
class sender {
public:
    virtual ~sender() {}

    //! Request an item from the sender
    virtual bool try_get( T & ) { return false; }

    //! Reserves an item in the sender
    virtual bool try_reserve( T & ) { return false; }

    //! Releases the reserved item
    virtual bool try_release( ) { return false; }

    //! Consumes the reserved item
    virtual bool try_consume( ) { return false; }

protected:
    //! The output type of this sender
    typedef T output_type;

    //! The successor type for this node
    typedef receiver<T> successor_type;

    //! Add a new successor to this node
    virtual bool register_successor( successor_type &r ) = 0;

    //! Removes a successor from this node
    virtual bool remove_successor( successor_type &r ) = 0;

    template<typename C>
    friend bool register_successor(sender<C>& s, receiver<C>& r);

    template<typename C>
    friend bool remove_successor  (sender<C>& s, receiver<C>& r);
};  // class sender<T>

template<typename C>
bool register_successor(sender<C>& s, receiver<C>& r) {
    return s.register_successor(r);
}

template<typename C>
bool remove_successor(sender<C>& s, receiver<C>& r) {
    return s.remove_successor(r);
}

//! Pure virtual template class that defines a receiver of messages of type T
template< typename T >
class receiver {
public:
    //! Destructor
    virtual ~receiver() {}

    //! Put an item to the receiver
    bool try_put( const T& t ) {
        graph_task *res = try_put_task(t);
        if (!res) return false;
        if (res != SUCCESSFULLY_ENQUEUED) spawn_in_graph_arena(graph_reference(), *res);
        return true;
    }

    //! put item to successor; return task to run the successor if possible.
protected:
    //! The input type of this receiver
    typedef T input_type;

    //! The predecessor type for this node
    typedef sender<T> predecessor_type;

    template< typename R, typename B > friend class run_and_put_task;
    template< typename X, typename Y > friend class broadcast_cache;
    template< typename X, typename Y > friend class round_robin_cache;
    virtual graph_task *try_put_task(const T& t) = 0;
    virtual graph& graph_reference() const = 0;

    template<typename TT, typename M> friend class successor_cache;
    virtual bool is_continue_receiver() { return false; }

    // TODO revamp: reconsider the inheritance and move node priority out of receiver
    virtual node_priority_t priority() const { return no_priority; }

    //! Add a predecessor to the node
    virtual bool register_predecessor( predecessor_type & ) { return false; }

    //! Remove a predecessor from the node
    virtual bool remove_predecessor( predecessor_type & ) { return false; }

    template <typename C>
    friend bool register_predecessor(receiver<C>& r, sender<C>& s);
    template <typename C>
    friend bool remove_predecessor  (receiver<C>& r, sender<C>& s);
}; // class receiver<T>

template <typename C>
bool register_predecessor(receiver<C>& r, sender<C>& s) {
    return r.register_predecessor(s);
}

template <typename C>
bool remove_predecessor(receiver<C>& r, sender<C>& s) {
    return r.remove_predecessor(s);
}

//! Base class for receivers of completion messages
/** These receivers automatically reset, but cannot be explicitly waited on */
class continue_receiver : public receiver< continue_msg > {
protected:

    //! Constructor
    explicit continue_receiver( int number_of_predecessors, node_priority_t a_priority ) {
        my_predecessor_count = my_initial_predecessor_count = number_of_predecessors;
        my_current_count = 0;
        my_priority = a_priority;
    }

    //! Copy constructor
    continue_receiver( const continue_receiver& src ) : receiver<continue_msg>() {
        my_predecessor_count = my_initial_predecessor_count = src.my_initial_predecessor_count;
        my_current_count = 0;
        my_priority = src.my_priority;
    }

    //! Increments the trigger threshold
    bool register_predecessor( predecessor_type & ) override {
        spin_mutex::scoped_lock l(my_mutex);
        ++my_predecessor_count;
        return true;
    }

    //! Decrements the trigger threshold
    /** Does not check to see if the removal of the predecessor now makes the current count
        exceed the new threshold.  So removing a predecessor while the graph is active can cause
        unexpected results. */
    bool remove_predecessor( predecessor_type & ) override {
        spin_mutex::scoped_lock l(my_mutex);
        --my_predecessor_count;
        return true;
    }

    //! The input type
    typedef continue_msg input_type;

    //! The predecessor type for this node
    typedef receiver<input_type>::predecessor_type predecessor_type;

    template< typename R, typename B > friend class run_and_put_task;
    template<typename X, typename Y> friend class broadcast_cache;
    template<typename X, typename Y> friend class round_robin_cache;
    // execute body is supposed to be too small to create a task for.
    graph_task* try_put_task( const input_type & ) override {
        {
            spin_mutex::scoped_lock l(my_mutex);
            if ( ++my_current_count < my_predecessor_count )
                return SUCCESSFULLY_ENQUEUED;
            else
                my_current_count = 0;
        }
        graph_task* res = execute();
        return res? res : SUCCESSFULLY_ENQUEUED;
    }

    spin_mutex my_mutex;
    int my_predecessor_count;
    int my_current_count;
    int my_initial_predecessor_count;
    node_priority_t my_priority;
    // the friend declaration in the base class did not eliminate the "protected class"
    // error in gcc 4.1.2
    template<typename U, typename V> friend class limiter_node;

    virtual void reset_receiver( reset_flags f ) {
        my_current_count = 0;
        if (f & rf_clear_edges) {
            my_predecessor_count = my_initial_predecessor_count;
        }
    }

    //! Does whatever should happen when the threshold is reached
    /** This should be very fast or else spawn a task.  This is
        called while the sender is blocked in the try_put(). */
    virtual graph_task* execute() = 0;
    template<typename TT, typename M> friend class successor_cache;
    bool is_continue_receiver() override { return true; }

    node_priority_t priority() const override { return my_priority; }
}; // class continue_receiver

#if __TBB_PREVIEW_MESSAGE_BASED_KEY_MATCHING
    template <typename K, typename T>
    K key_from_message( const T &t ) {
        return t.key();
    }
#endif /* __TBB_PREVIEW_MESSAGE_BASED_KEY_MATCHING */

} // d1
} // detail
} // tbb

#include "detail/_flow_graph_trace_impl.h"
#include "detail/_hash_compare.h"

namespace tbb {
namespace detail {
namespace d1 {

#include "detail/_flow_graph_body_impl.h"
#include "detail/_flow_graph_cache_impl.h"
#include "detail/_flow_graph_types_impl.h"

using namespace graph_policy_namespace;

template <typename C, typename N>
graph_iterator<C,N>::graph_iterator(C *g, bool begin) : my_graph(g), current_node(NULL)
{
    if (begin) current_node = my_graph->my_nodes;
    //else it is an end iterator by default
}

template <typename C, typename N>
typename graph_iterator<C,N>::reference graph_iterator<C,N>::operator*() const {
    __TBB_ASSERT(current_node, "graph_iterator at end");
    return *operator->();
}

template <typename C, typename N>
typename graph_iterator<C,N>::pointer graph_iterator<C,N>::operator->() const {
    return current_node;
}

template <typename C, typename N>
void graph_iterator<C,N>::internal_forward() {
    if (current_node) current_node = current_node->next;
}

//! Constructs a graph with isolated task_group_context
inline graph::graph() : my_wait_context(0), my_nodes(NULL), my_nodes_last(NULL), my_task_arena(NULL) {
    prepare_task_arena();
    own_context = true;
    cancelled = false;
    caught_exception = false;
    my_context = new (r1::cache_aligned_allocate(sizeof(task_group_context))) task_group_context(FLOW_TASKS);
    fgt_graph(this);
    my_is_active = true;
}

inline graph::graph(task_group_context& use_this_context) :
    my_wait_context(0), my_context(&use_this_context), my_nodes(NULL), my_nodes_last(NULL), my_task_arena(NULL) {
    prepare_task_arena();
    own_context = false;
    cancelled = false;
    caught_exception = false;
    fgt_graph(this);
    my_is_active = true;
}

inline graph::~graph() {
    wait_for_all();
    if (own_context) {
        my_context->~task_group_context();
        r1::cache_aligned_deallocate(my_context);
    }
    delete my_task_arena;
}

inline void graph::reserve_wait() {
    my_wait_context.reserve();
    fgt_reserve_wait(this);
}

inline void graph::release_wait() {
    fgt_release_wait(this);
    my_wait_context.release();
}

inline void graph::register_node(graph_node *n) {
    n->next = NULL;
    {
        spin_mutex::scoped_lock lock(nodelist_mutex);
        n->prev = my_nodes_last;
        if (my_nodes_last) my_nodes_last->next = n;
        my_nodes_last = n;
        if (!my_nodes) my_nodes = n;
    }
}

inline void graph::remove_node(graph_node *n) {
    {
        spin_mutex::scoped_lock lock(nodelist_mutex);
        __TBB_ASSERT(my_nodes && my_nodes_last, "graph::remove_node: Error: no registered nodes");
        if (n->prev) n->prev->next = n->next;
        if (n->next) n->next->prev = n->prev;
        if (my_nodes_last == n) my_nodes_last = n->prev;
        if (my_nodes == n) my_nodes = n->next;
    }
    n->prev = n->next = NULL;
}

inline void graph::reset( reset_flags f ) {
    // reset context
    deactivate_graph(*this);

    my_context->reset();
    cancelled = false;
    caught_exception = false;
    // reset all the nodes comprising the graph
    for(iterator ii = begin(); ii != end(); ++ii) {
        graph_node *my_p = &(*ii);
        my_p->reset_node(f);
    }
    // Reattach the arena. Might be useful to run the graph in a particular task_arena
    // while not limiting graph lifetime to a single task_arena::execute() call.
    prepare_task_arena( /*reinit=*/true );
    activate_graph(*this);
}

inline void graph::cancel() {
    my_context->cancel_group_execution();
}

inline graph::iterator graph::begin() { return iterator(this, true); }

inline graph::iterator graph::end() { return iterator(this, false); }

inline graph::const_iterator graph::begin() const { return const_iterator(this, true); }

inline graph::const_iterator graph::end() const { return const_iterator(this, false); }

inline graph::const_iterator graph::cbegin() const { return const_iterator(this, true); }

inline graph::const_iterator graph::cend() const { return const_iterator(this, false); }

inline graph_node::graph_node(graph& g) : my_graph(g) {
    my_graph.register_node(this);
}

inline graph_node::~graph_node() {
    my_graph.remove_node(this);
}

#include "detail/_flow_graph_node_impl.h"


//! An executable node that acts as a source, i.e. it has no predecessors

template < typename Output >
class input_node : public graph_node, public sender< Output > {
public:
    //! The type of the output message, which is complete
    typedef Output output_type;

    //! The type of successors of this node
    typedef typename sender<output_type>::successor_type successor_type;

    // Input node has no input type
    typedef null_type input_type;

    //! Constructor for a node with a successor
    template< typename Body >
     __TBB_NOINLINE_SYM input_node( graph &g, Body body )
         : graph_node(g), my_active(false)
         , my_body( new input_body_leaf< output_type, Body>(body) )
         , my_init_body( new input_body_leaf< output_type, Body>(body) )
         , my_successors(this), my_reserved(false), my_has_cached_item(false)
    {
        fgt_node_with_body(CODEPTR(), FLOW_INPUT_NODE, &this->my_graph,
                           static_cast<sender<output_type> *>(this), this->my_body);
    }

#if __TBB_PREVIEW_FLOW_GRAPH_NODE_SET
    template <typename Body, typename... Successors>
    input_node( const node_set<order::preceding, Successors...>& successors, Body body )
        : input_node(successors.graph_reference(), body)
    {
        make_edges(*this, successors);
    }
#endif

    //! Copy constructor
    __TBB_NOINLINE_SYM input_node( const input_node& src )
        : graph_node(src.my_graph), sender<Output>()
        , my_active(false)
        , my_body(src.my_init_body->clone()), my_init_body(src.my_init_body->clone())
        , my_successors(this), my_reserved(false), my_has_cached_item(false)
    {
        fgt_node_with_body(CODEPTR(), FLOW_INPUT_NODE, &this->my_graph,
                           static_cast<sender<output_type> *>(this), this->my_body);
    }

    //! The destructor
    ~input_node() { delete my_body; delete my_init_body; }

    //! Add a new successor to this node
    bool register_successor( successor_type &r ) override {
        spin_mutex::scoped_lock lock(my_mutex);
        my_successors.register_successor(r);
        if ( my_active )
            spawn_put();
        return true;
    }

    //! Removes a successor from this node
    bool remove_successor( successor_type &r ) override {
        spin_mutex::scoped_lock lock(my_mutex);
        my_successors.remove_successor(r);
        return true;
    }

    //! Request an item from the node
    bool try_get( output_type &v ) override {
        spin_mutex::scoped_lock lock(my_mutex);
        if ( my_reserved )
            return false;

        if ( my_has_cached_item ) {
            v = my_cached_item;
            my_has_cached_item = false;
            return true;
        }
        // we've been asked to provide an item, but we have none.  enqueue a task to
        // provide one.
        if ( my_active )
            spawn_put();
        return false;
    }

    //! Reserves an item.
    bool try_reserve( output_type &v ) override {
        spin_mutex::scoped_lock lock(my_mutex);
        if ( my_reserved ) {
            return false;
        }

        if ( my_has_cached_item ) {
            v = my_cached_item;
            my_reserved = true;
            return true;
        } else {
            return false;
        }
    }

    //! Release a reserved item.
    /** true = item has been released and so remains in sender, dest must request or reserve future items */
    bool try_release( ) override {
        spin_mutex::scoped_lock lock(my_mutex);
        __TBB_ASSERT( my_reserved && my_has_cached_item, "releasing non-existent reservation" );
        my_reserved = false;
        if(!my_successors.empty())
            spawn_put();
        return true;
    }

    //! Consumes a reserved item
    bool try_consume( ) override {
        spin_mutex::scoped_lock lock(my_mutex);
        __TBB_ASSERT( my_reserved && my_has_cached_item, "consuming non-existent reservation" );
        my_reserved = false;
        my_has_cached_item = false;
        if ( !my_successors.empty() ) {
            spawn_put();
        }
        return true;
    }

    //! Activates a node that was created in the inactive state
    void activate() {
        spin_mutex::scoped_lock lock(my_mutex);
        my_active = true;
        if (!my_successors.empty())
            spawn_put();
    }

    template<typename Body>
    Body copy_function_object() {
        input_body<output_type> &body_ref = *this->my_body;
        return dynamic_cast< input_body_leaf<output_type, Body> & >(body_ref).get_body();
    }

protected:

    //! resets the input_node to its initial state
    void reset_node( reset_flags f) override {
        my_active = false;
        my_reserved = false;
        my_has_cached_item = false;

        if(f & rf_clear_edges) my_successors.clear();
        if(f & rf_reset_bodies) {
            input_body<output_type> *tmp = my_init_body->clone();
            delete my_body;
            my_body = tmp;
        }
    }

private:
    spin_mutex my_mutex;
    bool my_active;
    input_body<output_type> *my_body;
    input_body<output_type> *my_init_body;
    broadcast_cache< output_type > my_successors;
    bool my_reserved;
    bool my_has_cached_item;
    output_type my_cached_item;

    // used by apply_body_bypass, can invoke body of node.
    bool try_reserve_apply_body(output_type &v) {
        spin_mutex::scoped_lock lock(my_mutex);
        if ( my_reserved ) {
            return false;
        }
        if ( !my_has_cached_item ) {
            flow_control control;

            fgt_begin_body( my_body );

            my_cached_item = (*my_body)(control);
            my_has_cached_item = !control.is_pipeline_stopped;

            fgt_end_body( my_body );
        }
        if ( my_has_cached_item ) {
            v = my_cached_item;
            my_reserved = true;
            return true;
        } else {
            return false;
        }
    }

    graph_task* create_put_task() {
        small_object_allocator allocator{};
        typedef input_node_task_bypass< input_node<output_type> > task_type;
        graph_task* t = allocator.new_object<task_type>(my_graph, allocator, *this);
        my_graph.reserve_wait();
        return t;
    }

    //! Spawns a task that applies the body
    void spawn_put( ) {
        if(is_graph_active(this->my_graph)) {
            spawn_in_graph_arena(this->my_graph, *create_put_task());
        }
    }

    friend class input_node_task_bypass< input_node<output_type> >;
    //! Applies the body.  Returning SUCCESSFULLY_ENQUEUED okay; forward_task_bypass will handle it.
    graph_task* apply_body_bypass( ) {
        output_type v;
        if ( !try_reserve_apply_body(v) )
            return NULL;

        graph_task *last_task = my_successors.try_put_task(v);
        if ( last_task )
            try_consume();
        else
            try_release();
        return last_task;
    }
};  // class input_node

//! Implements a function node that supports Input -> Output
template<typename Input, typename Output = continue_msg, typename Policy = queueing>
class function_node
    : public graph_node
    , public function_input< Input, Output, Policy, cache_aligned_allocator<Input> >
    , public function_output<Output>
{
    typedef cache_aligned_allocator<Input> internals_allocator;

public:
    typedef Input input_type;
    typedef Output output_type;
    typedef function_input<input_type,output_type,Policy,internals_allocator> input_impl_type;
    typedef function_input_queue<input_type, internals_allocator> input_queue_type;
    typedef function_output<output_type> fOutput_type;
    typedef typename input_impl_type::predecessor_type predecessor_type;
    typedef typename fOutput_type::successor_type successor_type;

    using input_impl_type::my_predecessors;

    //! Constructor
    // input_queue_type is allocated here, but destroyed in the function_input_base.
    // TODO: pass the graph_buffer_policy to the function_input_base so it can all
    // be done in one place.  This would be an interface-breaking change.
    template< typename Body >
     __TBB_NOINLINE_SYM function_node( graph &g, size_t concurrency,
                   Body body, Policy = Policy(), node_priority_t a_priority = no_priority )
        : graph_node(g), input_impl_type(g, concurrency, body, a_priority),
          fOutput_type(g) {
        fgt_node_with_body( CODEPTR(), FLOW_FUNCTION_NODE, &this->my_graph,
                static_cast<receiver<input_type> *>(this), static_cast<sender<output_type> *>(this), this->my_body );
    }

    template <typename Body>
    function_node( graph& g, size_t concurrency, Body body, node_priority_t a_priority )
        : function_node(g, concurrency, body, Policy(), a_priority) {}

#if __TBB_PREVIEW_FLOW_GRAPH_NODE_SET
    template <typename Body, typename... Args>
    function_node( const node_set<Args...>& nodes, size_t concurrency, Body body,
                   Policy p = Policy(), node_priority_t a_priority = no_priority )
        : function_node(nodes.graph_reference(), concurrency, body, p, a_priority) {
        make_edges_in_order(nodes, *this);
    }

    template <typename Body, typename... Args>
    function_node( const node_set<Args...>& nodes, size_t concurrency, Body body, node_priority_t a_priority )
        : function_node(nodes, concurrency, body, Policy(), a_priority) {}
#endif // __TBB_PREVIEW_FLOW_GRAPH_NODE_SET

    //! Copy constructor
    __TBB_NOINLINE_SYM function_node( const function_node& src ) :
        graph_node(src.my_graph),
        input_impl_type(src),
        fOutput_type(src.my_graph) {
        fgt_node_with_body( CODEPTR(), FLOW_FUNCTION_NODE, &this->my_graph,
                static_cast<receiver<input_type> *>(this), static_cast<sender<output_type> *>(this), this->my_body );
    }

protected:
    template< typename R, typename B > friend class run_and_put_task;
    template<typename X, typename Y> friend class broadcast_cache;
    template<typename X, typename Y> friend class round_robin_cache;
    using input_impl_type::try_put_task;

    broadcast_cache<output_type> &successors () override { return fOutput_type::my_successors; }

    void reset_node(reset_flags f) override {
        input_impl_type::reset_function_input(f);
        // TODO: use clear() instead.
        if(f & rf_clear_edges) {
            successors().clear();
            my_predecessors.clear();
        }
        __TBB_ASSERT(!(f & rf_clear_edges) || successors().empty(), "function_node successors not empty");
        __TBB_ASSERT(this->my_predecessors.empty(), "function_node predecessors not empty");
    }

};  // class function_node

//! implements a function node that supports Input -> (set of outputs)
// Output is a tuple of output types.
template<typename Input, typename Output, typename Policy = queueing>
class multifunction_node :
    public graph_node,
    public multifunction_input
    <
        Input,
        typename wrap_tuple_elements<
            std::tuple_size<Output>::value,  // #elements in tuple
            multifunction_output,  // wrap this around each element
            Output // the tuple providing the types
        >::type,
        Policy,
        cache_aligned_allocator<Input>
    >
{
    typedef cache_aligned_allocator<Input> internals_allocator;

protected:
    static const int N = std::tuple_size<Output>::value;
public:
    typedef Input input_type;
    typedef null_type output_type;
    typedef typename wrap_tuple_elements<N,multifunction_output, Output>::type output_ports_type;
    typedef multifunction_input<
        input_type, output_ports_type, Policy, internals_allocator> input_impl_type;
    typedef function_input_queue<input_type, internals_allocator> input_queue_type;
private:
    using input_impl_type::my_predecessors;
public:
    template<typename Body>
    __TBB_NOINLINE_SYM multifunction_node(
        graph &g, size_t concurrency,
        Body body, Policy = Policy(), node_priority_t a_priority = no_priority
    ) : graph_node(g), input_impl_type(g, concurrency, body, a_priority) {
        fgt_multioutput_node_with_body<N>(
            CODEPTR(), FLOW_MULTIFUNCTION_NODE,
            &this->my_graph, static_cast<receiver<input_type> *>(this),
            this->output_ports(), this->my_body
        );
    }

    template <typename Body>
    __TBB_NOINLINE_SYM multifunction_node(graph& g, size_t concurrency, Body body, node_priority_t a_priority)
        : multifunction_node(g, concurrency, body, Policy(), a_priority) {}

#if __TBB_PREVIEW_FLOW_GRAPH_NODE_SET
    template <typename Body, typename... Args>
    __TBB_NOINLINE_SYM multifunction_node(const node_set<Args...>& nodes, size_t concurrency, Body body,
                       Policy p = Policy(), node_priority_t a_priority = no_priority)
        : multifunction_node(nodes.graph_reference(), concurrency, body, p, a_priority) {
        make_edges_in_order(nodes, *this);
    }

    template <typename Body, typename... Args>
    __TBB_NOINLINE_SYM multifunction_node(const node_set<Args...>& nodes, size_t concurrency, Body body, node_priority_t a_priority)
        : multifunction_node(nodes, concurrency, body, Policy(), a_priority) {}
#endif // __TBB_PREVIEW_FLOW_GRAPH_NODE_SET

    __TBB_NOINLINE_SYM multifunction_node( const multifunction_node &other) :
        graph_node(other.my_graph), input_impl_type(other) {
        fgt_multioutput_node_with_body<N>( CODEPTR(), FLOW_MULTIFUNCTION_NODE,
                &this->my_graph, static_cast<receiver<input_type> *>(this),
                this->output_ports(), this->my_body );
    }

    // all the guts are in multifunction_input...
protected:
    void reset_node(reset_flags f) override { input_impl_type::reset(f); }
};  // multifunction_node

//! split_node: accepts a tuple as input, forwards each element of the tuple to its
//  successors.  The node has unlimited concurrency, so it does not reject inputs.
template<typename TupleType>
class split_node : public graph_node, public receiver<TupleType> {
    static const int N = std::tuple_size<TupleType>::value;
    typedef receiver<TupleType> base_type;
public:
    typedef TupleType input_type;
    typedef typename wrap_tuple_elements<
            N,  // #elements in tuple
            multifunction_output,  // wrap this around each element
            TupleType // the tuple providing the types
        >::type  output_ports_type;

    __TBB_NOINLINE_SYM explicit split_node(graph &g)
        : graph_node(g),
          my_output_ports(init_output_ports<output_ports_type>::call(g, my_output_ports))
    {
        fgt_multioutput_node<N>(CODEPTR(), FLOW_SPLIT_NODE, &this->my_graph,
            static_cast<receiver<input_type> *>(this), this->output_ports());
    }

#if __TBB_PREVIEW_FLOW_GRAPH_NODE_SET
    template <typename... Args>
    __TBB_NOINLINE_SYM split_node(const node_set<Args...>& nodes) : split_node(nodes.graph_reference()) {
        make_edges_in_order(nodes, *this);
    }
#endif

    __TBB_NOINLINE_SYM split_node(const split_node& other)
        : graph_node(other.my_graph), base_type(other),
          my_output_ports(init_output_ports<output_ports_type>::call(other.my_graph, my_output_ports))
    {
        fgt_multioutput_node<N>(CODEPTR(), FLOW_SPLIT_NODE, &this->my_graph,
            static_cast<receiver<input_type> *>(this), this->output_ports());
    }

    output_ports_type &output_ports() { return my_output_ports; }

protected:
    graph_task *try_put_task(const TupleType& t) override {
        // Sending split messages in parallel is not justified, as overheads would prevail.
        // Also, we do not have successors here. So we just tell the task returned here is successful.
        return emit_element<N>::emit_this(this->my_graph, t, output_ports());
    }
    void reset_node(reset_flags f) override {
        if (f & rf_clear_edges)
            clear_element<N>::clear_this(my_output_ports);

        __TBB_ASSERT(!(f & rf_clear_edges) || clear_element<N>::this_empty(my_output_ports), "split_node reset failed");
    }
    graph& graph_reference() const override {
        return my_graph;
    }

private:
    output_ports_type my_output_ports;
};

//! Implements an executable node that supports continue_msg -> Output
template <typename Output, typename Policy = Policy<void> >
class continue_node : public graph_node, public continue_input<Output, Policy>,
                      public function_output<Output> {
public:
    typedef continue_msg input_type;
    typedef Output output_type;
    typedef continue_input<Output, Policy> input_impl_type;
    typedef function_output<output_type> fOutput_type;
    typedef typename input_impl_type::predecessor_type predecessor_type;
    typedef typename fOutput_type::successor_type successor_type;

    //! Constructor for executable node with continue_msg -> Output
    template <typename Body >
    __TBB_NOINLINE_SYM continue_node(
        graph &g,
        Body body, Policy = Policy(), node_priority_t a_priority = no_priority
    ) : graph_node(g), input_impl_type( g, body, a_priority ),
        fOutput_type(g) {
        fgt_node_with_body( CODEPTR(), FLOW_CONTINUE_NODE, &this->my_graph,

                                           static_cast<receiver<input_type> *>(this),
                                           static_cast<sender<output_type> *>(this), this->my_body );
    }

    template <typename Body>
    continue_node( graph& g, Body body, node_priority_t a_priority )
        : continue_node(g, body, Policy(), a_priority) {}

#if __TBB_PREVIEW_FLOW_GRAPH_NODE_SET
    template <typename Body, typename... Args>
    continue_node( const node_set<Args...>& nodes, Body body,
                   Policy p = Policy(), node_priority_t a_priority = no_priority )
        : continue_node(nodes.graph_reference(), body, p, a_priority ) {
        make_edges_in_order(nodes, *this);
    }
    template <typename Body, typename... Args>
    continue_node( const node_set<Args...>& nodes, Body body, node_priority_t a_priority)
        : continue_node(nodes, body, Policy(), a_priority) {}
#endif // __TBB_PREVIEW_FLOW_GRAPH_NODE_SET

    //! Constructor for executable node with continue_msg -> Output
    template <typename Body >
    __TBB_NOINLINE_SYM continue_node(
        graph &g, int number_of_predecessors,
        Body body, Policy = Policy(), node_priority_t a_priority = no_priority
    ) : graph_node(g)
      , input_impl_type(g, number_of_predecessors, body, a_priority),
        fOutput_type(g) {
        fgt_node_with_body( CODEPTR(), FLOW_CONTINUE_NODE, &this->my_graph,
                                           static_cast<receiver<input_type> *>(this),
                                           static_cast<sender<output_type> *>(this), this->my_body );
    }

    template <typename Body>
    continue_node( graph& g, int number_of_predecessors, Body body, node_priority_t a_priority)
        : continue_node(g, number_of_predecessors, body, Policy(), a_priority) {}

#if __TBB_PREVIEW_FLOW_GRAPH_NODE_SET
    template <typename Body, typename... Args>
    continue_node( const node_set<Args...>& nodes, int number_of_predecessors,
                   Body body, Policy p = Policy(), node_priority_t a_priority = no_priority )
        : continue_node(nodes.graph_reference(), number_of_predecessors, body, p, a_priority) {
        make_edges_in_order(nodes, *this);
    }

    template <typename Body, typename... Args>
    continue_node( const node_set<Args...>& nodes, int number_of_predecessors,
                   Body body, node_priority_t a_priority )
        : continue_node(nodes, number_of_predecessors, body, Policy(), a_priority) {}
#endif

    //! Copy constructor
    __TBB_NOINLINE_SYM continue_node( const continue_node& src ) :
        graph_node(src.my_graph), input_impl_type(src),
        function_output<Output>(src.my_graph) {
        fgt_node_with_body( CODEPTR(), FLOW_CONTINUE_NODE, &this->my_graph,
                                           static_cast<receiver<input_type> *>(this),
                                           static_cast<sender<output_type> *>(this), this->my_body );
    }

protected:
    template< typename R, typename B > friend class run_and_put_task;
    template<typename X, typename Y> friend class broadcast_cache;
    template<typename X, typename Y> friend class round_robin_cache;
    using input_impl_type::try_put_task;
    broadcast_cache<output_type> &successors () override { return fOutput_type::my_successors; }

    void reset_node(reset_flags f) override {
        input_impl_type::reset_receiver(f);
        if(f & rf_clear_edges)successors().clear();
        __TBB_ASSERT(!(f & rf_clear_edges) || successors().empty(), "continue_node not reset");
    }
};  // continue_node

//! Forwards messages of type T to all successors
template <typename T>
class broadcast_node : public graph_node, public receiver<T>, public sender<T> {
public:
    typedef T input_type;
    typedef T output_type;
    typedef typename receiver<input_type>::predecessor_type predecessor_type;
    typedef typename sender<output_type>::successor_type successor_type;
private:
    broadcast_cache<input_type> my_successors;
public:

    __TBB_NOINLINE_SYM explicit broadcast_node(graph& g) : graph_node(g), my_successors(this) {
        fgt_node( CODEPTR(), FLOW_BROADCAST_NODE, &this->my_graph,
                  static_cast<receiver<input_type> *>(this), static_cast<sender<output_type> *>(this) );
    }

#if __TBB_PREVIEW_FLOW_GRAPH_NODE_SET
    template <typename... Args>
    broadcast_node(const node_set<Args...>& nodes) : broadcast_node(nodes.graph_reference()) {
        make_edges_in_order(nodes, *this);
    }
#endif

    // Copy constructor
    __TBB_NOINLINE_SYM broadcast_node( const broadcast_node& src ) : broadcast_node(src.my_graph) {}

    //! Adds a successor
    bool register_successor( successor_type &r ) override {
        my_successors.register_successor( r );
        return true;
    }

    //! Removes s as a successor
    bool remove_successor( successor_type &r ) override {
        my_successors.remove_successor( r );
        return true;
    }

protected:
    template< typename R, typename B > friend class run_and_put_task;
    template<typename X, typename Y> friend class broadcast_cache;
    template<typename X, typename Y> friend class round_robin_cache;
    //! build a task to run the successor if possible.  Default is old behavior.
    graph_task *try_put_task(const T& t) override {
        graph_task *new_task = my_successors.try_put_task(t);
        if (!new_task) new_task = SUCCESSFULLY_ENQUEUED;
        return new_task;
    }

    graph& graph_reference() const override {
        return my_graph;
    }

    void reset_node(reset_flags f) override {
        if (f&rf_clear_edges) {
           my_successors.clear();
        }
        __TBB_ASSERT(!(f & rf_clear_edges) || my_successors.empty(), "Error resetting broadcast_node");
    }
};  // broadcast_node

//! Forwards messages in arbitrary order
template <typename T>
class buffer_node
    : public graph_node
    , public reservable_item_buffer< T, cache_aligned_allocator<T> >
    , public receiver<T>, public sender<T>
{
    typedef cache_aligned_allocator<T> internals_allocator;

public:
    typedef T input_type;
    typedef T output_type;
    typedef typename receiver<input_type>::predecessor_type predecessor_type;
    typedef typename sender<output_type>::successor_type successor_type;
    typedef buffer_node<T> class_type;

protected:
    typedef size_t size_type;
    round_robin_cache< T, null_rw_mutex > my_successors;

    friend class forward_task_bypass< class_type >;

    enum op_type {reg_succ, rem_succ, req_item, res_item, rel_res, con_res, put_item, try_fwd_task
    };

    // implements the aggregator_operation concept
    class buffer_operation : public aggregated_operation< buffer_operation > {
    public:
        char type;
        T* elem;
        graph_task* ltask;
        successor_type *r;

        buffer_operation(const T& e, op_type t) : type(char(t))
                                                  , elem(const_cast<T*>(&e)) , ltask(NULL)
        {}
        buffer_operation(op_type t) : type(char(t)),  ltask(NULL) {}
    };

    bool forwarder_busy;
    typedef aggregating_functor<class_type, buffer_operation> handler_type;
    friend class aggregating_functor<class_type, buffer_operation>;
    aggregator< handler_type, buffer_operation> my_aggregator;

    virtual void handle_operations(buffer_operation *op_list) {
        handle_operations_impl(op_list, this);
    }

    template<typename derived_type>
    void handle_operations_impl(buffer_operation *op_list, derived_type* derived) {
        __TBB_ASSERT(static_cast<class_type*>(derived) == this, "'this' is not a base class for derived");

        buffer_operation *tmp = NULL;
        bool try_forwarding = false;
        while (op_list) {
            tmp = op_list;
            op_list = op_list->next;
            switch (tmp->type) {
            case reg_succ: internal_reg_succ(tmp); try_forwarding = true; break;
            case rem_succ: internal_rem_succ(tmp); break;
            case req_item: internal_pop(tmp); break;
            case res_item: internal_reserve(tmp); break;
            case rel_res:  internal_release(tmp); try_forwarding = true; break;
            case con_res:  internal_consume(tmp); try_forwarding = true; break;
            case put_item: try_forwarding = internal_push(tmp); break;
            case try_fwd_task: internal_forward_task(tmp); break;
            }
        }

        derived->order();

        if (try_forwarding && !forwarder_busy) {
            if(is_graph_active(this->my_graph)) {
                forwarder_busy = true;
                typedef forward_task_bypass<class_type> task_type;
                small_object_allocator allocator{};
                graph_task* new_task = allocator.new_object<task_type>(graph_reference(), allocator, *this);
                my_graph.reserve_wait();
                // tmp should point to the last item handled by the aggregator.  This is the operation
                // the handling thread enqueued.  So modifying that record will be okay.
                // TODO revamp: check that the issue is still present
                // workaround for icc bug  (at least 12.0 and 13.0)
                // error: function "tbb::flow::interfaceX::combine_tasks" cannot be called with the given argument list
                //        argument types are: (graph, graph_task *, graph_task *)
                graph_task *z = tmp->ltask;
                graph &g = this->my_graph;
                tmp->ltask = combine_tasks(g, z, new_task);  // in case the op generated a task
            }
        }
    }  // handle_operations

    inline graph_task *grab_forwarding_task( buffer_operation &op_data) {
        return op_data.ltask;
    }

    inline bool enqueue_forwarding_task(buffer_operation &op_data) {
        graph_task *ft = grab_forwarding_task(op_data);
        if(ft) {
            spawn_in_graph_arena(graph_reference(), *ft);
            return true;
        }
        return false;
    }

    //! This is executed by an enqueued task, the "forwarder"
    virtual graph_task *forward_task() {
        buffer_operation op_data(try_fwd_task);
        graph_task *last_task = NULL;
        do {
            op_data.status = WAIT;
            op_data.ltask = NULL;
            my_aggregator.execute(&op_data);

            // workaround for icc bug
            graph_task *xtask = op_data.ltask;
            graph& g = this->my_graph;
            last_task = combine_tasks(g, last_task, xtask);
        } while (op_data.status ==SUCCEEDED);
        return last_task;
    }

    //! Register successor
    virtual void internal_reg_succ(buffer_operation *op) {
        my_successors.register_successor(*(op->r));
        op->status.store(SUCCEEDED, std::memory_order_release);
    }

    //! Remove successor
    virtual void internal_rem_succ(buffer_operation *op) {
        my_successors.remove_successor(*(op->r));
        op->status.store(SUCCEEDED, std::memory_order_release);
    }

private:
    void order() {}

    bool is_item_valid() {
        return this->my_item_valid(this->my_tail - 1);
    }

    void try_put_and_add_task(graph_task*& last_task) {
        graph_task *new_task = my_successors.try_put_task(this->back());
        if (new_task) {
            // workaround for icc bug
            graph& g = this->my_graph;
            last_task = combine_tasks(g, last_task, new_task);
            this->destroy_back();
        }
    }

protected:
    //! Tries to forward valid items to successors
    virtual void internal_forward_task(buffer_operation *op) {
        internal_forward_task_impl(op, this);
    }

    template<typename derived_type>
    void internal_forward_task_impl(buffer_operation *op, derived_type* derived) {
        __TBB_ASSERT(static_cast<class_type*>(derived) == this, "'this' is not a base class for derived");

        if (this->my_reserved || !derived->is_item_valid()) {
            op->status.store(FAILED, std::memory_order_release);
            this->forwarder_busy = false;
            return;
        }
        // Try forwarding, giving each successor a chance
        graph_task* last_task = NULL;
        size_type counter = my_successors.size();
        for (; counter > 0 && derived->is_item_valid(); --counter)
            derived->try_put_and_add_task(last_task);

        op->ltask = last_task;  // return task
        if (last_task && !counter) {
            op->status.store(SUCCEEDED, std::memory_order_release);
        }
        else {
            op->status.store(FAILED, std::memory_order_release);
            forwarder_busy = false;
        }
    }

    virtual bool internal_push(buffer_operation *op) {
        this->push_back(*(op->elem));
        op->status.store(SUCCEEDED, std::memory_order_release);
        return true;
    }

    virtual void internal_pop(buffer_operation *op) {
        if(this->pop_back(*(op->elem))) {
            op->status.store(SUCCEEDED, std::memory_order_release);
        }
        else {
            op->status.store(FAILED, std::memory_order_release);
        }
    }

    virtual void internal_reserve(buffer_operation *op) {
        if(this->reserve_front(*(op->elem))) {
            op->status.store(SUCCEEDED, std::memory_order_release);
        }
        else {
            op->status.store(FAILED, std::memory_order_release);
        }
    }

    virtual void internal_consume(buffer_operation *op) {
        this->consume_front();
        op->status.store(SUCCEEDED, std::memory_order_release);
    }

    virtual void internal_release(buffer_operation *op) {
        this->release_front();
        op->status.store(SUCCEEDED, std::memory_order_release);
    }

public:
    //! Constructor
    __TBB_NOINLINE_SYM explicit buffer_node( graph &g )
        : graph_node(g), reservable_item_buffer<T, internals_allocator>(), receiver<T>(),
          sender<T>(), my_successors(this), forwarder_busy(false)
    {
        my_aggregator.initialize_handler(handler_type(this));
        fgt_node( CODEPTR(), FLOW_BUFFER_NODE, &this->my_graph,
                                 static_cast<receiver<input_type> *>(this), static_cast<sender<output_type> *>(this) );
    }

#if __TBB_PREVIEW_FLOW_GRAPH_NODE_SET
    template <typename... Args>
    buffer_node(const node_set<Args...>& nodes) : buffer_node(nodes.graph_reference()) {
        make_edges_in_order(nodes, *this);
    }
#endif

    //! Copy constructor
    __TBB_NOINLINE_SYM buffer_node( const buffer_node& src ) : buffer_node(src.my_graph) {}

    //
    // message sender implementation
    //

    //! Adds a new successor.
    /** Adds successor r to the list of successors; may forward tasks.  */
    bool register_successor( successor_type &r ) override {
        buffer_operation op_data(reg_succ);
        op_data.r = &r;
        my_aggregator.execute(&op_data);
        (void)enqueue_forwarding_task(op_data);
        return true;
    }

    //! Removes a successor.
    /** Removes successor r from the list of successors.
        It also calls r.remove_predecessor(*this) to remove this node as a predecessor. */
    bool remove_successor( successor_type &r ) override {
        // TODO revamp: investigate why full qualification is necessary here
        tbb::detail::d1::remove_predecessor(r, *this);
        buffer_operation op_data(rem_succ);
        op_data.r = &r;
        my_aggregator.execute(&op_data);
        // even though this operation does not cause a forward, if we are the handler, and
        // a forward is scheduled, we may be the first to reach this point after the aggregator,
        // and so should check for the task.
        (void)enqueue_forwarding_task(op_data);
        return true;
    }

    //! Request an item from the buffer_node
    /**  true = v contains the returned item<BR>
         false = no item has been returned */
    bool try_get( T &v ) override {
        buffer_operation op_data(req_item);
        op_data.elem = &v;
        my_aggregator.execute(&op_data);
        (void)enqueue_forwarding_task(op_data);
        return (op_data.status==SUCCEEDED);
    }

    //! Reserves an item.
    /**  false = no item can be reserved<BR>
         true = an item is reserved */
    bool try_reserve( T &v ) override {
        buffer_operation op_data(res_item);
        op_data.elem = &v;
        my_aggregator.execute(&op_data);
        (void)enqueue_forwarding_task(op_data);
        return (op_data.status==SUCCEEDED);
    }

    //! Release a reserved item.
    /**  true = item has been released and so remains in sender */
    bool try_release() override {
        buffer_operation op_data(rel_res);
        my_aggregator.execute(&op_data);
        (void)enqueue_forwarding_task(op_data);
        return true;
    }

    //! Consumes a reserved item.
    /** true = item is removed from sender and reservation removed */
    bool try_consume() override {
        buffer_operation op_data(con_res);
        my_aggregator.execute(&op_data);
        (void)enqueue_forwarding_task(op_data);
        return true;
    }

protected:

    template< typename R, typename B > friend class run_and_put_task;
    template<typename X, typename Y> friend class broadcast_cache;
    template<typename X, typename Y> friend class round_robin_cache;
    //! receive an item, return a task *if possible
    graph_task *try_put_task(const T &t) override {
        buffer_operation op_data(t, put_item);
        my_aggregator.execute(&op_data);
        graph_task *ft = grab_forwarding_task(op_data);
        // sequencer_nodes can return failure (if an item has been previously inserted)
        // We have to spawn the returned task if our own operation fails.

        if(ft && op_data.status ==FAILED) {
            // we haven't succeeded queueing the item, but for some reason the
            // call returned a task (if another request resulted in a successful
            // forward this could happen.)  Queue the task and reset the pointer.
            spawn_in_graph_arena(graph_reference(), *ft); ft = NULL;
        }
        else if(!ft && op_data.status ==SUCCEEDED) {
            ft = SUCCESSFULLY_ENQUEUED;
        }
        return ft;
    }

    graph& graph_reference() const override {
        return my_graph;
    }

protected:
    void reset_node( reset_flags f) override {
        reservable_item_buffer<T, internals_allocator>::reset();
        // TODO: just clear structures
        if (f&rf_clear_edges) {
            my_successors.clear();
        }
        forwarder_busy = false;
    }
};  // buffer_node

//! Forwards messages in FIFO order
template <typename T>
class queue_node : public buffer_node<T> {
protected:
    typedef buffer_node<T> base_type;
    typedef typename base_type::size_type size_type;
    typedef typename base_type::buffer_operation queue_operation;
    typedef queue_node class_type;

private:
    template<typename> friend class buffer_node;

    bool is_item_valid() {
        return this->my_item_valid(this->my_head);
    }

    void try_put_and_add_task(graph_task*& last_task) {
        graph_task *new_task = this->my_successors.try_put_task(this->front());
        if (new_task) {
            // workaround for icc bug
            graph& graph_ref = this->graph_reference();
            last_task = combine_tasks(graph_ref, last_task, new_task);
            this->destroy_front();
        }
    }

protected:
    void internal_forward_task(queue_operation *op) override {
        this->internal_forward_task_impl(op, this);
    }

    void internal_pop(queue_operation *op) override {
        if ( this->my_reserved || !this->my_item_valid(this->my_head)){
            op->status.store(FAILED, std::memory_order_release);
        }
        else {
            this->pop_front(*(op->elem));
            op->status.store(SUCCEEDED, std::memory_order_release);
        }
    }
    void internal_reserve(queue_operation *op) override {
        if (this->my_reserved || !this->my_item_valid(this->my_head)) {
            op->status.store(FAILED, std::memory_order_release);
        }
        else {
            this->reserve_front(*(op->elem));
            op->status.store(SUCCEEDED, std::memory_order_release);
        }
    }
    void internal_consume(queue_operation *op) override {
        this->consume_front();
        op->status.store(SUCCEEDED, std::memory_order_release);
    }

public:
    typedef T input_type;
    typedef T output_type;
    typedef typename receiver<input_type>::predecessor_type predecessor_type;
    typedef typename sender<output_type>::successor_type successor_type;

    //! Constructor
    __TBB_NOINLINE_SYM explicit queue_node( graph &g ) : base_type(g) {
        fgt_node( CODEPTR(), FLOW_QUEUE_NODE, &(this->my_graph),
                                 static_cast<receiver<input_type> *>(this),
                                 static_cast<sender<output_type> *>(this) );
    }

#if __TBB_PREVIEW_FLOW_GRAPH_NODE_SET
    template <typename... Args>
    queue_node( const node_set<Args...>& nodes) : queue_node(nodes.graph_reference()) {
        make_edges_in_order(nodes, *this);
    }
#endif

    //! Copy constructor
    __TBB_NOINLINE_SYM queue_node( const queue_node& src) : base_type(src) {
        fgt_node( CODEPTR(), FLOW_QUEUE_NODE, &(this->my_graph),
                                 static_cast<receiver<input_type> *>(this),
                                 static_cast<sender<output_type> *>(this) );
    }


protected:
    void reset_node( reset_flags f) override {
        base_type::reset_node(f);
    }
};  // queue_node

//! Forwards messages in sequence order
template <typename T>
class sequencer_node : public queue_node<T> {
    function_body< T, size_t > *my_sequencer;
    // my_sequencer should be a benign function and must be callable
    // from a parallel context.  Does this mean it needn't be reset?
public:
    typedef T input_type;
    typedef T output_type;
    typedef typename receiver<input_type>::predecessor_type predecessor_type;
    typedef typename sender<output_type>::successor_type successor_type;

    //! Constructor
    template< typename Sequencer >
    __TBB_NOINLINE_SYM sequencer_node( graph &g, const Sequencer& s ) : queue_node<T>(g),
        my_sequencer(new function_body_leaf< T, size_t, Sequencer>(s) ) {
        fgt_node( CODEPTR(), FLOW_SEQUENCER_NODE, &(this->my_graph),
                                 static_cast<receiver<input_type> *>(this),
                                 static_cast<sender<output_type> *>(this) );
    }

#if __TBB_PREVIEW_FLOW_GRAPH_NODE_SET
    template <typename Sequencer, typename... Args>
    sequencer_node( const node_set<Args...>& nodes, const Sequencer& s)
        : sequencer_node(nodes.graph_reference(), s) {
        make_edges_in_order(nodes, *this);
    }
#endif

    //! Copy constructor
    __TBB_NOINLINE_SYM sequencer_node( const sequencer_node& src ) : queue_node<T>(src),
        my_sequencer( src.my_sequencer->clone() ) {
        fgt_node( CODEPTR(), FLOW_SEQUENCER_NODE, &(this->my_graph),
                                 static_cast<receiver<input_type> *>(this),
                                 static_cast<sender<output_type> *>(this) );
    }

    //! Destructor
    ~sequencer_node() { delete my_sequencer; }

protected:
    typedef typename buffer_node<T>::size_type size_type;
    typedef typename buffer_node<T>::buffer_operation sequencer_operation;

private:
    bool internal_push(sequencer_operation *op) override {
        size_type tag = (*my_sequencer)(*(op->elem));
#if !TBB_DEPRECATED_SEQUENCER_DUPLICATES
        if (tag < this->my_head) {
            // have already emitted a message with this tag
            op->status.store(FAILED, std::memory_order_release);
            return false;
        }
#endif
        // cannot modify this->my_tail now; the buffer would be inconsistent.
        size_t new_tail = (tag+1 > this->my_tail) ? tag+1 : this->my_tail;

        if (this->size(new_tail) > this->capacity()) {
            this->grow_my_array(this->size(new_tail));
        }
        this->my_tail = new_tail;

        const op_stat res = this->place_item(tag, *(op->elem)) ? SUCCEEDED : FAILED;
        op->status.store(res, std::memory_order_release);
        return res ==SUCCEEDED;
    }
};  // sequencer_node

//! Forwards messages in priority order
template<typename T, typename Compare = std::less<T>>
class priority_queue_node : public buffer_node<T> {
public:
    typedef T input_type;
    typedef T output_type;
    typedef buffer_node<T> base_type;
    typedef priority_queue_node class_type;
    typedef typename receiver<input_type>::predecessor_type predecessor_type;
    typedef typename sender<output_type>::successor_type successor_type;

    //! Constructor
    __TBB_NOINLINE_SYM explicit priority_queue_node( graph &g, const Compare& comp = Compare() )
        : buffer_node<T>(g), compare(comp), mark(0) {
        fgt_node( CODEPTR(), FLOW_PRIORITY_QUEUE_NODE, &(this->my_graph),
                                 static_cast<receiver<input_type> *>(this),
                                 static_cast<sender<output_type> *>(this) );
    }

#if __TBB_PREVIEW_FLOW_GRAPH_NODE_SET
    template <typename... Args>
    priority_queue_node(const node_set<Args...>& nodes, const Compare& comp = Compare())
        : priority_queue_node(nodes.graph_reference(), comp) {
        make_edges_in_order(nodes, *this);
    }
#endif

    //! Copy constructor
    __TBB_NOINLINE_SYM priority_queue_node( const priority_queue_node &src )
        : buffer_node<T>(src), mark(0)
    {
        fgt_node( CODEPTR(), FLOW_PRIORITY_QUEUE_NODE, &(this->my_graph),
                                 static_cast<receiver<input_type> *>(this),
                                 static_cast<sender<output_type> *>(this) );
    }

protected:

    void reset_node( reset_flags f) override {
        mark = 0;
        base_type::reset_node(f);
    }

    typedef typename buffer_node<T>::size_type size_type;
    typedef typename buffer_node<T>::item_type item_type;
    typedef typename buffer_node<T>::buffer_operation prio_operation;

    //! Tries to forward valid items to successors
    void internal_forward_task(prio_operation *op) override {
        this->internal_forward_task_impl(op, this);
    }

    void handle_operations(prio_operation *op_list) override {
        this->handle_operations_impl(op_list, this);
    }

    bool internal_push(prio_operation *op) override {
        prio_push(*(op->elem));
        op->status.store(SUCCEEDED, std::memory_order_release);
        return true;
    }

    void internal_pop(prio_operation *op) override {
        // if empty or already reserved, don't pop
        if ( this->my_reserved == true || this->my_tail == 0 ) {
            op->status.store(FAILED, std::memory_order_release);
            return;
        }

        *(op->elem) = prio();
        op->status.store(SUCCEEDED, std::memory_order_release);
        prio_pop();

    }

    // pops the highest-priority item, saves copy
    void internal_reserve(prio_operation *op) override {
        if (this->my_reserved == true || this->my_tail == 0) {
            op->status.store(FAILED, std::memory_order_release);
            return;
        }
        this->my_reserved = true;
        *(op->elem) = prio();
        reserved_item = *(op->elem);
        op->status.store(SUCCEEDED, std::memory_order_release);
        prio_pop();
    }

    void internal_consume(prio_operation *op) override {
        op->status.store(SUCCEEDED, std::memory_order_release);
        this->my_reserved = false;
        reserved_item = input_type();
    }

    void internal_release(prio_operation *op) override {
        op->status.store(SUCCEEDED, std::memory_order_release);
        prio_push(reserved_item);
        this->my_reserved = false;
        reserved_item = input_type();
    }

private:
    template<typename> friend class buffer_node;

    void order() {
        if (mark < this->my_tail) heapify();
        __TBB_ASSERT(mark == this->my_tail, "mark unequal after heapify");
    }

    bool is_item_valid() {
        return this->my_tail > 0;
    }

    void try_put_and_add_task(graph_task*& last_task) {
        graph_task * new_task = this->my_successors.try_put_task(this->prio());
        if (new_task) {
            // workaround for icc bug
            graph& graph_ref = this->graph_reference();
            last_task = combine_tasks(graph_ref, last_task, new_task);
            prio_pop();
        }
    }

private:
    Compare compare;
    size_type mark;

    input_type reserved_item;

    // in case a reheap has not been done after a push, check if the mark item is higher than the 0'th item
    bool prio_use_tail() {
        __TBB_ASSERT(mark <= this->my_tail, "mark outside bounds before test");
        return mark < this->my_tail && compare(this->get_my_item(0), this->get_my_item(this->my_tail - 1));
    }

    // prio_push: checks that the item will fit, expand array if necessary, put at end
    void prio_push(const T &src) {
        if ( this->my_tail >= this->my_array_size )
            this->grow_my_array( this->my_tail + 1 );
        (void) this->place_item(this->my_tail, src);
        ++(this->my_tail);
        __TBB_ASSERT(mark < this->my_tail, "mark outside bounds after push");
    }

    // prio_pop: deletes highest priority item from the array, and if it is item
    // 0, move last item to 0 and reheap.  If end of array, just destroy and decrement tail
    // and mark.  Assumes the array has already been tested for emptiness; no failure.
    void prio_pop()  {
        if (prio_use_tail()) {
            // there are newly pushed elements; last one higher than top
            // copy the data
            this->destroy_item(this->my_tail-1);
            --(this->my_tail);
            __TBB_ASSERT(mark <= this->my_tail, "mark outside bounds after pop");
            return;
        }
        this->destroy_item(0);
        if(this->my_tail > 1) {
            // push the last element down heap
            __TBB_ASSERT(this->my_item_valid(this->my_tail - 1), NULL);
            this->move_item(0,this->my_tail - 1);
        }
        --(this->my_tail);
        if(mark > this->my_tail) --mark;
        if (this->my_tail > 1) // don't reheap for heap of size 1
            reheap();
        __TBB_ASSERT(mark <= this->my_tail, "mark outside bounds after pop");
    }

    const T& prio() {
        return this->get_my_item(prio_use_tail() ? this->my_tail-1 : 0);
    }

    // turn array into heap
    void heapify() {
        if(this->my_tail == 0) {
            mark = 0;
            return;
        }
        if (!mark) mark = 1;
        for (; mark<this->my_tail; ++mark) { // for each unheaped element
            size_type cur_pos = mark;
            input_type to_place;
            this->fetch_item(mark,to_place);
            do { // push to_place up the heap
                size_type parent = (cur_pos-1)>>1;
                if (!compare(this->get_my_item(parent), to_place))
                    break;
                this->move_item(cur_pos, parent);
                cur_pos = parent;
            } while( cur_pos );
            (void) this->place_item(cur_pos, to_place);
        }
    }

    // otherwise heapified array with new root element; rearrange to heap
    void reheap() {
        size_type cur_pos=0, child=1;
        while (child < mark) {
            size_type target = child;
            if (child+1<mark &&
                compare(this->get_my_item(child),
                        this->get_my_item(child+1)))
                ++target;
            // target now has the higher priority child
            if (compare(this->get_my_item(target),
                        this->get_my_item(cur_pos)))
                break;
            // swap
            this->swap_items(cur_pos, target);
            cur_pos = target;
            child = (cur_pos<<1)+1;
        }
    }
};  // priority_queue_node

//! Forwards messages only if the threshold has not been reached
/** This node forwards items until its threshold is reached.
    It contains no buffering.  If the downstream node rejects, the
    message is dropped. */
template< typename T, typename DecrementType=continue_msg >
class limiter_node : public graph_node, public receiver< T >, public sender< T > {
public:
    typedef T input_type;
    typedef T output_type;
    typedef typename receiver<input_type>::predecessor_type predecessor_type;
    typedef typename sender<output_type>::successor_type successor_type;
    //TODO: There is a lack of predefined types for its controlling "decrementer" port. It should be fixed later.

private:
    size_t my_threshold;
    size_t my_count; // number of successful puts
    size_t my_tries; // number of active put attempts
    reservable_predecessor_cache< T, spin_mutex > my_predecessors;
    spin_mutex my_mutex;
    broadcast_cache< T > my_successors;

    //! The internal receiver< DecrementType > that adjusts the count
    threshold_regulator< limiter_node<T, DecrementType>, DecrementType > decrement;

    graph_task* decrement_counter( long long delta ) {
        {
            spin_mutex::scoped_lock lock(my_mutex);
            if( delta > 0 && size_t(delta) > my_count )
                my_count = 0;
            else if( delta < 0 && size_t(delta) > my_threshold - my_count )
                my_count = my_threshold;
            else
                my_count -= size_t(delta); // absolute value of delta is sufficiently small
        }
        return forward_task();
    }

    // Let threshold_regulator call decrement_counter()
    friend class threshold_regulator< limiter_node<T, DecrementType>, DecrementType >;

    friend class forward_task_bypass< limiter_node<T,DecrementType> >;

    bool check_conditions() {  // always called under lock
        return ( my_count + my_tries < my_threshold && !my_predecessors.empty() && !my_successors.empty() );
    }

    // only returns a valid task pointer or NULL, never SUCCESSFULLY_ENQUEUED
    graph_task* forward_task() {
        input_type v;
        graph_task* rval = NULL;
        bool reserved = false;
            {
                spin_mutex::scoped_lock lock(my_mutex);
                if ( check_conditions() )
                    ++my_tries;
                else
                    return NULL;
            }

        //SUCCESS
        // if we can reserve and can put, we consume the reservation
        // we increment the count and decrement the tries
        if ( (my_predecessors.try_reserve(v)) == true ){
            reserved=true;
            if ( (rval = my_successors.try_put_task(v)) != NULL ){
                {
                    spin_mutex::scoped_lock lock(my_mutex);
                    ++my_count;
                    --my_tries;
                    my_predecessors.try_consume();
                    if ( check_conditions() ) {
                        if ( is_graph_active(this->my_graph) ) {
                            typedef forward_task_bypass<limiter_node<T, DecrementType>> task_type;
                            small_object_allocator allocator{};
                            graph_task* rtask = allocator.new_object<task_type>( my_graph, allocator, *this );
                            my_graph.reserve_wait();
                            spawn_in_graph_arena(graph_reference(), *rtask);
                        }
                    }
                }
                return rval;
            }
        }
        //FAILURE
        //if we can't reserve, we decrement the tries
        //if we can reserve but can't put, we decrement the tries and release the reservation
        {
            spin_mutex::scoped_lock lock(my_mutex);
            --my_tries;
            if (reserved) my_predecessors.try_release();
            if ( check_conditions() ) {
                if ( is_graph_active(this->my_graph) ) {
                    small_object_allocator allocator{};
                    typedef forward_task_bypass<limiter_node<T, DecrementType>> task_type;
                    graph_task* t = allocator.new_object<task_type>(my_graph, allocator, *this);
                    my_graph.reserve_wait();
                    __TBB_ASSERT(!rval, "Have two tasks to handle");
                    return t;
                }
            }
            return rval;
        }
    }

    void initialize() {
        fgt_node(
            CODEPTR(), FLOW_LIMITER_NODE, &this->my_graph,
            static_cast<receiver<input_type> *>(this), static_cast<receiver<DecrementType> *>(&decrement),
            static_cast<sender<output_type> *>(this)
        );
    }

public:
    //! Constructor
    limiter_node(graph &g, size_t threshold)
        : graph_node(g), my_threshold(threshold), my_count(0), my_tries(0), my_predecessors(this)
        , my_successors(this), decrement(this)
    {
        initialize();
    }

#if __TBB_PREVIEW_FLOW_GRAPH_NODE_SET
    template <typename... Args>
    limiter_node(const node_set<Args...>& nodes, size_t threshold)
        : limiter_node(nodes.graph_reference(), threshold) {
        make_edges_in_order(nodes, *this);
    }
#endif

    //! Copy constructor
    limiter_node( const limiter_node& src ) : limiter_node(src.my_graph, src.my_threshold) {}

    //! The interface for accessing internal receiver< DecrementType > that adjusts the count
    receiver<DecrementType>& decrementer() { return decrement; }

    //! Replace the current successor with this new successor
    bool register_successor( successor_type &r ) override {
        spin_mutex::scoped_lock lock(my_mutex);
        bool was_empty = my_successors.empty();
        my_successors.register_successor(r);
        //spawn a forward task if this is the only successor
        if ( was_empty && !my_predecessors.empty() && my_count + my_tries < my_threshold ) {
            if ( is_graph_active(this->my_graph) ) {
                small_object_allocator allocator{};
                typedef forward_task_bypass<limiter_node<T, DecrementType>> task_type;
                graph_task* t = allocator.new_object<task_type>(my_graph, allocator, *this);
                my_graph.reserve_wait();
                spawn_in_graph_arena(graph_reference(), *t);
            }
        }
        return true;
    }

    //! Removes a successor from this node
    /** r.remove_predecessor(*this) is also called. */
    bool remove_successor( successor_type &r ) override {
        // TODO revamp: investigate why qualification is needed for remove_predecessor() call
        tbb::detail::d1::remove_predecessor(r, *this);
        my_successors.remove_successor(r);
        return true;
    }

    //! Adds src to the list of cached predecessors.
    bool register_predecessor( predecessor_type &src ) override {
        spin_mutex::scoped_lock lock(my_mutex);
        my_predecessors.add( src );
        if ( my_count + my_tries < my_threshold && !my_successors.empty() && is_graph_active(this->my_graph) ) {
            small_object_allocator allocator{};
            typedef forward_task_bypass<limiter_node<T, DecrementType>> task_type;
            graph_task* t = allocator.new_object<task_type>(my_graph, allocator, *this);
            my_graph.reserve_wait();
            spawn_in_graph_arena(graph_reference(), *t);
        }
        return true;
    }

    //! Removes src from the list of cached predecessors.
    bool remove_predecessor( predecessor_type &src ) override {
        my_predecessors.remove( src );
        return true;
    }

protected:

    template< typename R, typename B > friend class run_and_put_task;
    template<typename X, typename Y> friend class broadcast_cache;
    template<typename X, typename Y> friend class round_robin_cache;
    //! Puts an item to this receiver
    graph_task* try_put_task( const T &t ) override {
        {
            spin_mutex::scoped_lock lock(my_mutex);
            if ( my_count + my_tries >= my_threshold )
                return NULL;
            else
                ++my_tries;
        }

        graph_task* rtask = my_successors.try_put_task(t);

        if ( !rtask ) {  // try_put_task failed.
            spin_mutex::scoped_lock lock(my_mutex);
            --my_tries;
            if (check_conditions() && is_graph_active(this->my_graph)) {
                small_object_allocator allocator{};
                typedef forward_task_bypass<limiter_node<T, DecrementType>> task_type;
                rtask = allocator.new_object<task_type>(my_graph, allocator, *this);
                my_graph.reserve_wait();
            }
        }
        else {
            spin_mutex::scoped_lock lock(my_mutex);
            ++my_count;
            --my_tries;
             }
        return rtask;
    }

    graph& graph_reference() const override { return my_graph; }

    void reset_node( reset_flags f) override {
        my_count = 0;
        if(f & rf_clear_edges) {
            my_predecessors.clear();
            my_successors.clear();
        }
        else
        {
            my_predecessors.reset( );
        }
        decrement.reset_receiver(f);
    }
};  // limiter_node

#include "detail/_flow_graph_join_impl.h"

template<typename OutputTuple, typename JP=queueing> class join_node;

template<typename OutputTuple>
class join_node<OutputTuple,reserving>: public unfolded_join_node<std::tuple_size<OutputTuple>::value, reserving_port, OutputTuple, reserving> {
private:
    static const int N = std::tuple_size<OutputTuple>::value;
    typedef unfolded_join_node<N, reserving_port, OutputTuple, reserving> unfolded_type;
public:
    typedef OutputTuple output_type;
    typedef typename unfolded_type::input_ports_type input_ports_type;
     __TBB_NOINLINE_SYM explicit join_node(graph &g) : unfolded_type(g) {
        fgt_multiinput_node<N>( CODEPTR(), FLOW_JOIN_NODE_RESERVING, &this->my_graph,
                                            this->input_ports(), static_cast< sender< output_type > *>(this) );
    }

#if __TBB_PREVIEW_FLOW_GRAPH_NODE_SET
    template <typename... Args>
    __TBB_NOINLINE_SYM join_node(const node_set<Args...>& nodes, reserving = reserving()) : join_node(nodes.graph_reference()) {
        make_edges_in_order(nodes, *this);
    }
#endif

    __TBB_NOINLINE_SYM join_node(const join_node &other) : unfolded_type(other) {
        fgt_multiinput_node<N>( CODEPTR(), FLOW_JOIN_NODE_RESERVING, &this->my_graph,
                                            this->input_ports(), static_cast< sender< output_type > *>(this) );
    }

};

template<typename OutputTuple>
class join_node<OutputTuple,queueing>: public unfolded_join_node<std::tuple_size<OutputTuple>::value, queueing_port, OutputTuple, queueing> {
private:
    static const int N = std::tuple_size<OutputTuple>::value;
    typedef unfolded_join_node<N, queueing_port, OutputTuple, queueing> unfolded_type;
public:
    typedef OutputTuple output_type;
    typedef typename unfolded_type::input_ports_type input_ports_type;
     __TBB_NOINLINE_SYM explicit join_node(graph &g) : unfolded_type(g) {
        fgt_multiinput_node<N>( CODEPTR(), FLOW_JOIN_NODE_QUEUEING, &this->my_graph,
                                            this->input_ports(), static_cast< sender< output_type > *>(this) );
    }

#if __TBB_PREVIEW_FLOW_GRAPH_NODE_SET
    template <typename... Args>
    __TBB_NOINLINE_SYM join_node(const node_set<Args...>& nodes, queueing = queueing()) : join_node(nodes.graph_reference()) {
        make_edges_in_order(nodes, *this);
    }
#endif

    __TBB_NOINLINE_SYM join_node(const join_node &other) : unfolded_type(other) {
        fgt_multiinput_node<N>( CODEPTR(), FLOW_JOIN_NODE_QUEUEING, &this->my_graph,
                                            this->input_ports(), static_cast< sender< output_type > *>(this) );
    }

};

// template for key_matching join_node
// tag_matching join_node is a specialization of key_matching, and is source-compatible.
template<typename OutputTuple, typename K, typename KHash>
class join_node<OutputTuple, key_matching<K, KHash> > : public unfolded_join_node<std::tuple_size<OutputTuple>::value,
      key_matching_port, OutputTuple, key_matching<K,KHash> > {
private:
    static const int N = std::tuple_size<OutputTuple>::value;
    typedef unfolded_join_node<N, key_matching_port, OutputTuple, key_matching<K,KHash> > unfolded_type;
public:
    typedef OutputTuple output_type;
    typedef typename unfolded_type::input_ports_type input_ports_type;

#if __TBB_PREVIEW_MESSAGE_BASED_KEY_MATCHING
    join_node(graph &g) : unfolded_type(g) {}
#endif  /* __TBB_PREVIEW_MESSAGE_BASED_KEY_MATCHING */

    template<typename __TBB_B0, typename __TBB_B1>
     __TBB_NOINLINE_SYM join_node(graph &g, __TBB_B0 b0, __TBB_B1 b1) : unfolded_type(g, b0, b1) {
        fgt_multiinput_node<N>( CODEPTR(), FLOW_JOIN_NODE_TAG_MATCHING, &this->my_graph,
                                                           this->input_ports(), static_cast< sender< output_type > *>(this) );
    }
    template<typename __TBB_B0, typename __TBB_B1, typename __TBB_B2>
     __TBB_NOINLINE_SYM join_node(graph &g, __TBB_B0 b0, __TBB_B1 b1, __TBB_B2 b2) : unfolded_type(g, b0, b1, b2) {
        fgt_multiinput_node<N>( CODEPTR(), FLOW_JOIN_NODE_TAG_MATCHING, &this->my_graph,
                                                           this->input_ports(), static_cast< sender< output_type > *>(this) );
    }
    template<typename __TBB_B0, typename __TBB_B1, typename __TBB_B2, typename __TBB_B3>
     __TBB_NOINLINE_SYM join_node(graph &g, __TBB_B0 b0, __TBB_B1 b1, __TBB_B2 b2, __TBB_B3 b3) : unfolded_type(g, b0, b1, b2, b3) {
        fgt_multiinput_node<N>( CODEPTR(), FLOW_JOIN_NODE_TAG_MATCHING, &this->my_graph,
                                                           this->input_ports(), static_cast< sender< output_type > *>(this) );
    }
    template<typename __TBB_B0, typename __TBB_B1, typename __TBB_B2, typename __TBB_B3, typename __TBB_B4>
     __TBB_NOINLINE_SYM join_node(graph &g, __TBB_B0 b0, __TBB_B1 b1, __TBB_B2 b2, __TBB_B3 b3, __TBB_B4 b4) :
            unfolded_type(g, b0, b1, b2, b3, b4) {
        fgt_multiinput_node<N>( CODEPTR(), FLOW_JOIN_NODE_TAG_MATCHING, &this->my_graph,
                                                           this->input_ports(), static_cast< sender< output_type > *>(this) );
    }
#if __TBB_VARIADIC_MAX >= 6
    template<typename __TBB_B0, typename __TBB_B1, typename __TBB_B2, typename __TBB_B3, typename __TBB_B4,
        typename __TBB_B5>
     __TBB_NOINLINE_SYM join_node(graph &g, __TBB_B0 b0, __TBB_B1 b1, __TBB_B2 b2, __TBB_B3 b3, __TBB_B4 b4, __TBB_B5 b5) :
            unfolded_type(g, b0, b1, b2, b3, b4, b5) {
        fgt_multiinput_node<N>( CODEPTR(), FLOW_JOIN_NODE_TAG_MATCHING, &this->my_graph,
                                                           this->input_ports(), static_cast< sender< output_type > *>(this) );
    }
#endif
#if __TBB_VARIADIC_MAX >= 7
    template<typename __TBB_B0, typename __TBB_B1, typename __TBB_B2, typename __TBB_B3, typename __TBB_B4,
        typename __TBB_B5, typename __TBB_B6>
     __TBB_NOINLINE_SYM join_node(graph &g, __TBB_B0 b0, __TBB_B1 b1, __TBB_B2 b2, __TBB_B3 b3, __TBB_B4 b4, __TBB_B5 b5, __TBB_B6 b6) :
            unfolded_type(g, b0, b1, b2, b3, b4, b5, b6) {
        fgt_multiinput_node<N>( CODEPTR(), FLOW_JOIN_NODE_TAG_MATCHING, &this->my_graph,
                                                           this->input_ports(), static_cast< sender< output_type > *>(this) );
    }
#endif
#if __TBB_VARIADIC_MAX >= 8
    template<typename __TBB_B0, typename __TBB_B1, typename __TBB_B2, typename __TBB_B3, typename __TBB_B4,
        typename __TBB_B5, typename __TBB_B6, typename __TBB_B7>
     __TBB_NOINLINE_SYM join_node(graph &g, __TBB_B0 b0, __TBB_B1 b1, __TBB_B2 b2, __TBB_B3 b3, __TBB_B4 b4, __TBB_B5 b5, __TBB_B6 b6,
            __TBB_B7 b7) : unfolded_type(g, b0, b1, b2, b3, b4, b5, b6, b7) {
        fgt_multiinput_node<N>( CODEPTR(), FLOW_JOIN_NODE_TAG_MATCHING, &this->my_graph,
                                                           this->input_ports(), static_cast< sender< output_type > *>(this) );
    }
#endif
#if __TBB_VARIADIC_MAX >= 9
    template<typename __TBB_B0, typename __TBB_B1, typename __TBB_B2, typename __TBB_B3, typename __TBB_B4,
        typename __TBB_B5, typename __TBB_B6, typename __TBB_B7, typename __TBB_B8>
     __TBB_NOINLINE_SYM join_node(graph &g, __TBB_B0 b0, __TBB_B1 b1, __TBB_B2 b2, __TBB_B3 b3, __TBB_B4 b4, __TBB_B5 b5, __TBB_B6 b6,
            __TBB_B7 b7, __TBB_B8 b8) : unfolded_type(g, b0, b1, b2, b3, b4, b5, b6, b7, b8) {
        fgt_multiinput_node<N>( CODEPTR(), FLOW_JOIN_NODE_TAG_MATCHING, &this->my_graph,
                                                           this->input_ports(), static_cast< sender< output_type > *>(this) );
    }
#endif
#if __TBB_VARIADIC_MAX >= 10
    template<typename __TBB_B0, typename __TBB_B1, typename __TBB_B2, typename __TBB_B3, typename __TBB_B4,
        typename __TBB_B5, typename __TBB_B6, typename __TBB_B7, typename __TBB_B8, typename __TBB_B9>
     __TBB_NOINLINE_SYM join_node(graph &g, __TBB_B0 b0, __TBB_B1 b1, __TBB_B2 b2, __TBB_B3 b3, __TBB_B4 b4, __TBB_B5 b5, __TBB_B6 b6,
            __TBB_B7 b7, __TBB_B8 b8, __TBB_B9 b9) : unfolded_type(g, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9) {
        fgt_multiinput_node<N>( CODEPTR(), FLOW_JOIN_NODE_TAG_MATCHING, &this->my_graph,
                                                           this->input_ports(), static_cast< sender< output_type > *>(this) );
    }
#endif

#if __TBB_PREVIEW_FLOW_GRAPH_NODE_SET
    template <
#if (__clang_major__ == 3 && __clang_minor__ == 4)
        // clang 3.4 misdeduces 'Args...' for 'node_set' while it can cope with template template parameter.
        template<typename...> class node_set,
#endif
        typename... Args, typename... Bodies
    >
    __TBB_NOINLINE_SYM join_node(const node_set<Args...>& nodes, Bodies... bodies)
        : join_node(nodes.graph_reference(), bodies...) {
        make_edges_in_order(nodes, *this);
    }
#endif

    __TBB_NOINLINE_SYM join_node(const join_node &other) : unfolded_type(other) {
        fgt_multiinput_node<N>( CODEPTR(), FLOW_JOIN_NODE_TAG_MATCHING, &this->my_graph,
                                                           this->input_ports(), static_cast< sender< output_type > *>(this) );
    }

};

// indexer node
#include "detail/_flow_graph_indexer_impl.h"

// TODO: Implement interface with variadic template or tuple
template<typename T0, typename T1=null_type, typename T2=null_type, typename T3=null_type,
                      typename T4=null_type, typename T5=null_type, typename T6=null_type,
                      typename T7=null_type, typename T8=null_type, typename T9=null_type> class indexer_node;

//indexer node specializations
template<typename T0>
class indexer_node<T0> : public unfolded_indexer_node<std::tuple<T0> > {
private:
    static const int N = 1;
public:
    typedef std::tuple<T0> InputTuple;
    typedef tagged_msg<size_t, T0> output_type;
    typedef unfolded_indexer_node<InputTuple> unfolded_type;
    __TBB_NOINLINE_SYM indexer_node(graph& g) : unfolded_type(g) {
        fgt_multiinput_node<N>( CODEPTR(), FLOW_INDEXER_NODE, &this->my_graph,
                                           this->input_ports(), static_cast< sender< output_type > *>(this) );
    }

#if __TBB_PREVIEW_FLOW_GRAPH_NODE_SET
    template <typename... Args>
    indexer_node(const node_set<Args...>& nodes) : indexer_node(nodes.graph_reference()) {
        make_edges_in_order(nodes, *this);
    }
#endif

    // Copy constructor
    __TBB_NOINLINE_SYM indexer_node( const indexer_node& other ) : unfolded_type(other) {
        fgt_multiinput_node<N>( CODEPTR(), FLOW_INDEXER_NODE, &this->my_graph,
                                           this->input_ports(), static_cast< sender< output_type > *>(this) );
    }
};

template<typename T0, typename T1>
class indexer_node<T0, T1> : public unfolded_indexer_node<std::tuple<T0, T1> > {
private:
    static const int N = 2;
public:
    typedef std::tuple<T0, T1> InputTuple;
    typedef tagged_msg<size_t, T0, T1> output_type;
    typedef unfolded_indexer_node<InputTuple> unfolded_type;
    __TBB_NOINLINE_SYM indexer_node(graph& g) : unfolded_type(g) {
        fgt_multiinput_node<N>( CODEPTR(), FLOW_INDEXER_NODE, &this->my_graph,
                                           this->input_ports(), static_cast< sender< output_type > *>(this) );
    }

#if __TBB_PREVIEW_FLOW_GRAPH_NODE_SET
    template <typename... Args>
    indexer_node(const node_set<Args...>& nodes) : indexer_node(nodes.graph_reference()) {
        make_edges_in_order(nodes, *this);
    }
#endif

    // Copy constructor
    __TBB_NOINLINE_SYM indexer_node( const indexer_node& other ) : unfolded_type(other) {
        fgt_multiinput_node<N>( CODEPTR(), FLOW_INDEXER_NODE, &this->my_graph,
                                           this->input_ports(), static_cast< sender< output_type > *>(this) );
    }

};

template<typename T0, typename T1, typename T2>
class indexer_node<T0, T1, T2> : public unfolded_indexer_node<std::tuple<T0, T1, T2> > {
private:
    static const int N = 3;
public:
    typedef std::tuple<T0, T1, T2> InputTuple;
    typedef tagged_msg<size_t, T0, T1, T2> output_type;
    typedef unfolded_indexer_node<InputTuple> unfolded_type;
    __TBB_NOINLINE_SYM indexer_node(graph& g) : unfolded_type(g) {
        fgt_multiinput_node<N>( CODEPTR(), FLOW_INDEXER_NODE, &this->my_graph,
                                           this->input_ports(), static_cast< sender< output_type > *>(this) );
    }

#if __TBB_PREVIEW_FLOW_GRAPH_NODE_SET
    template <typename... Args>
    indexer_node(const node_set<Args...>& nodes) : indexer_node(nodes.graph_reference()) {
        make_edges_in_order(nodes, *this);
    }
#endif

    // Copy constructor
    __TBB_NOINLINE_SYM indexer_node( const indexer_node& other ) : unfolded_type(other) {
        fgt_multiinput_node<N>( CODEPTR(), FLOW_INDEXER_NODE, &this->my_graph,
                                           this->input_ports(), static_cast< sender< output_type > *>(this) );
    }

};

template<typename T0, typename T1, typename T2, typename T3>
class indexer_node<T0, T1, T2, T3> : public unfolded_indexer_node<std::tuple<T0, T1, T2, T3> > {
private:
    static const int N = 4;
public:
    typedef std::tuple<T0, T1, T2, T3> InputTuple;
    typedef tagged_msg<size_t, T0, T1, T2, T3> output_type;
    typedef unfolded_indexer_node<InputTuple> unfolded_type;
    __TBB_NOINLINE_SYM indexer_node(graph& g) : unfolded_type(g) {
        fgt_multiinput_node<N>( CODEPTR(), FLOW_INDEXER_NODE, &this->my_graph,
                                           this->input_ports(), static_cast< sender< output_type > *>(this) );
    }

#if __TBB_PREVIEW_FLOW_GRAPH_NODE_SET
    template <typename... Args>
    indexer_node(const node_set<Args...>& nodes) : indexer_node(nodes.graph_reference()) {
        make_edges_in_order(nodes, *this);
    }
#endif

    // Copy constructor
    __TBB_NOINLINE_SYM indexer_node( const indexer_node& other ) : unfolded_type(other) {
        fgt_multiinput_node<N>( CODEPTR(), FLOW_INDEXER_NODE, &this->my_graph,
                                           this->input_ports(), static_cast< sender< output_type > *>(this) );
    }

};

template<typename T0, typename T1, typename T2, typename T3, typename T4>
class indexer_node<T0, T1, T2, T3, T4> : public unfolded_indexer_node<std::tuple<T0, T1, T2, T3, T4> > {
private:
    static const int N = 5;
public:
    typedef std::tuple<T0, T1, T2, T3, T4> InputTuple;
    typedef tagged_msg<size_t, T0, T1, T2, T3, T4> output_type;
    typedef unfolded_indexer_node<InputTuple> unfolded_type;
    __TBB_NOINLINE_SYM indexer_node(graph& g) : unfolded_type(g) {
        fgt_multiinput_node<N>( CODEPTR(), FLOW_INDEXER_NODE, &this->my_graph,
                                           this->input_ports(), static_cast< sender< output_type > *>(this) );
    }

#if __TBB_PREVIEW_FLOW_GRAPH_NODE_SET
    template <typename... Args>
    indexer_node(const node_set<Args...>& nodes) : indexer_node(nodes.graph_reference()) {
        make_edges_in_order(nodes, *this);
    }
#endif

    // Copy constructor
    __TBB_NOINLINE_SYM indexer_node( const indexer_node& other ) : unfolded_type(other) {
        fgt_multiinput_node<N>( CODEPTR(), FLOW_INDEXER_NODE, &this->my_graph,
                                           this->input_ports(), static_cast< sender< output_type > *>(this) );
    }

};

#if __TBB_VARIADIC_MAX >= 6
template<typename T0, typename T1, typename T2, typename T3, typename T4, typename T5>
class indexer_node<T0, T1, T2, T3, T4, T5> : public unfolded_indexer_node<std::tuple<T0, T1, T2, T3, T4, T5> > {
private:
    static const int N = 6;
public:
    typedef std::tuple<T0, T1, T2, T3, T4, T5> InputTuple;
    typedef tagged_msg<size_t, T0, T1, T2, T3, T4, T5> output_type;
    typedef unfolded_indexer_node<InputTuple> unfolded_type;
    __TBB_NOINLINE_SYM indexer_node(graph& g) : unfolded_type(g) {
        fgt_multiinput_node<N>( CODEPTR(), FLOW_INDEXER_NODE, &this->my_graph,
                                           this->input_ports(), static_cast< sender< output_type > *>(this) );
    }

#if __TBB_PREVIEW_FLOW_GRAPH_NODE_SET
    template <typename... Args>
    indexer_node(const node_set<Args...>& nodes) : indexer_node(nodes.graph_reference()) {
        make_edges_in_order(nodes, *this);
    }
#endif

    // Copy constructor
    __TBB_NOINLINE_SYM indexer_node( const indexer_node& other ) : unfolded_type(other) {
        fgt_multiinput_node<N>( CODEPTR(), FLOW_INDEXER_NODE, &this->my_graph,
                                           this->input_ports(), static_cast< sender< output_type > *>(this) );
    }

};
#endif //variadic max 6

#if __TBB_VARIADIC_MAX >= 7
template<typename T0, typename T1, typename T2, typename T3, typename T4, typename T5,
         typename T6>
class indexer_node<T0, T1, T2, T3, T4, T5, T6> : public unfolded_indexer_node<std::tuple<T0, T1, T2, T3, T4, T5, T6> > {
private:
    static const int N = 7;
public:
    typedef std::tuple<T0, T1, T2, T3, T4, T5, T6> InputTuple;
    typedef tagged_msg<size_t, T0, T1, T2, T3, T4, T5, T6> output_type;
    typedef unfolded_indexer_node<InputTuple> unfolded_type;
    __TBB_NOINLINE_SYM indexer_node(graph& g) : unfolded_type(g) {
        fgt_multiinput_node<N>( CODEPTR(), FLOW_INDEXER_NODE, &this->my_graph,
                                           this->input_ports(), static_cast< sender< output_type > *>(this) );
    }

#if __TBB_PREVIEW_FLOW_GRAPH_NODE_SET
    template <typename... Args>
    indexer_node(const node_set<Args...>& nodes) : indexer_node(nodes.graph_reference()) {
        make_edges_in_order(nodes, *this);
    }
#endif

    // Copy constructor
    __TBB_NOINLINE_SYM indexer_node( const indexer_node& other ) : unfolded_type(other) {
        fgt_multiinput_node<N>( CODEPTR(), FLOW_INDEXER_NODE, &this->my_graph,
                                           this->input_ports(), static_cast< sender< output_type > *>(this) );
    }

};
#endif //variadic max 7

#if __TBB_VARIADIC_MAX >= 8
template<typename T0, typename T1, typename T2, typename T3, typename T4, typename T5,
         typename T6, typename T7>
class indexer_node<T0, T1, T2, T3, T4, T5, T6, T7> : public unfolded_indexer_node<std::tuple<T0, T1, T2, T3, T4, T5, T6, T7> > {
private:
    static const int N = 8;
public:
    typedef std::tuple<T0, T1, T2, T3, T4, T5, T6, T7> InputTuple;
    typedef tagged_msg<size_t, T0, T1, T2, T3, T4, T5, T6, T7> output_type;
    typedef unfolded_indexer_node<InputTuple> unfolded_type;
    indexer_node(graph& g) : unfolded_type(g) {
        fgt_multiinput_node<N>( CODEPTR(), FLOW_INDEXER_NODE, &this->my_graph,
                                           this->input_ports(), static_cast< sender< output_type > *>(this) );
    }

#if __TBB_PREVIEW_FLOW_GRAPH_NODE_SET
    template <typename... Args>
    indexer_node(const node_set<Args...>& nodes) : indexer_node(nodes.graph_reference()) {
        make_edges_in_order(nodes, *this);
    }
#endif

    // Copy constructor
    indexer_node( const indexer_node& other ) : unfolded_type(other) {
        fgt_multiinput_node<N>( CODEPTR(), FLOW_INDEXER_NODE, &this->my_graph,
                                           this->input_ports(), static_cast< sender< output_type > *>(this) );
    }

};
#endif //variadic max 8

#if __TBB_VARIADIC_MAX >= 9
template<typename T0, typename T1, typename T2, typename T3, typename T4, typename T5,
         typename T6, typename T7, typename T8>
class indexer_node<T0, T1, T2, T3, T4, T5, T6, T7, T8> : public unfolded_indexer_node<std::tuple<T0, T1, T2, T3, T4, T5, T6, T7, T8> > {
private:
    static const int N = 9;
public:
    typedef std::tuple<T0, T1, T2, T3, T4, T5, T6, T7, T8> InputTuple;
    typedef tagged_msg<size_t, T0, T1, T2, T3, T4, T5, T6, T7, T8> output_type;
    typedef unfolded_indexer_node<InputTuple> unfolded_type;
    __TBB_NOINLINE_SYM indexer_node(graph& g) : unfolded_type(g) {
        fgt_multiinput_node<N>( CODEPTR(), FLOW_INDEXER_NODE, &this->my_graph,
                                           this->input_ports(), static_cast< sender< output_type > *>(this) );
    }

#if __TBB_PREVIEW_FLOW_GRAPH_NODE_SET
    template <typename... Args>
    indexer_node(const node_set<Args...>& nodes) : indexer_node(nodes.graph_reference()) {
        make_edges_in_order(nodes, *this);
    }
#endif

    // Copy constructor
    __TBB_NOINLINE_SYM indexer_node( const indexer_node& other ) : unfolded_type(other) {
        fgt_multiinput_node<N>( CODEPTR(), FLOW_INDEXER_NODE, &this->my_graph,
                                           this->input_ports(), static_cast< sender< output_type > *>(this) );
    }

};
#endif //variadic max 9

#if __TBB_VARIADIC_MAX >= 10
template<typename T0, typename T1, typename T2, typename T3, typename T4, typename T5,
         typename T6, typename T7, typename T8, typename T9>
class indexer_node/*default*/ : public unfolded_indexer_node<std::tuple<T0, T1, T2, T3, T4, T5, T6, T7, T8, T9> > {
private:
    static const int N = 10;
public:
    typedef std::tuple<T0, T1, T2, T3, T4, T5, T6, T7, T8, T9> InputTuple;
    typedef tagged_msg<size_t, T0, T1, T2, T3, T4, T5, T6, T7, T8, T9> output_type;
    typedef unfolded_indexer_node<InputTuple> unfolded_type;
    __TBB_NOINLINE_SYM indexer_node(graph& g) : unfolded_type(g) {
        fgt_multiinput_node<N>( CODEPTR(), FLOW_INDEXER_NODE, &this->my_graph,
                                           this->input_ports(), static_cast< sender< output_type > *>(this) );
    }

#if __TBB_PREVIEW_FLOW_GRAPH_NODE_SET
    template <typename... Args>
    indexer_node(const node_set<Args...>& nodes) : indexer_node(nodes.graph_reference()) {
        make_edges_in_order(nodes, *this);
    }
#endif

    // Copy constructor
    __TBB_NOINLINE_SYM indexer_node( const indexer_node& other ) : unfolded_type(other) {
        fgt_multiinput_node<N>( CODEPTR(), FLOW_INDEXER_NODE, &this->my_graph,
                                           this->input_ports(), static_cast< sender< output_type > *>(this) );
    }

};
#endif //variadic max 10

template< typename T >
inline void internal_make_edge( sender<T> &p, receiver<T> &s ) {
    register_successor(p, s);
    fgt_make_edge( &p, &s );
}

//! Makes an edge between a single predecessor and a single successor
template< typename T >
inline void make_edge( sender<T> &p, receiver<T> &s ) {
    internal_make_edge( p, s );
}

//Makes an edge from port 0 of a multi-output predecessor to port 0 of a multi-input successor.
template< typename T, typename V,
          typename = typename T::output_ports_type, typename = typename V::input_ports_type >
inline void make_edge( T& output, V& input) {
    make_edge(std::get<0>(output.output_ports()), std::get<0>(input.input_ports()));
}

//Makes an edge from port 0 of a multi-output predecessor to a receiver.
template< typename T, typename R,
          typename = typename T::output_ports_type >
inline void make_edge( T& output, receiver<R>& input) {
     make_edge(std::get<0>(output.output_ports()), input);
}

//Makes an edge from a sender to port 0 of a multi-input successor.
template< typename S,  typename V,
          typename = typename V::input_ports_type >
inline void make_edge( sender<S>& output, V& input) {
     make_edge(output, std::get<0>(input.input_ports()));
}

template< typename T >
inline void internal_remove_edge( sender<T> &p, receiver<T> &s ) {
    remove_successor( p, s );
    fgt_remove_edge( &p, &s );
}

//! Removes an edge between a single predecessor and a single successor
template< typename T >
inline void remove_edge( sender<T> &p, receiver<T> &s ) {
    internal_remove_edge( p, s );
}

//Removes an edge between port 0 of a multi-output predecessor and port 0 of a multi-input successor.
template< typename T, typename V,
          typename = typename T::output_ports_type, typename = typename V::input_ports_type >
inline void remove_edge( T& output, V& input) {
    remove_edge(std::get<0>(output.output_ports()), std::get<0>(input.input_ports()));
}

//Removes an edge between port 0 of a multi-output predecessor and a receiver.
template< typename T, typename R,
          typename = typename T::output_ports_type >
inline void remove_edge( T& output, receiver<R>& input) {
     remove_edge(std::get<0>(output.output_ports()), input);
}
//Removes an edge between a sender and port 0 of a multi-input successor.
template< typename S,  typename V,
          typename = typename V::input_ports_type >
inline void remove_edge( sender<S>& output, V& input) {
     remove_edge(output, std::get<0>(input.input_ports()));
}

//! Returns a copy of the body from a function or continue node
template< typename Body, typename Node >
Body copy_body( Node &n ) {
    return n.template copy_function_object<Body>();
}

//composite_node
template< typename InputTuple, typename OutputTuple > class composite_node;

template< typename... InputTypes, typename... OutputTypes>
class composite_node <std::tuple<InputTypes...>, std::tuple<OutputTypes...> > : public graph_node {

public:
    typedef std::tuple< receiver<InputTypes>&... > input_ports_type;
    typedef std::tuple< sender<OutputTypes>&... > output_ports_type;

private:
    std::unique_ptr<input_ports_type> my_input_ports;
    std::unique_ptr<output_ports_type> my_output_ports;

    static const size_t NUM_INPUTS = sizeof...(InputTypes);
    static const size_t NUM_OUTPUTS = sizeof...(OutputTypes);

protected:
    void reset_node(reset_flags) override {}

public:
    composite_node( graph &g ) : graph_node(g) {
        fgt_multiinput_multioutput_node( CODEPTR(), FLOW_COMPOSITE_NODE, this, &this->my_graph );
    }

    template<typename T1, typename T2>
    void set_external_ports(T1&& input_ports_tuple, T2&& output_ports_tuple) {
        static_assert(NUM_INPUTS == std::tuple_size<input_ports_type>::value, "number of arguments does not match number of input ports");
        static_assert(NUM_OUTPUTS == std::tuple_size<output_ports_type>::value, "number of arguments does not match number of output ports");

        fgt_internal_input_alias_helper<T1, NUM_INPUTS>::alias_port( this, input_ports_tuple);
        fgt_internal_output_alias_helper<T2, NUM_OUTPUTS>::alias_port( this, output_ports_tuple);

        my_input_ports.reset( new input_ports_type(std::forward<T1>(input_ports_tuple)) );
        my_output_ports.reset( new output_ports_type(std::forward<T2>(output_ports_tuple)) );
    }

    template< typename... NodeTypes >
    void add_visible_nodes(const NodeTypes&... n) { add_nodes_impl(this, true, n...); }

    template< typename... NodeTypes >
    void add_nodes(const NodeTypes&... n) { add_nodes_impl(this, false, n...); }


    input_ports_type& input_ports() {
         __TBB_ASSERT(my_input_ports, "input ports not set, call set_external_ports to set input ports");
         return *my_input_ports;
    }

    output_ports_type& output_ports() {
         __TBB_ASSERT(my_output_ports, "output ports not set, call set_external_ports to set output ports");
         return *my_output_ports;
    }
};  // class composite_node

//composite_node with only input ports
template< typename... InputTypes>
class composite_node <std::tuple<InputTypes...>, std::tuple<> > : public graph_node {
public:
    typedef std::tuple< receiver<InputTypes>&... > input_ports_type;

private:
    std::unique_ptr<input_ports_type> my_input_ports;
    static const size_t NUM_INPUTS = sizeof...(InputTypes);

protected:
    void reset_node(reset_flags) override {}

public:
    composite_node( graph &g ) : graph_node(g) {
        fgt_composite( CODEPTR(), this, &g );
    }

   template<typename T>
   void set_external_ports(T&& input_ports_tuple) {
       static_assert(NUM_INPUTS == std::tuple_size<input_ports_type>::value, "number of arguments does not match number of input ports");

       fgt_internal_input_alias_helper<T, NUM_INPUTS>::alias_port( this, input_ports_tuple);

       my_input_ports.reset( new input_ports_type(std::forward<T>(input_ports_tuple)) );
   }

    template< typename... NodeTypes >
    void add_visible_nodes(const NodeTypes&... n) { add_nodes_impl(this, true, n...); }

    template< typename... NodeTypes >
    void add_nodes( const NodeTypes&... n) { add_nodes_impl(this, false, n...); }


    input_ports_type& input_ports() {
         __TBB_ASSERT(my_input_ports, "input ports not set, call set_external_ports to set input ports");
         return *my_input_ports;
    }

};  // class composite_node

//composite_nodes with only output_ports
template<typename... OutputTypes>
class composite_node <std::tuple<>, std::tuple<OutputTypes...> > : public graph_node {
public:
    typedef std::tuple< sender<OutputTypes>&... > output_ports_type;

private:
    std::unique_ptr<output_ports_type> my_output_ports;
    static const size_t NUM_OUTPUTS = sizeof...(OutputTypes);

protected:
    void reset_node(reset_flags) override {}

public:
    __TBB_NOINLINE_SYM composite_node( graph &g ) : graph_node(g) {
        fgt_composite( CODEPTR(), this, &g );
    }

   template<typename T>
   void set_external_ports(T&& output_ports_tuple) {
       static_assert(NUM_OUTPUTS == std::tuple_size<output_ports_type>::value, "number of arguments does not match number of output ports");

       fgt_internal_output_alias_helper<T, NUM_OUTPUTS>::alias_port( this, output_ports_tuple);

       my_output_ports.reset( new output_ports_type(std::forward<T>(output_ports_tuple)) );
   }

    template<typename... NodeTypes >
    void add_visible_nodes(const NodeTypes&... n) { add_nodes_impl(this, true, n...); }

    template<typename... NodeTypes >
    void add_nodes(const NodeTypes&... n) { add_nodes_impl(this, false, n...); }


    output_ports_type& output_ports() {
         __TBB_ASSERT(my_output_ports, "output ports not set, call set_external_ports to set output ports");
         return *my_output_ports;
    }

};  // class composite_node

template<typename Gateway>
class async_body_base: no_assign {
public:
    typedef Gateway gateway_type;

    async_body_base(gateway_type *gateway): my_gateway(gateway) { }
    void set_gateway(gateway_type *gateway) {
        my_gateway = gateway;
    }

protected:
    gateway_type *my_gateway;
};

template<typename Input, typename Ports, typename Gateway, typename Body>
class async_body: public async_body_base<Gateway> {
public:
    typedef async_body_base<Gateway> base_type;
    typedef Gateway gateway_type;

    async_body(const Body &body, gateway_type *gateway)
        : base_type(gateway), my_body(body) { }

    void operator()( const Input &v, Ports & ) {
        my_body(v, *this->my_gateway);
    }

    Body get_body() { return my_body; }

private:
    Body my_body;
};

//! Implements async node
template < typename Input, typename Output,
           typename Policy = queueing_lightweight >
class async_node
    : public multifunction_node< Input, std::tuple< Output >, Policy >, public sender< Output >
{
    typedef multifunction_node< Input, std::tuple< Output >, Policy > base_type;
    typedef multifunction_input<
        Input, typename base_type::output_ports_type, Policy, cache_aligned_allocator<Input>> mfn_input_type;

public:
    typedef Input input_type;
    typedef Output output_type;
    typedef receiver<input_type> receiver_type;
    typedef receiver<output_type> successor_type;
    typedef sender<input_type> predecessor_type;
    typedef receiver_gateway<output_type> gateway_type;
    typedef async_body_base<gateway_type> async_body_base_type;
    typedef typename base_type::output_ports_type output_ports_type;

private:
    class receiver_gateway_impl: public receiver_gateway<Output> {
    public:
        receiver_gateway_impl(async_node* node): my_node(node) {}
        void reserve_wait() override {
            fgt_async_reserve(static_cast<typename async_node::receiver_type *>(my_node), &my_node->my_graph);
            my_node->my_graph.reserve_wait();
        }

        void release_wait() override {
            async_node* n = my_node;
            graph* g = &n->my_graph;
            g->release_wait();
            fgt_async_commit(static_cast<typename async_node::receiver_type *>(n), g);
        }

        //! Implements gateway_type::try_put for an external activity to submit a message to FG
        bool try_put(const Output &i) override {
            return my_node->try_put_impl(i);
        }

    private:
        async_node* my_node;
    } my_gateway;

    //The substitute of 'this' for member construction, to prevent compiler warnings
    async_node* self() { return this; }

    //! Implements gateway_type::try_put for an external activity to submit a message to FG
    bool try_put_impl(const Output &i) {
        multifunction_output<Output> &port_0 = output_port<0>(*this);
        broadcast_cache<output_type>& port_successors = port_0.successors();
        fgt_async_try_put_begin(this, &port_0);
        // TODO revamp: change to std::list<graph_task*>
        graph_task_list tasks;
        bool is_at_least_one_put_successful = port_successors.gather_successful_try_puts(i, tasks);
        __TBB_ASSERT( is_at_least_one_put_successful || tasks.empty(),
                      "Return status is inconsistent with the method operation." );

        while( !tasks.empty() ) {
            enqueue_in_graph_arena(this->my_graph, tasks.pop_front());
        }
        fgt_async_try_put_end(this, &port_0);
        return is_at_least_one_put_successful;
    }

public:
    template<typename Body>
    __TBB_NOINLINE_SYM async_node(
        graph &g, size_t concurrency,
        Body body, Policy = Policy(), node_priority_t a_priority = no_priority
    ) : base_type(
        g, concurrency,
        async_body<Input, typename base_type::output_ports_type, gateway_type, Body>
        (body, &my_gateway), a_priority ), my_gateway(self()) {
        fgt_multioutput_node_with_body<1>(
            CODEPTR(), FLOW_ASYNC_NODE,
            &this->my_graph, static_cast<receiver<input_type> *>(this),
            this->output_ports(), this->my_body
        );
    }

    template <typename Body, typename... Args>
    __TBB_NOINLINE_SYM async_node(graph& g, size_t concurrency, Body body, node_priority_t a_priority)
        : async_node(g, concurrency, body, Policy(), a_priority) {}

#if __TBB_PREVIEW_FLOW_GRAPH_NODE_SET
    template <typename Body, typename... Args>
    __TBB_NOINLINE_SYM async_node(
        const node_set<Args...>& nodes, size_t concurrency, Body body,
        Policy = Policy(), node_priority_t a_priority = no_priority )
        : async_node(nodes.graph_reference(), concurrency, body, a_priority) {
        make_edges_in_order(nodes, *this);
    }

    template <typename Body, typename... Args>
    __TBB_NOINLINE_SYM async_node(const node_set<Args...>& nodes, size_t concurrency, Body body, node_priority_t a_priority)
        : async_node(nodes, concurrency, body, Policy(), a_priority) {}
#endif // __TBB_PREVIEW_FLOW_GRAPH_NODE_SET

    __TBB_NOINLINE_SYM async_node( const async_node &other ) : base_type(other), sender<Output>(), my_gateway(self()) {
        static_cast<async_body_base_type*>(this->my_body->get_body_ptr())->set_gateway(&my_gateway);
        static_cast<async_body_base_type*>(this->my_init_body->get_body_ptr())->set_gateway(&my_gateway);

        fgt_multioutput_node_with_body<1>( CODEPTR(), FLOW_ASYNC_NODE,
                &this->my_graph, static_cast<receiver<input_type> *>(this),
                this->output_ports(), this->my_body );
    }

    gateway_type& gateway() {
        return my_gateway;
    }

    // Define sender< Output >

    //! Add a new successor to this node
    bool register_successor(successor_type&) override {
        __TBB_ASSERT(false, "Successors must be registered only via ports");
        return false;
    }

    //! Removes a successor from this node
    bool remove_successor(successor_type&) override {
        __TBB_ASSERT(false, "Successors must be removed only via ports");
        return false;
    }

    template<typename Body>
    Body copy_function_object() {
        typedef multifunction_body<input_type, typename base_type::output_ports_type> mfn_body_type;
        typedef async_body<Input, typename base_type::output_ports_type, gateway_type, Body> async_body_type;
        mfn_body_type &body_ref = *this->my_body;
        async_body_type ab = *static_cast<async_body_type*>(dynamic_cast< multifunction_body_leaf<input_type, typename base_type::output_ports_type, async_body_type> & >(body_ref).get_body_ptr());
        return ab.get_body();
    }

protected:

    void reset_node( reset_flags f) override {
       base_type::reset_node(f);
    }
};

#include "detail/_flow_graph_node_set_impl.h"

template< typename T >
class overwrite_node : public graph_node, public receiver<T>, public sender<T> {
public:
    typedef T input_type;
    typedef T output_type;
    typedef typename receiver<input_type>::predecessor_type predecessor_type;
    typedef typename sender<output_type>::successor_type successor_type;

    __TBB_NOINLINE_SYM explicit overwrite_node(graph &g)
        : graph_node(g), my_successors(this), my_buffer_is_valid(false)
    {
        fgt_node( CODEPTR(), FLOW_OVERWRITE_NODE, &this->my_graph,
                  static_cast<receiver<input_type> *>(this), static_cast<sender<output_type> *>(this) );
    }

#if __TBB_PREVIEW_FLOW_GRAPH_NODE_SET
    template <typename... Args>
    overwrite_node(const node_set<Args...>& nodes) : overwrite_node(nodes.graph_reference()) {
        make_edges_in_order(nodes, *this);
    }
#endif

    //! Copy constructor; doesn't take anything from src; default won't work
    __TBB_NOINLINE_SYM overwrite_node( const overwrite_node& src ) : overwrite_node(src.my_graph) {}

    ~overwrite_node() {}

    bool register_successor( successor_type &s ) override {
        spin_mutex::scoped_lock l( my_mutex );
        if (my_buffer_is_valid && is_graph_active( my_graph )) {
            // We have a valid value that must be forwarded immediately.
            bool ret = s.try_put( my_buffer );
            if ( ret ) {
                // We add the successor that accepted our put
                my_successors.register_successor( s );
            } else {
                // In case of reservation a race between the moment of reservation and register_successor can appear,
                // because failed reserve does not mean that register_successor is not ready to put a message immediately.
                // We have some sort of infinite loop: reserving node tries to set pull state for the edge,
                // but overwrite_node tries to return push state back. That is why we have to break this loop with task creation.
                small_object_allocator allocator{};
                typedef register_predecessor_task task_type;
                graph_task* t = allocator.new_object<task_type>(graph_reference(), allocator, *this, s);
                graph_reference().reserve_wait();
                spawn_in_graph_arena( my_graph, *t );
            }
        } else {
            // No valid value yet, just add as successor
            my_successors.register_successor( s );
        }
        return true;
    }

    bool remove_successor( successor_type &s ) override {
        spin_mutex::scoped_lock l( my_mutex );
        my_successors.remove_successor(s);
        return true;
    }

    bool try_get( input_type &v ) override {
        spin_mutex::scoped_lock l( my_mutex );
        if ( my_buffer_is_valid ) {
            v = my_buffer;
            return true;
        }
        return false;
    }

    //! Reserves an item
    bool try_reserve( T &v ) override {
        return try_get(v);
    }

    //! Releases the reserved item
    bool try_release() override { return true; }

    //! Consumes the reserved item
    bool try_consume() override { return true; }

    bool is_valid() {
       spin_mutex::scoped_lock l( my_mutex );
       return my_buffer_is_valid;
    }

    void clear() {
       spin_mutex::scoped_lock l( my_mutex );
       my_buffer_is_valid = false;
    }

protected:

    template< typename R, typename B > friend class run_and_put_task;
    template<typename X, typename Y> friend class broadcast_cache;
    template<typename X, typename Y> friend class round_robin_cache;
    graph_task* try_put_task( const input_type &v ) override {
        spin_mutex::scoped_lock l( my_mutex );
        return try_put_task_impl(v);
    }

    graph_task * try_put_task_impl(const input_type &v) {
        my_buffer = v;
        my_buffer_is_valid = true;
        graph_task* rtask = my_successors.try_put_task(v);
        if (!rtask) rtask = SUCCESSFULLY_ENQUEUED;
        return rtask;
    }

    graph& graph_reference() const override {
        return my_graph;
    }

    //! Breaks an infinite loop between the node reservation and register_successor call
    struct register_predecessor_task : public graph_task {
        register_predecessor_task(
            graph& g, small_object_allocator& allocator, predecessor_type& owner, successor_type& succ)
            : graph_task(g, allocator), o(owner), s(succ) {};

        task* execute(execution_data& ed) override {
            // TODO revamp: investigate why qualification is needed for register_successor() call
            using tbb::detail::d1::register_predecessor;
            using tbb::detail::d1::register_successor;
            if ( !register_predecessor(s, o) ) {
                register_successor(o, s);
            }
            finalize(ed);
            return nullptr;
        }

        predecessor_type& o;
        successor_type& s;
    };

    spin_mutex my_mutex;
    broadcast_cache< input_type, null_rw_mutex > my_successors;
    input_type my_buffer;
    bool my_buffer_is_valid;

    void reset_node( reset_flags f) override {
        my_buffer_is_valid = false;
       if (f&rf_clear_edges) {
           my_successors.clear();
       }
    }
};  // overwrite_node

template< typename T >
class write_once_node : public overwrite_node<T> {
public:
    typedef T input_type;
    typedef T output_type;
    typedef overwrite_node<T> base_type;
    typedef typename receiver<input_type>::predecessor_type predecessor_type;
    typedef typename sender<output_type>::successor_type successor_type;

    //! Constructor
    __TBB_NOINLINE_SYM explicit write_once_node(graph& g) : base_type(g) {
        fgt_node( CODEPTR(), FLOW_WRITE_ONCE_NODE, &(this->my_graph),
                                 static_cast<receiver<input_type> *>(this),
                                 static_cast<sender<output_type> *>(this) );
    }

#if __TBB_PREVIEW_FLOW_GRAPH_NODE_SET
    template <typename... Args>
    write_once_node(const node_set<Args...>& nodes) : write_once_node(nodes.graph_reference()) {
        make_edges_in_order(nodes, *this);
    }
#endif

    //! Copy constructor: call base class copy constructor
    __TBB_NOINLINE_SYM write_once_node( const write_once_node& src ) : base_type(src) {
        fgt_node( CODEPTR(), FLOW_WRITE_ONCE_NODE, &(this->my_graph),
                                 static_cast<receiver<input_type> *>(this),
                                 static_cast<sender<output_type> *>(this) );
    }

protected:
    template< typename R, typename B > friend class run_and_put_task;
    template<typename X, typename Y> friend class broadcast_cache;
    template<typename X, typename Y> friend class round_robin_cache;
    graph_task *try_put_task( const T &v ) override {
        spin_mutex::scoped_lock l( this->my_mutex );
        return this->my_buffer_is_valid ? NULL : this->try_put_task_impl(v);
    }
}; // write_once_node

inline void set_name(const graph& g, const char *name) {
    fgt_graph_desc(&g, name);
}

template <typename Output>
inline void set_name(const input_node<Output>& node, const char *name) {
    fgt_node_desc(&node, name);
}

template <typename Input, typename Output, typename Policy>
inline void set_name(const function_node<Input, Output, Policy>& node, const char *name) {
    fgt_node_desc(&node, name);
}

template <typename Output, typename Policy>
inline void set_name(const continue_node<Output,Policy>& node, const char *name) {
    fgt_node_desc(&node, name);
}

template <typename T>
inline void set_name(const broadcast_node<T>& node, const char *name) {
    fgt_node_desc(&node, name);
}

template <typename T>
inline void set_name(const buffer_node<T>& node, const char *name) {
    fgt_node_desc(&node, name);
}

template <typename T>
inline void set_name(const queue_node<T>& node, const char *name) {
    fgt_node_desc(&node, name);
}

template <typename T>
inline void set_name(const sequencer_node<T>& node, const char *name) {
    fgt_node_desc(&node, name);
}

template <typename T, typename Compare>
inline void set_name(const priority_queue_node<T, Compare>& node, const char *name) {
    fgt_node_desc(&node, name);
}

template <typename T, typename DecrementType>
inline void set_name(const limiter_node<T, DecrementType>& node, const char *name) {
    fgt_node_desc(&node, name);
}

template <typename OutputTuple, typename JP>
inline void set_name(const join_node<OutputTuple, JP>& node, const char *name) {
    fgt_node_desc(&node, name);
}

template <typename... Types>
inline void set_name(const indexer_node<Types...>& node, const char *name) {
    fgt_node_desc(&node, name);
}

template <typename T>
inline void set_name(const overwrite_node<T>& node, const char *name) {
    fgt_node_desc(&node, name);
}

template <typename T>
inline void set_name(const write_once_node<T>& node, const char *name) {
    fgt_node_desc(&node, name);
}

template<typename Input, typename Output, typename Policy>
inline void set_name(const multifunction_node<Input, Output, Policy>& node, const char *name) {
    fgt_multioutput_node_desc(&node, name);
}

template<typename TupleType>
inline void set_name(const split_node<TupleType>& node, const char *name) {
    fgt_multioutput_node_desc(&node, name);
}

template< typename InputTuple, typename OutputTuple >
inline void set_name(const composite_node<InputTuple, OutputTuple>& node, const char *name) {
    fgt_multiinput_multioutput_node_desc(&node, name);
}

template<typename Input, typename Output, typename Policy>
inline void set_name(const async_node<Input, Output, Policy>& node, const char *name)
{
    fgt_multioutput_node_desc(&node, name);
}
} // d1
} // detail
} // tbb


// Include deduction guides for node classes
#include "detail/_flow_graph_nodes_deduction.h"

namespace tbb {
namespace flow {
inline namespace v1 {
    using detail::d1::receiver;
    using detail::d1::sender;

    using detail::d1::serial;
    using detail::d1::unlimited;

    using detail::d1::reset_flags;
    using detail::d1::rf_reset_protocol;
    using detail::d1::rf_reset_bodies;
    using detail::d1::rf_clear_edges;

    using detail::d1::graph;
    using detail::d1::graph_node;
    using detail::d1::continue_msg;

    using detail::d1::input_node;
    using detail::d1::function_node;
    using detail::d1::multifunction_node;
    using detail::d1::split_node;
    using detail::d1::output_port;
    using detail::d1::indexer_node;
    using detail::d1::tagged_msg;
    using detail::d1::cast_to;
    using detail::d1::is_a;
    using detail::d1::continue_node;
    using detail::d1::overwrite_node;
    using detail::d1::write_once_node;
    using detail::d1::broadcast_node;
    using detail::d1::buffer_node;
    using detail::d1::queue_node;
    using detail::d1::sequencer_node;
    using detail::d1::priority_queue_node;
    using detail::d1::limiter_node;
    using namespace detail::d1::graph_policy_namespace;
    using detail::d1::join_node;
    using detail::d1::input_port;
    using detail::d1::copy_body;
    using detail::d1::make_edge;
    using detail::d1::remove_edge;
    using detail::d1::tag_value;
    using detail::d1::composite_node;
    using detail::d1::async_node;
    using detail::d1::node_priority_t;
    using detail::d1::no_priority;

#if __TBB_PREVIEW_FLOW_GRAPH_NODE_SET
    using detail::d1::follows;
    using detail::d1::precedes;
    using detail::d1::make_node_set;
    using detail::d1::make_edges;
#endif

} // v1
} // flow

    using detail::d1::flow_control;

namespace profiling {
    using detail::d1::set_name;
} // profiling

} // tbb


#if TBB_USE_PROFILING_TOOLS  && ( __linux__ || __APPLE__ )
   // We don't do pragma pop here, since it still gives warning on the USER side
   #undef __TBB_NOINLINE_SYM
#endif

#endif // __TBB_flow_graph_H
