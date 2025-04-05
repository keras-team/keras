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

#ifndef __TBB__flow_graph_body_impl_H
#define __TBB__flow_graph_body_impl_H

#ifndef __TBB_flow_graph_H
#error Do not #include this internal file directly; use public TBB headers instead.
#endif

// included in namespace tbb::detail::d1 (in flow_graph.h)

typedef std::uint64_t tag_value;


// TODO revamp: find out if there is already helper for has_policy.
template<typename ... Policies> struct Policy {};

template<typename ... Policies> struct has_policy;

template<typename ExpectedPolicy, typename FirstPolicy, typename ...Policies>
struct has_policy<ExpectedPolicy, FirstPolicy, Policies...> :
    std::integral_constant<bool, has_policy<ExpectedPolicy, FirstPolicy>::value ||
                                 has_policy<ExpectedPolicy, Policies...>::value> {};

template<typename ExpectedPolicy, typename SinglePolicy>
struct has_policy<ExpectedPolicy, SinglePolicy> :
    std::integral_constant<bool, std::is_same<ExpectedPolicy, SinglePolicy>::value> {};

template<typename ExpectedPolicy, typename ...Policies>
struct has_policy<ExpectedPolicy, Policy<Policies...> > : has_policy<ExpectedPolicy, Policies...> {};

namespace graph_policy_namespace {

    struct rejecting { };
    struct reserving { };
    struct queueing  { };
    struct lightweight  { };

    // K == type of field used for key-matching.  Each tag-matching port will be provided
    // functor that, given an object accepted by the port, will return the
    /// field of type K being used for matching.
    template<typename K, typename KHash=tbb_hash_compare<typename std::decay<K>::type > >
    struct key_matching {
        typedef K key_type;
        typedef typename std::decay<K>::type base_key_type;
        typedef KHash hash_compare_type;
    };

    // old tag_matching join's new specifier
    typedef key_matching<tag_value> tag_matching;

    // Aliases for Policy combinations
    typedef Policy<queueing, lightweight> queueing_lightweight;
    typedef Policy<rejecting, lightweight> rejecting_lightweight;

} // namespace graph_policy_namespace

// -------------- function_body containers ----------------------

//! A functor that takes no input and generates a value of type Output
template< typename Output >
class input_body : no_assign {
public:
    virtual ~input_body() {}
    virtual Output operator()(flow_control& fc) = 0;
    virtual input_body* clone() = 0;
};

//! The leaf for input_body
template< typename Output, typename Body>
class input_body_leaf : public input_body<Output> {
public:
    input_body_leaf( const Body &_body ) : body(_body) { }
    Output operator()(flow_control& fc) override { return body(fc); }
    input_body_leaf* clone() override {
        return new input_body_leaf< Output, Body >(body);
    }
    Body get_body() { return body; }
private:
    Body body;
};

//! A functor that takes an Input and generates an Output
template< typename Input, typename Output >
class function_body : no_assign {
public:
    virtual ~function_body() {}
    virtual Output operator()(const Input &input) = 0;
    virtual function_body* clone() = 0;
};

//! the leaf for function_body
template <typename Input, typename Output, typename B>
class function_body_leaf : public function_body< Input, Output > {
public:
    function_body_leaf( const B &_body ) : body(_body) { }
    Output operator()(const Input &i) override { return body(i); }
    B get_body() { return body; }
    function_body_leaf* clone() override {
        return new function_body_leaf< Input, Output, B >(body);
    }
private:
    B body;
};

//! the leaf for function_body specialized for Input and output of continue_msg
template <typename B>
class function_body_leaf< continue_msg, continue_msg, B> : public function_body< continue_msg, continue_msg > {
public:
    function_body_leaf( const B &_body ) : body(_body) { }
    continue_msg operator()( const continue_msg &i ) override {
        body(i);
        return i;
    }
    B get_body() { return body; }
    function_body_leaf* clone() override {
        return new function_body_leaf< continue_msg, continue_msg, B >(body);
    }
private:
    B body;
};

//! the leaf for function_body specialized for Output of continue_msg
template <typename Input, typename B>
class function_body_leaf< Input, continue_msg, B> : public function_body< Input, continue_msg > {
public:
    function_body_leaf( const B &_body ) : body(_body) { }
    continue_msg operator()(const Input &i) override {
        body(i);
        return continue_msg();
    }
    B get_body() { return body; }
    function_body_leaf* clone() override {
        return new function_body_leaf< Input, continue_msg, B >(body);
    }
private:
    B body;
};

//! the leaf for function_body specialized for Input of continue_msg
template <typename Output, typename B>
class function_body_leaf< continue_msg, Output, B > : public function_body< continue_msg, Output > {
public:
    function_body_leaf( const B &_body ) : body(_body) { }
    Output operator()(const continue_msg &i) override {
        return body(i);
    }
    B get_body() { return body; }
    function_body_leaf* clone() override {
        return new function_body_leaf< continue_msg, Output, B >(body);
    }
private:
    B body;
};

//! function_body that takes an Input and a set of output ports
template<typename Input, typename OutputSet>
class multifunction_body : no_assign {
public:
    virtual ~multifunction_body () {}
    virtual void operator()(const Input &/* input*/, OutputSet &/*oset*/) = 0;
    virtual multifunction_body* clone() = 0;
    virtual void* get_body_ptr() = 0;
};

//! leaf for multifunction.  OutputSet can be a std::tuple or a vector.
template<typename Input, typename OutputSet, typename B >
class multifunction_body_leaf : public multifunction_body<Input, OutputSet> {
public:
    multifunction_body_leaf(const B &_body) : body(_body) { }
    void operator()(const Input &input, OutputSet &oset) override {
        body(input, oset); // body may explicitly put() to one or more of oset.
    }
    void* get_body_ptr() override { return &body; }
    multifunction_body_leaf* clone() override {
        return new multifunction_body_leaf<Input, OutputSet,B>(body);
    }

private:
    B body;
};

// ------ function bodies for hash_buffers and key-matching joins.

template<typename Input, typename Output>
class type_to_key_function_body : no_assign {
    public:
        virtual ~type_to_key_function_body() {}
        virtual Output operator()(const Input &input) = 0;  // returns an Output
        virtual type_to_key_function_body* clone() = 0;
};

// specialization for ref output
template<typename Input, typename Output>
class type_to_key_function_body<Input,Output&> : no_assign {
    public:
        virtual ~type_to_key_function_body() {}
        virtual const Output & operator()(const Input &input) = 0;  // returns a const Output&
        virtual type_to_key_function_body* clone() = 0;
};

template <typename Input, typename Output, typename B>
class type_to_key_function_body_leaf : public type_to_key_function_body<Input, Output> {
public:
    type_to_key_function_body_leaf( const B &_body ) : body(_body) { }
    Output operator()(const Input &i) override { return body(i); }
    type_to_key_function_body_leaf* clone() override {
        return new type_to_key_function_body_leaf< Input, Output, B>(body);
    }
private:
    B body;
};

template <typename Input, typename Output, typename B>
class type_to_key_function_body_leaf<Input,Output&,B> : public type_to_key_function_body< Input, Output&> {
public:
    type_to_key_function_body_leaf( const B &_body ) : body(_body) { }
    const Output& operator()(const Input &i) override {
        return body(i);
    }
    type_to_key_function_body_leaf* clone() override {
        return new type_to_key_function_body_leaf< Input, Output&, B>(body);
    }
private:
    B body;
};

// --------------------------- end of function_body containers ------------------------

// --------------------------- node task bodies ---------------------------------------

//! A task that calls a node's forward_task function
template< typename NodeType >
class forward_task_bypass : public graph_task {
    NodeType &my_node;
public:
    forward_task_bypass( graph& g, small_object_allocator& allocator, NodeType &n
                         , node_priority_t node_priority = no_priority
    ) : graph_task(g, allocator, node_priority),
    my_node(n) {}

    task* execute(execution_data& ed) override {
        graph_task* next_task = my_node.forward_task();
        if (SUCCESSFULLY_ENQUEUED == next_task)
            next_task = nullptr;
        else if (next_task)
            next_task = prioritize_task(my_node.graph_reference(), *next_task);
        finalize(ed);
        return next_task;
    }
};

//! A task that calls a node's apply_body_bypass function, passing in an input of type Input
//  return the task* unless it is SUCCESSFULLY_ENQUEUED, in which case return NULL
template< typename NodeType, typename Input >
class apply_body_task_bypass : public graph_task {
    NodeType &my_node;
    Input my_input;
public:

    apply_body_task_bypass( graph& g, small_object_allocator& allocator, NodeType &n, const Input &i
                            , node_priority_t node_priority = no_priority
    ) : graph_task(g, allocator, node_priority),
        my_node(n), my_input(i) {}

    task* execute(execution_data& ed) override {
        graph_task* next_task = my_node.apply_body_bypass( my_input );
        if (SUCCESSFULLY_ENQUEUED == next_task)
            next_task = nullptr;
        else if (next_task)
            next_task = prioritize_task(my_node.graph_reference(), *next_task);
        finalize(ed);
        return next_task;

    }
};

//! A task that calls a node's apply_body_bypass function with no input
template< typename NodeType >
class input_node_task_bypass : public graph_task {
    NodeType &my_node;
public:
    input_node_task_bypass( graph& g, small_object_allocator& allocator, NodeType &n )
        : graph_task(g, allocator), my_node(n) {}

    task* execute(execution_data& ed) override {
        graph_task* next_task = my_node.apply_body_bypass( );
        if (SUCCESSFULLY_ENQUEUED == next_task)
            next_task = nullptr;
        else if (next_task)
            next_task = prioritize_task(my_node.graph_reference(), *next_task);
        finalize(ed);
        return next_task;
    }

};

// ------------------------ end of node task bodies -----------------------------------

template<typename T, typename DecrementType, typename DummyType = void>
class threshold_regulator;

template<typename T, typename DecrementType>
class threshold_regulator<T, DecrementType,
                  typename std::enable_if<std::is_integral<DecrementType>::value>::type>
    : public receiver<DecrementType>, no_copy
{
    T* my_node;
protected:

    graph_task* try_put_task( const DecrementType& value ) override {
        graph_task* result = my_node->decrement_counter( value );
        if( !result )
            result = SUCCESSFULLY_ENQUEUED;
        return result;
    }

    graph& graph_reference() const override {
        return my_node->my_graph;
    }

    template<typename U, typename V> friend class limiter_node;
    void reset_receiver( reset_flags ) {}

public:
    threshold_regulator(T* owner) : my_node(owner) {
        // Do not work with the passed pointer here as it may not be fully initialized yet
    }
};

template<typename T>
class threshold_regulator<T, continue_msg, void> : public continue_receiver, no_copy {

    T *my_node;

    graph_task* execute() override {
        return my_node->decrement_counter( 1 );
    }

protected:

    graph& graph_reference() const override {
        return my_node->my_graph;
    }

public:

    typedef continue_msg input_type;
    typedef continue_msg output_type;
    threshold_regulator(T* owner)
        : continue_receiver( /*number_of_predecessors=*/0, no_priority ), my_node(owner)
    {
        // Do not work with the passed pointer here as it may not be fully initialized yet
    }
};

#endif // __TBB__flow_graph_body_impl_H
