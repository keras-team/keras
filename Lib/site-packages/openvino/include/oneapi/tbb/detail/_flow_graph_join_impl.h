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

#ifndef __TBB__flow_graph_join_impl_H
#define __TBB__flow_graph_join_impl_H

#ifndef __TBB_flow_graph_H
#error Do not #include this internal file directly; use public TBB headers instead.
#endif

// included into namespace tbb::detail::d1

    struct forwarding_base : no_assign {
        forwarding_base(graph &g) : graph_ref(g) {}
        virtual ~forwarding_base() {}
        graph& graph_ref;
    };

    struct queueing_forwarding_base : forwarding_base {
        using forwarding_base::forwarding_base;
        // decrement_port_count may create a forwarding task.  If we cannot handle the task
        // ourselves, ask decrement_port_count to deal with it.
        virtual graph_task* decrement_port_count(bool handle_task) = 0;
    };

    struct reserving_forwarding_base : forwarding_base {
        using forwarding_base::forwarding_base;
        // decrement_port_count may create a forwarding task.  If we cannot handle the task
        // ourselves, ask decrement_port_count to deal with it.
        virtual graph_task* decrement_port_count() = 0;
        virtual void increment_port_count() = 0;
    };

    // specialization that lets us keep a copy of the current_key for building results.
    // KeyType can be a reference type.
    template<typename KeyType>
    struct matching_forwarding_base : public forwarding_base {
        typedef typename std::decay<KeyType>::type current_key_type;
        matching_forwarding_base(graph &g) : forwarding_base(g) { }
        virtual graph_task* increment_key_count(current_key_type const & /*t*/) = 0;
        current_key_type current_key; // so ports can refer to FE's desired items
    };

    template< int N >
    struct join_helper {

        template< typename TupleType, typename PortType >
        static inline void set_join_node_pointer(TupleType &my_input, PortType *port) {
            std::get<N-1>( my_input ).set_join_node_pointer(port);
            join_helper<N-1>::set_join_node_pointer( my_input, port );
        }
        template< typename TupleType >
        static inline void consume_reservations( TupleType &my_input ) {
            std::get<N-1>( my_input ).consume();
            join_helper<N-1>::consume_reservations( my_input );
        }

        template< typename TupleType >
        static inline void release_my_reservation( TupleType &my_input ) {
            std::get<N-1>( my_input ).release();
        }

        template <typename TupleType>
        static inline void release_reservations( TupleType &my_input) {
            join_helper<N-1>::release_reservations(my_input);
            release_my_reservation(my_input);
        }

        template< typename InputTuple, typename OutputTuple >
        static inline bool reserve( InputTuple &my_input, OutputTuple &out) {
            if ( !std::get<N-1>( my_input ).reserve( std::get<N-1>( out ) ) ) return false;
            if ( !join_helper<N-1>::reserve( my_input, out ) ) {
                release_my_reservation( my_input );
                return false;
            }
            return true;
        }

        template<typename InputTuple, typename OutputTuple>
        static inline bool get_my_item( InputTuple &my_input, OutputTuple &out) {
            bool res = std::get<N-1>(my_input).get_item(std::get<N-1>(out) ); // may fail
            return join_helper<N-1>::get_my_item(my_input, out) && res;       // do get on other inputs before returning
        }

        template<typename InputTuple, typename OutputTuple>
        static inline bool get_items(InputTuple &my_input, OutputTuple &out) {
            return get_my_item(my_input, out);
        }

        template<typename InputTuple>
        static inline void reset_my_port(InputTuple &my_input) {
            join_helper<N-1>::reset_my_port(my_input);
            std::get<N-1>(my_input).reset_port();
        }

        template<typename InputTuple>
        static inline void reset_ports(InputTuple& my_input) {
            reset_my_port(my_input);
        }

        template<typename InputTuple, typename KeyFuncTuple>
        static inline void set_key_functors(InputTuple &my_input, KeyFuncTuple &my_key_funcs) {
            std::get<N-1>(my_input).set_my_key_func(std::get<N-1>(my_key_funcs));
            std::get<N-1>(my_key_funcs) = nullptr;
            join_helper<N-1>::set_key_functors(my_input, my_key_funcs);
        }

        template< typename KeyFuncTuple>
        static inline void copy_key_functors(KeyFuncTuple &my_inputs, KeyFuncTuple &other_inputs) {
            __TBB_ASSERT(
                std::get<N-1>(other_inputs).get_my_key_func(),
                "key matching join node should not be instantiated without functors."
            );
            std::get<N-1>(my_inputs).set_my_key_func(std::get<N-1>(other_inputs).get_my_key_func()->clone());
            join_helper<N-1>::copy_key_functors(my_inputs, other_inputs);
        }

        template<typename InputTuple>
        static inline void reset_inputs(InputTuple &my_input, reset_flags f) {
            join_helper<N-1>::reset_inputs(my_input, f);
            std::get<N-1>(my_input).reset_receiver(f);
        }
    };  // join_helper<N>

    template< >
    struct join_helper<1> {

        template< typename TupleType, typename PortType >
        static inline void set_join_node_pointer(TupleType &my_input, PortType *port) {
            std::get<0>( my_input ).set_join_node_pointer(port);
        }

        template< typename TupleType >
        static inline void consume_reservations( TupleType &my_input ) {
            std::get<0>( my_input ).consume();
        }

        template< typename TupleType >
        static inline void release_my_reservation( TupleType &my_input ) {
            std::get<0>( my_input ).release();
        }

        template<typename TupleType>
        static inline void release_reservations( TupleType &my_input) {
            release_my_reservation(my_input);
        }

        template< typename InputTuple, typename OutputTuple >
        static inline bool reserve( InputTuple &my_input, OutputTuple &out) {
            return std::get<0>( my_input ).reserve( std::get<0>( out ) );
        }

        template<typename InputTuple, typename OutputTuple>
        static inline bool get_my_item( InputTuple &my_input, OutputTuple &out) {
            return std::get<0>(my_input).get_item(std::get<0>(out));
        }

        template<typename InputTuple, typename OutputTuple>
        static inline bool get_items(InputTuple &my_input, OutputTuple &out) {
            return get_my_item(my_input, out);
        }

        template<typename InputTuple>
        static inline void reset_my_port(InputTuple &my_input) {
            std::get<0>(my_input).reset_port();
        }

        template<typename InputTuple>
        static inline void reset_ports(InputTuple& my_input) {
            reset_my_port(my_input);
        }

        template<typename InputTuple, typename KeyFuncTuple>
        static inline void set_key_functors(InputTuple &my_input, KeyFuncTuple &my_key_funcs) {
            std::get<0>(my_input).set_my_key_func(std::get<0>(my_key_funcs));
            std::get<0>(my_key_funcs) = nullptr;
        }

        template< typename KeyFuncTuple>
        static inline void copy_key_functors(KeyFuncTuple &my_inputs, KeyFuncTuple &other_inputs) {
            __TBB_ASSERT(
                std::get<0>(other_inputs).get_my_key_func(),
                "key matching join node should not be instantiated without functors."
            );
            std::get<0>(my_inputs).set_my_key_func(std::get<0>(other_inputs).get_my_key_func()->clone());
        }
        template<typename InputTuple>
        static inline void reset_inputs(InputTuple &my_input, reset_flags f) {
            std::get<0>(my_input).reset_receiver(f);
        }
    };  // join_helper<1>

    //! The two-phase join port
    template< typename T >
    class reserving_port : public receiver<T> {
    public:
        typedef T input_type;
        typedef typename receiver<input_type>::predecessor_type predecessor_type;

    private:
        // ----------- Aggregator ------------
        enum op_type { reg_pred, rem_pred, res_item, rel_res, con_res
        };
        typedef reserving_port<T> class_type;

        class reserving_port_operation : public aggregated_operation<reserving_port_operation> {
        public:
            char type;
            union {
                T *my_arg;
                predecessor_type *my_pred;
            };
            reserving_port_operation(const T& e, op_type t) :
                type(char(t)), my_arg(const_cast<T*>(&e)) {}
            reserving_port_operation(const predecessor_type &s, op_type t) : type(char(t)),
                my_pred(const_cast<predecessor_type *>(&s)) {}
            reserving_port_operation(op_type t) : type(char(t)) {}
        };

        typedef aggregating_functor<class_type, reserving_port_operation> handler_type;
        friend class aggregating_functor<class_type, reserving_port_operation>;
        aggregator<handler_type, reserving_port_operation> my_aggregator;

        void handle_operations(reserving_port_operation* op_list) {
            reserving_port_operation *current;
            bool was_missing_predecessors = false;
            while(op_list) {
                current = op_list;
                op_list = op_list->next;
                switch(current->type) {
                case reg_pred:
                    was_missing_predecessors = my_predecessors.empty();
                    my_predecessors.add(*(current->my_pred));
                    if ( was_missing_predecessors ) {
                        (void) my_join->decrement_port_count(); // may try to forward
                    }
                    current->status.store( SUCCEEDED, std::memory_order_release);
                    break;
                case rem_pred:
                    if ( !my_predecessors.empty() ) {
                        my_predecessors.remove(*(current->my_pred));
                        if ( my_predecessors.empty() ) // was the last predecessor
                            my_join->increment_port_count();
                    }
                    // TODO: consider returning failure if there were no predecessors to remove
                    current->status.store( SUCCEEDED, std::memory_order_release );
                    break;
                case res_item:
                    if ( reserved ) {
                        current->status.store( FAILED, std::memory_order_release);
                    }
                    else if ( my_predecessors.try_reserve( *(current->my_arg) ) ) {
                        reserved = true;
                        current->status.store( SUCCEEDED, std::memory_order_release);
                    } else {
                        if ( my_predecessors.empty() ) {
                            my_join->increment_port_count();
                        }
                        current->status.store( FAILED, std::memory_order_release);
                    }
                    break;
                case rel_res:
                    reserved = false;
                    my_predecessors.try_release( );
                    current->status.store( SUCCEEDED, std::memory_order_release);
                    break;
                case con_res:
                    reserved = false;
                    my_predecessors.try_consume( );
                    current->status.store( SUCCEEDED, std::memory_order_release);
                    break;
                }
            }
        }

    protected:
        template< typename R, typename B > friend class run_and_put_task;
        template<typename X, typename Y> friend class broadcast_cache;
        template<typename X, typename Y> friend class round_robin_cache;
        graph_task* try_put_task( const T & ) override {
            return nullptr;
        }

        graph& graph_reference() const override {
            return my_join->graph_ref;
        }

    public:

        //! Constructor
        reserving_port() : my_join(nullptr), my_predecessors(this), reserved(false) {
            my_aggregator.initialize_handler(handler_type(this));
        }

        // copy constructor
        reserving_port(const reserving_port& /* other */) = delete;

        void set_join_node_pointer(reserving_forwarding_base *join) {
            my_join = join;
        }

        //! Add a predecessor
        bool register_predecessor( predecessor_type &src ) override {
            reserving_port_operation op_data(src, reg_pred);
            my_aggregator.execute(&op_data);
            return op_data.status == SUCCEEDED;
        }

        //! Remove a predecessor
        bool remove_predecessor( predecessor_type &src ) override {
            reserving_port_operation op_data(src, rem_pred);
            my_aggregator.execute(&op_data);
            return op_data.status == SUCCEEDED;
        }

        //! Reserve an item from the port
        bool reserve( T &v ) {
            reserving_port_operation op_data(v, res_item);
            my_aggregator.execute(&op_data);
            return op_data.status == SUCCEEDED;
        }

        //! Release the port
        void release( ) {
            reserving_port_operation op_data(rel_res);
            my_aggregator.execute(&op_data);
        }

        //! Complete use of the port
        void consume( ) {
            reserving_port_operation op_data(con_res);
            my_aggregator.execute(&op_data);
        }

        void reset_receiver( reset_flags f) {
            if(f & rf_clear_edges) my_predecessors.clear();
            else
            my_predecessors.reset();
            reserved = false;
            __TBB_ASSERT(!(f&rf_clear_edges) || my_predecessors.empty(), "port edges not removed");
        }

    private:
#if __TBB_PREVIEW_FLOW_GRAPH_NODE_SET
        friend class get_graph_helper;
#endif

        reserving_forwarding_base *my_join;
        reservable_predecessor_cache< T, null_mutex > my_predecessors;
        bool reserved;
    };  // reserving_port

    //! queueing join_port
    template<typename T>
    class queueing_port : public receiver<T>, public item_buffer<T> {
    public:
        typedef T input_type;
        typedef typename receiver<input_type>::predecessor_type predecessor_type;
        typedef queueing_port<T> class_type;

    // ----------- Aggregator ------------
    private:
        enum op_type { get__item, res_port, try__put_task
        };

        class queueing_port_operation : public aggregated_operation<queueing_port_operation> {
        public:
            char type;
            T my_val;
            T* my_arg;
            graph_task* bypass_t;
            // constructor for value parameter
            queueing_port_operation(const T& e, op_type t) :
                type(char(t)), my_val(e)
                , bypass_t(nullptr)
            {}
            // constructor for pointer parameter
            queueing_port_operation(const T* p, op_type t) :
                type(char(t)), my_arg(const_cast<T*>(p))
                , bypass_t(nullptr)
            {}
            // constructor with no parameter
            queueing_port_operation(op_type t) : type(char(t))
                , bypass_t(nullptr)
            {}
        };

        typedef aggregating_functor<class_type, queueing_port_operation> handler_type;
        friend class aggregating_functor<class_type, queueing_port_operation>;
        aggregator<handler_type, queueing_port_operation> my_aggregator;

        void handle_operations(queueing_port_operation* op_list) {
            queueing_port_operation *current;
            bool was_empty;
            while(op_list) {
                current = op_list;
                op_list = op_list->next;
                switch(current->type) {
                case try__put_task: {
                        graph_task* rtask = nullptr;
                        was_empty = this->buffer_empty();
                        this->push_back(current->my_val);
                        if (was_empty) rtask = my_join->decrement_port_count(false);
                        else
                            rtask = SUCCESSFULLY_ENQUEUED;
                        current->bypass_t = rtask;
                        current->status.store( SUCCEEDED, std::memory_order_release);
                    }
                    break;
                case get__item:
                    if(!this->buffer_empty()) {
                        *(current->my_arg) = this->front();
                        current->status.store( SUCCEEDED, std::memory_order_release);
                    }
                    else {
                        current->status.store( FAILED, std::memory_order_release);
                    }
                    break;
                case res_port:
                    __TBB_ASSERT(this->my_item_valid(this->my_head), "No item to reset");
                    this->destroy_front();
                    if(this->my_item_valid(this->my_head)) {
                        (void)my_join->decrement_port_count(true);
                    }
                    current->status.store( SUCCEEDED, std::memory_order_release);
                    break;
                }
            }
        }
    // ------------ End Aggregator ---------------

    protected:
        template< typename R, typename B > friend class run_and_put_task;
        template<typename X, typename Y> friend class broadcast_cache;
        template<typename X, typename Y> friend class round_robin_cache;
        graph_task* try_put_task(const T &v) override {
            queueing_port_operation op_data(v, try__put_task);
            my_aggregator.execute(&op_data);
            __TBB_ASSERT(op_data.status == SUCCEEDED || !op_data.bypass_t, "inconsistent return from aggregator");
            if(!op_data.bypass_t) return SUCCESSFULLY_ENQUEUED;
            return op_data.bypass_t;
        }

        graph& graph_reference() const override {
            return my_join->graph_ref;
        }

    public:

        //! Constructor
        queueing_port() : item_buffer<T>() {
            my_join = nullptr;
            my_aggregator.initialize_handler(handler_type(this));
        }

        //! copy constructor
        queueing_port(const queueing_port& /* other */) = delete;

        //! record parent for tallying available items
        void set_join_node_pointer(queueing_forwarding_base *join) {
            my_join = join;
        }

        bool get_item( T &v ) {
            queueing_port_operation op_data(&v, get__item);
            my_aggregator.execute(&op_data);
            return op_data.status == SUCCEEDED;
        }

        // reset_port is called when item is accepted by successor, but
        // is initiated by join_node.
        void reset_port() {
            queueing_port_operation op_data(res_port);
            my_aggregator.execute(&op_data);
            return;
        }

        void reset_receiver(reset_flags) {
            item_buffer<T>::reset();
        }

    private:
#if __TBB_PREVIEW_FLOW_GRAPH_NODE_SET
        friend class get_graph_helper;
#endif

        queueing_forwarding_base *my_join;
    };  // queueing_port

#include "_flow_graph_tagged_buffer_impl.h"

    template<typename K>
    struct count_element {
        K my_key;
        size_t my_value;
    };

    // method to access the key in the counting table
    // the ref has already been removed from K
    template< typename K >
    struct key_to_count_functor {
        typedef count_element<K> table_item_type;
        const K& operator()(const table_item_type& v) { return v.my_key; }
    };

    // the ports can have only one template parameter.  We wrap the types needed in
    // a traits type
    template< class TraitsType >
    class key_matching_port :
        public receiver<typename TraitsType::T>,
        public hash_buffer< typename TraitsType::K, typename TraitsType::T, typename TraitsType::TtoK,
                typename TraitsType::KHash > {
    public:
        typedef TraitsType traits;
        typedef key_matching_port<traits> class_type;
        typedef typename TraitsType::T input_type;
        typedef typename TraitsType::K key_type;
        typedef typename std::decay<key_type>::type noref_key_type;
        typedef typename receiver<input_type>::predecessor_type predecessor_type;
        typedef typename TraitsType::TtoK type_to_key_func_type;
        typedef typename TraitsType::KHash hash_compare_type;
        typedef hash_buffer< key_type, input_type, type_to_key_func_type, hash_compare_type > buffer_type;

    private:
// ----------- Aggregator ------------
    private:
        enum op_type { try__put, get__item, res_port
        };

        class key_matching_port_operation : public aggregated_operation<key_matching_port_operation> {
        public:
            char type;
            input_type my_val;
            input_type *my_arg;
            // constructor for value parameter
            key_matching_port_operation(const input_type& e, op_type t) :
                type(char(t)), my_val(e) {}
            // constructor for pointer parameter
            key_matching_port_operation(const input_type* p, op_type t) :
                type(char(t)), my_arg(const_cast<input_type*>(p)) {}
            // constructor with no parameter
            key_matching_port_operation(op_type t) : type(char(t)) {}
        };

        typedef aggregating_functor<class_type, key_matching_port_operation> handler_type;
        friend class aggregating_functor<class_type, key_matching_port_operation>;
        aggregator<handler_type, key_matching_port_operation> my_aggregator;

        void handle_operations(key_matching_port_operation* op_list) {
            key_matching_port_operation *current;
            while(op_list) {
                current = op_list;
                op_list = op_list->next;
                switch(current->type) {
                case try__put: {
                        bool was_inserted = this->insert_with_key(current->my_val);
                        // return failure if a duplicate insertion occurs
                        current->status.store( was_inserted ? SUCCEEDED : FAILED, std::memory_order_release);
                    }
                    break;
                case get__item:
                    // use current_key from FE for item
                    if(!this->find_with_key(my_join->current_key, *(current->my_arg))) {
                        __TBB_ASSERT(false, "Failed to find item corresponding to current_key.");
                    }
                    current->status.store( SUCCEEDED, std::memory_order_release);
                    break;
                case res_port:
                    // use current_key from FE for item
                    this->delete_with_key(my_join->current_key);
                    current->status.store( SUCCEEDED, std::memory_order_release);
                    break;
                }
            }
        }
// ------------ End Aggregator ---------------
    protected:
        template< typename R, typename B > friend class run_and_put_task;
        template<typename X, typename Y> friend class broadcast_cache;
        template<typename X, typename Y> friend class round_robin_cache;
        graph_task* try_put_task(const input_type& v) override {
            key_matching_port_operation op_data(v, try__put);
            graph_task* rtask = nullptr;
            my_aggregator.execute(&op_data);
            if(op_data.status == SUCCEEDED) {
                rtask = my_join->increment_key_count((*(this->get_key_func()))(v));  // may spawn
                // rtask has to reflect the return status of the try_put
                if(!rtask) rtask = SUCCESSFULLY_ENQUEUED;
            }
            return rtask;
        }

        graph& graph_reference() const override {
            return my_join->graph_ref;
        }

    public:

        key_matching_port() : receiver<input_type>(), buffer_type() {
            my_join = nullptr;
            my_aggregator.initialize_handler(handler_type(this));
        }

        // copy constructor
        key_matching_port(const key_matching_port& /*other*/) = delete;
#if __INTEL_COMPILER <= 2021
        // Suppress superfluous diagnostic about virtual keyword absence in a destructor of an inherited
        // class while the parent class has the virtual keyword for the destrocutor.
        virtual
#endif
        ~key_matching_port() { }

        void set_join_node_pointer(forwarding_base *join) {
            my_join = dynamic_cast<matching_forwarding_base<key_type>*>(join);
        }

        void set_my_key_func(type_to_key_func_type *f) { this->set_key_func(f); }

        type_to_key_func_type* get_my_key_func() { return this->get_key_func(); }

        bool get_item( input_type &v ) {
            // aggregator uses current_key from FE for Key
            key_matching_port_operation op_data(&v, get__item);
            my_aggregator.execute(&op_data);
            return op_data.status == SUCCEEDED;
        }

        // reset_port is called when item is accepted by successor, but
        // is initiated by join_node.
        void reset_port() {
            key_matching_port_operation op_data(res_port);
            my_aggregator.execute(&op_data);
            return;
        }

        void reset_receiver(reset_flags ) {
            buffer_type::reset();
        }

    private:
        // my_join forwarding base used to count number of inputs that
        // received key.
        matching_forwarding_base<key_type> *my_join;
    };  // key_matching_port

    using namespace graph_policy_namespace;

    template<typename JP, typename InputTuple, typename OutputTuple>
    class join_node_base;

    //! join_node_FE : implements input port policy
    template<typename JP, typename InputTuple, typename OutputTuple>
    class join_node_FE;

    template<typename InputTuple, typename OutputTuple>
    class join_node_FE<reserving, InputTuple, OutputTuple> : public reserving_forwarding_base {
    public:
        static const int N = std::tuple_size<OutputTuple>::value;
        typedef OutputTuple output_type;
        typedef InputTuple input_type;
        typedef join_node_base<reserving, InputTuple, OutputTuple> base_node_type; // for forwarding

        join_node_FE(graph &g) : reserving_forwarding_base(g), my_node(nullptr) {
            ports_with_no_inputs = N;
            join_helper<N>::set_join_node_pointer(my_inputs, this);
        }

        join_node_FE(const join_node_FE& other) : reserving_forwarding_base((other.reserving_forwarding_base::graph_ref)), my_node(nullptr) {
            ports_with_no_inputs = N;
            join_helper<N>::set_join_node_pointer(my_inputs, this);
        }

        void set_my_node(base_node_type *new_my_node) { my_node = new_my_node; }

       void increment_port_count() override {
            ++ports_with_no_inputs;
        }

        // if all input_ports have predecessors, spawn forward to try and consume tuples
        graph_task* decrement_port_count() override {
            if(ports_with_no_inputs.fetch_sub(1) == 1) {
                if(is_graph_active(this->graph_ref)) {
                    small_object_allocator allocator{};
                    typedef forward_task_bypass<base_node_type> task_type;
                    graph_task* t = allocator.new_object<task_type>(graph_ref, allocator, *my_node);
                    graph_ref.reserve_wait();
                    spawn_in_graph_arena(this->graph_ref, *t);
                }
            }
            return nullptr;
        }

        input_type &input_ports() { return my_inputs; }

    protected:

        void reset(  reset_flags f) {
            // called outside of parallel contexts
            ports_with_no_inputs = N;
            join_helper<N>::reset_inputs(my_inputs, f);
        }

        // all methods on input ports should be called under mutual exclusion from join_node_base.

        bool tuple_build_may_succeed() {
            return !ports_with_no_inputs;
        }

        bool try_to_make_tuple(output_type &out) {
            if(ports_with_no_inputs) return false;
            return join_helper<N>::reserve(my_inputs, out);
        }

        void tuple_accepted() {
            join_helper<N>::consume_reservations(my_inputs);
        }
        void tuple_rejected() {
            join_helper<N>::release_reservations(my_inputs);
        }

        input_type my_inputs;
        base_node_type *my_node;
        std::atomic<std::size_t> ports_with_no_inputs;
    };  // join_node_FE<reserving, ... >

    template<typename InputTuple, typename OutputTuple>
    class join_node_FE<queueing, InputTuple, OutputTuple> : public queueing_forwarding_base {
    public:
        static const int N = std::tuple_size<OutputTuple>::value;
        typedef OutputTuple output_type;
        typedef InputTuple input_type;
        typedef join_node_base<queueing, InputTuple, OutputTuple> base_node_type; // for forwarding

        join_node_FE(graph &g) : queueing_forwarding_base(g), my_node(nullptr) {
            ports_with_no_items = N;
            join_helper<N>::set_join_node_pointer(my_inputs, this);
        }

        join_node_FE(const join_node_FE& other) : queueing_forwarding_base((other.queueing_forwarding_base::graph_ref)), my_node(nullptr) {
            ports_with_no_items = N;
            join_helper<N>::set_join_node_pointer(my_inputs, this);
        }

        // needed for forwarding
        void set_my_node(base_node_type *new_my_node) { my_node = new_my_node; }

        void reset_port_count() {
            ports_with_no_items = N;
        }

        // if all input_ports have items, spawn forward to try and consume tuples
        graph_task* decrement_port_count(bool handle_task) override
        {
            if(ports_with_no_items.fetch_sub(1) == 1) {
                if(is_graph_active(this->graph_ref)) {
                    small_object_allocator allocator{};
                    typedef forward_task_bypass<base_node_type> task_type;
                    graph_task* t = allocator.new_object<task_type>(graph_ref, allocator, *my_node);
                    graph_ref.reserve_wait();
                    if( !handle_task )
                        return t;
                    spawn_in_graph_arena(this->graph_ref, *t);
                }
            }
            return nullptr;
        }

        input_type &input_ports() { return my_inputs; }

    protected:

        void reset(  reset_flags f) {
            reset_port_count();
            join_helper<N>::reset_inputs(my_inputs, f );
        }

        // all methods on input ports should be called under mutual exclusion from join_node_base.

        bool tuple_build_may_succeed() {
            return !ports_with_no_items;
        }

        bool try_to_make_tuple(output_type &out) {
            if(ports_with_no_items) return false;
            return join_helper<N>::get_items(my_inputs, out);
        }

        void tuple_accepted() {
            reset_port_count();
            join_helper<N>::reset_ports(my_inputs);
        }
        void tuple_rejected() {
            // nothing to do.
        }

        input_type my_inputs;
        base_node_type *my_node;
        std::atomic<std::size_t> ports_with_no_items;
    };  // join_node_FE<queueing, ...>

    // key_matching join front-end.
    template<typename InputTuple, typename OutputTuple, typename K, typename KHash>
    class join_node_FE<key_matching<K,KHash>, InputTuple, OutputTuple> : public matching_forwarding_base<K>,
             // buffer of key value counts
              public hash_buffer<   // typedefed below to key_to_count_buffer_type
                  typename std::decay<K>::type&,        // force ref type on K
                  count_element<typename std::decay<K>::type>,
                  type_to_key_function_body<
                      count_element<typename std::decay<K>::type>,
                      typename std::decay<K>::type& >,
                  KHash >,
             // buffer of output items
             public item_buffer<OutputTuple> {
    public:
        static const int N = std::tuple_size<OutputTuple>::value;
        typedef OutputTuple output_type;
        typedef InputTuple input_type;
        typedef K key_type;
        typedef typename std::decay<key_type>::type unref_key_type;
        typedef KHash key_hash_compare;
        // must use K without ref.
        typedef count_element<unref_key_type> count_element_type;
        // method that lets us refer to the key of this type.
        typedef key_to_count_functor<unref_key_type> key_to_count_func;
        typedef type_to_key_function_body< count_element_type, unref_key_type&> TtoK_function_body_type;
        typedef type_to_key_function_body_leaf<count_element_type, unref_key_type&, key_to_count_func> TtoK_function_body_leaf_type;
        // this is the type of the special table that keeps track of the number of discrete
        // elements corresponding to each key that we've seen.
        typedef hash_buffer< unref_key_type&, count_element_type, TtoK_function_body_type, key_hash_compare >
                 key_to_count_buffer_type;
        typedef item_buffer<output_type> output_buffer_type;
        typedef join_node_base<key_matching<key_type,key_hash_compare>, InputTuple, OutputTuple> base_node_type; // for forwarding
        typedef matching_forwarding_base<key_type> forwarding_base_type;

// ----------- Aggregator ------------
        // the aggregator is only needed to serialize the access to the hash table.
        // and the output_buffer_type base class
    private:
        enum op_type { res_count, inc_count, may_succeed, try_make };
        typedef join_node_FE<key_matching<key_type,key_hash_compare>, InputTuple, OutputTuple> class_type;

        class key_matching_FE_operation : public aggregated_operation<key_matching_FE_operation> {
        public:
            char type;
            unref_key_type my_val;
            output_type* my_output;
            graph_task* bypass_t;
            // constructor for value parameter
            key_matching_FE_operation(const unref_key_type& e , op_type t) : type(char(t)), my_val(e),
                 my_output(nullptr), bypass_t(nullptr) {}
            key_matching_FE_operation(output_type *p, op_type t) : type(char(t)), my_output(p), bypass_t(nullptr) {}
            // constructor with no parameter
            key_matching_FE_operation(op_type t) : type(char(t)), my_output(nullptr), bypass_t(nullptr) {}
        };

        typedef aggregating_functor<class_type, key_matching_FE_operation> handler_type;
        friend class aggregating_functor<class_type, key_matching_FE_operation>;
        aggregator<handler_type, key_matching_FE_operation> my_aggregator;

        // called from aggregator, so serialized
        // returns a task pointer if the a task would have been enqueued but we asked that
        // it be returned.  Otherwise returns nullptr.
        graph_task* fill_output_buffer(unref_key_type &t) {
            output_type l_out;
            graph_task* rtask = nullptr;
            bool do_fwd = this->buffer_empty() && is_graph_active(this->graph_ref);
            this->current_key = t;
            this->delete_with_key(this->current_key);   // remove the key
            if(join_helper<N>::get_items(my_inputs, l_out)) {  //  <== call back
                this->push_back(l_out);
                if(do_fwd) {  // we enqueue if receiving an item from predecessor, not if successor asks for item
                    small_object_allocator allocator{};
                    typedef forward_task_bypass<base_node_type> task_type;
                    rtask = allocator.new_object<task_type>(this->graph_ref, allocator, *my_node);
                    this->graph_ref.reserve_wait();
                    do_fwd = false;
                }
                // retire the input values
                join_helper<N>::reset_ports(my_inputs);  //  <== call back
            }
            else {
                __TBB_ASSERT(false, "should have had something to push");
            }
            return rtask;
        }

        void handle_operations(key_matching_FE_operation* op_list) {
            key_matching_FE_operation *current;
            while(op_list) {
                current = op_list;
                op_list = op_list->next;
                switch(current->type) {
                case res_count:  // called from BE
                    {
                        this->destroy_front();
                        current->status.store( SUCCEEDED, std::memory_order_release);
                    }
                    break;
                case inc_count: {  // called from input ports
                        count_element_type *p = 0;
                        unref_key_type &t = current->my_val;
                        if(!(this->find_ref_with_key(t,p))) {
                            count_element_type ev;
                            ev.my_key = t;
                            ev.my_value = 0;
                            this->insert_with_key(ev);
                            bool found = this->find_ref_with_key(t, p);
                            __TBB_ASSERT_EX(found, "should find key after inserting it");
                        }
                        if(++(p->my_value) == size_t(N)) {
                            current->bypass_t = fill_output_buffer(t);
                        }
                    }
                    current->status.store( SUCCEEDED, std::memory_order_release);
                    break;
                case may_succeed:  // called from BE
                    current->status.store( this->buffer_empty() ? FAILED : SUCCEEDED, std::memory_order_release);
                    break;
                case try_make:  // called from BE
                    if(this->buffer_empty()) {
                        current->status.store( FAILED, std::memory_order_release);
                    }
                    else {
                        *(current->my_output) = this->front();
                        current->status.store( SUCCEEDED, std::memory_order_release);
                    }
                    break;
                }
            }
        }
// ------------ End Aggregator ---------------

    public:
        template<typename FunctionTuple>
        join_node_FE(graph &g, FunctionTuple &TtoK_funcs) : forwarding_base_type(g), my_node(nullptr) {
            join_helper<N>::set_join_node_pointer(my_inputs, this);
            join_helper<N>::set_key_functors(my_inputs, TtoK_funcs);
            my_aggregator.initialize_handler(handler_type(this));
                    TtoK_function_body_type *cfb = new TtoK_function_body_leaf_type(key_to_count_func());
            this->set_key_func(cfb);
        }

        join_node_FE(const join_node_FE& other) : forwarding_base_type((other.forwarding_base_type::graph_ref)), key_to_count_buffer_type(),
        output_buffer_type() {
            my_node = nullptr;
            join_helper<N>::set_join_node_pointer(my_inputs, this);
            join_helper<N>::copy_key_functors(my_inputs, const_cast<input_type &>(other.my_inputs));
            my_aggregator.initialize_handler(handler_type(this));
            TtoK_function_body_type *cfb = new TtoK_function_body_leaf_type(key_to_count_func());
            this->set_key_func(cfb);
        }

        // needed for forwarding
        void set_my_node(base_node_type *new_my_node) { my_node = new_my_node; }

        void reset_port_count() {  // called from BE
            key_matching_FE_operation op_data(res_count);
            my_aggregator.execute(&op_data);
            return;
        }

        // if all input_ports have items, spawn forward to try and consume tuples
        // return a task if we are asked and did create one.
        graph_task *increment_key_count(unref_key_type const & t) override {  // called from input_ports
            key_matching_FE_operation op_data(t, inc_count);
            my_aggregator.execute(&op_data);
            return op_data.bypass_t;
        }

        input_type &input_ports() { return my_inputs; }

    protected:

        void reset(  reset_flags f ) {
            // called outside of parallel contexts
            join_helper<N>::reset_inputs(my_inputs, f);

            key_to_count_buffer_type::reset();
            output_buffer_type::reset();
        }

        // all methods on input ports should be called under mutual exclusion from join_node_base.

        bool tuple_build_may_succeed() {  // called from back-end
            key_matching_FE_operation op_data(may_succeed);
            my_aggregator.execute(&op_data);
            return op_data.status == SUCCEEDED;
        }

        // cannot lock while calling back to input_ports.  current_key will only be set
        // and reset under the aggregator, so it will remain consistent.
        bool try_to_make_tuple(output_type &out) {
            key_matching_FE_operation op_data(&out,try_make);
            my_aggregator.execute(&op_data);
            return op_data.status == SUCCEEDED;
        }

        void tuple_accepted() {
            reset_port_count();  // reset current_key after ports reset.
        }

        void tuple_rejected() {
            // nothing to do.
        }

        input_type my_inputs;  // input ports
        base_node_type *my_node;
    }; // join_node_FE<key_matching<K,KHash>, InputTuple, OutputTuple>

    //! join_node_base
    template<typename JP, typename InputTuple, typename OutputTuple>
    class join_node_base : public graph_node, public join_node_FE<JP, InputTuple, OutputTuple>,
                           public sender<OutputTuple> {
    protected:
        using graph_node::my_graph;
    public:
        typedef OutputTuple output_type;

        typedef typename sender<output_type>::successor_type successor_type;
        typedef join_node_FE<JP, InputTuple, OutputTuple> input_ports_type;
        using input_ports_type::tuple_build_may_succeed;
        using input_ports_type::try_to_make_tuple;
        using input_ports_type::tuple_accepted;
        using input_ports_type::tuple_rejected;

    private:
        // ----------- Aggregator ------------
        enum op_type { reg_succ, rem_succ, try__get, do_fwrd, do_fwrd_bypass
        };
        typedef join_node_base<JP,InputTuple,OutputTuple> class_type;

        class join_node_base_operation : public aggregated_operation<join_node_base_operation> {
        public:
            char type;
            union {
                output_type *my_arg;
                successor_type *my_succ;
            };
            graph_task* bypass_t;
            join_node_base_operation(const output_type& e, op_type t) : type(char(t)),
                my_arg(const_cast<output_type*>(&e)), bypass_t(nullptr) {}
            join_node_base_operation(const successor_type &s, op_type t) : type(char(t)),
                my_succ(const_cast<successor_type *>(&s)), bypass_t(nullptr) {}
            join_node_base_operation(op_type t) : type(char(t)), bypass_t(nullptr) {}
        };

        typedef aggregating_functor<class_type, join_node_base_operation> handler_type;
        friend class aggregating_functor<class_type, join_node_base_operation>;
        bool forwarder_busy;
        aggregator<handler_type, join_node_base_operation> my_aggregator;

        void handle_operations(join_node_base_operation* op_list) {
            join_node_base_operation *current;
            while(op_list) {
                current = op_list;
                op_list = op_list->next;
                switch(current->type) {
                case reg_succ: {
                        my_successors.register_successor(*(current->my_succ));
                        if(tuple_build_may_succeed() && !forwarder_busy && is_graph_active(my_graph)) {
                            small_object_allocator allocator{};
                            typedef forward_task_bypass< join_node_base<JP, InputTuple, OutputTuple> > task_type;
                            graph_task* t = allocator.new_object<task_type>(my_graph, allocator, *this);
                            my_graph.reserve_wait();
                            spawn_in_graph_arena(my_graph, *t);
                            forwarder_busy = true;
                        }
                        current->status.store( SUCCEEDED, std::memory_order_release);
                    }
                    break;
                case rem_succ:
                    my_successors.remove_successor(*(current->my_succ));
                    current->status.store( SUCCEEDED, std::memory_order_release);
                    break;
                case try__get:
                    if(tuple_build_may_succeed()) {
                        if(try_to_make_tuple(*(current->my_arg))) {
                            tuple_accepted();
                            current->status.store( SUCCEEDED, std::memory_order_release);
                        }
                        else current->status.store( FAILED, std::memory_order_release);
                    }
                    else current->status.store( FAILED, std::memory_order_release);
                    break;
                case do_fwrd_bypass: {
                        bool build_succeeded;
                        graph_task *last_task = nullptr;
                        output_type out;
                        // forwarding must be exclusive, because try_to_make_tuple and tuple_accepted
                        // are separate locked methods in the FE.  We could conceivably fetch the front
                        // of the FE queue, then be swapped out, have someone else consume the FE's
                        // object, then come back, forward, and then try to remove it from the queue
                        // again. Without reservation of the FE, the methods accessing it must be locked.
                        // We could remember the keys of the objects we forwarded, and then remove
                        // them from the input ports after forwarding is complete?
                        if(tuple_build_may_succeed()) {  // checks output queue of FE
                            do {
                                build_succeeded = try_to_make_tuple(out);  // fetch front_end of queue
                                if(build_succeeded) {
                                    graph_task *new_task = my_successors.try_put_task(out);
                                    last_task = combine_tasks(my_graph, last_task, new_task);
                                    if(new_task) {
                                        tuple_accepted();
                                    }
                                    else {
                                        tuple_rejected();
                                        build_succeeded = false;
                                    }
                                }
                            } while(build_succeeded);
                        }
                        current->bypass_t = last_task;
                        current->status.store( SUCCEEDED, std::memory_order_release);
                        forwarder_busy = false;
                    }
                    break;
                }
            }
        }
        // ---------- end aggregator -----------
    public:
        join_node_base(graph &g)
            : graph_node(g), input_ports_type(g), forwarder_busy(false), my_successors(this)
        {
            input_ports_type::set_my_node(this);
            my_aggregator.initialize_handler(handler_type(this));
        }

        join_node_base(const join_node_base& other) :
            graph_node(other.graph_node::my_graph), input_ports_type(other),
            sender<OutputTuple>(), forwarder_busy(false), my_successors(this)
        {
            input_ports_type::set_my_node(this);
            my_aggregator.initialize_handler(handler_type(this));
        }

        template<typename FunctionTuple>
        join_node_base(graph &g, FunctionTuple f)
            : graph_node(g), input_ports_type(g, f), forwarder_busy(false), my_successors(this)
        {
            input_ports_type::set_my_node(this);
            my_aggregator.initialize_handler(handler_type(this));
        }

        bool register_successor(successor_type &r) override {
            join_node_base_operation op_data(r, reg_succ);
            my_aggregator.execute(&op_data);
            return op_data.status == SUCCEEDED;
        }

        bool remove_successor( successor_type &r) override {
            join_node_base_operation op_data(r, rem_succ);
            my_aggregator.execute(&op_data);
            return op_data.status == SUCCEEDED;
        }

        bool try_get( output_type &v) override {
            join_node_base_operation op_data(v, try__get);
            my_aggregator.execute(&op_data);
            return op_data.status == SUCCEEDED;
        }

    protected:
        void reset_node(reset_flags f) override {
            input_ports_type::reset(f);
            if(f & rf_clear_edges) my_successors.clear();
        }

    private:
        broadcast_cache<output_type, null_rw_mutex> my_successors;

        friend class forward_task_bypass< join_node_base<JP, InputTuple, OutputTuple> >;
        graph_task *forward_task() {
            join_node_base_operation op_data(do_fwrd_bypass);
            my_aggregator.execute(&op_data);
            return op_data.bypass_t;
        }

    };  // join_node_base

    // join base class type generator
    template<int N, template<class> class PT, typename OutputTuple, typename JP>
    struct join_base {
        typedef join_node_base<JP, typename wrap_tuple_elements<N,PT,OutputTuple>::type, OutputTuple> type;
    };

    template<int N, typename OutputTuple, typename K, typename KHash>
    struct join_base<N, key_matching_port, OutputTuple, key_matching<K,KHash> > {
        typedef key_matching<K, KHash> key_traits_type;
        typedef K key_type;
        typedef KHash key_hash_compare;
        typedef join_node_base< key_traits_type,
                // ports type
                typename wrap_key_tuple_elements<N,key_matching_port,key_traits_type,OutputTuple>::type,
                OutputTuple > type;
    };

    //! unfolded_join_node : passes input_ports_type to join_node_base.  We build the input port type
    //  using tuple_element.  The class PT is the port type (reserving_port, queueing_port, key_matching_port)
    //  and should match the typename.

    template<int N, template<class> class PT, typename OutputTuple, typename JP>
    class unfolded_join_node : public join_base<N,PT,OutputTuple,JP>::type {
    public:
        typedef typename wrap_tuple_elements<N, PT, OutputTuple>::type input_ports_type;
        typedef OutputTuple output_type;
    private:
        typedef join_node_base<JP, input_ports_type, output_type > base_type;
    public:
        unfolded_join_node(graph &g) : base_type(g) {}
        unfolded_join_node(const unfolded_join_node &other) : base_type(other) {}
    };

#if __TBB_PREVIEW_MESSAGE_BASED_KEY_MATCHING
    template <typename K, typename T>
    struct key_from_message_body {
        K operator()(const T& t) const {
            return key_from_message<K>(t);
        }
    };
    // Adds const to reference type
    template <typename K, typename T>
    struct key_from_message_body<K&,T> {
        const K& operator()(const T& t) const {
            return key_from_message<const K&>(t);
        }
    };
#endif /* __TBB_PREVIEW_MESSAGE_BASED_KEY_MATCHING */
    // key_matching unfolded_join_node.  This must be a separate specialization because the constructors
    // differ.

    template<typename OutputTuple, typename K, typename KHash>
    class unfolded_join_node<2,key_matching_port,OutputTuple,key_matching<K,KHash> > : public
            join_base<2,key_matching_port,OutputTuple,key_matching<K,KHash> >::type {
        typedef typename std::tuple_element<0, OutputTuple>::type T0;
        typedef typename std::tuple_element<1, OutputTuple>::type T1;
    public:
        typedef typename wrap_key_tuple_elements<2,key_matching_port,key_matching<K,KHash>,OutputTuple>::type input_ports_type;
        typedef OutputTuple output_type;
    private:
        typedef join_node_base<key_matching<K,KHash>, input_ports_type, output_type > base_type;
        typedef type_to_key_function_body<T0, K> *f0_p;
        typedef type_to_key_function_body<T1, K> *f1_p;
        typedef std::tuple< f0_p, f1_p > func_initializer_type;
    public:
#if __TBB_PREVIEW_MESSAGE_BASED_KEY_MATCHING
        unfolded_join_node(graph &g) : base_type(g,
                func_initializer_type(
                    new type_to_key_function_body_leaf<T0, K, key_from_message_body<K,T0> >(key_from_message_body<K,T0>()),
                    new type_to_key_function_body_leaf<T1, K, key_from_message_body<K,T1> >(key_from_message_body<K,T1>())
                    ) ) {
        }
#endif /* __TBB_PREVIEW_MESSAGE_BASED_KEY_MATCHING */
        template<typename Body0, typename Body1>
        unfolded_join_node(graph &g, Body0 body0, Body1 body1) : base_type(g,
                func_initializer_type(
                    new type_to_key_function_body_leaf<T0, K, Body0>(body0),
                    new type_to_key_function_body_leaf<T1, K, Body1>(body1)
                    ) ) {
            static_assert(std::tuple_size<OutputTuple>::value == 2, "wrong number of body initializers");
        }
        unfolded_join_node(const unfolded_join_node &other) : base_type(other) {}
    };

    template<typename OutputTuple, typename K, typename KHash>
    class unfolded_join_node<3,key_matching_port,OutputTuple,key_matching<K,KHash> > : public
            join_base<3,key_matching_port,OutputTuple,key_matching<K,KHash> >::type {
        typedef typename std::tuple_element<0, OutputTuple>::type T0;
        typedef typename std::tuple_element<1, OutputTuple>::type T1;
        typedef typename std::tuple_element<2, OutputTuple>::type T2;
    public:
        typedef typename wrap_key_tuple_elements<3,key_matching_port,key_matching<K,KHash>,OutputTuple>::type input_ports_type;
        typedef OutputTuple output_type;
    private:
        typedef join_node_base<key_matching<K,KHash>, input_ports_type, output_type > base_type;
        typedef type_to_key_function_body<T0, K> *f0_p;
        typedef type_to_key_function_body<T1, K> *f1_p;
        typedef type_to_key_function_body<T2, K> *f2_p;
        typedef std::tuple< f0_p, f1_p, f2_p > func_initializer_type;
    public:
#if __TBB_PREVIEW_MESSAGE_BASED_KEY_MATCHING
        unfolded_join_node(graph &g) : base_type(g,
                func_initializer_type(
                    new type_to_key_function_body_leaf<T0, K, key_from_message_body<K,T0> >(key_from_message_body<K,T0>()),
                    new type_to_key_function_body_leaf<T1, K, key_from_message_body<K,T1> >(key_from_message_body<K,T1>()),
                    new type_to_key_function_body_leaf<T2, K, key_from_message_body<K,T2> >(key_from_message_body<K,T2>())
                    ) ) {
        }
#endif /* __TBB_PREVIEW_MESSAGE_BASED_KEY_MATCHING */
        template<typename Body0, typename Body1, typename Body2>
        unfolded_join_node(graph &g, Body0 body0, Body1 body1, Body2 body2) : base_type(g,
                func_initializer_type(
                    new type_to_key_function_body_leaf<T0, K, Body0>(body0),
                    new type_to_key_function_body_leaf<T1, K, Body1>(body1),
                    new type_to_key_function_body_leaf<T2, K, Body2>(body2)
                    ) ) {
            static_assert(std::tuple_size<OutputTuple>::value == 3, "wrong number of body initializers");
        }
        unfolded_join_node(const unfolded_join_node &other) : base_type(other) {}
    };

    template<typename OutputTuple, typename K, typename KHash>
    class unfolded_join_node<4,key_matching_port,OutputTuple,key_matching<K,KHash> > : public
            join_base<4,key_matching_port,OutputTuple,key_matching<K,KHash> >::type {
        typedef typename std::tuple_element<0, OutputTuple>::type T0;
        typedef typename std::tuple_element<1, OutputTuple>::type T1;
        typedef typename std::tuple_element<2, OutputTuple>::type T2;
        typedef typename std::tuple_element<3, OutputTuple>::type T3;
    public:
        typedef typename wrap_key_tuple_elements<4,key_matching_port,key_matching<K,KHash>,OutputTuple>::type input_ports_type;
        typedef OutputTuple output_type;
    private:
        typedef join_node_base<key_matching<K,KHash>, input_ports_type, output_type > base_type;
        typedef type_to_key_function_body<T0, K> *f0_p;
        typedef type_to_key_function_body<T1, K> *f1_p;
        typedef type_to_key_function_body<T2, K> *f2_p;
        typedef type_to_key_function_body<T3, K> *f3_p;
        typedef std::tuple< f0_p, f1_p, f2_p, f3_p > func_initializer_type;
    public:
#if __TBB_PREVIEW_MESSAGE_BASED_KEY_MATCHING
        unfolded_join_node(graph &g) : base_type(g,
                func_initializer_type(
                    new type_to_key_function_body_leaf<T0, K, key_from_message_body<K,T0> >(key_from_message_body<K,T0>()),
                    new type_to_key_function_body_leaf<T1, K, key_from_message_body<K,T1> >(key_from_message_body<K,T1>()),
                    new type_to_key_function_body_leaf<T2, K, key_from_message_body<K,T2> >(key_from_message_body<K,T2>()),
                    new type_to_key_function_body_leaf<T3, K, key_from_message_body<K,T3> >(key_from_message_body<K,T3>())
                    ) ) {
        }
#endif /* __TBB_PREVIEW_MESSAGE_BASED_KEY_MATCHING */
        template<typename Body0, typename Body1, typename Body2, typename Body3>
        unfolded_join_node(graph &g, Body0 body0, Body1 body1, Body2 body2, Body3 body3) : base_type(g,
                func_initializer_type(
                    new type_to_key_function_body_leaf<T0, K, Body0>(body0),
                    new type_to_key_function_body_leaf<T1, K, Body1>(body1),
                    new type_to_key_function_body_leaf<T2, K, Body2>(body2),
                    new type_to_key_function_body_leaf<T3, K, Body3>(body3)
                    ) ) {
            static_assert(std::tuple_size<OutputTuple>::value == 4, "wrong number of body initializers");
        }
        unfolded_join_node(const unfolded_join_node &other) : base_type(other) {}
    };

    template<typename OutputTuple, typename K, typename KHash>
    class unfolded_join_node<5,key_matching_port,OutputTuple,key_matching<K,KHash> > : public
            join_base<5,key_matching_port,OutputTuple,key_matching<K,KHash> >::type {
        typedef typename std::tuple_element<0, OutputTuple>::type T0;
        typedef typename std::tuple_element<1, OutputTuple>::type T1;
        typedef typename std::tuple_element<2, OutputTuple>::type T2;
        typedef typename std::tuple_element<3, OutputTuple>::type T3;
        typedef typename std::tuple_element<4, OutputTuple>::type T4;
    public:
        typedef typename wrap_key_tuple_elements<5,key_matching_port,key_matching<K,KHash>,OutputTuple>::type input_ports_type;
        typedef OutputTuple output_type;
    private:
        typedef join_node_base<key_matching<K,KHash> , input_ports_type, output_type > base_type;
        typedef type_to_key_function_body<T0, K> *f0_p;
        typedef type_to_key_function_body<T1, K> *f1_p;
        typedef type_to_key_function_body<T2, K> *f2_p;
        typedef type_to_key_function_body<T3, K> *f3_p;
        typedef type_to_key_function_body<T4, K> *f4_p;
        typedef std::tuple< f0_p, f1_p, f2_p, f3_p, f4_p > func_initializer_type;
    public:
#if __TBB_PREVIEW_MESSAGE_BASED_KEY_MATCHING
        unfolded_join_node(graph &g) : base_type(g,
                func_initializer_type(
                    new type_to_key_function_body_leaf<T0, K, key_from_message_body<K,T0> >(key_from_message_body<K,T0>()),
                    new type_to_key_function_body_leaf<T1, K, key_from_message_body<K,T1> >(key_from_message_body<K,T1>()),
                    new type_to_key_function_body_leaf<T2, K, key_from_message_body<K,T2> >(key_from_message_body<K,T2>()),
                    new type_to_key_function_body_leaf<T3, K, key_from_message_body<K,T3> >(key_from_message_body<K,T3>()),
                    new type_to_key_function_body_leaf<T4, K, key_from_message_body<K,T4> >(key_from_message_body<K,T4>())
                    ) ) {
        }
#endif /* __TBB_PREVIEW_MESSAGE_BASED_KEY_MATCHING */
        template<typename Body0, typename Body1, typename Body2, typename Body3, typename Body4>
        unfolded_join_node(graph &g, Body0 body0, Body1 body1, Body2 body2, Body3 body3, Body4 body4) : base_type(g,
                func_initializer_type(
                    new type_to_key_function_body_leaf<T0, K, Body0>(body0),
                    new type_to_key_function_body_leaf<T1, K, Body1>(body1),
                    new type_to_key_function_body_leaf<T2, K, Body2>(body2),
                    new type_to_key_function_body_leaf<T3, K, Body3>(body3),
                    new type_to_key_function_body_leaf<T4, K, Body4>(body4)
                    ) ) {
            static_assert(std::tuple_size<OutputTuple>::value == 5, "wrong number of body initializers");
        }
        unfolded_join_node(const unfolded_join_node &other) : base_type(other) {}
    };

#if __TBB_VARIADIC_MAX >= 6
    template<typename OutputTuple, typename K, typename KHash>
    class unfolded_join_node<6,key_matching_port,OutputTuple,key_matching<K,KHash> > : public
            join_base<6,key_matching_port,OutputTuple,key_matching<K,KHash> >::type {
        typedef typename std::tuple_element<0, OutputTuple>::type T0;
        typedef typename std::tuple_element<1, OutputTuple>::type T1;
        typedef typename std::tuple_element<2, OutputTuple>::type T2;
        typedef typename std::tuple_element<3, OutputTuple>::type T3;
        typedef typename std::tuple_element<4, OutputTuple>::type T4;
        typedef typename std::tuple_element<5, OutputTuple>::type T5;
    public:
        typedef typename wrap_key_tuple_elements<6,key_matching_port,key_matching<K,KHash>,OutputTuple>::type input_ports_type;
        typedef OutputTuple output_type;
    private:
        typedef join_node_base<key_matching<K,KHash> , input_ports_type, output_type > base_type;
        typedef type_to_key_function_body<T0, K> *f0_p;
        typedef type_to_key_function_body<T1, K> *f1_p;
        typedef type_to_key_function_body<T2, K> *f2_p;
        typedef type_to_key_function_body<T3, K> *f3_p;
        typedef type_to_key_function_body<T4, K> *f4_p;
        typedef type_to_key_function_body<T5, K> *f5_p;
        typedef std::tuple< f0_p, f1_p, f2_p, f3_p, f4_p, f5_p > func_initializer_type;
    public:
#if __TBB_PREVIEW_MESSAGE_BASED_KEY_MATCHING
        unfolded_join_node(graph &g) : base_type(g,
                func_initializer_type(
                    new type_to_key_function_body_leaf<T0, K, key_from_message_body<K,T0> >(key_from_message_body<K,T0>()),
                    new type_to_key_function_body_leaf<T1, K, key_from_message_body<K,T1> >(key_from_message_body<K,T1>()),
                    new type_to_key_function_body_leaf<T2, K, key_from_message_body<K,T2> >(key_from_message_body<K,T2>()),
                    new type_to_key_function_body_leaf<T3, K, key_from_message_body<K,T3> >(key_from_message_body<K,T3>()),
                    new type_to_key_function_body_leaf<T4, K, key_from_message_body<K,T4> >(key_from_message_body<K,T4>()),
                    new type_to_key_function_body_leaf<T5, K, key_from_message_body<K,T5> >(key_from_message_body<K,T5>())
                    ) ) {
        }
#endif /* __TBB_PREVIEW_MESSAGE_BASED_KEY_MATCHING */
        template<typename Body0, typename Body1, typename Body2, typename Body3, typename Body4, typename Body5>
        unfolded_join_node(graph &g, Body0 body0, Body1 body1, Body2 body2, Body3 body3, Body4 body4, Body5 body5)
                : base_type(g, func_initializer_type(
                    new type_to_key_function_body_leaf<T0, K, Body0>(body0),
                    new type_to_key_function_body_leaf<T1, K, Body1>(body1),
                    new type_to_key_function_body_leaf<T2, K, Body2>(body2),
                    new type_to_key_function_body_leaf<T3, K, Body3>(body3),
                    new type_to_key_function_body_leaf<T4, K, Body4>(body4),
                    new type_to_key_function_body_leaf<T5, K, Body5>(body5)
                    ) ) {
            static_assert(std::tuple_size<OutputTuple>::value == 6, "wrong number of body initializers");
        }
        unfolded_join_node(const unfolded_join_node &other) : base_type(other) {}
    };
#endif

#if __TBB_VARIADIC_MAX >= 7
    template<typename OutputTuple, typename K, typename KHash>
    class unfolded_join_node<7,key_matching_port,OutputTuple,key_matching<K,KHash> > : public
            join_base<7,key_matching_port,OutputTuple,key_matching<K,KHash> >::type {
        typedef typename std::tuple_element<0, OutputTuple>::type T0;
        typedef typename std::tuple_element<1, OutputTuple>::type T1;
        typedef typename std::tuple_element<2, OutputTuple>::type T2;
        typedef typename std::tuple_element<3, OutputTuple>::type T3;
        typedef typename std::tuple_element<4, OutputTuple>::type T4;
        typedef typename std::tuple_element<5, OutputTuple>::type T5;
        typedef typename std::tuple_element<6, OutputTuple>::type T6;
    public:
        typedef typename wrap_key_tuple_elements<7,key_matching_port,key_matching<K,KHash>,OutputTuple>::type input_ports_type;
        typedef OutputTuple output_type;
    private:
        typedef join_node_base<key_matching<K,KHash> , input_ports_type, output_type > base_type;
        typedef type_to_key_function_body<T0, K> *f0_p;
        typedef type_to_key_function_body<T1, K> *f1_p;
        typedef type_to_key_function_body<T2, K> *f2_p;
        typedef type_to_key_function_body<T3, K> *f3_p;
        typedef type_to_key_function_body<T4, K> *f4_p;
        typedef type_to_key_function_body<T5, K> *f5_p;
        typedef type_to_key_function_body<T6, K> *f6_p;
        typedef std::tuple< f0_p, f1_p, f2_p, f3_p, f4_p, f5_p, f6_p > func_initializer_type;
    public:
#if __TBB_PREVIEW_MESSAGE_BASED_KEY_MATCHING
        unfolded_join_node(graph &g) : base_type(g,
                func_initializer_type(
                    new type_to_key_function_body_leaf<T0, K, key_from_message_body<K,T0> >(key_from_message_body<K,T0>()),
                    new type_to_key_function_body_leaf<T1, K, key_from_message_body<K,T1> >(key_from_message_body<K,T1>()),
                    new type_to_key_function_body_leaf<T2, K, key_from_message_body<K,T2> >(key_from_message_body<K,T2>()),
                    new type_to_key_function_body_leaf<T3, K, key_from_message_body<K,T3> >(key_from_message_body<K,T3>()),
                    new type_to_key_function_body_leaf<T4, K, key_from_message_body<K,T4> >(key_from_message_body<K,T4>()),
                    new type_to_key_function_body_leaf<T5, K, key_from_message_body<K,T5> >(key_from_message_body<K,T5>()),
                    new type_to_key_function_body_leaf<T6, K, key_from_message_body<K,T6> >(key_from_message_body<K,T6>())
                    ) ) {
        }
#endif /* __TBB_PREVIEW_MESSAGE_BASED_KEY_MATCHING */
        template<typename Body0, typename Body1, typename Body2, typename Body3, typename Body4,
                 typename Body5, typename Body6>
        unfolded_join_node(graph &g, Body0 body0, Body1 body1, Body2 body2, Body3 body3, Body4 body4,
                Body5 body5, Body6 body6) : base_type(g, func_initializer_type(
                    new type_to_key_function_body_leaf<T0, K, Body0>(body0),
                    new type_to_key_function_body_leaf<T1, K, Body1>(body1),
                    new type_to_key_function_body_leaf<T2, K, Body2>(body2),
                    new type_to_key_function_body_leaf<T3, K, Body3>(body3),
                    new type_to_key_function_body_leaf<T4, K, Body4>(body4),
                    new type_to_key_function_body_leaf<T5, K, Body5>(body5),
                    new type_to_key_function_body_leaf<T6, K, Body6>(body6)
                    ) ) {
            static_assert(std::tuple_size<OutputTuple>::value == 7, "wrong number of body initializers");
        }
        unfolded_join_node(const unfolded_join_node &other) : base_type(other) {}
    };
#endif

#if __TBB_VARIADIC_MAX >= 8
    template<typename OutputTuple, typename K, typename KHash>
    class unfolded_join_node<8,key_matching_port,OutputTuple,key_matching<K,KHash> > : public
            join_base<8,key_matching_port,OutputTuple,key_matching<K,KHash> >::type {
        typedef typename std::tuple_element<0, OutputTuple>::type T0;
        typedef typename std::tuple_element<1, OutputTuple>::type T1;
        typedef typename std::tuple_element<2, OutputTuple>::type T2;
        typedef typename std::tuple_element<3, OutputTuple>::type T3;
        typedef typename std::tuple_element<4, OutputTuple>::type T4;
        typedef typename std::tuple_element<5, OutputTuple>::type T5;
        typedef typename std::tuple_element<6, OutputTuple>::type T6;
        typedef typename std::tuple_element<7, OutputTuple>::type T7;
    public:
        typedef typename wrap_key_tuple_elements<8,key_matching_port,key_matching<K,KHash>,OutputTuple>::type input_ports_type;
        typedef OutputTuple output_type;
    private:
        typedef join_node_base<key_matching<K,KHash> , input_ports_type, output_type > base_type;
        typedef type_to_key_function_body<T0, K> *f0_p;
        typedef type_to_key_function_body<T1, K> *f1_p;
        typedef type_to_key_function_body<T2, K> *f2_p;
        typedef type_to_key_function_body<T3, K> *f3_p;
        typedef type_to_key_function_body<T4, K> *f4_p;
        typedef type_to_key_function_body<T5, K> *f5_p;
        typedef type_to_key_function_body<T6, K> *f6_p;
        typedef type_to_key_function_body<T7, K> *f7_p;
        typedef std::tuple< f0_p, f1_p, f2_p, f3_p, f4_p, f5_p, f6_p, f7_p > func_initializer_type;
    public:
#if __TBB_PREVIEW_MESSAGE_BASED_KEY_MATCHING
        unfolded_join_node(graph &g) : base_type(g,
                func_initializer_type(
                    new type_to_key_function_body_leaf<T0, K, key_from_message_body<K,T0> >(key_from_message_body<K,T0>()),
                    new type_to_key_function_body_leaf<T1, K, key_from_message_body<K,T1> >(key_from_message_body<K,T1>()),
                    new type_to_key_function_body_leaf<T2, K, key_from_message_body<K,T2> >(key_from_message_body<K,T2>()),
                    new type_to_key_function_body_leaf<T3, K, key_from_message_body<K,T3> >(key_from_message_body<K,T3>()),
                    new type_to_key_function_body_leaf<T4, K, key_from_message_body<K,T4> >(key_from_message_body<K,T4>()),
                    new type_to_key_function_body_leaf<T5, K, key_from_message_body<K,T5> >(key_from_message_body<K,T5>()),
                    new type_to_key_function_body_leaf<T6, K, key_from_message_body<K,T6> >(key_from_message_body<K,T6>()),
                    new type_to_key_function_body_leaf<T7, K, key_from_message_body<K,T7> >(key_from_message_body<K,T7>())
                    ) ) {
        }
#endif /* __TBB_PREVIEW_MESSAGE_BASED_KEY_MATCHING */
        template<typename Body0, typename Body1, typename Body2, typename Body3, typename Body4,
                 typename Body5, typename Body6, typename Body7>
        unfolded_join_node(graph &g, Body0 body0, Body1 body1, Body2 body2, Body3 body3, Body4 body4,
                Body5 body5, Body6 body6, Body7 body7) : base_type(g, func_initializer_type(
                    new type_to_key_function_body_leaf<T0, K, Body0>(body0),
                    new type_to_key_function_body_leaf<T1, K, Body1>(body1),
                    new type_to_key_function_body_leaf<T2, K, Body2>(body2),
                    new type_to_key_function_body_leaf<T3, K, Body3>(body3),
                    new type_to_key_function_body_leaf<T4, K, Body4>(body4),
                    new type_to_key_function_body_leaf<T5, K, Body5>(body5),
                    new type_to_key_function_body_leaf<T6, K, Body6>(body6),
                    new type_to_key_function_body_leaf<T7, K, Body7>(body7)
                    ) ) {
            static_assert(std::tuple_size<OutputTuple>::value == 8, "wrong number of body initializers");
        }
        unfolded_join_node(const unfolded_join_node &other) : base_type(other) {}
    };
#endif

#if __TBB_VARIADIC_MAX >= 9
    template<typename OutputTuple, typename K, typename KHash>
    class unfolded_join_node<9,key_matching_port,OutputTuple,key_matching<K,KHash> > : public
            join_base<9,key_matching_port,OutputTuple,key_matching<K,KHash> >::type {
        typedef typename std::tuple_element<0, OutputTuple>::type T0;
        typedef typename std::tuple_element<1, OutputTuple>::type T1;
        typedef typename std::tuple_element<2, OutputTuple>::type T2;
        typedef typename std::tuple_element<3, OutputTuple>::type T3;
        typedef typename std::tuple_element<4, OutputTuple>::type T4;
        typedef typename std::tuple_element<5, OutputTuple>::type T5;
        typedef typename std::tuple_element<6, OutputTuple>::type T6;
        typedef typename std::tuple_element<7, OutputTuple>::type T7;
        typedef typename std::tuple_element<8, OutputTuple>::type T8;
    public:
        typedef typename wrap_key_tuple_elements<9,key_matching_port,key_matching<K,KHash>,OutputTuple>::type input_ports_type;
        typedef OutputTuple output_type;
    private:
        typedef join_node_base<key_matching<K,KHash> , input_ports_type, output_type > base_type;
        typedef type_to_key_function_body<T0, K> *f0_p;
        typedef type_to_key_function_body<T1, K> *f1_p;
        typedef type_to_key_function_body<T2, K> *f2_p;
        typedef type_to_key_function_body<T3, K> *f3_p;
        typedef type_to_key_function_body<T4, K> *f4_p;
        typedef type_to_key_function_body<T5, K> *f5_p;
        typedef type_to_key_function_body<T6, K> *f6_p;
        typedef type_to_key_function_body<T7, K> *f7_p;
        typedef type_to_key_function_body<T8, K> *f8_p;
        typedef std::tuple< f0_p, f1_p, f2_p, f3_p, f4_p, f5_p, f6_p, f7_p, f8_p > func_initializer_type;
    public:
#if __TBB_PREVIEW_MESSAGE_BASED_KEY_MATCHING
        unfolded_join_node(graph &g) : base_type(g,
                func_initializer_type(
                    new type_to_key_function_body_leaf<T0, K, key_from_message_body<K,T0> >(key_from_message_body<K,T0>()),
                    new type_to_key_function_body_leaf<T1, K, key_from_message_body<K,T1> >(key_from_message_body<K,T1>()),
                    new type_to_key_function_body_leaf<T2, K, key_from_message_body<K,T2> >(key_from_message_body<K,T2>()),
                    new type_to_key_function_body_leaf<T3, K, key_from_message_body<K,T3> >(key_from_message_body<K,T3>()),
                    new type_to_key_function_body_leaf<T4, K, key_from_message_body<K,T4> >(key_from_message_body<K,T4>()),
                    new type_to_key_function_body_leaf<T5, K, key_from_message_body<K,T5> >(key_from_message_body<K,T5>()),
                    new type_to_key_function_body_leaf<T6, K, key_from_message_body<K,T6> >(key_from_message_body<K,T6>()),
                    new type_to_key_function_body_leaf<T7, K, key_from_message_body<K,T7> >(key_from_message_body<K,T7>()),
                    new type_to_key_function_body_leaf<T8, K, key_from_message_body<K,T8> >(key_from_message_body<K,T8>())
                    ) ) {
        }
#endif /* __TBB_PREVIEW_MESSAGE_BASED_KEY_MATCHING */
        template<typename Body0, typename Body1, typename Body2, typename Body3, typename Body4,
                 typename Body5, typename Body6, typename Body7, typename Body8>
        unfolded_join_node(graph &g, Body0 body0, Body1 body1, Body2 body2, Body3 body3, Body4 body4,
                Body5 body5, Body6 body6, Body7 body7, Body8 body8) : base_type(g, func_initializer_type(
                    new type_to_key_function_body_leaf<T0, K, Body0>(body0),
                    new type_to_key_function_body_leaf<T1, K, Body1>(body1),
                    new type_to_key_function_body_leaf<T2, K, Body2>(body2),
                    new type_to_key_function_body_leaf<T3, K, Body3>(body3),
                    new type_to_key_function_body_leaf<T4, K, Body4>(body4),
                    new type_to_key_function_body_leaf<T5, K, Body5>(body5),
                    new type_to_key_function_body_leaf<T6, K, Body6>(body6),
                    new type_to_key_function_body_leaf<T7, K, Body7>(body7),
                    new type_to_key_function_body_leaf<T8, K, Body8>(body8)
                    ) ) {
            static_assert(std::tuple_size<OutputTuple>::value == 9, "wrong number of body initializers");
        }
        unfolded_join_node(const unfolded_join_node &other) : base_type(other) {}
    };
#endif

#if __TBB_VARIADIC_MAX >= 10
    template<typename OutputTuple, typename K, typename KHash>
    class unfolded_join_node<10,key_matching_port,OutputTuple,key_matching<K,KHash> > : public
            join_base<10,key_matching_port,OutputTuple,key_matching<K,KHash> >::type {
        typedef typename std::tuple_element<0, OutputTuple>::type T0;
        typedef typename std::tuple_element<1, OutputTuple>::type T1;
        typedef typename std::tuple_element<2, OutputTuple>::type T2;
        typedef typename std::tuple_element<3, OutputTuple>::type T3;
        typedef typename std::tuple_element<4, OutputTuple>::type T4;
        typedef typename std::tuple_element<5, OutputTuple>::type T5;
        typedef typename std::tuple_element<6, OutputTuple>::type T6;
        typedef typename std::tuple_element<7, OutputTuple>::type T7;
        typedef typename std::tuple_element<8, OutputTuple>::type T8;
        typedef typename std::tuple_element<9, OutputTuple>::type T9;
    public:
        typedef typename wrap_key_tuple_elements<10,key_matching_port,key_matching<K,KHash>,OutputTuple>::type input_ports_type;
        typedef OutputTuple output_type;
    private:
        typedef join_node_base<key_matching<K,KHash> , input_ports_type, output_type > base_type;
        typedef type_to_key_function_body<T0, K> *f0_p;
        typedef type_to_key_function_body<T1, K> *f1_p;
        typedef type_to_key_function_body<T2, K> *f2_p;
        typedef type_to_key_function_body<T3, K> *f3_p;
        typedef type_to_key_function_body<T4, K> *f4_p;
        typedef type_to_key_function_body<T5, K> *f5_p;
        typedef type_to_key_function_body<T6, K> *f6_p;
        typedef type_to_key_function_body<T7, K> *f7_p;
        typedef type_to_key_function_body<T8, K> *f8_p;
        typedef type_to_key_function_body<T9, K> *f9_p;
        typedef std::tuple< f0_p, f1_p, f2_p, f3_p, f4_p, f5_p, f6_p, f7_p, f8_p, f9_p > func_initializer_type;
    public:
#if __TBB_PREVIEW_MESSAGE_BASED_KEY_MATCHING
        unfolded_join_node(graph &g) : base_type(g,
                func_initializer_type(
                    new type_to_key_function_body_leaf<T0, K, key_from_message_body<K,T0> >(key_from_message_body<K,T0>()),
                    new type_to_key_function_body_leaf<T1, K, key_from_message_body<K,T1> >(key_from_message_body<K,T1>()),
                    new type_to_key_function_body_leaf<T2, K, key_from_message_body<K,T2> >(key_from_message_body<K,T2>()),
                    new type_to_key_function_body_leaf<T3, K, key_from_message_body<K,T3> >(key_from_message_body<K,T3>()),
                    new type_to_key_function_body_leaf<T4, K, key_from_message_body<K,T4> >(key_from_message_body<K,T4>()),
                    new type_to_key_function_body_leaf<T5, K, key_from_message_body<K,T5> >(key_from_message_body<K,T5>()),
                    new type_to_key_function_body_leaf<T6, K, key_from_message_body<K,T6> >(key_from_message_body<K,T6>()),
                    new type_to_key_function_body_leaf<T7, K, key_from_message_body<K,T7> >(key_from_message_body<K,T7>()),
                    new type_to_key_function_body_leaf<T8, K, key_from_message_body<K,T8> >(key_from_message_body<K,T8>()),
                    new type_to_key_function_body_leaf<T9, K, key_from_message_body<K,T9> >(key_from_message_body<K,T9>())
                    ) ) {
        }
#endif /* __TBB_PREVIEW_MESSAGE_BASED_KEY_MATCHING */
        template<typename Body0, typename Body1, typename Body2, typename Body3, typename Body4,
            typename Body5, typename Body6, typename Body7, typename Body8, typename Body9>
        unfolded_join_node(graph &g, Body0 body0, Body1 body1, Body2 body2, Body3 body3, Body4 body4,
                Body5 body5, Body6 body6, Body7 body7, Body8 body8, Body9 body9) : base_type(g, func_initializer_type(
                    new type_to_key_function_body_leaf<T0, K, Body0>(body0),
                    new type_to_key_function_body_leaf<T1, K, Body1>(body1),
                    new type_to_key_function_body_leaf<T2, K, Body2>(body2),
                    new type_to_key_function_body_leaf<T3, K, Body3>(body3),
                    new type_to_key_function_body_leaf<T4, K, Body4>(body4),
                    new type_to_key_function_body_leaf<T5, K, Body5>(body5),
                    new type_to_key_function_body_leaf<T6, K, Body6>(body6),
                    new type_to_key_function_body_leaf<T7, K, Body7>(body7),
                    new type_to_key_function_body_leaf<T8, K, Body8>(body8),
                    new type_to_key_function_body_leaf<T9, K, Body9>(body9)
                    ) ) {
            static_assert(std::tuple_size<OutputTuple>::value == 10, "wrong number of body initializers");
        }
        unfolded_join_node(const unfolded_join_node &other) : base_type(other) {}
    };
#endif

    //! templated function to refer to input ports of the join node
    template<size_t N, typename JNT>
    typename std::tuple_element<N, typename JNT::input_ports_type>::type &input_port(JNT &jn) {
        return std::get<N>(jn.input_ports());
    }

#endif // __TBB__flow_graph_join_impl_H
