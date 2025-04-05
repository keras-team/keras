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

#ifndef __TBB__flow_graph_indexer_impl_H
#define __TBB__flow_graph_indexer_impl_H

#ifndef __TBB_flow_graph_H
#error Do not #include this internal file directly; use public TBB headers instead.
#endif

// included in namespace tbb::detail::d1

#include "_flow_graph_types_impl.h"

    // Output of the indexer_node is a tbb::flow::tagged_msg, and will be of
    // the form  tagged_msg<tag, result>
    // where the value of tag will indicate which result was put to the
    // successor.

    template<typename IndexerNodeBaseType, typename T, size_t K>
    graph_task* do_try_put(const T &v, void *p) {
        typename IndexerNodeBaseType::output_type o(K, v);
        return reinterpret_cast<IndexerNodeBaseType *>(p)->try_put_task(&o);
    }

    template<typename TupleTypes,int N>
    struct indexer_helper {
        template<typename IndexerNodeBaseType, typename PortTuple>
        static inline void set_indexer_node_pointer(PortTuple &my_input, IndexerNodeBaseType *p, graph& g) {
            typedef typename std::tuple_element<N-1, TupleTypes>::type T;
            graph_task* (*indexer_node_put_task)(const T&, void *) = do_try_put<IndexerNodeBaseType, T, N-1>;
            std::get<N-1>(my_input).set_up(p, indexer_node_put_task, g);
            indexer_helper<TupleTypes,N-1>::template set_indexer_node_pointer<IndexerNodeBaseType,PortTuple>(my_input, p, g);
        }
    };

    template<typename TupleTypes>
    struct indexer_helper<TupleTypes,1> {
        template<typename IndexerNodeBaseType, typename PortTuple>
        static inline void set_indexer_node_pointer(PortTuple &my_input, IndexerNodeBaseType *p, graph& g) {
            typedef typename std::tuple_element<0, TupleTypes>::type T;
            graph_task* (*indexer_node_put_task)(const T&, void *) = do_try_put<IndexerNodeBaseType, T, 0>;
            std::get<0>(my_input).set_up(p, indexer_node_put_task, g);
        }
    };

    template<typename T>
    class indexer_input_port : public receiver<T> {
    private:
        void* my_indexer_ptr;
        typedef graph_task* (* forward_function_ptr)(T const &, void* );
        forward_function_ptr my_try_put_task;
        graph* my_graph;
    public:
        void set_up(void* p, forward_function_ptr f, graph& g) {
            my_indexer_ptr = p;
            my_try_put_task = f;
            my_graph = &g;
        }

    protected:
        template< typename R, typename B > friend class run_and_put_task;
        template<typename X, typename Y> friend class broadcast_cache;
        template<typename X, typename Y> friend class round_robin_cache;
        graph_task* try_put_task(const T &v) override {
            return my_try_put_task(v, my_indexer_ptr);
        }

        graph& graph_reference() const override {
            return *my_graph;
        }
    };

    template<typename InputTuple, typename OutputType, typename StructTypes>
    class indexer_node_FE {
    public:
        static const int N = std::tuple_size<InputTuple>::value;
        typedef OutputType output_type;
        typedef InputTuple input_type;

        // Some versions of Intel(R) C++ Compiler fail to generate an implicit constructor for the class which has std::tuple as a member.
        indexer_node_FE() : my_inputs() {}

        input_type &input_ports() { return my_inputs; }
    protected:
        input_type my_inputs;
    };

    //! indexer_node_base
    template<typename InputTuple, typename OutputType, typename StructTypes>
    class indexer_node_base : public graph_node, public indexer_node_FE<InputTuple, OutputType,StructTypes>,
                           public sender<OutputType> {
    protected:
       using graph_node::my_graph;
    public:
        static const size_t N = std::tuple_size<InputTuple>::value;
        typedef OutputType output_type;
        typedef StructTypes tuple_types;
        typedef typename sender<output_type>::successor_type successor_type;
        typedef indexer_node_FE<InputTuple, output_type,StructTypes> input_ports_type;

    private:
        // ----------- Aggregator ------------
        enum op_type { reg_succ, rem_succ, try__put_task
        };
        typedef indexer_node_base<InputTuple,output_type,StructTypes> class_type;

        class indexer_node_base_operation : public aggregated_operation<indexer_node_base_operation> {
        public:
            char type;
            union {
                output_type const *my_arg;
                successor_type *my_succ;
                graph_task* bypass_t;
            };
            indexer_node_base_operation(const output_type* e, op_type t) :
                type(char(t)), my_arg(e) {}
            indexer_node_base_operation(const successor_type &s, op_type t) : type(char(t)),
                my_succ(const_cast<successor_type *>(&s)) {}
        };

        typedef aggregating_functor<class_type, indexer_node_base_operation> handler_type;
        friend class aggregating_functor<class_type, indexer_node_base_operation>;
        aggregator<handler_type, indexer_node_base_operation> my_aggregator;

        void handle_operations(indexer_node_base_operation* op_list) {
            indexer_node_base_operation *current;
            while(op_list) {
                current = op_list;
                op_list = op_list->next;
                switch(current->type) {

                case reg_succ:
                    my_successors.register_successor(*(current->my_succ));
                    current->status.store( SUCCEEDED, std::memory_order_release);
                    break;

                case rem_succ:
                    my_successors.remove_successor(*(current->my_succ));
                    current->status.store( SUCCEEDED, std::memory_order_release);
                    break;
                case try__put_task: {
                        current->bypass_t = my_successors.try_put_task(*(current->my_arg));
                        current->status.store( SUCCEEDED, std::memory_order_release);  // return of try_put_task actual return value
                    }
                    break;
                }
            }
        }
        // ---------- end aggregator -----------
    public:
        indexer_node_base(graph& g) : graph_node(g), input_ports_type(), my_successors(this) {
            indexer_helper<StructTypes,N>::set_indexer_node_pointer(this->my_inputs, this, g);
            my_aggregator.initialize_handler(handler_type(this));
        }

        indexer_node_base(const indexer_node_base& other)
            : graph_node(other.my_graph), input_ports_type(), sender<output_type>(), my_successors(this)
        {
            indexer_helper<StructTypes,N>::set_indexer_node_pointer(this->my_inputs, this, other.my_graph);
            my_aggregator.initialize_handler(handler_type(this));
        }

        bool register_successor(successor_type &r) override {
            indexer_node_base_operation op_data(r, reg_succ);
            my_aggregator.execute(&op_data);
            return op_data.status == SUCCEEDED;
        }

        bool remove_successor( successor_type &r) override {
            indexer_node_base_operation op_data(r, rem_succ);
            my_aggregator.execute(&op_data);
            return op_data.status == SUCCEEDED;
        }

        graph_task* try_put_task(output_type const *v) { // not a virtual method in this class
            indexer_node_base_operation op_data(v, try__put_task);
            my_aggregator.execute(&op_data);
            return op_data.bypass_t;
        }

    protected:
        void reset_node(reset_flags f) override {
            if(f & rf_clear_edges) {
                my_successors.clear();
            }
        }

    private:
        broadcast_cache<output_type, null_rw_mutex> my_successors;
    };  //indexer_node_base


    template<int N, typename InputTuple> struct input_types;

    template<typename InputTuple>
    struct input_types<1, InputTuple> {
        typedef typename std::tuple_element<0, InputTuple>::type first_type;
        typedef tagged_msg<size_t, first_type > type;
    };

    template<typename InputTuple>
    struct input_types<2, InputTuple> {
        typedef typename std::tuple_element<0, InputTuple>::type first_type;
        typedef typename std::tuple_element<1, InputTuple>::type second_type;
        typedef tagged_msg<size_t, first_type, second_type> type;
    };

    template<typename InputTuple>
    struct input_types<3, InputTuple> {
        typedef typename std::tuple_element<0, InputTuple>::type first_type;
        typedef typename std::tuple_element<1, InputTuple>::type second_type;
        typedef typename std::tuple_element<2, InputTuple>::type third_type;
        typedef tagged_msg<size_t, first_type, second_type, third_type> type;
    };

    template<typename InputTuple>
    struct input_types<4, InputTuple> {
        typedef typename std::tuple_element<0, InputTuple>::type first_type;
        typedef typename std::tuple_element<1, InputTuple>::type second_type;
        typedef typename std::tuple_element<2, InputTuple>::type third_type;
        typedef typename std::tuple_element<3, InputTuple>::type fourth_type;
        typedef tagged_msg<size_t, first_type, second_type, third_type,
                                                      fourth_type> type;
    };

    template<typename InputTuple>
    struct input_types<5, InputTuple> {
        typedef typename std::tuple_element<0, InputTuple>::type first_type;
        typedef typename std::tuple_element<1, InputTuple>::type second_type;
        typedef typename std::tuple_element<2, InputTuple>::type third_type;
        typedef typename std::tuple_element<3, InputTuple>::type fourth_type;
        typedef typename std::tuple_element<4, InputTuple>::type fifth_type;
        typedef tagged_msg<size_t, first_type, second_type, third_type,
                                                      fourth_type, fifth_type> type;
    };

    template<typename InputTuple>
    struct input_types<6, InputTuple> {
        typedef typename std::tuple_element<0, InputTuple>::type first_type;
        typedef typename std::tuple_element<1, InputTuple>::type second_type;
        typedef typename std::tuple_element<2, InputTuple>::type third_type;
        typedef typename std::tuple_element<3, InputTuple>::type fourth_type;
        typedef typename std::tuple_element<4, InputTuple>::type fifth_type;
        typedef typename std::tuple_element<5, InputTuple>::type sixth_type;
        typedef tagged_msg<size_t, first_type, second_type, third_type,
                                                      fourth_type, fifth_type, sixth_type> type;
    };

    template<typename InputTuple>
    struct input_types<7, InputTuple> {
        typedef typename std::tuple_element<0, InputTuple>::type first_type;
        typedef typename std::tuple_element<1, InputTuple>::type second_type;
        typedef typename std::tuple_element<2, InputTuple>::type third_type;
        typedef typename std::tuple_element<3, InputTuple>::type fourth_type;
        typedef typename std::tuple_element<4, InputTuple>::type fifth_type;
        typedef typename std::tuple_element<5, InputTuple>::type sixth_type;
        typedef typename std::tuple_element<6, InputTuple>::type seventh_type;
        typedef tagged_msg<size_t, first_type, second_type, third_type,
                                                      fourth_type, fifth_type, sixth_type,
                                                      seventh_type> type;
    };


    template<typename InputTuple>
    struct input_types<8, InputTuple> {
        typedef typename std::tuple_element<0, InputTuple>::type first_type;
        typedef typename std::tuple_element<1, InputTuple>::type second_type;
        typedef typename std::tuple_element<2, InputTuple>::type third_type;
        typedef typename std::tuple_element<3, InputTuple>::type fourth_type;
        typedef typename std::tuple_element<4, InputTuple>::type fifth_type;
        typedef typename std::tuple_element<5, InputTuple>::type sixth_type;
        typedef typename std::tuple_element<6, InputTuple>::type seventh_type;
        typedef typename std::tuple_element<7, InputTuple>::type eighth_type;
        typedef tagged_msg<size_t, first_type, second_type, third_type,
                                                      fourth_type, fifth_type, sixth_type,
                                                      seventh_type, eighth_type> type;
    };


    template<typename InputTuple>
    struct input_types<9, InputTuple> {
        typedef typename std::tuple_element<0, InputTuple>::type first_type;
        typedef typename std::tuple_element<1, InputTuple>::type second_type;
        typedef typename std::tuple_element<2, InputTuple>::type third_type;
        typedef typename std::tuple_element<3, InputTuple>::type fourth_type;
        typedef typename std::tuple_element<4, InputTuple>::type fifth_type;
        typedef typename std::tuple_element<5, InputTuple>::type sixth_type;
        typedef typename std::tuple_element<6, InputTuple>::type seventh_type;
        typedef typename std::tuple_element<7, InputTuple>::type eighth_type;
        typedef typename std::tuple_element<8, InputTuple>::type nineth_type;
        typedef tagged_msg<size_t, first_type, second_type, third_type,
                                                      fourth_type, fifth_type, sixth_type,
                                                      seventh_type, eighth_type, nineth_type> type;
    };

    template<typename InputTuple>
    struct input_types<10, InputTuple> {
        typedef typename std::tuple_element<0, InputTuple>::type first_type;
        typedef typename std::tuple_element<1, InputTuple>::type second_type;
        typedef typename std::tuple_element<2, InputTuple>::type third_type;
        typedef typename std::tuple_element<3, InputTuple>::type fourth_type;
        typedef typename std::tuple_element<4, InputTuple>::type fifth_type;
        typedef typename std::tuple_element<5, InputTuple>::type sixth_type;
        typedef typename std::tuple_element<6, InputTuple>::type seventh_type;
        typedef typename std::tuple_element<7, InputTuple>::type eighth_type;
        typedef typename std::tuple_element<8, InputTuple>::type nineth_type;
        typedef typename std::tuple_element<9, InputTuple>::type tenth_type;
        typedef tagged_msg<size_t, first_type, second_type, third_type,
                                                      fourth_type, fifth_type, sixth_type,
                                                      seventh_type, eighth_type, nineth_type,
                                                      tenth_type> type;
    };

    // type generators
    template<typename OutputTuple>
    struct indexer_types : public input_types<std::tuple_size<OutputTuple>::value, OutputTuple> {
        static const int N = std::tuple_size<OutputTuple>::value;
        typedef typename input_types<N, OutputTuple>::type output_type;
        typedef typename wrap_tuple_elements<N,indexer_input_port,OutputTuple>::type input_ports_type;
        typedef indexer_node_FE<input_ports_type,output_type,OutputTuple> indexer_FE_type;
        typedef indexer_node_base<input_ports_type, output_type, OutputTuple> indexer_base_type;
    };

    template<class OutputTuple>
    class unfolded_indexer_node : public indexer_types<OutputTuple>::indexer_base_type {
    public:
        typedef typename indexer_types<OutputTuple>::input_ports_type input_ports_type;
        typedef OutputTuple tuple_types;
        typedef typename indexer_types<OutputTuple>::output_type output_type;
    private:
        typedef typename indexer_types<OutputTuple>::indexer_base_type base_type;
    public:
        unfolded_indexer_node(graph& g) : base_type(g) {}
        unfolded_indexer_node(const unfolded_indexer_node &other) : base_type(other) {}
    };

#endif  /* __TBB__flow_graph_indexer_impl_H */
