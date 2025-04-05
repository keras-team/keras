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

#ifndef __TBB_parallel_filters_H
#define __TBB_parallel_filters_H

#include "_config.h"
#include "_task.h"
#include "_pipeline_filters_deduction.h"
#include "../tbb_allocator.h"

#include <cstddef>
#include <cstdint>

namespace tbb {
namespace detail {

namespace d1 {
class base_filter;
}

namespace r1 {
void __TBB_EXPORTED_FUNC set_end_of_input(d1::base_filter&);
class pipeline;
class stage_task;
class input_buffer;
}

namespace d1 {
class filter_node;

//! A stage in a pipeline.
/** @ingroup algorithms */
class base_filter{
private:
    //! Value used to mark "not in pipeline"
    static base_filter* not_in_pipeline() { return reinterpret_cast<base_filter*>(std::intptr_t(-1)); }
public:
    //! The lowest bit 0 is for parallel vs serial
    static constexpr  unsigned int filter_is_serial = 0x1;

    //! 2nd bit distinguishes ordered vs unordered filters.
    static constexpr  unsigned int filter_is_out_of_order = 0x1<<1;

    //! 3rd bit marks input filters emitting small objects
    static constexpr  unsigned int filter_may_emit_null = 0x1<<2;

    base_filter(const base_filter&) = delete;
    base_filter& operator=(const base_filter&) = delete;

protected:
    explicit base_filter( unsigned int m ) :
        next_filter_in_pipeline(not_in_pipeline()),
        my_input_buffer(nullptr),
        my_filter_mode(m),
        my_pipeline(nullptr)
    {}

    // signal end-of-input for concrete_filters
    void set_end_of_input() {
        r1::set_end_of_input(*this);
    }

public:
    //! True if filter is serial.
    bool is_serial() const {
        return bool( my_filter_mode & filter_is_serial );
    }

    //! True if filter must receive stream in order.
    bool is_ordered() const {
        return (my_filter_mode & filter_is_serial) && !(my_filter_mode & filter_is_out_of_order);
    }

    //! true if an input filter can emit null
    bool object_may_be_null() {
        return ( my_filter_mode & filter_may_emit_null ) == filter_may_emit_null;
    }

    //! Operate on an item from the input stream, and return item for output stream.
    /** Returns nullptr if filter is a sink. */
    virtual void* operator()( void* item ) = 0;

    //! Destroy filter.
    virtual ~base_filter() {};

    //! Destroys item if pipeline was cancelled.
    /** Required to prevent memory leaks.
        Note it can be called concurrently even for serial filters.*/
    virtual void finalize( void* /*item*/ ) {}

private:
    //! Pointer to next filter in the pipeline.
    base_filter* next_filter_in_pipeline;

    //! Buffer for incoming tokens, or nullptr if not required.
    /** The buffer is required if the filter is serial. */
    r1::input_buffer* my_input_buffer;

    friend class r1::stage_task;
    friend class r1::pipeline;
    friend void r1::set_end_of_input(d1::base_filter&);

    //! Storage for filter mode and dynamically checked implementation version.
    const unsigned int my_filter_mode;

    //! Pointer to the pipeline.
    r1::pipeline* my_pipeline;
};

template<typename Body, typename InputType, typename OutputType >
class concrete_filter;

//! input_filter control to signal end-of-input for parallel_pipeline
class flow_control {
    bool is_pipeline_stopped = false;
    flow_control() = default;
    template<typename Body, typename InputType, typename OutputType > friend class concrete_filter;
    template<typename Output> friend class input_node;
public:
    void stop() { is_pipeline_stopped = true; }
};

// Emulate std::is_trivially_copyable (false positives not allowed, false negatives suboptimal but safe).
#if __TBB_CPP11_TYPE_PROPERTIES_PRESENT
template<typename T> using tbb_trivially_copyable = std::is_trivially_copyable<T>;
#else
template<typename T> struct tbb_trivially_copyable                      { enum { value = false }; };
template<typename T> struct tbb_trivially_copyable <         T*       > { enum { value = true  }; };
template<>           struct tbb_trivially_copyable <         bool     > { enum { value = true  }; };
template<>           struct tbb_trivially_copyable <         char     > { enum { value = true  }; };
template<>           struct tbb_trivially_copyable <  signed char     > { enum { value = true  }; };
template<>           struct tbb_trivially_copyable <unsigned char     > { enum { value = true  }; };
template<>           struct tbb_trivially_copyable <         short    > { enum { value = true  }; };
template<>           struct tbb_trivially_copyable <unsigned short    > { enum { value = true  }; };
template<>           struct tbb_trivially_copyable <         int      > { enum { value = true  }; };
template<>           struct tbb_trivially_copyable <unsigned int      > { enum { value = true  }; };
template<>           struct tbb_trivially_copyable <         long     > { enum { value = true  }; };
template<>           struct tbb_trivially_copyable <unsigned long     > { enum { value = true  }; };
template<>           struct tbb_trivially_copyable <         long long> { enum { value = true  }; };
template<>           struct tbb_trivially_copyable <unsigned long long> { enum { value = true  }; };
template<>           struct tbb_trivially_copyable <         float    > { enum { value = true  }; };
template<>           struct tbb_trivially_copyable <         double   > { enum { value = true  }; };
template<>           struct tbb_trivially_copyable <    long double   > { enum { value = true  }; };
#endif // __TBB_CPP11_TYPE_PROPERTIES_PRESENT

template<typename T>
struct use_allocator {
   static constexpr bool value = sizeof(T) > sizeof(void *) || !tbb_trivially_copyable<T>::value;
};

// A helper class to customize how a type is passed between filters.
// Usage: token_helper<T, use_allocator<T>::value>
template<typename T, bool Allocate> struct token_helper;

// using tbb_allocator
template<typename T>
struct token_helper<T, true> {
    using pointer = T*;
    using value_type = T;
    static pointer create_token(value_type && source) {
        return new (r1::allocate_memory(sizeof(T))) T(std::move(source));
    }
    static value_type & token(pointer & t) { return *t; }
    static void * cast_to_void_ptr(pointer ref) { return reinterpret_cast<void *>(ref); }
    static pointer cast_from_void_ptr(void * ref) { return reinterpret_cast<pointer>(ref); }
    static void destroy_token(pointer token) {
        token->~value_type();
        r1::deallocate_memory(token);
    }
};

// pointer specialization
template<typename T>
struct token_helper<T*, false> {
    using pointer = T*;
    using value_type = T*;
    static pointer create_token(const value_type & source) { return source; }
    static value_type & token(pointer & t) { return t; }
    static void * cast_to_void_ptr(pointer ref) { return reinterpret_cast<void *>(ref); }
    static pointer cast_from_void_ptr(void * ref) { return reinterpret_cast<pointer>(ref); }
    static void destroy_token( pointer /*token*/) {}
};

// converting type to and from void*, passing objects directly
template<typename T>
struct token_helper<T, false> {
    typedef union {
        T actual_value;
        void * void_overlay;
    } type_to_void_ptr_map;
    using pointer = T;  // not really a pointer in this case.
    using value_type = T;
    static pointer create_token(const value_type & source) { return source; }
    static value_type & token(pointer & t) { return t; }
    static void * cast_to_void_ptr(pointer ref) {
        type_to_void_ptr_map mymap;
        mymap.void_overlay = nullptr;
        mymap.actual_value = ref;
        return mymap.void_overlay;
    }
    static pointer cast_from_void_ptr(void * ref) {
        type_to_void_ptr_map mymap;
        mymap.void_overlay = ref;
        return mymap.actual_value;
    }
    static void destroy_token( pointer /*token*/) {}
};

// intermediate
template<typename InputType,  typename OutputType, typename Body>
class concrete_filter: public base_filter {
    const Body& my_body;
    using input_helper = token_helper<InputType, use_allocator<InputType >::value>;
    using input_pointer = typename input_helper::pointer;
    using output_helper = token_helper<OutputType, use_allocator<OutputType>::value>;
    using output_pointer = typename output_helper::pointer;

    void* operator()(void* input) override {
        input_pointer temp_input = input_helper::cast_from_void_ptr(input);
        output_pointer temp_output = output_helper::create_token(my_body(std::move(input_helper::token(temp_input))));
        input_helper::destroy_token(temp_input);
        return output_helper::cast_to_void_ptr(temp_output);
    }

    void finalize(void * input) override {
        input_pointer temp_input = input_helper::cast_from_void_ptr(input);
        input_helper::destroy_token(temp_input);
    }

public:
    concrete_filter(unsigned int m, const Body& body) : base_filter(m), my_body(body) {}
};

// input
template<typename OutputType, typename Body>
class concrete_filter<void, OutputType, Body>: public base_filter {
    const Body& my_body;
    using output_helper = token_helper<OutputType, use_allocator<OutputType>::value>;
    using output_pointer = typename output_helper::pointer;

    void* operator()(void*) override {
        flow_control control;
        output_pointer temp_output = output_helper::create_token(my_body(control));
        if(control.is_pipeline_stopped) {
            output_helper::destroy_token(temp_output);
            set_end_of_input();
            return nullptr;
        }
        return output_helper::cast_to_void_ptr(temp_output);
    }

public:
    concrete_filter(unsigned int m, const Body& body) :
        base_filter(m | filter_may_emit_null),
        my_body(body)
    {}
};

// output
template<typename InputType, typename Body>
class concrete_filter<InputType, void, Body>: public base_filter {
    const Body& my_body;
    using input_helper = token_helper<InputType, use_allocator<InputType >::value>;
    using input_pointer = typename input_helper::pointer;

    void* operator()(void* input) override {
        input_pointer temp_input = input_helper::cast_from_void_ptr(input);
        my_body(std::move(input_helper::token(temp_input)));
        input_helper::destroy_token(temp_input);
        return nullptr;
    }
    void finalize(void* input) override {
        input_pointer temp_input = input_helper::cast_from_void_ptr(input);
        input_helper::destroy_token(temp_input);
    }

public:
    concrete_filter(unsigned int m, const Body& body) : base_filter(m), my_body(body) {}
};

template<typename Body>
class concrete_filter<void, void, Body>: public base_filter {
    const Body& my_body;

    void* operator()(void*) override {
        flow_control control;
        my_body(control);
        void* output = control.is_pipeline_stopped ? nullptr : (void*)(std::intptr_t)-1;
        return output;
    }
public:
    concrete_filter(unsigned int m, const Body& body) : base_filter(m), my_body(body) {}
};

class filter_node_ptr {
    filter_node * my_node;

public:
    filter_node_ptr() : my_node(nullptr) {}
    filter_node_ptr(filter_node *);
    ~filter_node_ptr();
    filter_node_ptr(const filter_node_ptr &);
    filter_node_ptr(filter_node_ptr &&);
    void operator=(filter_node *);
    void operator=(const filter_node_ptr &);
    void operator=(filter_node_ptr &&);
    filter_node& operator*() const;
    operator bool() const;
};

//! Abstract base class that represents a node in a parse tree underlying a filter class.
/** These nodes are always heap-allocated and can be shared by filter objects. */
class filter_node {
    /** Count must be atomic because it is hidden state for user, but might be shared by threads. */
    std::atomic<std::intptr_t> ref_count;
public:
    filter_node_ptr left;
    filter_node_ptr right;
protected:
    filter_node() : ref_count(0), left(nullptr), right(nullptr) {
#ifdef __TBB_TEST_FILTER_NODE_COUNT
        ++(__TBB_TEST_FILTER_NODE_COUNT);
#endif
    }
public:
    filter_node(const filter_node_ptr& x, const filter_node_ptr& y) : filter_node(){
        left = x;
        right = y;
    }
    filter_node(const filter_node&) = delete;
    filter_node& operator=(const filter_node&) = delete;

    //! Add concrete_filter to pipeline
    virtual base_filter* create_filter() const {
        __TBB_ASSERT(false, "method of non-leaf was called");
        return nullptr;
    }

    //! Increment reference count
    void add_ref() { ref_count.fetch_add(1, std::memory_order_relaxed); }

    //! Decrement reference count and delete if it becomes zero.
    void remove_ref() {
        __TBB_ASSERT(ref_count>0,"ref_count underflow");
        if( ref_count.fetch_sub(1, std::memory_order_relaxed) == 1 ) {
            this->~filter_node();
            r1::deallocate_memory(this);
        }
    }

    virtual ~filter_node() {
#ifdef __TBB_TEST_FILTER_NODE_COUNT
        --(__TBB_TEST_FILTER_NODE_COUNT);
#endif
    }
};

inline filter_node_ptr::filter_node_ptr(filter_node * nd) : my_node(nd) {
    if (my_node) {
        my_node->add_ref();
    }
}

inline filter_node_ptr::~filter_node_ptr() {
    if (my_node) {
        my_node->remove_ref();
    }
}

inline filter_node_ptr::filter_node_ptr(const filter_node_ptr & rhs) : my_node(rhs.my_node) {
    if (my_node) {
        my_node->add_ref();
    }
}

inline filter_node_ptr::filter_node_ptr(filter_node_ptr && rhs) : my_node(rhs.my_node) {
    rhs.my_node = nullptr;
}

inline void filter_node_ptr::operator=(filter_node * rhs) {
    // Order of operations below carefully chosen so that reference counts remain correct
    // in unlikely event that remove_ref throws exception.
    filter_node* old = my_node;
    my_node = rhs;
    if (my_node) {
        my_node->add_ref();
    }
    if (old) {
        old->remove_ref();
    }
}

inline void filter_node_ptr::operator=(const filter_node_ptr & rhs) {
    *this = rhs.my_node;
}

inline void filter_node_ptr::operator=(filter_node_ptr && rhs) {
    filter_node* old = my_node;
    my_node = rhs.my_node;
    rhs.my_node = nullptr;
    if (old) {
        old->remove_ref();
    }
}

inline filter_node& filter_node_ptr::operator*() const{
    __TBB_ASSERT(my_node,"NULL node is used");
    return *my_node;
}

inline filter_node_ptr::operator bool() const {
    return my_node != nullptr;
}

//! Node in parse tree representing result of make_filter.
template<typename InputType, typename OutputType, typename Body>
class filter_node_leaf: public filter_node {
    const unsigned int my_mode;
    const Body my_body;
    base_filter* create_filter() const override {
        return new(r1::allocate_memory(sizeof(concrete_filter<InputType, OutputType, Body>))) concrete_filter<InputType, OutputType, Body>(my_mode,my_body);
    }
public:
    filter_node_leaf( unsigned int m, const Body& b ) : my_mode(m), my_body(b) {}
};


template <typename Body, typename Input = typename body_types<decltype(&Body::operator())>::input_type>
using filter_input = typename std::conditional<std::is_same<Input, flow_control>::value, void, Input>::type;

template <typename Body>
using filter_output = typename body_types<decltype(&Body::operator())>::output_type;

} // namespace d1
} // namespace detail
} // namespace tbb


#endif /* __TBB_parallel_filters_H */
