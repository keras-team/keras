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

#ifndef __TBB_concurrent_priority_queue_H
#define __TBB_concurrent_priority_queue_H

#include "detail/_namespace_injection.h"
#include "detail/_aggregator.h"
#include "detail/_template_helpers.h"
#include "detail/_allocator_traits.h"
#include "detail/_range_common.h"
#include "detail/_exception.h"
#include "detail/_utils.h"
#include "detail/_containers_helpers.h"
#include "cache_aligned_allocator.h"
#include <vector>
#include <iterator>
#include <functional>
#include <utility>
#include <initializer_list>
#include <type_traits>

namespace tbb {
namespace detail {
namespace d1 {

template <typename T, typename Compare = std::less<T>, typename Allocator = cache_aligned_allocator<T>>
class concurrent_priority_queue {
public:
    using value_type = T;
    using reference = T&;
    using const_reference = const T&;

    using size_type = std::size_t;
    using difference_type = std::ptrdiff_t;

    using allocator_type = Allocator;

    concurrent_priority_queue() : concurrent_priority_queue(allocator_type{}) {}

    explicit concurrent_priority_queue( const allocator_type& alloc )
        : mark(0), my_size(0), my_compare(), data(alloc)
    {
        my_aggregator.initialize_handler(functor{this});
    }

    explicit concurrent_priority_queue( const Compare& compare, const allocator_type& alloc = allocator_type() )
        : mark(0), my_size(0), my_compare(compare), data(alloc)
    {
        my_aggregator.initialize_handler(functor{this});
    }

    explicit concurrent_priority_queue( size_type init_capacity, const allocator_type& alloc = allocator_type() )
        : mark(0), my_size(0), my_compare(), data(alloc)
    {
        data.reserve(init_capacity);
        my_aggregator.initialize_handler(functor{this});
    }

    explicit concurrent_priority_queue( size_type init_capacity, const Compare& compare, const allocator_type& alloc = allocator_type() )
        : mark(0), my_size(0), my_compare(compare), data(alloc)
    {
        data.reserve(init_capacity);
        my_aggregator.initialize_handler(functor{this});
    }

    template <typename InputIterator>
    concurrent_priority_queue( InputIterator begin, InputIterator end, const Compare& compare, const allocator_type& alloc = allocator_type() )
        : mark(0), my_compare(compare), data(begin, end, alloc)
    {
        my_aggregator.initialize_handler(functor{this});
        heapify();
        my_size.store(data.size(), std::memory_order_relaxed);
    }

    template <typename InputIterator>
    concurrent_priority_queue( InputIterator begin, InputIterator end, const allocator_type& alloc = allocator_type() )
        : concurrent_priority_queue(begin, end, Compare(), alloc) {}

    concurrent_priority_queue( std::initializer_list<value_type> init, const Compare& compare, const allocator_type& alloc = allocator_type() )
        : concurrent_priority_queue(init.begin(), init.end(), compare, alloc) {}

    concurrent_priority_queue( std::initializer_list<value_type> init, const allocator_type& alloc = allocator_type() )
        : concurrent_priority_queue(init, Compare(), alloc) {}

    concurrent_priority_queue( const concurrent_priority_queue& other )
        : mark(other.mark), my_size(other.my_size.load(std::memory_order_relaxed)), my_compare(other.my_compare),
          data(other.data)
    {
        my_aggregator.initialize_handler(functor{this});
    }

    concurrent_priority_queue( const concurrent_priority_queue& other, const allocator_type& alloc )
        : mark(other.mark), my_size(other.my_size.load(std::memory_order_relaxed)), my_compare(other.my_compare),
          data(other.data, alloc)
    {
        my_aggregator.initialize_handler(functor{this});
    }

    concurrent_priority_queue( concurrent_priority_queue&& other )
        : mark(other.mark), my_size(other.my_size.load(std::memory_order_relaxed)), my_compare(other.my_compare),
          data(std::move(other.data))
    {
        my_aggregator.initialize_handler(functor{this});
    }

    concurrent_priority_queue( concurrent_priority_queue&& other, const allocator_type& alloc )
        : mark(other.mark), my_size(other.my_size.load(std::memory_order_relaxed)), my_compare(other.my_compare),
          data(std::move(other.data), alloc)
    {
        my_aggregator.initialize_handler(functor{this});
    }

    concurrent_priority_queue& operator=( const concurrent_priority_queue& other ) {
        if (this != &other) {
            data = other.data;
            mark = other.mark;
            my_size.store(other.my_size.load(std::memory_order_relaxed), std::memory_order_relaxed);
        }
        return *this;
    }

    concurrent_priority_queue& operator=( concurrent_priority_queue&& other ) {
        if (this != &other) {
            // TODO: check if exceptions from std::vector::operator=(vector&&) should be handled separately
            data = std::move(other.data);
            mark = other.mark;
            my_size.store(other.my_size.load(std::memory_order_relaxed), std::memory_order_relaxed);
        }
        return *this;
    }

    concurrent_priority_queue& operator=( std::initializer_list<value_type> init ) {
        assign(init.begin(), init.end());
        return *this;
    }

    template <typename InputIterator>
    void assign( InputIterator begin, InputIterator end ) {
        data.assign(begin, end);
        mark = 0;
        my_size.store(data.size(), std::memory_order_relaxed);
        heapify();
    }

    void assign( std::initializer_list<value_type> init ) {
        assign(init.begin(), init.end());
    }

    /* Returned value may not reflect results of pending operations.
       This operation reads shared data and will trigger a race condition. */
    __TBB_nodiscard bool empty() const { return size() == 0; }

    // Returns the current number of elements contained in the queue
    /* Returned value may not reflect results of pending operations.
       This operation reads shared data and will trigger a race condition. */
    size_type size() const { return my_size.load(std::memory_order_relaxed); }

    /* This operation can be safely used concurrently with other push, try_pop or emplace operations. */
    void push( const value_type& value ) {
        cpq_operation op_data(value, PUSH_OP);
        my_aggregator.execute(&op_data);
        if (op_data.status == FAILED)
            throw_exception(exception_id::bad_alloc);
    }

    /* This operation can be safely used concurrently with other push, try_pop or emplace operations. */
    void push( value_type&& value ) {
        cpq_operation op_data(value, PUSH_RVALUE_OP);
        my_aggregator.execute(&op_data);
        if (op_data.status == FAILED)
            throw_exception(exception_id::bad_alloc);
    }

    /* This operation can be safely used concurrently with other push, try_pop or emplace operations. */
    template <typename... Args>
    void emplace( Args&&... args ) {
        // TODO: support uses allocator construction in this place
        push(value_type(std::forward<Args>(args)...));
    }

    // Gets a reference to and removes highest priority element
    /* If a highest priority element was found, sets elem and returns true,
       otherwise returns false.
       This operation can be safely used concurrently with other push, try_pop or emplace operations. */
    bool try_pop( value_type& value ) {
        cpq_operation op_data(value, POP_OP);
        my_aggregator.execute(&op_data);
        return op_data.status == SUCCEEDED;
    }

    // This operation affects the whole container => it is not thread-safe
    void clear() {
        data.clear();
        mark = 0;
        my_size.store(0, std::memory_order_relaxed);
    }

    // This operation affects the whole container => it is not thread-safe
    void swap( concurrent_priority_queue& other ) {
        if (this != &other) {
            using std::swap;
            swap(data, other.data);
            swap(mark, other.mark);

            size_type sz = my_size.load(std::memory_order_relaxed);
            my_size.store(other.my_size.load(std::memory_order_relaxed), std::memory_order_relaxed);
            other.my_size.store(sz, std::memory_order_relaxed);
        }
    }

    allocator_type get_allocator() const { return data.get_allocator(); }
private:
    enum operation_type {INVALID_OP, PUSH_OP, POP_OP, PUSH_RVALUE_OP};
    enum operation_status {WAIT = 0, SUCCEEDED, FAILED};

    class cpq_operation : public aggregated_operation<cpq_operation> {
    public:
        operation_type type;
        union {
            value_type* elem;
            size_type sz;
        };
        cpq_operation( const value_type& value, operation_type t )
            : type(t), elem(const_cast<value_type*>(&value)) {}
    }; // class cpq_operation

    class functor {
        concurrent_priority_queue* my_cpq;
    public:
        functor() : my_cpq(nullptr) {}
        functor( concurrent_priority_queue* cpq ) : my_cpq(cpq) {}

        void operator()(cpq_operation* op_list) {
            __TBB_ASSERT(my_cpq != nullptr, "Invalid functor");
            my_cpq->handle_operations(op_list);
        }
    }; // class functor

    void handle_operations( cpq_operation* op_list ) {
        call_itt_notify(acquired, this);
        cpq_operation* tmp, *pop_list = nullptr;
        __TBB_ASSERT(mark == data.size(), NULL);

        // First pass processes all constant (amortized; reallocation may happen) time pushes and pops.
        while(op_list) {
            // ITT note: &(op_list->status) tag is used to cover accesses to op_list
            // node. This thread is going to handle the operation, and so will acquire it
            // and perform the associated operation w/o triggering a race condition; the
            // thread that created the operation is waiting on the status field, so when
            // this thread is done with the operation, it will perform a
            // store_with_release to give control back to the waiting thread in
            // aggregator::insert_operation.
            // TODO: enable
            call_itt_notify(acquired, &(op_list->status));
            __TBB_ASSERT(op_list->type != INVALID_OP, NULL);

            tmp = op_list;
            op_list = op_list->next.load(std::memory_order_relaxed);
            if (tmp->type == POP_OP) {
                if (mark < data.size() &&
                    my_compare(data[0], data.back()))
                {
                    // there are newly pushed elems and the last one is higher than top
                    *(tmp->elem) = std::move(data.back());
                    my_size.store(my_size.load(std::memory_order_relaxed) - 1, std::memory_order_relaxed);
                    tmp->status.store(uintptr_t(SUCCEEDED), std::memory_order_release);

                    data.pop_back();
                    __TBB_ASSERT(mark <= data.size(), NULL);
                } else { // no convenient item to pop; postpone
                    tmp->next.store(pop_list, std::memory_order_relaxed);
                    pop_list = tmp;
                }
            } else { // PUSH_OP or PUSH_RVALUE_OP
                __TBB_ASSERT(tmp->type == PUSH_OP || tmp->type == PUSH_RVALUE_OP, "Unknown operation");
#if TBB_USE_EXCEPTIONS
                try
#endif
                {
                    if (tmp->type == PUSH_OP) {
                        push_back_helper(*(tmp->elem));
                    } else {
                        data.push_back(std::move(*(tmp->elem)));
                    }
                    my_size.store(my_size.load(std::memory_order_relaxed) + 1, std::memory_order_relaxed);
                    tmp->status.store(uintptr_t(SUCCEEDED), std::memory_order_release);
                }
#if TBB_USE_EXCEPTIONS
                catch(...) {
                    tmp->status.store(uintptr_t(FAILED), std::memory_order_release);
                }
#endif
            }
        }

        // Second pass processes pop operations
        while(pop_list) {
            tmp = pop_list;
            pop_list = pop_list->next.load(std::memory_order_relaxed);
            __TBB_ASSERT(tmp->type == POP_OP, NULL);
            if (data.empty()) {
                tmp->status.store(uintptr_t(FAILED), std::memory_order_release);
            } else {
                __TBB_ASSERT(mark <= data.size(), NULL);
                if (mark < data.size() &&
                    my_compare(data[0], data.back()))
                {
                    // there are newly pushed elems and the last one is higher than top
                    *(tmp->elem) = std::move(data.back());
                    my_size.store(my_size.load(std::memory_order_relaxed) - 1, std::memory_order_relaxed);
                    tmp->status.store(uintptr_t(SUCCEEDED), std::memory_order_release);
                    data.pop_back();
                } else { // extract top and push last element down heap
                    *(tmp->elem) = std::move(data[0]);
                    my_size.store(my_size.load(std::memory_order_relaxed) - 1, std::memory_order_relaxed);
                    tmp->status.store(uintptr_t(SUCCEEDED), std::memory_order_release);
                    reheap();
                }
            }
        }

        // heapify any leftover pushed elements before doing the next
        // batch of operations
        if (mark < data.size()) heapify();
        __TBB_ASSERT(mark == data.size(), NULL);
        call_itt_notify(releasing, this);
    }

    // Merge unsorted elements into heap
    void heapify() {
        if (!mark && data.size() > 0) mark = 1;
        for (; mark < data.size(); ++mark) {
            // for each unheapified element under size
            size_type cur_pos = mark;
            value_type to_place = std::move(data[mark]);
            do { // push to_place up the heap
                size_type parent = (cur_pos - 1) >> 1;
                if (!my_compare(data[parent], to_place))
                    break;
                data[cur_pos] = std::move(data[parent]);
                cur_pos = parent;
            } while(cur_pos);
            data[cur_pos] = std::move(to_place);
        }
    }

    // Re-heapify after an extraction
    // Re-heapify by pushing last element down the heap from the root.
    void reheap() {
        size_type cur_pos = 0, child = 1;

        while(child < mark) {
            size_type target = child;
            if (child + 1 < mark && my_compare(data[child], data[child + 1]))
                ++target;
            // target now has the higher priority child
            if (my_compare(data[target], data.back()))
                break;
            data[cur_pos] = std::move(data[target]);
            cur_pos = target;
            child = (cur_pos << 1) + 1;
        }
        if (cur_pos != data.size() - 1)
            data[cur_pos] = std::move(data.back());
        data.pop_back();
        if (mark > data.size()) mark = data.size();
    }

    void push_back_helper( const T& value ) {
        push_back_helper_impl(value, std::is_copy_constructible<T>{});
    }

    void push_back_helper_impl( const T& value, /*is_copy_constructible = */std::true_type ) {
        data.push_back(value);
    }

    void push_back_helper_impl( const T&, /*is_copy_constructible = */std::false_type ) {
        __TBB_ASSERT(false, "error: calling tbb::concurrent_priority_queue.push(const value_type&) for move-only type");
    }

    using aggregator_type = aggregator<functor, cpq_operation>;

    aggregator_type my_aggregator;
    // Padding added to avoid false sharing
    char padding1[max_nfs_size - sizeof(aggregator_type)];
    // The point at which unsorted elements begin
    size_type mark;
    std::atomic<size_type> my_size;
    Compare my_compare;

    // Padding added to avoid false sharing
    char padding2[max_nfs_size - (2*sizeof(size_type)) - sizeof(Compare)];
    //! Storage for the heap of elements in queue, plus unheapified elements
    /** data has the following structure:

         binary unheapified
          heap   elements
        ____|_______|____
        |       |       |
        v       v       v
        [_|...|_|_|...|_| |...| ]
         0       ^       ^       ^
                 |       |       |__capacity
                 |       |__my_size
                 |__mark

        Thus, data stores the binary heap starting at position 0 through
        mark-1 (it may be empty).  Then there are 0 or more elements
        that have not yet been inserted into the heap, in positions
        mark through my_size-1. */

    using vector_type = std::vector<value_type, allocator_type>;
    vector_type data;

    friend bool operator==( const concurrent_priority_queue& lhs,
                            const concurrent_priority_queue& rhs )
    {
        return lhs.data == rhs.data;
    }

#if !__TBB_CPP20_COMPARISONS_PRESENT
    friend bool operator!=( const concurrent_priority_queue& lhs,
                            const concurrent_priority_queue& rhs )
    {
        return !(lhs == rhs);
    }
#endif
}; // class concurrent_priority_queue

#if __TBB_CPP17_DEDUCTION_GUIDES_PRESENT
template <typename It,
          typename Comp = std::less<iterator_value_t<It>>,
          typename Alloc = tbb::cache_aligned_allocator<iterator_value_t<It>>,
          typename = std::enable_if_t<is_input_iterator_v<It>>,
          typename = std::enable_if_t<is_allocator_v<Alloc>>,
          typename = std::enable_if_t<!is_allocator_v<Comp>>>
concurrent_priority_queue( It, It, Comp = Comp(), Alloc = Alloc() )
-> concurrent_priority_queue<iterator_value_t<It>, Comp, Alloc>;

template <typename It, typename Alloc,
          typename = std::enable_if_t<is_input_iterator_v<It>>,
          typename = std::enable_if_t<is_allocator_v<Alloc>>>
concurrent_priority_queue( It, It, Alloc )
-> concurrent_priority_queue<iterator_value_t<It>, std::less<iterator_value_t<It>>, Alloc>;

template <typename T,
          typename Comp = std::less<T>,
          typename Alloc = tbb::cache_aligned_allocator<T>,
          typename = std::enable_if_t<is_allocator_v<Alloc>>,
          typename = std::enable_if_t<!is_allocator_v<Comp>>>
concurrent_priority_queue( std::initializer_list<T>, Comp = Comp(), Alloc = Alloc() )
-> concurrent_priority_queue<T, Comp, Alloc>;

template <typename T, typename Alloc,
          typename = std::enable_if_t<is_allocator_v<Alloc>>>
concurrent_priority_queue( std::initializer_list<T>, Alloc )
-> concurrent_priority_queue<T, std::less<T>, Alloc>;

#endif // __TBB_CPP17_DEDUCTION_GUIDES_PRESENT

template <typename T, typename Compare, typename Allocator>
void swap( concurrent_priority_queue<T, Compare, Allocator>& lhs,
           concurrent_priority_queue<T, Compare, Allocator>& rhs )
{
    lhs.swap(rhs);
}

} // namespace d1
} // namespace detail
inline namespace v1 {
using detail::d1::concurrent_priority_queue;

} // inline namespace v1
} // namespace tbb

#endif // __TBB_concurrent_priority_queue_H
