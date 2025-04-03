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

#ifndef __TBB_parallel_for_each_H
#define __TBB_parallel_for_each_H

#include "detail/_config.h"
#include "detail/_namespace_injection.h"
#include "detail/_exception.h"
#include "detail/_task.h"
#include "detail/_aligned_space.h"
#include "detail/_small_object_pool.h"

#include "parallel_for.h"
#include "task_group.h" // task_group_context

#include <iterator>
#include <type_traits>

namespace tbb {
namespace detail {
namespace d2 {
template<typename Body, typename Item> class feeder_impl;
} // namespace d2

namespace d1 {
//! Class the user supplied algorithm body uses to add new tasks
template<typename Item>
class feeder {
    feeder() {}
    feeder(const feeder&) = delete;
    void operator=( const feeder&) = delete;

    virtual ~feeder () {}
    virtual void internal_add_copy(const Item& item) = 0;
    virtual void internal_add_move(Item&& item) = 0;

    template<typename Body_, typename Item_> friend class detail::d2::feeder_impl;
public:
    //! Add a work item to a running parallel_for_each.
    void add(const Item& item) {internal_add_copy(item);}
    void add(Item&& item) {internal_add_move(std::move(item));}
};

} // namespace d1

namespace d2 {
using namespace tbb::detail::d1;
/** Selects one of the two possible forms of function call member operator.
    @ingroup algorithms **/
template<class Body>
struct parallel_for_each_operator_selector {
public:
    template<typename ItemArg, typename FeederArg>
    static auto call(const Body& body, ItemArg&& item, FeederArg*)
    -> decltype(body(std::forward<ItemArg>(item)), void()) {
        #if defined(_MSC_VER) && !defined(__INTEL_COMPILER)
        // Suppression of Microsoft non-standard extension warnings
        #pragma warning (push)
        #pragma warning (disable: 4239)
        #endif

        body(std::forward<ItemArg>(item));

        #if defined(_MSC_VER) && !defined(__INTEL_COMPILER)
        #pragma warning (push)
        #endif
    }

    template<typename ItemArg, typename FeederArg>
    static auto call(const Body& body, ItemArg&& item, FeederArg* feeder)
    -> decltype(body(std::forward<ItemArg>(item), *feeder), void()) {
        #if defined(_MSC_VER) && !defined(__INTEL_COMPILER)
        // Suppression of Microsoft non-standard extension warnings
        #pragma warning (push)
        #pragma warning (disable: 4239)
        #endif
        __TBB_ASSERT(feeder, "Feeder was not created but should be");

        body(std::forward<ItemArg>(item), *feeder);

        #if defined(_MSC_VER) && !defined(__INTEL_COMPILER)
        #pragma warning (push)
        #endif
    }
};

template<typename Body, typename Item>
struct feeder_item_task: public task {
    using feeder_type = feeder_impl<Body, Item>;

    template <typename ItemType>
    feeder_item_task(ItemType&& input_item, feeder_type& feeder, small_object_allocator& alloc) :
        item(std::forward<ItemType>(input_item)),
        my_feeder(feeder),
        my_allocator(alloc)
    {}

    void finalize(const execution_data& ed) {
        my_feeder.my_wait_context.release();
        my_allocator.delete_object(this, ed);
    }

    //! Hack for resolve ambiguity between calls to the body with and without moving the stored copy
    //! Executing body with moving the copy should have higher priority
    using first_priority = int;
    using second_priority = double;

    template <typename BodyType, typename ItemType, typename FeederType>
    static auto call(const BodyType& call_body, ItemType& call_item, FeederType& call_feeder, first_priority)
    -> decltype(parallel_for_each_operator_selector<Body>::call(call_body, std::move(call_item), &call_feeder), void())
    {
        parallel_for_each_operator_selector<Body>::call(call_body, std::move(call_item), &call_feeder);
    }

    template <typename BodyType, typename ItemType, typename FeederType>
    static void call(const BodyType& call_body, ItemType& call_item, FeederType& call_feeder, second_priority) {
        parallel_for_each_operator_selector<Body>::call(call_body, call_item, &call_feeder);
    }

    task* execute(execution_data& ed) override {
        call(my_feeder.my_body, item, my_feeder, first_priority{});
        finalize(ed);
        return nullptr;
    }

    task* cancel(execution_data& ed) override {
        finalize(ed);
        return nullptr;
    }

    Item item;
    feeder_type& my_feeder;
    small_object_allocator my_allocator;
}; // class feeder_item_task

/** Implements new task adding procedure.
    @ingroup algorithms **/
template<typename Body, typename Item>
class feeder_impl : public feeder<Item> {
    // Avoiding use of copy constructor in a virtual method if the type does not support it
    void internal_add_copy_impl(std::true_type, const Item& item) {
        using feeder_task = feeder_item_task<Body, Item>;
        small_object_allocator alloc;
        auto task = alloc.new_object<feeder_task>(item, *this, alloc);

        my_wait_context.reserve();
        spawn(*task, my_execution_context);
    }

    void internal_add_copy_impl(std::false_type, const Item&) {
        __TBB_ASSERT(false, "Overloading for r-value reference doesn't work or it's not movable and not copyable object");
    }

    void internal_add_copy(const Item& item) override {
        internal_add_copy_impl(typename std::is_copy_constructible<Item>::type(), item);
    }

    void internal_add_move(Item&& item) override {
        using feeder_task = feeder_item_task<Body, Item>;
        small_object_allocator alloc{};
        auto task = alloc.new_object<feeder_task>(std::move(item), *this, alloc);

        my_wait_context.reserve();
        spawn(*task, my_execution_context);
    }
public:
    feeder_impl(const Body& body, wait_context& w_context, task_group_context &context)
      : my_body(body),
        my_wait_context(w_context)
      , my_execution_context(context)
    {}

    const Body& my_body;
    wait_context& my_wait_context;
    task_group_context& my_execution_context;
}; // class feeder_impl

/** Execute computation under one element of the range
    @ingroup algorithms **/
template<typename Iterator, typename Body, typename Item>
struct for_each_iteration_task: public task {
    using feeder_type = feeder_impl<Body, Item>;

    for_each_iteration_task(Iterator input_item_ptr, const Body& body, feeder_impl<Body, Item>* feeder_ptr, wait_context& wait_context) :
        item_ptr(input_item_ptr), my_body(body), my_feeder_ptr(feeder_ptr), parent_wait_context(wait_context)
    {}

    void finalize() {
        parent_wait_context.release();
    }

    task* execute(execution_data&) override {
        parallel_for_each_operator_selector<Body>::call(my_body, *item_ptr, my_feeder_ptr);
        finalize();
        return nullptr;
    }

    task* cancel(execution_data&) override {
        finalize();
        return nullptr;
    }

    Iterator item_ptr;
    const Body& my_body;
    feeder_impl<Body, Item>* my_feeder_ptr;
    wait_context& parent_wait_context;
}; // class for_each_iteration_task

// Helper to get the type of the iterator to the internal sequence of copies
// If the element can be passed to the body as an rvalue - this iterator should be move_iterator
template <typename Body, typename Item, typename = void>
struct input_iteration_task_iterator_helper {
    // For input iterators we pass const lvalue reference to the body
    // It is prohibited to take non-constant lvalue references for input iterators
    using type = const Item*;
};

template <typename Body, typename Item>
struct input_iteration_task_iterator_helper<Body, Item,
    tbb::detail::void_t<decltype(parallel_for_each_operator_selector<Body>::call(std::declval<const Body&>(),
                                                                                 std::declval<Item&&>(),
                                                                                 std::declval<feeder_impl<Body, Item>*>()))>>
{
    using type = std::move_iterator<Item*>;
};

/** Split one block task to several(max_block_size) iteration tasks for input iterators
    @ingroup algorithms **/
template <typename Body, typename Item>
struct input_block_handling_task : public task {
    static constexpr size_t max_block_size = 4;

    using feeder_type = feeder_impl<Body, Item>;
    using iteration_task_iterator_type = typename input_iteration_task_iterator_helper<Body, Item>::type;
    using iteration_task = for_each_iteration_task<iteration_task_iterator_type, Body, Item>;

    input_block_handling_task(wait_context& root_wait_context, task_group_context& e_context,
                              const Body& body, feeder_impl<Body, Item>* feeder_ptr, small_object_allocator& alloc)
        :my_size(0), my_wait_context(0), my_root_wait_context(root_wait_context),
         my_execution_context(e_context), my_allocator(alloc)
    {
        auto item_it = block_iteration_space.begin();
        for (auto* it = task_pool.begin(); it != task_pool.end(); ++it) {
            new (it) iteration_task(iteration_task_iterator_type(item_it++), body, feeder_ptr, my_wait_context);
        }
    }

    void finalize(const execution_data& ed) {
        my_root_wait_context.release();
        my_allocator.delete_object(this, ed);
    }

    task* execute(execution_data& ed) override {
        __TBB_ASSERT( my_size > 0, "Negative size was passed to task");
        for (std::size_t counter = 1; counter < my_size; ++counter) {
            my_wait_context.reserve();
            spawn(*(task_pool.begin() + counter), my_execution_context);
        }
        my_wait_context.reserve();
        execute_and_wait(*task_pool.begin(), my_execution_context,
                         my_wait_context,    my_execution_context);

        // deallocate current task after children execution
        finalize(ed);
        return nullptr;
    }

    task* cancel(execution_data& ed) override {
        finalize(ed);
        return nullptr;
    }

    ~input_block_handling_task() {
        for(std::size_t counter = 0; counter < max_block_size; ++counter) {
            (task_pool.begin() + counter)->~iteration_task();
            (block_iteration_space.begin() + counter)->~Item();
        }
    }

    aligned_space<Item, max_block_size> block_iteration_space;
    aligned_space<iteration_task, max_block_size> task_pool;
    std::size_t my_size;
    wait_context my_wait_context;
    wait_context& my_root_wait_context;
    task_group_context& my_execution_context;
    small_object_allocator my_allocator;
}; // class input_block_handling_task

/** Split one block task to several(max_block_size) iteration tasks for forward iterators
    @ingroup algorithms **/
template <typename Iterator, typename Body, typename Item>
struct forward_block_handling_task : public task {
    static constexpr size_t max_block_size = 4;

    using iteration_task = for_each_iteration_task<Iterator, Body, Item>;

    forward_block_handling_task(Iterator first, std::size_t size,
                                wait_context& w_context, task_group_context& e_context,
                                const Body& body, feeder_impl<Body, Item>* feeder_ptr,
                                small_object_allocator& alloc)
        : my_size(size), my_wait_context(0), my_root_wait_context(w_context),
          my_execution_context(e_context), my_allocator(alloc)
    {
        auto* task_it = task_pool.begin();
        for (std::size_t i = 0; i < size; i++) {
            new (task_it++) iteration_task(first, body, feeder_ptr, my_wait_context);
            ++first;
        }
    }

    void finalize(const execution_data& ed) {
        my_root_wait_context.release();
        my_allocator.delete_object(this, ed);
    }

    task* execute(execution_data& ed) override {
        __TBB_ASSERT( my_size > 0, "Negative size was passed to task");
        for(std::size_t counter = 1; counter < my_size; ++counter) {
            my_wait_context.reserve();
            spawn(*(task_pool.begin() + counter), my_execution_context);
        }
        my_wait_context.reserve();
        execute_and_wait(*task_pool.begin(), my_execution_context,
                         my_wait_context,    my_execution_context);

        // deallocate current task after children execution
        finalize(ed);
        return nullptr;
    }

    task* cancel(execution_data& ed) override {
        finalize(ed);
        return nullptr;
    }

    ~forward_block_handling_task() {
        for(std::size_t counter = 0; counter < my_size; ++counter) {
            (task_pool.begin() + counter)->~iteration_task();
        }
    }

    aligned_space<iteration_task, max_block_size> task_pool;
    std::size_t my_size;
    wait_context my_wait_context;
    wait_context& my_root_wait_context;
    task_group_context& my_execution_context;
    small_object_allocator my_allocator;
}; // class forward_block_handling_task

/** Body for parallel_for algorithm.
  * Allows to redirect operations under random access iterators range to the parallel_for algorithm.
    @ingroup algorithms **/
template <typename Iterator, typename Body, typename Item>
class parallel_for_body_wrapper {
    Iterator my_first;
    const Body& my_body;
    feeder_impl<Body, Item>* my_feeder_ptr;
public:
    parallel_for_body_wrapper(Iterator first, const Body& body, feeder_impl<Body, Item>* feeder_ptr)
        : my_first(first), my_body(body), my_feeder_ptr(feeder_ptr) {}

    void operator()(tbb::blocked_range<std::size_t> range) const {
#if __INTEL_COMPILER
#pragma ivdep
#endif
        for (std::size_t count = range.begin(); count != range.end(); count++) {
            parallel_for_each_operator_selector<Body>::call(my_body, *(my_first + count),
                                                            my_feeder_ptr);
        }
    }
}; // class parallel_for_body_wrapper


/** Helper for getting iterators tag including inherited custom tags
    @ingroup algorithms */
template<typename It>
using tag = typename std::iterator_traits<It>::iterator_category;

template<typename It>
using iterator_tag_dispatch = typename
    std::conditional<
        std::is_base_of<std::random_access_iterator_tag, tag<It>>::value,
        std::random_access_iterator_tag,
        typename std::conditional<
            std::is_base_of<std::forward_iterator_tag, tag<It>>::value,
            std::forward_iterator_tag,
            std::input_iterator_tag
        >::type
    >::type;

template <typename Body, typename Iterator, typename Item>
using feeder_is_required = tbb::detail::void_t<decltype(std::declval<const Body>()(std::declval<typename std::iterator_traits<Iterator>::reference>(),
                                                                                   std::declval<feeder<Item>&>()))>;

// Creates feeder object only if the body can accept it
template <typename Iterator, typename Body, typename Item, typename = void>
struct feeder_holder {
    feeder_holder( wait_context&, task_group_context&, const Body& ) {}

    feeder_impl<Body, Item>* feeder_ptr() { return nullptr; }
}; // class feeder_holder

template <typename Iterator, typename Body, typename Item>
class feeder_holder<Iterator, Body, Item, feeder_is_required<Body, Iterator, Item>> {
public:
    feeder_holder( wait_context& w_context, task_group_context& context, const Body& body )
        : my_feeder(body, w_context, context) {}

    feeder_impl<Body, Item>* feeder_ptr() { return &my_feeder; }
private:
    feeder_impl<Body, Item> my_feeder;
}; // class feeder_holder

template <typename Iterator, typename Body, typename Item>
class for_each_root_task_base : public task {
public:
    for_each_root_task_base(Iterator first, Iterator last, const Body& body, wait_context& w_context, task_group_context& e_context)
        : my_first(first), my_last(last), my_wait_context(w_context), my_execution_context(e_context),
          my_body(body), my_feeder_holder(my_wait_context, my_execution_context, my_body)
    {
        my_wait_context.reserve();
    }
private:
    task* cancel(execution_data&) override {
        this->my_wait_context.release();
        return nullptr;
    }
protected:
    Iterator my_first;
    Iterator my_last;
    wait_context& my_wait_context;
    task_group_context& my_execution_context;
    const Body& my_body;
    feeder_holder<Iterator, Body, Item> my_feeder_holder;
}; // class for_each_root_task_base

/** parallel_for_each algorithm root task - most generic version
  * Splits input range to blocks
    @ingroup algorithms **/
template <typename Iterator, typename Body, typename Item, typename IteratorTag = iterator_tag_dispatch<Iterator>>
class for_each_root_task : public for_each_root_task_base<Iterator, Body, Item>
{
    using base_type = for_each_root_task_base<Iterator, Body, Item>;
public:
    using base_type::base_type;
private:
    task* execute(execution_data& ed) override {
        using block_handling_type = input_block_handling_task<Body, Item>;

        if (this->my_first == this->my_last) {
            this->my_wait_context.release();
            return nullptr;
        }

        this->my_wait_context.reserve();
        small_object_allocator alloc{};
        auto block_handling_task = alloc.new_object<block_handling_type>(ed, this->my_wait_context, this->my_execution_context,
                                                                         this->my_body, this->my_feeder_holder.feeder_ptr(),
                                                                         alloc);

        auto* block_iterator = block_handling_task->block_iteration_space.begin();
        for (; !(this->my_first == this->my_last) && block_handling_task->my_size < block_handling_type::max_block_size; ++this->my_first) {
            // Move semantics are automatically used when supported by the iterator
            new (block_iterator++) Item(*this->my_first);
            ++block_handling_task->my_size;
        }

        // Do not access this after spawn to avoid races
        spawn(*this, this->my_execution_context);
        return block_handling_task;
    }
}; // class for_each_root_task - most generic implementation

/** parallel_for_each algorithm root task - forward iterator based specialization
  * Splits input range to blocks
    @ingroup algorithms **/
template <typename Iterator, typename Body, typename Item>
class for_each_root_task<Iterator, Body, Item, std::forward_iterator_tag>
    : public for_each_root_task_base<Iterator, Body, Item>
{
    using base_type = for_each_root_task_base<Iterator, Body, Item>;
public:
    using base_type::base_type;
private:
    task* execute(execution_data& ed) override {
        using block_handling_type = forward_block_handling_task<Iterator, Body, Item>;
        if (this->my_first == this->my_last) {
            this->my_wait_context.release();
            return nullptr;
        }

        std::size_t block_size{0};
        Iterator first_block_element = this->my_first;
        for (; !(this->my_first == this->my_last) && block_size < block_handling_type::max_block_size; ++this->my_first) {
            ++block_size;
        }

        this->my_wait_context.reserve();
        small_object_allocator alloc{};
        auto block_handling_task = alloc.new_object<block_handling_type>(ed, first_block_element, block_size,
                                                                         this->my_wait_context, this->my_execution_context,
                                                                         this->my_body, this->my_feeder_holder.feeder_ptr(), alloc);

        // Do not access this after spawn to avoid races
        spawn(*this, this->my_execution_context);
        return block_handling_task;
    }
}; // class for_each_root_task - forward iterator based specialization

/** parallel_for_each algorithm root task - random access iterator based specialization
  * Splits input range to blocks
    @ingroup algorithms **/
template <typename Iterator, typename Body, typename Item>
class for_each_root_task<Iterator, Body, Item, std::random_access_iterator_tag>
    : public for_each_root_task_base<Iterator, Body, Item>
{
    using base_type = for_each_root_task_base<Iterator, Body, Item>;
public:
    using base_type::base_type;
private:
    task* execute(execution_data&) override {
        tbb::parallel_for(
            tbb::blocked_range<std::size_t>(0, std::distance(this->my_first, this->my_last)),
            parallel_for_body_wrapper<Iterator, Body, Item>(this->my_first, this->my_body, this->my_feeder_holder.feeder_ptr())
            , this->my_execution_context
        );

        this->my_wait_context.release();
        return nullptr;
    }
}; // class for_each_root_task - random access iterator based specialization

/** Helper for getting item type. If item type can be deduced from feeder - got it from feeder,
    if feeder is generic - got item type from range.
    @ingroup algorithms */
template<typename Body, typename Item, typename FeederArg>
auto feeder_argument_parser(void (Body::*)(Item, feeder<FeederArg>&) const) -> FeederArg;

template<typename Body, typename>
decltype(feeder_argument_parser<Body>(&Body::operator())) get_item_type_impl(int); // for (T, feeder<T>)
template<typename Body, typename Item> Item get_item_type_impl(...); // stub

template <typename Body, typename Item>
using get_item_type = decltype(get_item_type_impl<Body, Item>(0));

/** Implements parallel iteration over a range.
    @ingroup algorithms */
template<typename Iterator, typename Body>
void run_parallel_for_each( Iterator first, Iterator last, const Body& body, task_group_context& context)
{
    if (!(first == last)) {
        using ItemType = get_item_type<Body, typename std::iterator_traits<Iterator>::value_type>;
        wait_context w_context(0);

        for_each_root_task<Iterator, Body, ItemType> root_task(first, last, body, w_context, context);

        execute_and_wait(root_task, context, w_context, context);
    }
}

/** \page parallel_for_each_body_req Requirements on parallel_for_each body
    Class \c Body implementing the concept of parallel_for_each body must define:
    - \code
        B::operator()(
                cv_item_type item,
                feeder<item_type>& feeder
        ) const

        OR

        B::operator()( cv_item_type& item ) const
      \endcode                                               Process item.
                                                             May be invoked concurrently  for the same \c this but different \c item.

    - \code item_type( const item_type& ) \endcode
                                                             Copy a work item.
    - \code ~item_type() \endcode                            Destroy a work item
**/

/** \name parallel_for_each
    See also requirements on \ref parallel_for_each_body_req "parallel_for_each Body". **/
//@{
//! Parallel iteration over a range, with optional addition of more work.
/** @ingroup algorithms */
template<typename Iterator, typename Body>
void parallel_for_each(Iterator first, Iterator last, const Body& body) {
    task_group_context context(PARALLEL_FOR_EACH);
    run_parallel_for_each<Iterator, Body>(first, last, body, context);
}

template<typename Range, typename Body>
void parallel_for_each(Range& rng, const Body& body) {
    parallel_for_each(std::begin(rng), std::end(rng), body);
}

template<typename Range, typename Body>
void parallel_for_each(const Range& rng, const Body& body) {
    parallel_for_each(std::begin(rng), std::end(rng), body);
}

//! Parallel iteration over a range, with optional addition of more work and user-supplied context
/** @ingroup algorithms */
template<typename Iterator, typename Body>
void parallel_for_each(Iterator first, Iterator last, const Body& body, task_group_context& context) {
    run_parallel_for_each<Iterator, Body>(first, last, body, context);
}

template<typename Range, typename Body>
void parallel_for_each(Range& rng, const Body& body, task_group_context& context) {
    parallel_for_each(std::begin(rng), std::end(rng), body, context);
}

template<typename Range, typename Body>
void parallel_for_each(const Range& rng, const Body& body, task_group_context& context) {
    parallel_for_each(std::begin(rng), std::end(rng), body, context);
}

} // namespace d2
} // namespace detail
//! @endcond
//@}

inline namespace v1 {
using detail::d2::parallel_for_each;
using detail::d1::feeder;
} // namespace v1

} // namespace tbb

#endif /* __TBB_parallel_for_each_H */
