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

#ifndef __TBB_detail__concurrent_queue_base_H
#define __TBB_detail__concurrent_queue_base_H

#include "_utils.h"
#include "_exception.h"
#include "_machine.h"
#include "_allocator_traits.h"

#include "../profiling.h"
#include "../spin_mutex.h"
#include "../cache_aligned_allocator.h"

#include <atomic>

namespace tbb {
namespace detail {
namespace d1 {

using ticket_type = std::size_t;

template <typename Page>
inline bool is_valid_page(const Page p) {
    return reinterpret_cast<std::uintptr_t>(p) > 1;
}

template <typename T, typename Allocator>
struct concurrent_queue_rep;

template <typename Container, typename T, typename Allocator>
class micro_queue_pop_finalizer;

#if _MSC_VER && !defined(__INTEL_COMPILER)
// unary minus operator applied to unsigned type, result still unsigned
#pragma warning( push )
#pragma warning( disable: 4146 )
#endif

// A queue using simple locking.
// For efficiency, this class has no constructor.
// The caller is expected to zero-initialize it.
template <typename T, typename Allocator>
class micro_queue {
private:
    using queue_rep_type = concurrent_queue_rep<T, Allocator>;
    using self_type = micro_queue<T, Allocator>;
public:
    using size_type = std::size_t;
    using value_type = T;
    using reference = value_type&;
    using const_reference = const value_type&;

    using allocator_type = Allocator;
    using allocator_traits_type = tbb::detail::allocator_traits<allocator_type>;

    static constexpr size_type item_size = sizeof(T);
    static constexpr size_type items_per_page = item_size <=   8 ? 32 :
                                                item_size <=  16 ? 16 :
                                                item_size <=  32 ?  8 :
                                                item_size <=  64 ?  4 :
                                                item_size <= 128 ?  2 : 1;

    struct padded_page {
        padded_page() {}
        ~padded_page() {}

        reference operator[] (std::size_t index) {
            __TBB_ASSERT(index < items_per_page, "Index out of range");
            return items[index];
        }

        const_reference operator[] (std::size_t index) const {
            __TBB_ASSERT(index < items_per_page, "Index out of range");
            return items[index];
        }

        padded_page* next{ nullptr };
        std::atomic<std::uintptr_t> mask{};

        union {
            value_type items[items_per_page];
        };
    }; // struct padded_page

    using page_allocator_type = typename allocator_traits_type::template rebind_alloc<padded_page>;
protected:
    using page_allocator_traits = tbb::detail::allocator_traits<page_allocator_type>;

public:
    using item_constructor_type = void (*)(value_type* location, const void* src);
    micro_queue() = default;
    micro_queue( const micro_queue& ) = delete;
    micro_queue& operator=( const micro_queue& ) = delete;

    size_type prepare_page( ticket_type k, queue_rep_type& base, page_allocator_type page_allocator,
                            padded_page*& p ) {
        __TBB_ASSERT(p == nullptr, "Invalid page argument for prepare_page");
        k &= -queue_rep_type::n_queue;
        size_type index = modulo_power_of_two(k / queue_rep_type::n_queue, items_per_page);
        if (!index) {
            try_call( [&] {
                p = page_allocator_traits::allocate(page_allocator, 1);
            }).on_exception( [&] {
                ++base.n_invalid_entries;
                invalidate_page( k );
            });
            page_allocator_traits::construct(page_allocator, p);
        }

        if (tail_counter.load(std::memory_order_relaxed) != k) spin_wait_until_my_turn(tail_counter, k, base);
        call_itt_notify(acquired, &tail_counter);

        if (p) {
            spin_mutex::scoped_lock lock( page_mutex );
            padded_page* q = tail_page.load(std::memory_order_relaxed);
            if (is_valid_page(q)) {
                q->next = p;
            } else {
                head_page.store(p, std::memory_order_relaxed);
            }
            tail_page.store(p, std::memory_order_relaxed);;
        } else {
            p = tail_page.load(std::memory_order_acquire); // TODO may be relaxed ?
        }
        return index;
    }

    template<typename... Args>
    void push( ticket_type k, queue_rep_type& base, Args&&... args )
    {
        padded_page* p = nullptr;
        page_allocator_type page_allocator(base.get_allocator());
        size_type index = prepare_page(k, base, page_allocator, p);
        __TBB_ASSERT(p != nullptr, "Page was not prepared");

        // try_call API is not convenient here due to broken
        // variadic capture on GCC 4.8.5
        auto value_guard = make_raii_guard([&] {
            ++base.n_invalid_entries;
            call_itt_notify(releasing, &tail_counter);
            tail_counter.fetch_add(queue_rep_type::n_queue);
        });

        page_allocator_traits::construct(page_allocator, &(*p)[index], std::forward<Args>(args)...);
        // If no exception was thrown, mark item as present.
        p->mask.store(p->mask.load(std::memory_order_relaxed) | uintptr_t(1) << index, std::memory_order_relaxed);
        call_itt_notify(releasing, &tail_counter);

        value_guard.dismiss();
        tail_counter.fetch_add(queue_rep_type::n_queue);
    }

    void abort_push( ticket_type k, queue_rep_type& base) {
        padded_page* p = nullptr;
        prepare_page(k, base, base.get_allocator(), p);
        ++base.n_invalid_entries;
        tail_counter.fetch_add(queue_rep_type::n_queue);
    }

    bool pop( void* dst, ticket_type k, queue_rep_type& base ) {
        k &= -queue_rep_type::n_queue;
        if (head_counter.load(std::memory_order_relaxed) != k) spin_wait_until_eq(head_counter, k);
        call_itt_notify(acquired, &head_counter);
        if (tail_counter.load(std::memory_order_relaxed) == k) spin_wait_while_eq(tail_counter, k);
        call_itt_notify(acquired, &tail_counter);
        padded_page *p = head_page.load(std::memory_order_acquire);
        __TBB_ASSERT( p, nullptr );
        size_type index = modulo_power_of_two( k/queue_rep_type::n_queue, items_per_page );
        bool success = false;
        {
            page_allocator_type page_allocator(base.get_allocator());
            micro_queue_pop_finalizer<self_type, value_type, page_allocator_type> finalizer(*this, page_allocator,
                k + queue_rep_type::n_queue, index == items_per_page - 1 ? p : nullptr );
            if (p->mask.load(std::memory_order_relaxed) & (std::uintptr_t(1) << index)) {
                success = true;
                assign_and_destroy_item( dst, *p, index );
            } else {
                --base.n_invalid_entries;
            }
        }
        return success;
    }

    micro_queue& assign( const micro_queue& src, queue_rep_type& base,
        item_constructor_type construct_item )
    {
        head_counter.store(src.head_counter.load(std::memory_order_relaxed), std::memory_order_relaxed);
        tail_counter.store(src.tail_counter.load(std::memory_order_relaxed), std::memory_order_relaxed);

        const padded_page* srcp = src.head_page.load(std::memory_order_relaxed);
        if( is_valid_page(srcp) ) {
            ticket_type g_index = head_counter.load(std::memory_order_relaxed);
            size_type n_items  = (tail_counter.load(std::memory_order_relaxed) - head_counter.load(std::memory_order_relaxed))
                / queue_rep_type::n_queue;
            size_type index = modulo_power_of_two(head_counter.load(std::memory_order_relaxed) / queue_rep_type::n_queue, items_per_page);
            size_type end_in_first_page = (index+n_items < items_per_page) ? (index + n_items) : items_per_page;

            try_call( [&] {
                head_page.store(make_copy(base, srcp, index, end_in_first_page, g_index, construct_item), std::memory_order_relaxed);
            }).on_exception( [&] {
                head_counter.store(0, std::memory_order_relaxed);
                tail_counter.store(0, std::memory_order_relaxed);
            });
            padded_page* cur_page = head_page.load(std::memory_order_relaxed);

            try_call( [&] {
                if (srcp != src.tail_page.load(std::memory_order_relaxed)) {
                    for (srcp = srcp->next; srcp != src.tail_page.load(std::memory_order_relaxed); srcp=srcp->next ) {
                        cur_page->next = make_copy( base, srcp, 0, items_per_page, g_index, construct_item );
                        cur_page = cur_page->next;
                    }

                    __TBB_ASSERT(srcp == src.tail_page.load(std::memory_order_relaxed), nullptr );
                    size_type last_index = modulo_power_of_two(tail_counter.load(std::memory_order_relaxed) / queue_rep_type::n_queue, items_per_page);
                    if( last_index==0 ) last_index = items_per_page;

                    cur_page->next = make_copy( base, srcp, 0, last_index, g_index, construct_item );
                    cur_page = cur_page->next;
                }
                tail_page.store(cur_page, std::memory_order_relaxed);
            }).on_exception( [&] {
                padded_page* invalid_page = reinterpret_cast<padded_page*>(std::uintptr_t(1));
                tail_page.store(invalid_page, std::memory_order_relaxed);
            });
        } else {
            head_page.store(nullptr, std::memory_order_relaxed);
            tail_page.store(nullptr, std::memory_order_relaxed);
        }
        return *this;
    }

    padded_page* make_copy( queue_rep_type& base, const padded_page* src_page, size_type begin_in_page,
        size_type end_in_page, ticket_type& g_index, item_constructor_type construct_item )
    {
        page_allocator_type page_allocator(base.get_allocator());
        padded_page* new_page = page_allocator_traits::allocate(page_allocator, 1);
        new_page->next = nullptr;
        new_page->mask.store(src_page->mask.load(std::memory_order_relaxed), std::memory_order_relaxed);
        for (; begin_in_page!=end_in_page; ++begin_in_page, ++g_index) {
            if (new_page->mask.load(std::memory_order_relaxed) & uintptr_t(1) << begin_in_page) {
                copy_item(*new_page, begin_in_page, *src_page, begin_in_page, construct_item);
            }
        }
        return new_page;
    }

    void invalidate_page( ticket_type k )  {
        // Append an invalid page at address 1 so that no more pushes are allowed.
        padded_page* invalid_page = reinterpret_cast<padded_page*>(std::uintptr_t(1));
        {
            spin_mutex::scoped_lock lock( page_mutex );
            tail_counter.store(k + queue_rep_type::n_queue + 1, std::memory_order_relaxed);
            padded_page* q = tail_page.load(std::memory_order_relaxed);
            if (is_valid_page(q)) {
                q->next = invalid_page;
            } else {
                head_page.store(invalid_page, std::memory_order_relaxed);
            }
            tail_page.store(invalid_page, std::memory_order_relaxed);
        }
    }

    padded_page* get_tail_page() {
        return tail_page.load(std::memory_order_relaxed);
    }

    padded_page* get_head_page() {
        return head_page.load(std::memory_order_relaxed);
    }

    void set_tail_page( padded_page* pg ) {
        tail_page.store(pg, std::memory_order_relaxed);
    }

    void clear(queue_rep_type& base) {
        padded_page* curr_page = head_page.load(std::memory_order_relaxed);
        std::size_t index = head_counter.load(std::memory_order_relaxed);
        page_allocator_type page_allocator(base.get_allocator());

        while (curr_page) {
            for (; index != items_per_page - 1; ++index) {
                curr_page->operator[](index).~value_type();
            }
                padded_page* next_page = curr_page->next;
                page_allocator_traits::destroy(page_allocator, curr_page);
                page_allocator_traits::deallocate(page_allocator, curr_page, 1);
                curr_page = next_page;
        }

        padded_page* invalid_page = reinterpret_cast<padded_page*>(std::uintptr_t(1));
        head_page.store(invalid_page, std::memory_order_relaxed);
        tail_page.store(invalid_page, std::memory_order_relaxed);
    }

private:
    // template <typename U, typename A>
    friend class micro_queue_pop_finalizer<self_type, value_type, page_allocator_type>;

    // Class used to ensure exception-safety of method "pop"
    class destroyer  {
        value_type& my_value;
    public:
        destroyer( reference value ) : my_value(value) {}
        destroyer( const destroyer& ) = delete;
        destroyer& operator=( const destroyer& ) = delete;
        ~destroyer() {my_value.~T();}
    }; // class destroyer

    void copy_item( padded_page& dst, size_type dindex, const padded_page& src, size_type sindex,
        item_constructor_type construct_item )
    {
        auto& src_item = src[sindex];
        construct_item( &dst[dindex], static_cast<const void*>(&src_item) );
    }

    void assign_and_destroy_item( void* dst, padded_page& src, size_type index ) {
        auto& from = src[index];
        destroyer d(from);
        *static_cast<T*>(dst) = std::move(from);
    }

    void spin_wait_until_my_turn( std::atomic<ticket_type>& counter, ticket_type k, queue_rep_type& rb ) const {
        for (atomic_backoff b(true);; b.pause()) {
            ticket_type c = counter;
            if (c == k) return;
            else if (c & 1) {
                ++rb.n_invalid_entries;
                throw_exception( exception_id::bad_last_alloc);
            }
        }
    }

    std::atomic<padded_page*> head_page{};
    std::atomic<ticket_type> head_counter{};

    std::atomic<padded_page*> tail_page{};
    std::atomic<ticket_type> tail_counter{};

    spin_mutex page_mutex{};
}; // class micro_queue

#if _MSC_VER && !defined(__INTEL_COMPILER)
#pragma warning( pop )
#endif // warning 4146 is back

template <typename Container, typename T, typename Allocator>
class micro_queue_pop_finalizer {
public:
    using padded_page = typename Container::padded_page;
    using allocator_type = Allocator;
    using allocator_traits_type = tbb::detail::allocator_traits<allocator_type>;

    micro_queue_pop_finalizer( Container& queue, Allocator& alloc, ticket_type k, padded_page* p ) :
        my_ticket_type(k), my_queue(queue), my_page(p), allocator(alloc)
    {}

    micro_queue_pop_finalizer( const micro_queue_pop_finalizer& ) = delete;
    micro_queue_pop_finalizer& operator=( const micro_queue_pop_finalizer& ) = delete;

    ~micro_queue_pop_finalizer() {
        padded_page* p = my_page;
        if( is_valid_page(p) ) {
            spin_mutex::scoped_lock lock( my_queue.page_mutex );
            padded_page* q = p->next;
            my_queue.head_page.store(q, std::memory_order_relaxed);
            if( !is_valid_page(q) ) {
                my_queue.tail_page.store(nullptr, std::memory_order_relaxed);
            }
        }
        my_queue.head_counter.store(my_ticket_type, std::memory_order_relaxed);
        if ( is_valid_page(p) ) {
            allocator_traits_type::destroy(allocator, static_cast<padded_page*>(p));
            allocator_traits_type::deallocate(allocator, static_cast<padded_page*>(p), 1);
        }
    }
private:
    ticket_type my_ticket_type;
    Container& my_queue;
    padded_page* my_page;
    Allocator& allocator;
}; // class micro_queue_pop_finalizer

#if _MSC_VER && !defined(__INTEL_COMPILER)
// structure was padded due to alignment specifier
#pragma warning( push )
#pragma warning( disable: 4324 )
#endif

template <typename T, typename Allocator>
struct concurrent_queue_rep {
    using self_type = concurrent_queue_rep<T, Allocator>;
    using size_type = std::size_t;
    using micro_queue_type = micro_queue<T, Allocator>;
    using allocator_type = Allocator;
    using allocator_traits_type = tbb::detail::allocator_traits<allocator_type>;
    using padded_page = typename micro_queue_type::padded_page;
    using page_allocator_type = typename micro_queue_type::page_allocator_type;
    using item_constructor_type = typename micro_queue_type::item_constructor_type;
private:
    using page_allocator_traits = tbb::detail::allocator_traits<page_allocator_type>;
    using queue_allocator_type = typename allocator_traits_type::template rebind_alloc<self_type>;

public:
    // must be power of 2
    static constexpr size_type n_queue = 8;
    // Approximately n_queue/golden ratio
    static constexpr size_type phi = 3;
    static constexpr size_type item_size = micro_queue_type::item_size;
    static constexpr size_type items_per_page = micro_queue_type::items_per_page;

    concurrent_queue_rep( queue_allocator_type& alloc ) : my_queue_allocator(alloc)
    {}

    concurrent_queue_rep( const concurrent_queue_rep& ) = delete;
    concurrent_queue_rep& operator=( const concurrent_queue_rep& ) = delete;

    void clear() {
        page_allocator_type page_allocator(my_queue_allocator);
        for (size_type i = 0; i < n_queue; ++i) {
            padded_page* tail_page = array[i].get_tail_page();
            if( is_valid_page(tail_page) ) {
                __TBB_ASSERT(array[i].get_head_page() == tail_page, "at most one page should remain" );
                page_allocator_traits::destroy(page_allocator, static_cast<padded_page*>(tail_page));
                page_allocator_traits::deallocate(page_allocator, static_cast<padded_page*>(tail_page), 1);
                array[i].set_tail_page(nullptr);
            } else {
                __TBB_ASSERT(!is_valid_page(array[i].get_head_page()), "head page pointer corrupt?");
            }
        }
    }

    void assign( const concurrent_queue_rep& src, item_constructor_type construct_item ) {
        head_counter.store(src.head_counter.load(std::memory_order_relaxed), std::memory_order_relaxed);
        tail_counter.store(src.tail_counter.load(std::memory_order_relaxed), std::memory_order_relaxed);
        n_invalid_entries.store(src.n_invalid_entries.load(std::memory_order_relaxed), std::memory_order_relaxed);

        // copy or move micro_queues
        size_type queue_idx = 0;
        try_call( [&] {
            for (; queue_idx < n_queue; ++queue_idx) {
                array[queue_idx].assign(src.array[queue_idx], *this, construct_item);
            }
        }).on_exception( [&] {
            for (size_type i = 0; i < queue_idx + 1; ++i) {
                array[i].clear(*this);
            }
            head_counter.store(0, std::memory_order_relaxed);
            tail_counter.store(0, std::memory_order_relaxed);
            n_invalid_entries.store(0, std::memory_order_relaxed);
        });

        __TBB_ASSERT(head_counter.load(std::memory_order_relaxed) == src.head_counter.load(std::memory_order_relaxed) &&
                     tail_counter.load(std::memory_order_relaxed) == src.tail_counter.load(std::memory_order_relaxed),
                     "the source concurrent queue should not be concurrently modified." );
    }

    bool empty() const {
        ticket_type tc = tail_counter.load(std::memory_order_acquire);
        ticket_type hc = head_counter.load(std::memory_order_relaxed);
        // if tc!=r.tail_counter, the queue was not empty at some point between the two reads.
        return tc == tail_counter.load(std::memory_order_relaxed) &&
               std::ptrdiff_t(tc - hc - n_invalid_entries.load(std::memory_order_relaxed)) <= 0;
    }

    std::ptrdiff_t size() const {
        __TBB_ASSERT(sizeof(std::ptrdiff_t) <= sizeof(size_type), NULL);
        std::ptrdiff_t hc = head_counter.load(std::memory_order_acquire);
        std::ptrdiff_t tc = tail_counter.load(std::memory_order_relaxed);
        std::ptrdiff_t nie = n_invalid_entries.load(std::memory_order_relaxed);

        return tc - hc - nie;
    }

    queue_allocator_type& get_allocator() {
        return my_queue_allocator;
    }

    friend class micro_queue<T, Allocator>;

    // Map ticket_type to an array index
    static size_type index( ticket_type k ) {
        return k * phi % n_queue;
    }

    micro_queue_type& choose( ticket_type k ) {
        // The formula here approximates LRU in a cache-oblivious way.
        return array[index(k)];
    }

    alignas(max_nfs_size) micro_queue_type array[n_queue];

    alignas(max_nfs_size) std::atomic<ticket_type> head_counter{};
    alignas(max_nfs_size) std::atomic<ticket_type> tail_counter{};
    alignas(max_nfs_size) std::atomic<size_type> n_invalid_entries{};
    queue_allocator_type& my_queue_allocator;
}; // class concurrent_queue_rep

#if _MSC_VER && !defined(__INTEL_COMPILER)
#pragma warning( pop )
#endif

template <typename Value, typename Allocator>
class concurrent_queue_iterator_base {
    using queue_rep_type = concurrent_queue_rep<Value, Allocator>;
    using padded_page = typename queue_rep_type::padded_page;
protected:
    concurrent_queue_iterator_base() = default;

    concurrent_queue_iterator_base( const concurrent_queue_iterator_base& other ) {
        assign(other);
    }

    concurrent_queue_iterator_base( queue_rep_type* queue_rep )
        : my_queue_rep(queue_rep),
          my_head_counter(my_queue_rep->head_counter.load(std::memory_order_relaxed))
    {
        for (std::size_t i = 0; i < queue_rep_type::n_queue; ++i) {
            my_array[i] = my_queue_rep->array[i].get_head_page();
        }

        if (!get_item(my_item, my_head_counter)) advance();
    }

    void assign( const concurrent_queue_iterator_base& other ) {
        my_item = other.my_item;
        my_queue_rep = other.my_queue_rep;

        if (my_queue_rep != nullptr) {
            my_head_counter = other.my_head_counter;

            for (std::size_t i = 0; i < queue_rep_type::n_queue; ++i) {
                my_array[i] = other.my_array[i];
            }
        }
    }

    void advance() {
        __TBB_ASSERT(my_item, "Attempt to increment iterator past end of the queue");
        std::size_t k = my_head_counter;
#if TBB_USE_ASSERT
        Value* tmp;
        get_item(tmp, k);
        __TBB_ASSERT(my_item == tmp, nullptr);
#endif
        std::size_t i = modulo_power_of_two(k / queue_rep_type::n_queue, my_queue_rep->items_per_page);
        if (i == my_queue_rep->items_per_page - 1) {
            padded_page*& root = my_array[queue_rep_type::index(k)];
            root = root->next;
        }
        // Advance k
        my_head_counter = ++k;
        if (!get_item(my_item, k)) advance();
    }

    concurrent_queue_iterator_base& operator=( const concurrent_queue_iterator_base& other ) {
        this->assign(other);
        return *this;
    }

    bool get_item( Value*& item, std::size_t k ) {
        if (k == my_queue_rep->tail_counter.load(std::memory_order_relaxed)) {
            item = nullptr;
            return true;
        } else {
            padded_page* p = my_array[queue_rep_type::index(k)];
            __TBB_ASSERT(p, nullptr);
            std::size_t i = modulo_power_of_two(k / queue_rep_type::n_queue, my_queue_rep->items_per_page);
            item = &(*p)[i];
            return (p->mask & uintptr_t(1) << i) != 0;
        }
    }

    Value* my_item{ nullptr };
    queue_rep_type* my_queue_rep{ nullptr };
    ticket_type my_head_counter{};
    padded_page* my_array[queue_rep_type::n_queue];
}; // class concurrent_queue_iterator_base

struct concurrent_queue_iterator_provider {
    template <typename Iterator, typename Container>
    static Iterator get( const Container& container ) {
        return Iterator(container);
    }
}; // struct concurrent_queue_iterator_provider

template <typename Container, typename Value, typename Allocator>
class concurrent_queue_iterator : public concurrent_queue_iterator_base<typename std::remove_cv<Value>::type, Allocator> {
    using base_type = concurrent_queue_iterator_base<typename std::remove_cv<Value>::type, Allocator>;
public:
    using value_type = Value;
    using pointer = value_type*;
    using reference = value_type&;
    using difference_type = std::ptrdiff_t;
    using iterator_category = std::forward_iterator_tag;

    concurrent_queue_iterator() = default;

    /** If Value==Container::value_type, then this routine is the copy constructor.
        If Value==const Container::value_type, then this routine is a conversion constructor. */
    concurrent_queue_iterator( const concurrent_queue_iterator<Container, typename Container::value_type, Allocator>& other )
        : base_type(other) {}

private:
    concurrent_queue_iterator( const Container& container )
        : base_type(container.my_queue_representation) {}
public:
    concurrent_queue_iterator& operator=( const concurrent_queue_iterator<Container, typename Container::value_type, Allocator>& other ) {
        this->assign(other);
        return *this;
    }

    reference operator*() const {
        return *static_cast<pointer>(this->my_item);
    }

    pointer operator->() const { return &operator*(); }

    concurrent_queue_iterator& operator++() {
        this->advance();
        return *this;
    }

    concurrent_queue_iterator operator++(int) {
        concurrent_queue_iterator tmp = *this;
        ++*this;
        return tmp;
    }

    friend bool operator==( const concurrent_queue_iterator& lhs, const concurrent_queue_iterator& rhs ) {
        return lhs.my_item == rhs.my_item;
    }

    friend bool operator!=( const concurrent_queue_iterator& lhs, const concurrent_queue_iterator& rhs ) {
        return lhs.my_item != rhs.my_item;
    }
private:
    friend struct concurrent_queue_iterator_provider;
}; // class concurrent_queue_iterator

} // namespace d1
} // namespace detail
} // tbb

#endif // __TBB_detail__concurrent_queue_base_H
