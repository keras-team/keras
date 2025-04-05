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

#ifndef __TBB_concurrent_lru_cache_H
#define __TBB_concurrent_lru_cache_H

#if ! TBB_PREVIEW_CONCURRENT_LRU_CACHE
    #error Set TBB_PREVIEW_CONCURRENT_LRU_CACHE to include concurrent_lru_cache.h
#endif

#include "detail/_assert.h"
#include "detail/_aggregator.h"

#include <map>       // for std::map
#include <list>      // for std::list
#include <utility>   // for std::make_pair
#include <algorithm> // for std::find
#include <atomic>    // for std::atomic<bool>

namespace tbb {

namespace detail {
namespace d1 {

//-----------------------------------------------------------------------------
// Concurrent LRU cache
//-----------------------------------------------------------------------------

template<typename KeyT, typename ValT, typename KeyToValFunctorT = ValT (*) (KeyT)>
class concurrent_lru_cache : no_assign {
// incapsulated helper classes
private:
    struct handle_object;
    struct storage_map_value_type;

    struct aggregator_operation;
    struct retrieve_aggregator_operation;
    struct signal_end_of_usage_aggregator_operation;

// typedefs
public:
    using key_type = KeyT;
    using value_type = ValT;
    using pointer = ValT*;
    using reference = ValT&;
    using const_pointer = const ValT*;
    using const_reference = const ValT&;

    using value_function_type = KeyToValFunctorT;
    using handle = handle_object;
private:
    using lru_cache_type = concurrent_lru_cache<KeyT, ValT, KeyToValFunctorT>;

    using storage_map_type = std::map<key_type, storage_map_value_type>;
    using storage_map_iterator_type = typename storage_map_type::iterator;
    using storage_map_pointer_type = typename storage_map_type::pointer;
    using storage_map_reference_type = typename storage_map_type::reference;

    using history_list_type = std::list<storage_map_iterator_type>;
    using history_list_iterator_type = typename history_list_type::iterator;

    using aggregator_operation_type = aggregator_operation;
    using aggregator_function_type = aggregating_functor<lru_cache_type, aggregator_operation_type>;
    using aggregator_type = aggregator<aggregator_function_type, aggregator_operation_type>;

    friend class aggregating_functor<lru_cache_type,aggregator_operation_type>;

// fields
private:
    value_function_type my_value_function;
    aggregator_type my_aggregator;

    storage_map_type my_storage_map;            // storage map for used objects
    history_list_type my_history_list;          // history list for unused objects
    const std::size_t my_history_list_capacity; // history list's allowed capacity

// interface
public:

    concurrent_lru_cache(value_function_type value_function, std::size_t cache_capacity)
        : my_value_function(value_function), my_history_list_capacity(cache_capacity) {
        my_aggregator.initialize_handler(aggregator_function_type(this));
    }

    handle operator[](key_type key) {
        retrieve_aggregator_operation op(key);
        my_aggregator.execute(&op);

        if (op.is_new_value_needed()) {
            op.result().second.my_value = my_value_function(key);
            op.result().second.my_is_ready.store(true, std::memory_order_release);
        } else {
            spin_wait_while_eq(op.result().second.my_is_ready, false);
        }

        return handle(*this, op.result());
    }

private:

    void handle_operations(aggregator_operation* op_list) {
        while (op_list) {
            op_list->cast_and_handle(*this);
            aggregator_operation* prev_op = op_list;
            op_list = op_list->next;

            (prev_op->status).store(1, std::memory_order_release);
        }
    }

    void signal_end_of_usage(storage_map_reference_type map_record_ref) {
        signal_end_of_usage_aggregator_operation op(map_record_ref);
        my_aggregator.execute(&op);
    }

    void signal_end_of_usage_serial(storage_map_reference_type map_record_ref) {
        storage_map_iterator_type map_it = my_storage_map.find(map_record_ref.first);

        __TBB_ASSERT(map_it != my_storage_map.end(),
            "cache should not return past-end iterators to outer world");
        __TBB_ASSERT(&(*map_it) == &map_record_ref,
            "dangling reference has been returned to outside world: data race?");
        __TBB_ASSERT(std::find(my_history_list.begin(), my_history_list.end(), map_it) == my_history_list.end(),
            "object in use should not be in list of unused objects ");

        // if it was the last reference, put it to the LRU history
        if (! --(map_it->second.my_ref_counter)) {
            // if the LRU history is full, evict the oldest items to get space
            if (my_history_list.size() >= my_history_list_capacity) {
                std::size_t number_of_elements_to_evict = 1 + my_history_list.size() - my_history_list_capacity;

                for (std::size_t i = 0; i < number_of_elements_to_evict; ++i) {
                    storage_map_iterator_type map_it_to_evict = my_history_list.back();

                    __TBB_ASSERT(map_it_to_evict->second.my_ref_counter == 0,
                        "item to be evicted should not have a live references");

                    // TODO: can we use forward_list instead of list? pop_front / insert_after last
                    my_history_list.pop_back();
                    my_storage_map.erase(map_it_to_evict);
                }
            }

            // TODO: can we use forward_list instead of list? pop_front / insert_after last
            my_history_list.push_front(map_it);
            map_it->second.my_history_list_iterator = my_history_list.begin();
        }
    }

    storage_map_reference_type retrieve_serial(key_type key, bool& is_new_value_needed) {
        storage_map_iterator_type map_it = my_storage_map.find(key);

        if (map_it == my_storage_map.end()) {
            map_it = my_storage_map.emplace_hint(
                map_it, std::piecewise_construct, std::make_tuple(key), std::make_tuple(value_type(), 0, my_history_list.end(), false));
            is_new_value_needed = true;
        } else {
            history_list_iterator_type list_it = map_it->second.my_history_list_iterator;
            if (list_it != my_history_list.end()) {
                __TBB_ASSERT(map_it->second.my_ref_counter == 0,
                    "item to be evicted should not have a live references");

                // Item is going to be used. Therefore it is not a subject for eviction,
                // so we remove it from LRU history.
                my_history_list.erase(list_it);
                map_it->second.my_history_list_iterator = my_history_list.end();
            }
        }

        ++(map_it->second.my_ref_counter);
        return *map_it;
    }
};

//-----------------------------------------------------------------------------
// Value type for storage map in concurrent LRU cache
//-----------------------------------------------------------------------------

template<typename KeyT, typename ValT, typename KeyToValFunctorT>
struct concurrent_lru_cache<KeyT, ValT, KeyToValFunctorT>::storage_map_value_type {
//typedefs
public:
    using ref_counter_type = std::size_t;

// fields
public:
    value_type my_value;
    ref_counter_type my_ref_counter;
    history_list_iterator_type my_history_list_iterator;
    std::atomic<bool> my_is_ready;

// interface
public:
    storage_map_value_type(
        value_type const& value, ref_counter_type ref_counter,
        history_list_iterator_type history_list_iterator, bool is_ready)
        : my_value(value), my_ref_counter(ref_counter),
          my_history_list_iterator(history_list_iterator), my_is_ready(is_ready) {}
};

//-----------------------------------------------------------------------------
// Handle object for operator[] in concurrent LRU cache
//-----------------------------------------------------------------------------

template<typename KeyT, typename ValT, typename KeyToValFunctorT>
struct concurrent_lru_cache<KeyT, ValT, KeyToValFunctorT>::handle_object {
// fields
private:
    lru_cache_type* my_lru_cache_ptr;
    storage_map_pointer_type my_map_record_ptr;

// interface
public:
    handle_object()
        : my_lru_cache_ptr(nullptr), my_map_record_ptr(nullptr) {}
    handle_object(lru_cache_type& lru_cache_ref, storage_map_reference_type map_record_ref)
        : my_lru_cache_ptr(&lru_cache_ref), my_map_record_ptr(&map_record_ref) {}

    handle_object(handle_object&) = delete;
    void operator=(handle_object&) = delete;

    handle_object(handle_object&& other)
        : my_lru_cache_ptr(other.my_lru_cache_ptr), my_map_record_ptr(other.my_map_record_ptr) {

        __TBB_ASSERT(
            bool(other.my_lru_cache_ptr) == bool(other.my_map_record_ptr),
            "invalid state of moving object?");

        other.my_lru_cache_ptr = nullptr;
        other.my_map_record_ptr = nullptr;
    }

    handle_object& operator=(handle_object&& other) {
        __TBB_ASSERT(
            bool(other.my_lru_cache_ptr) == bool(other.my_map_record_ptr),
            "invalid state of moving object?");

        if (my_lru_cache_ptr)
            my_lru_cache_ptr->signal_end_of_usage(*my_map_record_ptr);

        my_lru_cache_ptr = other.my_lru_cache_ptr;
        my_map_record_ptr = other.my_map_record_ptr;
        other.my_lru_cache_ptr = nullptr;
        other.my_map_record_ptr = nullptr;

        return *this;
    }

    ~handle_object() {
        if (my_lru_cache_ptr)
            my_lru_cache_ptr->signal_end_of_usage(*my_map_record_ptr);
    }

    operator bool() const {
        return (my_lru_cache_ptr && my_map_record_ptr);
    }

    value_type& value() {
        __TBB_ASSERT(my_lru_cache_ptr, "get value from already moved object?");
        __TBB_ASSERT(my_map_record_ptr, "get value from an invalid or already moved object?");

        return my_map_record_ptr->second.my_value;
    }
};

//-----------------------------------------------------------------------------
// Aggregator operation for aggregator type in concurrent LRU cache
//-----------------------------------------------------------------------------

template<typename KeyT, typename ValT, typename KeyToValFunctorT>
struct concurrent_lru_cache<KeyT, ValT, KeyToValFunctorT>::aggregator_operation
    : aggregated_operation<aggregator_operation> {
// incapsulated helper classes
public:
    enum class op_type { retrieve, signal_end_of_usage };

// fields
private:
    op_type my_op;

// interface
public:
    aggregator_operation(op_type op) : my_op(op) {}

    // TODO: aggregator_operation can be implemented
    //   - as a statically typed variant type or CRTP? (static, dependent on the use case)
    //   - or use pointer to function and apply_visitor (dynamic)
    //   - or use virtual functions (dynamic)
    void cast_and_handle(lru_cache_type& lru_cache_ref) {
        if (my_op == op_type::retrieve)
            static_cast<retrieve_aggregator_operation*>(this)->handle(lru_cache_ref);
        else
            static_cast<signal_end_of_usage_aggregator_operation*>(this)->handle(lru_cache_ref);
    }
};

template<typename KeyT, typename ValT, typename KeyToValFunctorT>
struct concurrent_lru_cache<KeyT, ValT, KeyToValFunctorT>::retrieve_aggregator_operation
    : aggregator_operation, private no_assign {
public:
    key_type my_key;
    storage_map_pointer_type my_map_record_ptr;
    bool my_is_new_value_needed;

public:
    retrieve_aggregator_operation(key_type key)
        : aggregator_operation(aggregator_operation::op_type::retrieve),
          my_key(key), my_is_new_value_needed(false) {}

    void handle(lru_cache_type& lru_cache_ref) {
        my_map_record_ptr = &lru_cache_ref.retrieve_serial(my_key, my_is_new_value_needed);
    }

    storage_map_reference_type result() { return *my_map_record_ptr; }

    bool is_new_value_needed() { return my_is_new_value_needed; }
};

template<typename KeyT, typename ValT, typename KeyToValFunctorT>
struct concurrent_lru_cache<KeyT, ValT, KeyToValFunctorT>::signal_end_of_usage_aggregator_operation
    : aggregator_operation, private no_assign {

private:
    storage_map_reference_type my_map_record_ref;

public:
    signal_end_of_usage_aggregator_operation(storage_map_reference_type map_record_ref)
        : aggregator_operation(aggregator_operation::op_type::signal_end_of_usage),
          my_map_record_ref(map_record_ref) {}

    void handle(lru_cache_type& lru_cache_ref) {
        lru_cache_ref.signal_end_of_usage_serial(my_map_record_ref);
    }
};

// TODO: if we have guarantees that KeyToValFunctorT always have
//       ValT as a return type and KeyT as an argument type
//       we can deduce template parameters of concurrent_lru_cache
//       by pattern matching on KeyToValFunctorT

} // namespace d1
} // namespace detail

inline namespace v1 {

using detail::d1::concurrent_lru_cache;

} // inline namespace v1
} // namespace tbb

#endif // __TBB_concurrent_lru_cache_H
