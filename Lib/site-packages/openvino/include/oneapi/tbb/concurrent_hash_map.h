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

#ifndef __TBB_concurrent_hash_map_H
#define __TBB_concurrent_hash_map_H

#include "detail/_namespace_injection.h"
#include "detail/_utils.h"
#include "detail/_assert.h"
#include "detail/_allocator_traits.h"
#include "detail/_containers_helpers.h"
#include "detail/_template_helpers.h"
#include "detail/_hash_compare.h"
#include "detail/_range_common.h"
#include "tbb_allocator.h"
#include "spin_rw_mutex.h"

#include <atomic>
#include <initializer_list>
#include <tuple>
#include <iterator>
#include <utility>      // Need std::pair
#include <cstring>      // Need std::memset

namespace tbb {
namespace detail {
namespace d1 {

struct hash_map_node_base : no_copy {
    using mutex_type = spin_rw_mutex;
    // Scoped lock type for mutex
    using scoped_type = mutex_type::scoped_lock;
    // Next node in chain
    hash_map_node_base* next;
    mutex_type mutex;
};

// Incompleteness flag value
static hash_map_node_base* const rehash_req = reinterpret_cast<hash_map_node_base*>(std::size_t(3));
// Rehashed empty bucket flag
static hash_map_node_base* const empty_rehashed = reinterpret_cast<hash_map_node_base*>(std::size_t(0));

// base class of concurrent_hash_map

template <typename Allocator>
class hash_map_base {
public:
    using size_type = std::size_t;
    using hashcode_type = std::size_t;
    using segment_index_type = std::size_t;
    using node_base = hash_map_node_base;

    struct bucket : no_copy {
        using mutex_type = spin_rw_mutex;
        using scoped_type = mutex_type::scoped_lock;

        bucket() : node_list(nullptr) {}
        bucket( node_base* ptr ) : node_list(ptr) {}

        mutex_type mutex;
        std::atomic<node_base*> node_list;
    };

    using allocator_type = Allocator;
    using allocator_traits_type = tbb::detail::allocator_traits<allocator_type>;
    using bucket_allocator_type = typename allocator_traits_type::template rebind_alloc<bucket>;
    using bucket_allocator_traits = tbb::detail::allocator_traits<bucket_allocator_type>;

    // Count of segments in the first block
    static constexpr size_type embedded_block = 1;
    // Count of segments in the first block
    static constexpr size_type embedded_buckets = 1 << embedded_block;
    // Count of segments in the first block
    static constexpr size_type first_block = 8; //including embedded_block. perfect with bucket size 16, so the allocations are power of 4096
    // Size of a pointer / table size
    static constexpr size_type pointers_per_table = sizeof(segment_index_type) * 8; // one segment per bit

    using segment_ptr_type = bucket*;
    using atomic_segment_type = std::atomic<segment_ptr_type>;
    using segments_table_type = atomic_segment_type[pointers_per_table];

    hash_map_base( const allocator_type& alloc ) : my_allocator(alloc), my_mask(embedded_buckets - 1), my_size(0) {
        for (size_type i = 0; i != embedded_buckets; ++i) {
            my_embedded_segment[i].node_list.store(nullptr, std::memory_order_relaxed);
        }

        for (size_type segment_index = 0; segment_index < pointers_per_table; ++segment_index) {
            auto argument = segment_index < embedded_block ? my_embedded_segment + segment_base(segment_index) : nullptr;
            my_table[segment_index].store(argument, std::memory_order_relaxed);
        }

        __TBB_ASSERT( embedded_block <= first_block, "The first block number must include embedded blocks");
    }

    // segment index of given index in the array
    static segment_index_type segment_index_of( size_type index ) {
        return segment_index_type(tbb::detail::log2( index|1 ));
    }

    // the first array index of given segment
    static segment_index_type segment_base( segment_index_type k ) {
        return (segment_index_type(1) << k & ~segment_index_type(1));
    }

    // segment size except for k == 0
    static size_type segment_size( segment_index_type k ) {
        return size_type(1) << k; // fake value for k==0
    }

    // true if ptr is valid pointer
    static bool is_valid( void* ptr ) {
        return reinterpret_cast<uintptr_t>(ptr) > uintptr_t(63);
    }

    template <typename... Args>
    void init_buckets_impl( segment_ptr_type ptr, size_type sz, Args&&... args ) {
        for (size_type i = 0; i < sz; ++i) {
            bucket_allocator_traits::construct(my_allocator, ptr + i, std::forward<Args>(args)...);
        }
    }

    // Initialize buckets
    void init_buckets( segment_ptr_type ptr, size_type sz, bool is_initial ) {
        if (is_initial) {
            init_buckets_impl(ptr, sz);
        } else {
            init_buckets_impl(ptr, sz, reinterpret_cast<node_base*>(rehash_req));
        }
    }

    // Add node n to bucket b
    static void add_to_bucket( bucket* b, node_base* n ) {
        __TBB_ASSERT(b->node_list.load(std::memory_order_relaxed) != rehash_req, nullptr);
        n->next = b->node_list.load(std::memory_order_relaxed);
        b->node_list.store(n, std::memory_order_relaxed); // its under lock and flag is set
    }

    const bucket_allocator_type& get_allocator() const {
        return my_allocator;
    }

    bucket_allocator_type& get_allocator() {
        return my_allocator;
    }

    // Enable segment
    void enable_segment( segment_index_type k, bool is_initial = false ) {
        __TBB_ASSERT( k, "Zero segment must be embedded" );
        size_type sz;
        __TBB_ASSERT( !is_valid(my_table[k].load(std::memory_order_relaxed)), "Wrong concurrent assignment");
        if (k >= first_block) {
            sz = segment_size(k);
            segment_ptr_type ptr = nullptr;
            try_call( [&] {
                ptr = bucket_allocator_traits::allocate(my_allocator, sz);
            } ).on_exception( [&] {
                my_table[k].store(nullptr, std::memory_order_relaxed);
            });

            __TBB_ASSERT(ptr, nullptr);
            init_buckets(ptr, sz, is_initial);
            my_table[k].store(ptr, std::memory_order_release);
            sz <<= 1;// double it to get entire capacity of the container
        } else { // the first block
            __TBB_ASSERT( k == embedded_block, "Wrong segment index" );
            sz = segment_size(first_block);
            segment_ptr_type ptr = nullptr;
            try_call( [&] {
                ptr = bucket_allocator_traits::allocate(my_allocator, sz - embedded_buckets);
            } ).on_exception( [&] {
                my_table[k].store(nullptr, std::memory_order_relaxed);
            });

            __TBB_ASSERT(ptr, nullptr);
            init_buckets(ptr, sz - embedded_buckets, is_initial);
            ptr -= segment_base(embedded_block);
            for(segment_index_type i = embedded_block; i < first_block; i++) // calc the offsets
                my_table[i].store(ptr + segment_base(i), std::memory_order_release);
        }
        my_mask.store(sz-1, std::memory_order_release);
    }

    void delete_segment( segment_index_type s ) {
        segment_ptr_type buckets_ptr = my_table[s].load(std::memory_order_relaxed);
        size_type sz = segment_size( s ? s : 1 );

        size_type deallocate_size = 0;

        if (s >= first_block) { // the first segment or the next
            deallocate_size = sz;
        } else if (s == embedded_block && embedded_block != first_block) {
            deallocate_size = segment_size(first_block) - embedded_buckets;
        }

        for (size_type i = 0; i < deallocate_size; ++i) {
            bucket_allocator_traits::destroy(my_allocator, buckets_ptr + i);
        }
        if (deallocate_size != 0) {
            bucket_allocator_traits::deallocate(my_allocator, buckets_ptr, deallocate_size);
        }

        if (s >= embedded_block) my_table[s].store(nullptr, std::memory_order_relaxed);
    }

    // Get bucket by (masked) hashcode
    bucket *get_bucket( hashcode_type h ) const noexcept {
        segment_index_type s = segment_index_of( h );
        h -= segment_base(s);
        segment_ptr_type seg = my_table[s].load(std::memory_order_acquire);
        __TBB_ASSERT( is_valid(seg), "hashcode must be cut by valid mask for allocated segments" );
        return &seg[h];
    }

    // detail serial rehashing helper
    void mark_rehashed_levels( hashcode_type h ) noexcept {
        segment_index_type s = segment_index_of( h );
        while (segment_ptr_type seg = my_table[++s].load(std::memory_order_relaxed))
            if( seg[h].node_list.load(std::memory_order_relaxed) == rehash_req ) {
                seg[h].node_list.store(empty_rehashed, std::memory_order_relaxed);
                mark_rehashed_levels( h + ((hashcode_type)1<<s) ); // optimized segment_base(s)
            }
    }

    // Check for mask race
    // Splitting into two functions should help inlining
    inline bool check_mask_race( const hashcode_type h, hashcode_type &m ) const {
        hashcode_type m_now, m_old = m;
        m_now = my_mask.load(std::memory_order_acquire);
        if (m_old != m_now) {
            return check_rehashing_collision(h, m_old, m = m_now);
        }
        return false;
    }

    // Process mask race, check for rehashing collision
    bool check_rehashing_collision( const hashcode_type h, hashcode_type m_old, hashcode_type m ) const {
        __TBB_ASSERT(m_old != m, nullptr); // TODO?: m arg could be optimized out by passing h = h&m
        if( (h & m_old) != (h & m) ) { // mask changed for this hashcode, rare event
            // condition above proves that 'h' has some other bits set beside 'm_old'
            // find next applicable mask after m_old    //TODO: look at bsl instruction
            for( ++m_old; !(h & m_old); m_old <<= 1 ) // at maximum few rounds depending on the first block size
                ;
            m_old = (m_old<<1) - 1; // get full mask from a bit
            __TBB_ASSERT((m_old&(m_old+1))==0 && m_old <= m, nullptr);
            // check whether it is rehashing/ed
            if( get_bucket(h & m_old)->node_list.load(std::memory_order_acquire) != rehash_req ) {
                return true;
            }
        }
        return false;
    }

    // Insert a node and check for load factor. @return segment index to enable.
    segment_index_type insert_new_node( bucket *b, node_base *n, hashcode_type mask ) {
        size_type sz = ++my_size; // prefix form is to enforce allocation after the first item inserted
        add_to_bucket( b, n );
        // check load factor
        if( sz >= mask ) { // TODO: add custom load_factor
            segment_index_type new_seg = tbb::detail::log2( mask+1 ); //optimized segment_index_of
            __TBB_ASSERT( is_valid(my_table[new_seg-1].load(std::memory_order_relaxed)), "new allocations must not publish new mask until segment has allocated");
            static const segment_ptr_type is_allocating = segment_ptr_type(2);;
            segment_ptr_type disabled = nullptr;
            if (!(my_table[new_seg].load(std::memory_order_acquire))
                && my_table[new_seg].compare_exchange_strong(disabled, is_allocating))
                return new_seg; // The value must be processed
        }
        return 0;
    }

    // Prepare enough segments for number of buckets
    void reserve(size_type buckets) {
        if( !buckets-- ) return;
        bool is_initial = !my_size.load(std::memory_order_relaxed);
        for (size_type m = my_mask.load(std::memory_order_relaxed); buckets > m;
            m = my_mask.load(std::memory_order_relaxed))
        {
            enable_segment( segment_index_of( m+1 ), is_initial );
        }
    }

    // Swap hash_map_bases
    void internal_swap_content(hash_map_base &table) {
        using std::swap;
        swap_atomics_relaxed(my_mask, table.my_mask);
        swap_atomics_relaxed(my_size, table.my_size);

        for(size_type i = 0; i < embedded_buckets; i++) {
            auto temp = my_embedded_segment[i].node_list.load(std::memory_order_relaxed);
            my_embedded_segment[i].node_list.store(table.my_embedded_segment[i].node_list.load(std::memory_order_relaxed),
                std::memory_order_relaxed);
            table.my_embedded_segment[i].node_list.store(temp, std::memory_order_relaxed);
        }
        for(size_type i = embedded_block; i < pointers_per_table; i++) {
            auto temp = my_table[i].load(std::memory_order_relaxed);
            my_table[i].store(table.my_table[i].load(std::memory_order_relaxed),
                std::memory_order_relaxed);
            table.my_table[i].store(temp, std::memory_order_relaxed);
        }
    }

    void internal_move(hash_map_base&& other) {
        my_mask.store(other.my_mask.load(std::memory_order_relaxed), std::memory_order_relaxed);
        other.my_mask.store(embedded_buckets - 1, std::memory_order_relaxed);

        my_size.store(other.my_size.load(std::memory_order_relaxed), std::memory_order_relaxed);
        other.my_size.store(0, std::memory_order_relaxed);

        for (size_type i = 0; i < embedded_buckets; ++i) {
            my_embedded_segment[i].node_list.store(other.my_embedded_segment[i].node_list, std::memory_order_relaxed);
            other.my_embedded_segment[i].node_list.store(nullptr, std::memory_order_relaxed);
        }

        for (size_type i = embedded_block; i < pointers_per_table; ++i) {
            my_table[i].store(other.my_table[i].load(std::memory_order_relaxed),
                std::memory_order_relaxed);
            other.my_table[i].store(nullptr, std::memory_order_relaxed);
        }
    }

protected:

    bucket_allocator_type my_allocator;
    // Hash mask = sum of allocated segment sizes - 1
    std::atomic<hashcode_type> my_mask;
    // Size of container in stored items
    std::atomic<size_type> my_size; // It must be in separate cache line from my_mask due to performance effects
    // Zero segment
    bucket my_embedded_segment[embedded_buckets];
    // Segment pointers table. Also prevents false sharing between my_mask and my_size
    segments_table_type my_table;
};

template <typename Iterator>
class hash_map_range;

// Meets requirements of a forward iterator for STL
// Value is either the T or const T type of the container.
template <typename Container, typename Value>
class hash_map_iterator {
    using map_type = Container;
    using node = typename Container::node;
    using map_base = typename Container::base_type;
    using node_base = typename map_base::node_base;
    using bucket = typename map_base::bucket;
public:
    using value_type = Value;
    using size_type = typename Container::size_type;
    using difference_type = typename Container::difference_type;
    using pointer = value_type*;
    using reference = value_type&;
    using iterator_category = std::forward_iterator_tag;

    // Construct undefined iterator
    hash_map_iterator(): my_map(), my_index(), my_bucket(), my_node() {}
    hash_map_iterator( const hash_map_iterator<Container, typename Container::value_type>& other ) :
        my_map(other.my_map),
        my_index(other.my_index),
        my_bucket(other.my_bucket),
        my_node(other.my_node)
    {}

    hash_map_iterator& operator=( const hash_map_iterator<Container, typename Container::value_type>& other ) {
        my_map = other.my_map;
        my_index = other.my_index;
        my_bucket = other.my_bucket;
        my_node = other.my_node;
        return *this;
    }

    Value& operator*() const {
        __TBB_ASSERT( map_base::is_valid(my_node), "iterator uninitialized or at end of container?" );
        return my_node->value();
    }

    Value* operator->() const {return &operator*();}

    hash_map_iterator& operator++() {
        my_node = static_cast<node*>( my_node->next );
        if( !my_node ) advance_to_next_bucket();
        return *this;
    }

    // Post increment
    hash_map_iterator operator++(int) {
        hash_map_iterator old(*this);
        operator++();
        return old;
    }
private:
    template <typename C, typename T, typename U>
    friend bool operator==( const hash_map_iterator<C,T>& i, const hash_map_iterator<C,U>& j );

    template <typename C, typename T, typename U>
    friend bool operator!=( const hash_map_iterator<C,T>& i, const hash_map_iterator<C,U>& j );

    template <typename C, typename T, typename U>
    friend ptrdiff_t operator-( const hash_map_iterator<C,T>& i, const hash_map_iterator<C,U>& j );

    template <typename C, typename U>
    friend class hash_map_iterator;

    template <typename I>
    friend class hash_map_range;

    void advance_to_next_bucket() { // TODO?: refactor to iterator_base class
        size_t k = my_index+1;
        __TBB_ASSERT( my_bucket, "advancing an invalid iterator?");
        while (k <= my_map->my_mask.load(std::memory_order_relaxed)) {
            // Following test uses 2's-complement wizardry
            if( k&(k-2) ) // not the beginning of a segment
                ++my_bucket;
            else my_bucket = my_map->get_bucket( k );
            my_node = static_cast<node*>( my_bucket->node_list.load(std::memory_order_relaxed) );
            if( map_base::is_valid(my_node) ) {
                my_index = k; return;
            }
            ++k;
        }
        my_bucket = 0; my_node = 0; my_index = k; // the end
    }

    template <typename Key, typename T, typename HashCompare, typename A>
    friend class concurrent_hash_map;

    hash_map_iterator( const Container &map, std::size_t index, const bucket *b, node_base *n ) :
        my_map(&map), my_index(index), my_bucket(b), my_node(static_cast<node*>(n))
    {
        if( b && !map_base::is_valid(n) )
            advance_to_next_bucket();
    }

    // concurrent_hash_map over which we are iterating.
    const Container *my_map;
    // Index in hash table for current item
    size_t my_index;
    // Pointer to bucket
    const bucket* my_bucket;
    // Pointer to node that has current item
    node* my_node;
};

template <typename Container, typename T, typename U>
bool operator==( const hash_map_iterator<Container,T>& i, const hash_map_iterator<Container,U>& j ) {
    return i.my_node == j.my_node && i.my_map == j.my_map;
}

template <typename Container, typename T, typename U>
bool operator!=( const hash_map_iterator<Container,T>& i, const hash_map_iterator<Container,U>& j ) {
    return i.my_node != j.my_node || i.my_map != j.my_map;
}

// Range class used with concurrent_hash_map
template <typename Iterator>
class hash_map_range {
    using map_type = typename Iterator::map_type;
public:
    // Type for size of a range
    using size_type = std::size_t;
    using value_type = typename Iterator::value_type;
    using reference = typename Iterator::reference;
    using difference_type = typename Iterator::difference_type;
    using iterator = Iterator;

    // True if range is empty.
    bool empty() const {return my_begin == my_end;}

    // True if range can be partitioned into two subranges.
    bool is_divisible() const {
        return my_midpoint != my_end;
    }

    // Split range.
    hash_map_range( hash_map_range& r, split ) :
        my_end(r.my_end),
        my_grainsize(r.my_grainsize)
    {
        r.my_end = my_begin = r.my_midpoint;
        __TBB_ASSERT( !empty(), "Splitting despite the range is not divisible" );
        __TBB_ASSERT( !r.empty(), "Splitting despite the range is not divisible" );
        set_midpoint();
        r.set_midpoint();
    }

    // Init range with container and grainsize specified
    hash_map_range( const map_type &map, size_type grainsize_ = 1 ) :
        my_begin( Iterator( map, 0, map.my_embedded_segment, map.my_embedded_segment->node_list.load(std::memory_order_relaxed) ) ),
        my_end( Iterator( map, map.my_mask.load(std::memory_order_relaxed) + 1, 0, 0 ) ),
        my_grainsize( grainsize_ )
    {
        __TBB_ASSERT( grainsize_>0, "grainsize must be positive" );
        set_midpoint();
    }

    const Iterator begin() const { return my_begin; }
    const Iterator end() const { return my_end; }
    // The grain size for this range.
    size_type grainsize() const { return my_grainsize; }

private:
    Iterator my_begin;
    Iterator my_end;
    mutable Iterator my_midpoint;
    size_t my_grainsize;
    // Set my_midpoint to point approximately half way between my_begin and my_end.
    void set_midpoint() const;
    template <typename U> friend class hash_map_range;
};

template <typename Iterator>
void hash_map_range<Iterator>::set_midpoint() const {
    // Split by groups of nodes
    size_t m = my_end.my_index-my_begin.my_index;
    if( m > my_grainsize ) {
        m = my_begin.my_index + m/2u;
        auto b = my_begin.my_map->get_bucket(m);
        my_midpoint = Iterator(*my_begin.my_map,m,b,b->node_list.load(std::memory_order_relaxed));
    } else {
        my_midpoint = my_end;
    }
    __TBB_ASSERT( my_begin.my_index <= my_midpoint.my_index,
        "my_begin is after my_midpoint" );
    __TBB_ASSERT( my_midpoint.my_index <= my_end.my_index,
        "my_midpoint is after my_end" );
    __TBB_ASSERT( my_begin != my_midpoint || my_begin == my_end,
        "[my_begin, my_midpoint) range should not be empty" );
}

template <typename Key, typename T,
          typename HashCompare = tbb_hash_compare<Key>,
          typename Allocator = tbb_allocator<std::pair<const Key, T>>>
class concurrent_hash_map : protected hash_map_base<Allocator> {
    template <typename Container, typename Value>
    friend class hash_map_iterator;

    template <typename I>
    friend class hash_map_range;
    using allocator_traits_type = tbb::detail::allocator_traits<Allocator>;
public:
    using base_type = hash_map_base<Allocator>;
    using key_type = Key;
    using mapped_type = T;
    // type_identity is needed to disable implicit deduction guides for std::initializer_list constructors
    // and copy/move constructor with explicit allocator argument
    using allocator_type = tbb::detail::type_identity_t<Allocator>;
    using hash_compare_type = tbb::detail::type_identity_t<HashCompare>;
    using value_type = std::pair<const Key, T>;
    using size_type = typename base_type::size_type;
    using difference_type = std::ptrdiff_t;

    using pointer = typename allocator_traits_type::pointer;
    using const_pointer = typename allocator_traits_type::const_pointer;

    using reference = value_type&;
    using const_reference = const value_type&;
    using iterator = hash_map_iterator<concurrent_hash_map, value_type>;
    using const_iterator = hash_map_iterator<concurrent_hash_map, const value_type>;
    using range_type = hash_map_range<iterator>;
    using const_range_type = hash_map_range<const_iterator>;

protected:
    static_assert(std::is_same<value_type, typename Allocator::value_type>::value,
        "value_type of the container must be the same as its allocator's");

    friend class const_accessor;
    class node;
    using segment_index_type = typename base_type::segment_index_type;
    using segment_ptr_type = typename base_type::segment_ptr_type;
    using node_base = typename base_type::node_base;
    using bucket = typename base_type::bucket;
    using hashcode_type = typename base_type::hashcode_type;
    using bucket_allocator_type = typename base_type::bucket_allocator_type;
    using node_allocator_type = typename base_type::allocator_traits_type::template rebind_alloc<node>;
    using node_allocator_traits = tbb::detail::allocator_traits<node_allocator_type>;
    hash_compare_type my_hash_compare;

    class node : public node_base {
    public:
        node() {}
        ~node() {}
        pointer storage() { return &my_value; }
        value_type& value() { return *storage(); }
    private:
        union {
            value_type my_value;
        };
    };

    void delete_node( node_base *n ) {
        node_allocator_type node_allocator(this->get_allocator());
        node_allocator_traits::destroy(node_allocator, static_cast<node*>(n)->storage());
        node_allocator_traits::destroy(node_allocator, static_cast<node*>(n));
        node_allocator_traits::deallocate(node_allocator, static_cast<node*>(n), 1);
    }

    template <typename... Args>
    static node* create_node(bucket_allocator_type& allocator, Args&&... args) {
        node_allocator_type node_allocator(allocator);
        node* node_ptr = node_allocator_traits::allocate(node_allocator, 1);
        auto guard = make_raii_guard([&] {
            node_allocator_traits::destroy(node_allocator, node_ptr);
            node_allocator_traits::deallocate(node_allocator, node_ptr, 1);
        });

        node_allocator_traits::construct(node_allocator, node_ptr);
        node_allocator_traits::construct(node_allocator, node_ptr->storage(), std::forward<Args>(args)...);
        guard.dismiss();
        return node_ptr;
    }

    static node* allocate_node_copy_construct(bucket_allocator_type& allocator, const Key &key, const T * t){
        return create_node(allocator, key, *t);
    }

    static node* allocate_node_move_construct(bucket_allocator_type& allocator, const Key &key, const T * t){
        return create_node(allocator, key, std::move(*const_cast<T*>(t)));
    }

    static node* allocate_node_default_construct(bucket_allocator_type& allocator, const Key &key, const T * ){
        // Emplace construct an empty T object inside the pair
        return create_node(allocator, std::piecewise_construct,
                           std::forward_as_tuple(key), std::forward_as_tuple());
    }

    static node* do_not_allocate_node(bucket_allocator_type& , const Key &, const T * ){
        __TBB_ASSERT(false,"this dummy function should not be called");
        return nullptr;
    }

    node *search_bucket( const key_type &key, bucket *b ) const {
        node *n = static_cast<node*>( b->node_list.load(std::memory_order_relaxed) );
        while (this->is_valid(n) && !my_hash_compare.equal(key, n->value().first))
            n = static_cast<node*>( n->next );
        __TBB_ASSERT(n != rehash_req, "Search can be executed only for rehashed bucket");
        return n;
    }

    // bucket accessor is to find, rehash, acquire a lock, and access a bucket
    class bucket_accessor : public bucket::scoped_type {
        bucket *my_b;
    public:
        bucket_accessor( concurrent_hash_map *base, const hashcode_type h, bool writer = false ) { acquire( base, h, writer ); }
        // find a bucket by masked hashcode, optionally rehash, and acquire the lock
        inline void acquire( concurrent_hash_map *base, const hashcode_type h, bool writer = false ) {
            my_b = base->get_bucket( h );
            // TODO: actually, notification is unnecessary here, just hiding double-check
            if( my_b->node_list.load(std::memory_order_acquire) == rehash_req
                && bucket::scoped_type::try_acquire( my_b->mutex, /*write=*/true ) )
            {
                if( my_b->node_list.load(std::memory_order_relaxed) == rehash_req ) base->rehash_bucket( my_b, h ); //recursive rehashing
            }
            else bucket::scoped_type::acquire( my_b->mutex, writer );
            __TBB_ASSERT( my_b->node_list.load(std::memory_order_relaxed) != rehash_req, nullptr);
        }
        // check whether bucket is locked for write
        bool is_writer() { return bucket::scoped_type::m_is_writer; }
        // get bucket pointer
        bucket *operator() () { return my_b; }
    };

    // TODO refactor to hash_base
    void rehash_bucket( bucket *b_new, const hashcode_type hash ) {
        __TBB_ASSERT( *(intptr_t*)(&b_new->mutex), "b_new must be locked (for write)");
        __TBB_ASSERT( hash > 1, "The lowermost buckets can't be rehashed" );
        b_new->node_list.store(empty_rehashed, std::memory_order_release); // mark rehashed
        hashcode_type mask = (1u << tbb::detail::log2(hash)) - 1; // get parent mask from the topmost bit
        bucket_accessor b_old( this, hash & mask );

        mask = (mask<<1) | 1; // get full mask for new bucket
        __TBB_ASSERT( (mask&(mask+1))==0 && (hash & mask) == hash, nullptr );
    restart:
        node_base* prev = nullptr;
        node_base* curr = b_old()->node_list.load(std::memory_order_acquire);
        while (this->is_valid(curr)) {
            hashcode_type curr_node_hash = my_hash_compare.hash(static_cast<node*>(curr)->value().first);

            if ((curr_node_hash & mask) == hash) {
                if (!b_old.is_writer()) {
                    if (!b_old.upgrade_to_writer()) {
                        goto restart; // node ptr can be invalid due to concurrent erase
                    }
                }
                node_base* next = curr->next;
                // exclude from b_old
                if (prev == nullptr) {
                    b_old()->node_list.store(curr->next, std::memory_order_relaxed);
                } else {
                    prev->next = curr->next;
                }
                this->add_to_bucket(b_new, curr);
                curr = next;
            } else {
                prev = curr;
                curr = curr->next;
            }
        }
    }

public:

    class accessor;
    // Combines data access, locking, and garbage collection.
    class const_accessor : private node::scoped_type /*which derived from no_copy*/ {
        friend class concurrent_hash_map<Key,T,HashCompare,Allocator>;
        friend class accessor;
    public:
        // Type of value
        using value_type = const typename concurrent_hash_map::value_type;

        // True if result is empty.
        bool empty() const { return !my_node; }

        // Set to null
        void release() {
            if( my_node ) {
                node::scoped_type::release();
                my_node = 0;
            }
        }

        // Return reference to associated value in hash table.
        const_reference operator*() const {
            __TBB_ASSERT( my_node, "attempt to dereference empty accessor" );
            return my_node->value();
        }

        // Return pointer to associated value in hash table.
        const_pointer operator->() const {
            return &operator*();
        }

        // Create empty result
        const_accessor() : my_node(nullptr) {}

        // Destroy result after releasing the underlying reference.
        ~const_accessor() {
            my_node = nullptr; // scoped lock's release() is called in its destructor
        }
    protected:
        bool is_writer() { return node::scoped_type::m_is_writer; }
        node *my_node;
        hashcode_type my_hash;
    };

    // Allows write access to elements and combines data access, locking, and garbage collection.
    class accessor: public const_accessor {
    public:
        // Type of value
        using value_type = typename concurrent_hash_map::value_type;

        // Return reference to associated value in hash table.
        reference operator*() const {
            __TBB_ASSERT( this->my_node, "attempt to dereference empty accessor" );
            return this->my_node->value();
        }

        // Return pointer to associated value in hash table.
        pointer operator->() const {
            return &operator*();
        }
    };

    explicit concurrent_hash_map( const hash_compare_type& compare, const allocator_type& a = allocator_type() )
        : base_type(a)
        , my_hash_compare(compare)
    {}

    concurrent_hash_map() : concurrent_hash_map(hash_compare_type()) {}

    explicit concurrent_hash_map( const allocator_type& a )
        : concurrent_hash_map(hash_compare_type(), a)
    {}

    // Construct empty table with n preallocated buckets. This number serves also as initial concurrency level.
    concurrent_hash_map( size_type n, const allocator_type &a = allocator_type() )
        : concurrent_hash_map(a)
    {
        this->reserve(n);
    }

    concurrent_hash_map( size_type n, const hash_compare_type& compare, const allocator_type& a = allocator_type() )
        : concurrent_hash_map(compare, a)
    {
        this->reserve(n);
    }

    // Copy constructor
    concurrent_hash_map( const concurrent_hash_map &table )
        : concurrent_hash_map(node_allocator_traits::select_on_container_copy_construction(table.get_allocator()))
    {
        try_call( [&] {
            internal_copy(table);
        }).on_exception( [&] {
            this->clear();
        });
    }

    concurrent_hash_map( const concurrent_hash_map &table, const allocator_type &a)
        : concurrent_hash_map(a)
    {
        try_call( [&] {
            internal_copy(table);
        }).on_exception( [&] {
            this->clear();
        });
    }

    // Move constructor
    concurrent_hash_map( concurrent_hash_map &&table )
        : concurrent_hash_map(std::move(table.get_allocator()))
    {
        this->internal_move(std::move(table));
    }

    // Move constructor
    concurrent_hash_map( concurrent_hash_map &&table, const allocator_type &a )
        : concurrent_hash_map(a)
    {
        using is_equal_type = typename node_allocator_traits::is_always_equal;
        internal_move_construct_with_allocator(std::move(table), a, is_equal_type());
    }

    // Construction with copying iteration range and given allocator instance
    template <typename I>
    concurrent_hash_map( I first, I last, const allocator_type &a = allocator_type() )
        : concurrent_hash_map(a)
    {
        try_call( [&] {
            internal_copy(first, last, std::distance(first, last));
        }).on_exception( [&] {
            this->clear();
        });
    }

    template <typename I>
    concurrent_hash_map( I first, I last, const hash_compare_type& compare, const allocator_type& a = allocator_type() )
        : concurrent_hash_map(compare, a)
    {
        try_call( [&] {
            internal_copy(first, last, std::distance(first, last));
        }).on_exception( [&] {
            this->clear();
        });
    }

    concurrent_hash_map( std::initializer_list<value_type> il, const hash_compare_type& compare = hash_compare_type(), const allocator_type& a = allocator_type() )
        : concurrent_hash_map(compare, a)
    {
        try_call( [&] {
            internal_copy(il.begin(), il.end(), il.size());
        }).on_exception( [&] {
            this->clear();
        });
    }

    concurrent_hash_map( std::initializer_list<value_type> il, const allocator_type& a )
        : concurrent_hash_map(il, hash_compare_type(), a) {}

    // Assignment
    concurrent_hash_map& operator=( const concurrent_hash_map &table ) {
        if( this != &table ) {
            clear();
            copy_assign_allocators(this->my_allocator, table.my_allocator);
            internal_copy(table);
        }
        return *this;
    }

    // Move Assignment
    concurrent_hash_map& operator=( concurrent_hash_map &&table ) {
        if( this != &table ) {
            using pocma_type = typename node_allocator_traits::propagate_on_container_move_assignment;
            using is_equal_type = typename node_allocator_traits::is_always_equal;
            move_assign_allocators(this->my_allocator, table.my_allocator);
            internal_move_assign(std::move(table), tbb::detail::disjunction<is_equal_type, pocma_type>());
        }
        return *this;
    }

    // Assignment
    concurrent_hash_map& operator=( std::initializer_list<value_type> il ) {
        clear();
        internal_copy(il.begin(), il.end(), il.size());
        return *this;
    }

    // Rehashes and optionally resizes the whole table.
    /** Useful to optimize performance before or after concurrent operations.
        Also enables using of find() and count() concurrent methods in serial context. */
    void rehash(size_type sz = 0) {
        this->reserve(sz); // TODO: add reduction of number of buckets as well
        hashcode_type mask = this->my_mask.load(std::memory_order_relaxed);
        hashcode_type b = (mask+1)>>1; // size or first index of the last segment
        __TBB_ASSERT((b&(b-1))==0, nullptr); // zero or power of 2
        bucket *bp = this->get_bucket( b ); // only the last segment should be scanned for rehashing
        for(; b <= mask; b++, bp++ ) {
            node_base *n = bp->node_list.load(std::memory_order_relaxed);
            __TBB_ASSERT( this->is_valid(n) || n == empty_rehashed || n == rehash_req, "Broken detail structure" );
            __TBB_ASSERT( *reinterpret_cast<intptr_t*>(&bp->mutex) == 0, "concurrent or unexpectedly terminated operation during rehash() execution" );
            if( n == rehash_req ) { // rehash bucket, conditional because rehashing of a previous bucket may affect this one
                hashcode_type h = b; bucket *b_old = bp;
                do {
                    __TBB_ASSERT( h > 1, "The lowermost buckets can't be rehashed" );
                    hashcode_type m = ( 1u<<tbb::detail::log2( h ) ) - 1; // get parent mask from the topmost bit
                    b_old = this->get_bucket( h &= m );
                } while( b_old->node_list.load(std::memory_order_relaxed) == rehash_req );
                // now h - is index of the root rehashed bucket b_old
                this->mark_rehashed_levels( h ); // mark all non-rehashed children recursively across all segments
                node_base* prev = nullptr;
                node_base* curr = b_old->node_list.load(std::memory_order_relaxed);
                while (this->is_valid(curr)) {
                    hashcode_type curr_node_hash = my_hash_compare.hash(static_cast<node*>(curr)->value().first);

                    if ((curr_node_hash & mask) != h) { // should be rehashed
                        node_base* next = curr->next;
                        // exclude from b_old
                        if (prev == nullptr) {
                            b_old->node_list.store(curr->next, std::memory_order_relaxed);
                        } else {
                            prev->next = curr->next;
                        }
                        bucket *b_new = this->get_bucket(curr_node_hash & mask);
                        __TBB_ASSERT(b_new->node_list.load(std::memory_order_relaxed) != rehash_req, "hash() function changed for key in table or detail error" );
                        this->add_to_bucket(b_new, curr);
                        curr = next;
                    } else {
                        prev = curr;
                        curr = curr->next;
                    }
                }
            }
        }
    }

    // Clear table
    void clear() {
        hashcode_type m = this->my_mask.load(std::memory_order_relaxed);
        __TBB_ASSERT((m&(m+1))==0, "data structure is invalid");
        this->my_size.store(0, std::memory_order_relaxed);
        segment_index_type s = this->segment_index_of( m );
        __TBB_ASSERT( s+1 == this->pointers_per_table || !this->my_table[s+1].load(std::memory_order_relaxed), "wrong mask or concurrent grow" );
        do {
            __TBB_ASSERT(this->is_valid(this->my_table[s].load(std::memory_order_relaxed)), "wrong mask or concurrent grow" );
            segment_ptr_type buckets_ptr = this->my_table[s].load(std::memory_order_relaxed);
            size_type sz = this->segment_size( s ? s : 1 );
            for( segment_index_type i = 0; i < sz; i++ )
                for( node_base *n = buckets_ptr[i].node_list.load(std::memory_order_relaxed);
                    this->is_valid(n); n = buckets_ptr[i].node_list.load(std::memory_order_relaxed) )
                {
                    buckets_ptr[i].node_list.store(n->next, std::memory_order_relaxed);
                    delete_node( n );
                }
            this->delete_segment(s);
        } while(s-- > 0);
        this->my_mask.store(this->embedded_buckets - 1, std::memory_order_relaxed);
    }

    // Clear table and destroy it.
    ~concurrent_hash_map() { clear(); }

    //------------------------------------------------------------------------
    // Parallel algorithm support
    //------------------------------------------------------------------------
    range_type range( size_type grainsize=1 ) {
        return range_type( *this, grainsize );
    }
    const_range_type range( size_type grainsize=1 ) const {
        return const_range_type( *this, grainsize );
    }

    //------------------------------------------------------------------------
    // STL support - not thread-safe methods
    //------------------------------------------------------------------------
    iterator begin() { return iterator( *this, 0, this->my_embedded_segment, this->my_embedded_segment->node_list.load(std::memory_order_relaxed) ); }
    const_iterator begin() const { return const_iterator( *this, 0, this->my_embedded_segment, this->my_embedded_segment->node_list.load(std::memory_order_relaxed) ); }
    const_iterator cbegin() const { return const_iterator( *this, 0, this->my_embedded_segment, this->my_embedded_segment->node_list.load(std::memory_order_relaxed) ); }
    iterator end() { return iterator( *this, 0, 0, 0 ); }
    const_iterator end() const { return const_iterator( *this, 0, 0, 0 ); }
    const_iterator cend() const { return const_iterator( *this, 0, 0, 0 ); }
    std::pair<iterator, iterator> equal_range( const Key& key ) { return internal_equal_range( key, end() ); }
    std::pair<const_iterator, const_iterator> equal_range( const Key& key ) const { return internal_equal_range( key, end() ); }

    // Number of items in table.
    size_type size() const { return this->my_size.load(std::memory_order_acquire); }

    // True if size()==0.
    __TBB_nodiscard bool empty() const { return size() == 0; }

    // Upper bound on size.
    size_type max_size() const {
        return allocator_traits_type::max_size(base_type::get_allocator());
    }

    // Returns the current number of buckets
    size_type bucket_count() const { return this->my_mask.load(std::memory_order_relaxed) + 1; }

    // return allocator object
    allocator_type get_allocator() const { return base_type::get_allocator(); }

    // swap two instances. Iterators are invalidated
    void swap(concurrent_hash_map& table) {
        using pocs_type = typename node_allocator_traits::propagate_on_container_swap;
        using is_equal_type = typename node_allocator_traits::is_always_equal;
        swap_allocators(this->my_allocator, table.my_allocator);
        internal_swap(table, tbb::detail::disjunction<pocs_type, is_equal_type>());
    }

    //------------------------------------------------------------------------
    // concurrent map operations
    //------------------------------------------------------------------------

    // Return count of items (0 or 1)
    size_type count( const Key &key ) const {
        return const_cast<concurrent_hash_map*>(this)->lookup(/*insert*/false, key, nullptr, nullptr, /*write=*/false, &do_not_allocate_node );
    }

    // Find item and acquire a read lock on the item.
    /** Return true if item is found, false otherwise. */
    bool find( const_accessor &result, const Key &key ) const {
        result.release();
        return const_cast<concurrent_hash_map*>(this)->lookup(/*insert*/false, key, nullptr, &result, /*write=*/false, &do_not_allocate_node );
    }

    // Find item and acquire a write lock on the item.
    /** Return true if item is found, false otherwise. */
    bool find( accessor &result, const Key &key ) {
        result.release();
        return lookup(/*insert*/false, key, nullptr, &result, /*write=*/true, &do_not_allocate_node );
    }

    // Insert item (if not already present) and acquire a read lock on the item.
    /** Returns true if item is new. */
    bool insert( const_accessor &result, const Key &key ) {
        result.release();
        return lookup(/*insert*/true, key, nullptr, &result, /*write=*/false, &allocate_node_default_construct );
    }

    // Insert item (if not already present) and acquire a write lock on the item.
    /** Returns true if item is new. */
    bool insert( accessor &result, const Key &key ) {
        result.release();
        return lookup(/*insert*/true, key, nullptr, &result, /*write=*/true, &allocate_node_default_construct );
    }

    // Insert item by copying if there is no such key present already and acquire a read lock on the item.
    /** Returns true if item is new. */
    bool insert( const_accessor &result, const value_type &value ) {
        result.release();
        return lookup(/*insert*/true, value.first, &value.second, &result, /*write=*/false, &allocate_node_copy_construct );
    }

    // Insert item by copying if there is no such key present already and acquire a write lock on the item.
    /** Returns true if item is new. */
    bool insert( accessor &result, const value_type &value ) {
        result.release();
        return lookup(/*insert*/true, value.first, &value.second, &result, /*write=*/true, &allocate_node_copy_construct );
    }

    // Insert item by copying if there is no such key present already
    /** Returns true if item is inserted. */
    bool insert( const value_type &value ) {
        return lookup(/*insert*/true, value.first, &value.second, nullptr, /*write=*/false, &allocate_node_copy_construct );
    }

    // Insert item by copying if there is no such key present already and acquire a read lock on the item.
    /** Returns true if item is new. */
    bool insert( const_accessor &result, value_type && value ) {
        return generic_move_insert(result, std::move(value));
    }

    // Insert item by copying if there is no such key present already and acquire a write lock on the item.
    /** Returns true if item is new. */
    bool insert( accessor &result, value_type && value ) {
        return generic_move_insert(result, std::move(value));
    }

    // Insert item by copying if there is no such key present already
    /** Returns true if item is inserted. */
    bool insert( value_type && value ) {
        return generic_move_insert(accessor_not_used(), std::move(value));
    }

    // Insert item by copying if there is no such key present already and acquire a read lock on the item.
    /** Returns true if item is new. */
    template <typename... Args>
    bool emplace( const_accessor &result, Args&&... args ) {
        return generic_emplace(result, std::forward<Args>(args)...);
    }

    // Insert item by copying if there is no such key present already and acquire a write lock on the item.
    /** Returns true if item is new. */
    template <typename... Args>
    bool emplace( accessor &result, Args&&... args ) {
        return generic_emplace(result, std::forward<Args>(args)...);
    }

    // Insert item by copying if there is no such key present already
    /** Returns true if item is inserted. */
    template <typename... Args>
    bool emplace( Args&&... args ) {
        return generic_emplace(accessor_not_used(), std::forward<Args>(args)...);
    }

    // Insert range [first, last)
    template <typename I>
    void insert( I first, I last ) {
        for ( ; first != last; ++first )
            insert( *first );
    }

    // Insert initializer list
    void insert( std::initializer_list<value_type> il ) {
        insert( il.begin(), il.end() );
    }

    // Erase item.
    /** Return true if item was erased by particularly this call. */
    bool erase( const Key &key ) {
        node_base *erase_node;
        hashcode_type const hash = my_hash_compare.hash(key);
        hashcode_type mask = this->my_mask.load(std::memory_order_acquire);
    restart:
        {//lock scope
            // get bucket
            bucket_accessor b( this, hash & mask );
        search:
            node_base* prev = nullptr;
            erase_node = b()->node_list.load(std::memory_order_relaxed);
            while (this->is_valid(erase_node) && !my_hash_compare.equal(key, static_cast<node*>(erase_node)->value().first ) ) {
                prev = erase_node;
                erase_node = erase_node->next;
            }

            if (erase_node == nullptr) { // not found, but mask could be changed
                if (this->check_mask_race(hash, mask))
                    goto restart;
                return false;
            } else if (!b.is_writer() && !b.upgrade_to_writer()) {
                if (this->check_mask_race(hash, mask)) // contended upgrade, check mask
                    goto restart;
                goto search;
            }

            // remove from container
            if (prev == nullptr) {
                b()->node_list.store(erase_node->next, std::memory_order_relaxed);
            } else {
                prev->next = erase_node->next;
            }
            this->my_size--;
        }
        {
            typename node::scoped_type item_locker( erase_node->mutex, /*write=*/true );
        }
        // note: there should be no threads pretending to acquire this mutex again, do not try to upgrade const_accessor!
        delete_node(erase_node); // Only one thread can delete it due to write lock on the bucket
        return true;
    }

    // Erase item by const_accessor.
    /** Return true if item was erased by particularly this call. */
    bool erase( const_accessor& item_accessor ) {
        return exclude( item_accessor );
    }

    // Erase item by accessor.
    /** Return true if item was erased by particularly this call. */
    bool erase( accessor& item_accessor ) {
        return exclude( item_accessor );
    }

protected:
    // Insert or find item and optionally acquire a lock on the item.
    bool lookup( bool op_insert, const Key &key, const T *t, const_accessor *result, bool write, node* (*allocate_node)(bucket_allocator_type&,
        const Key&, const T*), node *tmp_n  = 0)
    {
        __TBB_ASSERT( !result || !result->my_node, nullptr );
        bool return_value;
        hashcode_type const h = my_hash_compare.hash( key );
        hashcode_type m = this->my_mask.load(std::memory_order_acquire);
        segment_index_type grow_segment = 0;
        node *n;
        restart:
        {//lock scope
            __TBB_ASSERT((m&(m+1))==0, "data structure is invalid");
            return_value = false;
            // get bucket
            bucket_accessor b( this, h & m );
            // find a node
            n = search_bucket( key, b() );
            if( op_insert ) {
                // [opt] insert a key
                if( !n ) {
                    if( !tmp_n ) {
                        tmp_n = allocate_node(base_type::get_allocator(), key, t);
                    }
                    if( !b.is_writer() && !b.upgrade_to_writer() ) { // TODO: improved insertion
                        // Rerun search_list, in case another thread inserted the item during the upgrade.
                        n = search_bucket( key, b() );
                        if( this->is_valid(n) ) { // unfortunately, it did
                            b.downgrade_to_reader();
                            goto exists;
                        }
                    }
                    if( this->check_mask_race(h, m) )
                        goto restart; // b.release() is done in ~b().
                    // insert and set flag to grow the container
                    grow_segment = this->insert_new_node( b(), n = tmp_n, m );
                    tmp_n = 0;
                    return_value = true;
                }
            } else { // find or count
                if( !n ) {
                    if( this->check_mask_race( h, m ) )
                        goto restart; // b.release() is done in ~b(). TODO: replace by continue
                    return false;
                }
                return_value = true;
            }
        exists:
            if( !result ) goto check_growth;
            // TODO: the following seems as generic/regular operation
            // acquire the item
            if( !result->try_acquire( n->mutex, write ) ) {
                for( tbb::detail::atomic_backoff backoff(true);; ) {
                    if( result->try_acquire( n->mutex, write ) ) break;
                    if( !backoff.bounded_pause() ) {
                        // the wait takes really long, restart the operation
                        b.release();
                        __TBB_ASSERT( !op_insert || !return_value, "Can't acquire new item in locked bucket?" );
                        yield();
                        m = this->my_mask.load(std::memory_order_acquire);
                        goto restart;
                    }
                }
            }
        }//lock scope
        result->my_node = n;
        result->my_hash = h;
    check_growth:
        // [opt] grow the container
        if( grow_segment ) {
            this->enable_segment( grow_segment );
        }
        if( tmp_n ) // if op_insert only
            delete_node( tmp_n );
        return return_value;
    }

    struct accessor_not_used { void release(){}};
    friend const_accessor* accessor_location( accessor_not_used const& ){ return nullptr;}
    friend const_accessor* accessor_location( const_accessor & a )      { return &a;}

    friend bool is_write_access_needed( accessor const& )           { return true;}
    friend bool is_write_access_needed( const_accessor const& )     { return false;}
    friend bool is_write_access_needed( accessor_not_used const& )  { return false;}

    template <typename Accessor>
    bool generic_move_insert( Accessor && result, value_type && value ) {
        result.release();
        return lookup(/*insert*/true, value.first, &value.second, accessor_location(result), is_write_access_needed(result), &allocate_node_move_construct );
    }

    template <typename Accessor, typename... Args>
    bool generic_emplace( Accessor && result, Args &&... args ) {
        result.release();
        node * node_ptr = create_node(base_type::get_allocator(), std::forward<Args>(args)...);
        return lookup(/*insert*/true, node_ptr->value().first, nullptr, accessor_location(result), is_write_access_needed(result), &do_not_allocate_node, node_ptr );
    }

    // delete item by accessor
    bool exclude( const_accessor &item_accessor ) {
        __TBB_ASSERT( item_accessor.my_node, nullptr );
        node_base *const exclude_node = item_accessor.my_node;
        hashcode_type const hash = item_accessor.my_hash;
        hashcode_type mask = this->my_mask.load(std::memory_order_acquire);
        do {
            // get bucket
            bucket_accessor b( this, hash & mask, /*writer=*/true );
            node_base* prev = nullptr;
            node_base* curr = b()->node_list.load(std::memory_order_relaxed);

            while (curr && curr != exclude_node) {
                prev = curr;
                curr = curr->next;
            }

            if (curr == nullptr) { // someone else was first
                if (this->check_mask_race(hash, mask))
                    continue;
                item_accessor.release();
                return false;
            }
            __TBB_ASSERT( curr == exclude_node, nullptr );
            // remove from container
            if (prev == nullptr) {
                b()->node_list.store(curr->next, std::memory_order_relaxed);
            } else {
                prev->next = curr->next;
            }

            this->my_size--;
            break;
        } while(true);
        if (!item_accessor.is_writer()) { // need to get exclusive lock
            item_accessor.upgrade_to_writer(); // return value means nothing here
        }

        item_accessor.release();
        delete_node(exclude_node); // Only one thread can delete it
        return true;
    }

    // Returns an iterator for an item defined by the key, or for the next item after it (if upper==true)
    template <typename I>
    std::pair<I, I> internal_equal_range( const Key& key, I end_ ) const {
        hashcode_type h = my_hash_compare.hash( key );
        hashcode_type m = this->my_mask.load(std::memory_order_relaxed);
        __TBB_ASSERT((m&(m+1))==0, "data structure is invalid");
        h &= m;
        bucket *b = this->get_bucket( h );
        while ( b->node_list.load(std::memory_order_relaxed) == rehash_req ) {
            m = ( 1u<<tbb::detail::log2( h ) ) - 1; // get parent mask from the topmost bit
            b = this->get_bucket( h &= m );
        }
        node *n = search_bucket( key, b );
        if( !n )
            return std::make_pair(end_, end_);
        iterator lower(*this, h, b, n), upper(lower);
        return std::make_pair(lower, ++upper);
    }

    // Copy "source" to *this, where *this must start out empty.
    void internal_copy( const concurrent_hash_map& source ) {
        hashcode_type mask = source.my_mask.load(std::memory_order_relaxed);
        if( this->my_mask.load(std::memory_order_relaxed) == mask ) { // optimized version
            this->reserve(source.my_size.load(std::memory_order_relaxed)); // TODO: load_factor?
            bucket *dst = 0, *src = 0;
            bool rehash_required = false;
            for( hashcode_type k = 0; k <= mask; k++ ) {
                if( k & (k-2) ) ++dst,src++; // not the beginning of a segment
                else { dst = this->get_bucket( k ); src = source.get_bucket( k ); }
                __TBB_ASSERT( dst->node_list.load(std::memory_order_relaxed) != rehash_req, "Invalid bucket in destination table");
                node *n = static_cast<node*>( src->node_list.load(std::memory_order_relaxed) );
                if( n == rehash_req ) { // source is not rehashed, items are in previous buckets
                    rehash_required = true;
                    dst->node_list.store(rehash_req, std::memory_order_relaxed);
                } else for(; n; n = static_cast<node*>( n->next ) ) {
                    node* node_ptr = create_node(base_type::get_allocator(), n->value().first, n->value().second);
                    this->add_to_bucket( dst, node_ptr);
                    this->my_size.fetch_add(1, std::memory_order_relaxed);
                }
            }
            if( rehash_required ) rehash();
        } else internal_copy(source.begin(), source.end(), source.my_size.load(std::memory_order_relaxed));
    }

    template <typename I>
    void internal_copy( I first, I last, size_type reserve_size ) {
        this->reserve(reserve_size); // TODO: load_factor?
        hashcode_type m = this->my_mask.load(std::memory_order_relaxed);
        for(; first != last; ++first) {
            hashcode_type h = my_hash_compare.hash( (*first).first );
            bucket *b = this->get_bucket( h & m );
            __TBB_ASSERT( b->node_list.load(std::memory_order_relaxed) != rehash_req, "Invalid bucket in destination table");
            node* node_ptr = create_node(base_type::get_allocator(), (*first).first, (*first).second);
            this->add_to_bucket( b, node_ptr );
            ++this->my_size; // TODO: replace by non-atomic op
        }
    }

    void internal_move_construct_with_allocator( concurrent_hash_map&& other, const allocator_type&,
                                                /*is_always_equal=*/std::true_type )
    {
        this->internal_move(std::move(other));
    }

    void internal_move_construct_with_allocator( concurrent_hash_map&& other, const allocator_type& a,
                                                /*is_always_equal=*/std::false_type )
    {
        if (a == other.get_allocator()){
            this->internal_move(std::move(other));
        } else {
            try_call( [&] {
                internal_copy(std::make_move_iterator(other.begin()), std::make_move_iterator(other.end()),
                    other.size());
            }).on_exception( [&] {
                this->clear();
            });
        }
    }

    void internal_move_assign( concurrent_hash_map&& other,
        /*is_always_equal || POCMA = */std::true_type)
    {
        this->internal_move(std::move(other));
    }

    void internal_move_assign(concurrent_hash_map&& other, /*is_always_equal=*/ std::false_type) {
        if (this->my_allocator == other.my_allocator) {
            this->internal_move(std::move(other));
        } else {
            //do per element move
            internal_copy(std::make_move_iterator(other.begin()), std::make_move_iterator(other.end()),
                other.size());
        }
    }

    void internal_swap(concurrent_hash_map& other, /*is_always_equal || POCS = */ std::true_type) {
        this->internal_swap_content(other);
    }

    void internal_swap(concurrent_hash_map& other, /*is_always_equal || POCS = */ std::false_type) {
        __TBB_ASSERT(this->my_allocator == other.my_allocator, nullptr);
        this->internal_swap_content(other);
    }

    // Fast find when no concurrent erasure is used. For internal use inside TBB only!
    /** Return pointer to item with given key, or nullptr if no such item exists.
        Must not be called concurrently with erasure operations. */
    const_pointer internal_fast_find( const Key& key ) const {
        hashcode_type h = my_hash_compare.hash( key );
        hashcode_type m = this->my_mask.load(std::memory_order_acquire);
        node *n;
    restart:
        __TBB_ASSERT((m&(m+1))==0, "data structure is invalid");
        bucket *b = this->get_bucket( h & m );
        // TODO: actually, notification is unnecessary here, just hiding double-check
        if( b->node_list.load(std::memory_order_acquire) == rehash_req )
        {
            typename bucket::scoped_type lock;
            if( lock.try_acquire( b->mutex, /*write=*/true ) ) {
                if( b->node_list.load(std::memory_order_relaxed) == rehash_req)
                    const_cast<concurrent_hash_map*>(this)->rehash_bucket( b, h & m ); //recursive rehashing
            }
            else lock.acquire( b->mutex, /*write=*/false );
            __TBB_ASSERT(b->node_list.load(std::memory_order_relaxed) != rehash_req,nullptr);
        }
        n = search_bucket( key, b );
        if( n )
            return n->storage();
        else if( this->check_mask_race( h, m ) )
            goto restart;
        return 0;
    }
};

#if __TBB_CPP17_DEDUCTION_GUIDES_PRESENT
template <typename It,
          typename HashCompare = tbb_hash_compare<iterator_key_t<It>>,
          typename Alloc = tbb_allocator<iterator_alloc_pair_t<It>>,
          typename = std::enable_if_t<is_input_iterator_v<It>>,
          typename = std::enable_if_t<is_allocator_v<Alloc>>,
          typename = std::enable_if_t<!is_allocator_v<HashCompare>>>
concurrent_hash_map( It, It, HashCompare = HashCompare(), Alloc = Alloc() )
-> concurrent_hash_map<iterator_key_t<It>, iterator_mapped_t<It>, HashCompare, Alloc>;

template <typename It, typename Alloc,
          typename = std::enable_if_t<is_input_iterator_v<It>>,
          typename = std::enable_if_t<is_allocator_v<Alloc>>>
concurrent_hash_map( It, It, Alloc )
-> concurrent_hash_map<iterator_key_t<It>, iterator_mapped_t<It>, tbb_hash_compare<iterator_key_t<It>>, Alloc>;

template <typename Key, typename T,
          typename HashCompare = tbb_hash_compare<std::remove_const_t<Key>>,
          typename Alloc = tbb_allocator<std::pair<const Key, T>>,
          typename = std::enable_if_t<is_allocator_v<Alloc>>,
          typename = std::enable_if_t<!is_allocator_v<HashCompare>>>
concurrent_hash_map( std::initializer_list<std::pair<Key, T>>, HashCompare = HashCompare(), Alloc = Alloc() )
-> concurrent_hash_map<std::remove_const_t<Key>, T, HashCompare, Alloc>;

template <typename Key, typename T, typename Alloc,
          typename = std::enable_if_t<is_allocator_v<Alloc>>>
concurrent_hash_map( std::initializer_list<std::pair<Key, T>>, Alloc )
-> concurrent_hash_map<std::remove_const_t<Key>, T, tbb_hash_compare<std::remove_const_t<Key>>, Alloc>;

#endif /* __TBB_CPP17_DEDUCTION_GUIDES_PRESENT */

template <typename Key, typename T, typename HashCompare, typename A1, typename A2>
inline bool operator==(const concurrent_hash_map<Key, T, HashCompare, A1> &a, const concurrent_hash_map<Key, T, HashCompare, A2> &b) {
    if(a.size() != b.size()) return false;
    typename concurrent_hash_map<Key, T, HashCompare, A1>::const_iterator i(a.begin()), i_end(a.end());
    typename concurrent_hash_map<Key, T, HashCompare, A2>::const_iterator j, j_end(b.end());
    for(; i != i_end; ++i) {
        j = b.equal_range(i->first).first;
        if( j == j_end || !(i->second == j->second) ) return false;
    }
    return true;
}

#if !__TBB_CPP20_COMPARISONS_PRESENT
template <typename Key, typename T, typename HashCompare, typename A1, typename A2>
inline bool operator!=(const concurrent_hash_map<Key, T, HashCompare, A1> &a, const concurrent_hash_map<Key, T, HashCompare, A2> &b)
{    return !(a == b); }
#endif // !__TBB_CPP20_COMPARISONS_PRESENT

template <typename Key, typename T, typename HashCompare, typename A>
inline void swap(concurrent_hash_map<Key, T, HashCompare, A> &a, concurrent_hash_map<Key, T, HashCompare, A> &b)
{    a.swap( b ); }

} // namespace d1
} // namespace detail

inline namespace v1 {
    using detail::split;
    using detail::d1::concurrent_hash_map;
    using detail::d1::tbb_hash_compare;
} // namespace v1

} // namespace tbb

#endif /* __TBB_concurrent_hash_map_H */
