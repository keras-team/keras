/*
    Copyright (c) 2019-2021 Intel Corporation

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

#ifndef __TBB_detail__node_handle_H
#define __TBB_detail__node_handle_H

#include "_allocator_traits.h"
#include "_assert.h"

namespace tbb {
namespace detail {
namespace d1 {

// A structure to access private node handle methods in internal TBB classes
// Regular friend declaration is not convenient because classes which use node handle
// can be placed in the different versioning namespaces.
struct node_handle_accessor {
    template <typename NodeHandleType>
    static typename NodeHandleType::node* get_node_ptr( NodeHandleType& nh ) {
        return nh.get_node_ptr();
    }

    template <typename NodeHandleType>
    static NodeHandleType construct( typename NodeHandleType::node* node_ptr ) {
        return NodeHandleType{node_ptr};
    }

    template <typename NodeHandleType>
    static void deactivate( NodeHandleType& nh ) {
        nh.deactivate();
    }
}; // struct node_handle_accessor

template<typename Value, typename Node, typename Allocator>
class node_handle_base {
public:
    using allocator_type = Allocator;
protected:
    using node = Node;
    using allocator_traits_type = tbb::detail::allocator_traits<allocator_type>;
public:

    node_handle_base() : my_node(nullptr), my_allocator() {}
    node_handle_base(node_handle_base&& nh) : my_node(nh.my_node),
                                              my_allocator(std::move(nh.my_allocator)) {
        nh.my_node = nullptr;
    }

    __TBB_nodiscard bool empty() const { return my_node == nullptr; }
    explicit operator bool() const { return my_node != nullptr; }

    ~node_handle_base() { internal_destroy(); }

    node_handle_base& operator=( node_handle_base&& nh ) {
        internal_destroy();
        my_node = nh.my_node;
        move_assign_allocators(my_allocator, nh.my_allocator);
        nh.deactivate();
        return *this;
    }

    void swap( node_handle_base& nh ) {
        using std::swap;
        swap(my_node, nh.my_node);
        swap_allocators(my_allocator, nh.my_allocator);
    }

    allocator_type get_allocator() const {
        return my_allocator;
    }

protected:
    node_handle_base( node* n ) : my_node(n) {}

    void internal_destroy() {
        if(my_node != nullptr) {
            allocator_traits_type::destroy(my_allocator, my_node->storage());
            typename allocator_traits_type::template rebind_alloc<node> node_allocator(my_allocator);
            node_allocator.deallocate(my_node, 1);
        }
    }

    node* get_node_ptr() { return my_node; }

    void deactivate() { my_node = nullptr; }

    node* my_node;
    allocator_type my_allocator;
};

// node handle for maps
template<typename Key, typename Value, typename Node, typename Allocator>
class node_handle : public node_handle_base<Value, Node, Allocator> {
    using base_type = node_handle_base<Value, Node, Allocator>;
public:
    using key_type = Key;
    using mapped_type = typename Value::second_type;
    using allocator_type = typename base_type::allocator_type;

    node_handle() = default;

    key_type& key() const {
        __TBB_ASSERT(!this->empty(), "Cannot get key from the empty node_type object");
        return *const_cast<key_type*>(&(this->my_node->value().first));
    }

    mapped_type& mapped() const {
        __TBB_ASSERT(!this->empty(), "Cannot get mapped value from the empty node_type object");
        return this->my_node->value().second;
    }

private:
    friend struct node_handle_accessor;

    node_handle( typename base_type::node* n ) : base_type(n) {}
}; // class node_handle

// node handle for sets
template<typename Key, typename Node, typename Allocator>
class node_handle<Key, Key, Node, Allocator> : public node_handle_base<Key, Node, Allocator> {
    using base_type = node_handle_base<Key, Node, Allocator>;
public:
    using value_type = Key;
    using allocator_type = typename base_type::allocator_type;

    node_handle() = default;

    value_type& value() const {
        __TBB_ASSERT(!this->empty(), "Cannot get value from the empty node_type object");
        return *const_cast<value_type*>(&(this->my_node->value()));
    }

private:
    friend struct node_handle_accessor;

    node_handle( typename base_type::node* n ) : base_type(n) {}
}; // class node_handle

template <typename Key, typename Value, typename Node, typename Allocator>
void swap( node_handle<Key, Value, Node, Allocator>& lhs,
           node_handle<Key, Value, Node, Allocator>& rhs ) {
    return lhs.swap(rhs);
}

} // namespace d1
} // namespace detail
} // namespace tbb

#endif // __TBB_detail__node_handle_H
