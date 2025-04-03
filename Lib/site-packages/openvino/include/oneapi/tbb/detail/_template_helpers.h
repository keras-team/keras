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

#ifndef __TBB_detail__template_helpers_H
#define __TBB_detail__template_helpers_H

#include "_utils.h"
#include "_config.h"

#include <cstddef>
#include <cstdint>

#include <type_traits>
#include <memory>
#include <iterator>

namespace tbb {
namespace detail {
inline namespace d0 {

// An internal implementation of void_t, which can be used in SFINAE contexts
template <typename...>
struct void_impl {
    using type = void;
}; // struct void_impl

template <typename... Args>
using void_t = typename void_impl<Args...>::type;

// Generic SFINAE helper for expression checks, based on the idea demonstrated in ISO C++ paper n4502
template <typename T, typename, template <typename> class... Checks>
struct supports_impl {
    using type = std::false_type;
};

template <typename T, template <typename> class... Checks>
struct supports_impl<T, void_t<Checks<T>...>, Checks...> {
    using type = std::true_type;
};

template <typename T, template <typename> class... Checks>
using supports = typename supports_impl<T, void, Checks...>::type;

//! A template to select either 32-bit or 64-bit constant as compile time, depending on machine word size.
template <unsigned u, unsigned long long ull >
struct select_size_t_constant {
    // Explicit cast is needed to avoid compiler warnings about possible truncation.
    // The value of the right size,   which is selected by ?:, is anyway not truncated or promoted.
    static const std::size_t value = (std::size_t)((sizeof(std::size_t)==sizeof(u)) ? u : ull);
};

// TODO: do we really need it?
//! Cast between unrelated pointer types.
/** This method should be used sparingly as a last resort for dealing with
  situations that inherently break strict ISO C++ aliasing rules. */
// T is a pointer type because it will be explicitly provided by the programmer as a template argument;
// U is a referent type to enable the compiler to check that "ptr" is a pointer, deducing U in the process.
template<typename T, typename U>
inline T punned_cast( U* ptr ) {
    std::uintptr_t x = reinterpret_cast<std::uintptr_t>(ptr);
    return reinterpret_cast<T>(x);
}

template<class T, size_t S, size_t R>
struct padded_base : T {
    char pad[S - R];
};
template<class T, size_t S> struct padded_base<T, S, 0> : T {};

//! Pads type T to fill out to a multiple of cache line size.
template<class T, size_t S = max_nfs_size>
struct padded : padded_base<T, S, sizeof(T) % S> {};

#if __TBB_CPP14_INTEGER_SEQUENCE_PRESENT

using std::index_sequence;
using std::make_index_sequence;

#else

template<std::size_t... S> class index_sequence {};

template<std::size_t N, std::size_t... S>
struct make_index_sequence_impl : make_index_sequence_impl < N - 1, N - 1, S... > {};

template<std::size_t... S>
struct make_index_sequence_impl <0, S...> {
    using type = index_sequence<S...>;
};

template<std::size_t N>
using make_index_sequence = typename make_index_sequence_impl<N>::type;

#endif /* __TBB_CPP14_INTEGER_SEQUENCE_PRESENT */

#if __TBB_CPP17_LOGICAL_OPERATIONS_PRESENT
using std::conjunction;
using std::disjunction;
#else // __TBB_CPP17_LOGICAL_OPERATIONS_PRESENT

template <typename...>
struct conjunction : std::true_type {};

template <typename First, typename... Args>
struct conjunction<First, Args...>
    : std::conditional<bool(First::value), conjunction<Args...>, First>::type {};

template <typename T>
struct conjunction<T> : T {};

template <typename...>
struct disjunction : std::false_type {};

template <typename First, typename... Args>
struct disjunction<First, Args...>
    : std::conditional<bool(First::value), First, disjunction<Args...>>::type {};

template <typename T>
struct disjunction<T> : T {};

#endif // __TBB_CPP17_LOGICAL_OPERATIONS_PRESENT

template <typename Iterator>
using iterator_value_t = typename std::iterator_traits<Iterator>::value_type;

template <typename Iterator>
using iterator_key_t = typename std::remove_const<typename iterator_value_t<Iterator>::first_type>::type;

template <typename Iterator>
using iterator_mapped_t = typename iterator_value_t<Iterator>::second_type;

template <typename Iterator>
using iterator_alloc_pair_t = std::pair<typename std::add_const<iterator_key_t<Iterator>>::type,
                                        iterator_mapped_t<Iterator>>;

template <typename A> using alloc_value_type = typename A::value_type;
template <typename A> using alloc_ptr_t = typename std::allocator_traits<A>::pointer;
template <typename A> using has_allocate = decltype(std::declval<alloc_ptr_t<A>&>() = std::declval<A>().allocate(0));
template <typename A> using has_deallocate = decltype(std::declval<A>().deallocate(std::declval<alloc_ptr_t<A>>(), 0));

// alloc_value_type should be checked first, because it can be used in other checks
template <typename T>
using is_allocator = supports<T, alloc_value_type, has_allocate, has_deallocate>;

#if __TBB_CPP17_DEDUCTION_GUIDES_PRESENT
template <typename T>
inline constexpr bool is_allocator_v = is_allocator<T>::value;
#endif

// Template class in which the "type" determines the type of the element number N in pack Args
template <std::size_t N, typename... Args>
struct pack_element {
    using type = void;
};

template <std::size_t N, typename T, typename... Args>
struct pack_element<N, T, Args...> {
    using type = typename pack_element<N-1, Args...>::type;
};

template <typename T, typename... Args>
struct pack_element<0, T, Args...> {
    using type = T;
};

template <std::size_t N, typename... Args>
using pack_element_t = typename pack_element<N, Args...>::type;

template <typename Func>
class raii_guard {
public:
    raii_guard( Func f ) : my_func(f), is_active(true) {}

    ~raii_guard() {
        if (is_active) {
            my_func();
        }
    }

    void dismiss() {
        is_active = false;
    }
private:
    Func my_func;
    bool is_active;
}; // class raii_guard

template <typename Func>
raii_guard<Func> make_raii_guard( Func f ) {
    return raii_guard<Func>(f);
}

template <typename Body>
struct try_call_proxy {
    try_call_proxy( Body b ) : body(b) {}

    template <typename OnExceptionBody>
    void on_exception( OnExceptionBody on_exception_body ) {
        auto guard = make_raii_guard(on_exception_body);
        body();
        guard.dismiss();
    }

    template <typename OnCompletionBody>
    void on_completion(OnCompletionBody on_completion_body) {
        auto guard = make_raii_guard(on_completion_body);
        body();
    }

    Body body;
}; // struct try_call_proxy

// Template helper function for API
// try_call(lambda1).on_exception(lambda2)
// Executes lambda1 and if it throws an exception - executes lambda2
template <typename Body>
try_call_proxy<Body> try_call( Body b ) {
    return try_call_proxy<Body>(b);
}

#if __TBB_CPP17_IS_SWAPPABLE_PRESENT
using std::is_nothrow_swappable;
using std::is_swappable;
#else // __TBB_CPP17_IS_SWAPPABLE_PRESENT
namespace is_swappable_detail {
using std::swap;

template <typename T>
using has_swap = decltype(swap(std::declval<T&>(), std::declval<T&>()));

#if _MSC_VER && _MSC_VER <= 1900 && !__INTEL_COMPILER
// Workaround for VS2015: it fails to instantiate noexcept(...) inside std::integral_constant.
template <typename T>
struct noexcept_wrapper {
    static const bool value = noexcept(swap(std::declval<T&>(), std::declval<T&>()));
};
template <typename T>
struct is_nothrow_swappable_impl : std::integral_constant<bool, noexcept_wrapper<T>::value> {};
#else
template <typename T>
struct is_nothrow_swappable_impl : std::integral_constant<bool, noexcept(swap(std::declval<T&>(), std::declval<T&>()))> {};
#endif
}

template <typename T>
struct is_swappable : supports<T, is_swappable_detail::has_swap> {};

template <typename T>
struct is_nothrow_swappable
    : conjunction<is_swappable<T>, is_swappable_detail::is_nothrow_swappable_impl<T>> {};
#endif // __TBB_CPP17_IS_SWAPPABLE_PRESENT

//! Allows to store a function parameter pack as a variable and later pass it to another function
template< typename... Types >
struct stored_pack;

template<>
struct stored_pack<>
{
    using pack_type = stored_pack<>;
    stored_pack() {}

    // Friend front-end functions
    template< typename F, typename Pack > friend void call(F&& f, Pack&& p);
    template< typename Ret, typename F, typename Pack > friend Ret call_and_return(F&& f, Pack&& p);

protected:
    // Ideally, ref-qualified non-static methods would be used,
    // but that would greatly reduce the set of compilers where it works.
    template< typename Ret, typename F, typename... Preceding >
    static Ret call(F&& f, const pack_type& /*pack*/, Preceding&&... params) {
        return std::forward<F>(f)(std::forward<Preceding>(params)...);
    }
    template< typename Ret, typename F, typename... Preceding >
    static Ret call(F&& f, pack_type&& /*pack*/, Preceding&&... params) {
        return std::forward<F>(f)(std::forward<Preceding>(params)...);
    }
};

template< typename T, typename... Types >
struct stored_pack<T, Types...> : stored_pack<Types...>
{
    using pack_type = stored_pack<T, Types...>;
    using pack_remainder = stored_pack<Types...>;

    // Since lifetime of original values is out of control, copies should be made.
    // Thus references should be stripped away from the deduced type.
    typename std::decay<T>::type leftmost_value;

    // Here rvalue references act in the same way as forwarding references,
    // as long as class template parameters were deduced via forwarding references.
    stored_pack(T&& t, Types&&... types)
    : pack_remainder(std::forward<Types>(types)...), leftmost_value(std::forward<T>(t)) {}

    // Friend front-end functions
    template< typename F, typename Pack > friend void call(F&& f, Pack&& p);
    template< typename Ret, typename F, typename Pack > friend Ret call_and_return(F&& f, Pack&& p);

protected:
    template< typename Ret, typename F, typename... Preceding >
    static Ret call(F&& f, pack_type& pack, Preceding&&... params) {
        return pack_remainder::template call<Ret>(
            std::forward<F>(f), static_cast<pack_remainder&>(pack),
            std::forward<Preceding>(params)... , pack.leftmost_value
        );
    }

    template< typename Ret, typename F, typename... Preceding >
    static Ret call(F&& f, pack_type&& pack, Preceding&&... params) {
        return pack_remainder::template call<Ret>(
            std::forward<F>(f), static_cast<pack_remainder&&>(pack),
            std::forward<Preceding>(params)... , std::move(pack.leftmost_value)
        );
    }
};

//! Calls the given function with arguments taken from a stored_pack
template< typename F, typename Pack >
void call(F&& f, Pack&& p) {
    std::decay<Pack>::type::template call<void>(std::forward<F>(f), std::forward<Pack>(p));
}

template< typename Ret, typename F, typename Pack >
Ret call_and_return(F&& f, Pack&& p) {
    return std::decay<Pack>::type::template call<Ret>(std::forward<F>(f), std::forward<Pack>(p));
}

template< typename... Types >
stored_pack<Types...> save_pack(Types&&... types) {
    return stored_pack<Types...>(std::forward<Types>(types)...);
}

// A structure with the value which is equal to Trait::value
// but can be used in the immediate context due to parameter T
template <typename Trait, typename T>
struct dependent_bool : std::integral_constant<bool, bool(Trait::value)> {};

template <typename Callable>
struct body_arg_detector;

template <typename Callable, typename ReturnType, typename Arg>
struct body_arg_detector<ReturnType(Callable::*)(Arg)> {
    using arg_type = Arg;
};

template <typename Callable, typename ReturnType, typename Arg>
struct body_arg_detector<ReturnType(Callable::*)(Arg) const> {
    using arg_type = Arg;
};

template <typename Callable>
struct argument_detector;

template <typename Callable>
struct argument_detector {
    using type = typename body_arg_detector<decltype(&Callable::operator())>::arg_type;
};

template <typename ReturnType, typename Arg>
struct argument_detector<ReturnType(*)(Arg)> {
    using type = Arg;
};

// Detects the argument type of callable, works for callable with one argument.
template <typename Callable>
using argument_type_of = typename argument_detector<typename std::decay<Callable>::type>::type;

template <typename T>
struct type_identity {
    using type = T;
};

template <typename T>
using type_identity_t = typename type_identity<T>::type;

} // inline namespace d0
} // namespace detail
} // namespace tbb

#endif // __TBB_detail__template_helpers_H

