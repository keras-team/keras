// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <iostream>
#include <sstream>
#include <stdexcept>

#include "openvino/core/core_visibility.hpp"
#include "openvino/core/deprecated.hpp"

namespace ov {

/// Base error for ov runtime errors.
class OPENVINO_API Exception : public std::runtime_error {
public:
    [[noreturn]] static void create(const char* file, int line, const std::string& explanation);
    virtual ~Exception();

    static const std::string default_msg;

protected:
    explicit Exception(const std::string& what_arg);

    static std::string make_what(const char* file,
                                 int line,
                                 const char* check_string,
                                 const std::string& context_info,
                                 const std::string& explanation);
};

static inline std::ostream& write_all_to_stream(std::ostream& str) {
    return str;
}

template <typename T, typename... TS>
std::ostream& write_all_to_stream(std::ostream& str, T&& arg, TS&&... args) {
    return write_all_to_stream(str << arg, std::forward<TS>(args)...);
}

template <class T,
          typename std::enable_if<!std::is_same<typename std::decay<T>::type, std::string>::value>::type* = nullptr>
std::string stringify(T&& arg) {
    std::stringstream stream;
    stream << arg;
    return stream.str();
}

template <class T,
          typename std::enable_if<std::is_same<typename std::decay<T>::type, std::string>::value>::type* = nullptr>
T& stringify(T&& arg) {
    return arg;
}

/// Base class for check failure exceptions.
class OPENVINO_API AssertFailure : public Exception {
public:
    [[noreturn]] static void create(const char* file,
                                    int line,
                                    const char* check_string,
                                    const std::string& context_info,
                                    const std::string& explanation);
    virtual ~AssertFailure();

protected:
    explicit AssertFailure(const std::string& what_arg) : ov::Exception(what_arg) {}
};

/// Exception class to be thrown on not implemented code
class OPENVINO_API NotImplemented : public AssertFailure {
public:
    [[noreturn]] static void create(const char* file, int line, const std::string& explanation);
    virtual ~NotImplemented();

    static const std::string default_msg;

protected:
    explicit NotImplemented(const std::string& what_arg) : ov::AssertFailure(what_arg) {}
};
}  // namespace ov

//
// Helper macro for defining custom check macros, which throw custom exception classes and provide
// useful context information (the check condition, source filename, line number, and any domain-
// specific context information [e.g., a summary of the node that was being processed at the time
// of the check]).
//
// For example (actually implemented in node.cpp), let's say we want to define a macro for
// checking conditions during node validation, usable as follows:
//
//    NODE_VALIDATION_CHECK(node_being_checked,
//                          node_being_checked->get_input_shape(0).size() == 1,
//                          "Node must have an input rank of 1, but got ",
//                          node_being_checked->get_input_shape(0).size(), ".");
//
// In case of failure, this will throw an exception of type NodeValidationFailure with a what()
// string something like:
//
//      Check 'node_being_checked->get_input_shape(0).size() == 1' failed at foo.cpp:123:
//      While validating node 'Broadcast[Broadcast_10](Reshape_9: float{1,3,4,5}) -> (??)':
//      Node must have an input of rank 1, but got 2.
//
// To implement this, he first step is to define a subclass of AssertFailure (let's say it's called
// MyFailure), which must have a constructor of the form:
//
//      MyFailure(const CheckLocInfo& check_loc_info,
//                T context_info, // "T" can be any type; you'll supply a function to convert "T"
//                                // to std::string
//                const std::string& explanation)
//
// Here, we define a custom class for node validation failures as follows:
//
//    static std::string node_validation_failure_loc_string(const Node* node)
//    {
//        std::stringstream ss;
//        ss << "While validating node '" << *node << "'";
//        return ss.str();
//    }
//
//    class NodeValidationFailure : public AssertFailure
//    {
//    public:
//        NodeValidationFailure(const CheckLocInfo& check_loc_info,
//                              const Node* node,
//                              const std::string& explanation)
//            : AssertFailure(check_loc_info, node_validation_failure_loc_string(node), explanation)
//        {
//        }
//    };
//
// Then, we define the macro NODE_VALIDATION_CHECK as follows:
//
// #define NODE_VALIDATION_CHECK(node, cond, ...) <backslash>
//     OPENVINO_ASSERT_HELPER(::ov::NodeValidationFailure, (node), (cond), ##__VA_ARGS__)
//
// The macro NODE_VALIDATION_CHECK can now be called on any condition, with a Node* pointer
// supplied to generate an informative error message via node_validation_failure_loc_string().
//
// Take care to fully qualify the exception class name in the macro body.
//
// The "..." may be filled with expressions of any type that has an "operator<<" overload for
// insertion into std::ostream.
//
#define OPENVINO_ASSERT_HELPER2(exc_class, ctx, check, ...)                      \
    do {                                                                         \
        if (!(check)) {                                                          \
            ::std::ostringstream ss___;                                          \
            ::ov::write_all_to_stream(ss___, __VA_ARGS__);                       \
            exc_class::create(__FILE__, __LINE__, (#check), (ctx), ss___.str()); \
        }                                                                        \
    } while (0)

#define OPENVINO_ASSERT_HELPER1(exc_class, ctx, check)                                      \
    do {                                                                                    \
        if (!(check)) {                                                                     \
            exc_class::create(__FILE__, __LINE__, (#check), (ctx), exc_class::default_msg); \
        }                                                                                   \
    } while (0)

#define OPENVINO_ASSERT_HELPER(exc_class, ctx, ...) CALL_OVERLOAD(OPENVINO_ASSERT_HELPER, exc_class, ctx, __VA_ARGS__)

// Helper macros for OPENVINO_THROW which is special case of OPENVINO_ASSERT_HELPER without some not required
// parameters for ov::Exception, as result reduce binary size.
#define OPENVINO_THROW_HELPER2(exc_class, ctx, ...)         \
    do {                                                    \
        ::std::ostringstream ss___;                         \
        ::ov::write_all_to_stream(ss___, __VA_ARGS__);      \
        exc_class::create(__FILE__, __LINE__, ss___.str()); \
    } while (0)

#define OPENVINO_THROW_HELPER1(exc_class, ctx, explanation)                  \
    do {                                                                     \
        exc_class::create(__FILE__, __LINE__, ::ov::stringify(explanation)); \
    } while (0)

#define OPENVINO_THROW_HELPER(exc_class, ctx, ...) CALL_OVERLOAD(OPENVINO_THROW_HELPER, exc_class, ctx, __VA_ARGS__)

/// \brief Macro to check whether a boolean condition holds.
/// \param cond Condition to check
/// \param ... Additional error message info to be added to the error message via the `<<`
///            stream-insertion operator. Note that the expressions here will be evaluated lazily,
///            i.e., only if the `cond` evaluates to `false`.
/// \throws ::ov::AssertFailure if `cond` is false.
#define OPENVINO_ASSERT(...) OPENVINO_ASSERT_HELPER(::ov::AssertFailure, ::ov::AssertFailure::default_msg, __VA_ARGS__)

/// \brief Macro to signal a code path that is unreachable in a successful execution. It's
/// implemented with OPENVINO_ASSERT macro.
/// \param ... Additional error message that should describe why that execution path is unreachable.
/// \throws ::ov::Exception if the macro is executed.
#define OPENVINO_THROW(...) OPENVINO_THROW_HELPER(::ov::Exception, ov::Exception::default_msg, __VA_ARGS__)

#define OPENVINO_THROW_NOT_IMPLEMENTED(...) \
    OPENVINO_THROW_HELPER(::ov::NotImplemented, ::ov::Exception::default_msg, __VA_ARGS__)

#define OPENVINO_NOT_IMPLEMENTED \
    OPENVINO_THROW_HELPER(::ov::NotImplemented, ::ov::Exception::default_msg, ::ov::Exception::default_msg)

#define GLUE(x, y) x y

#define RETURN_ARG_COUNT(_1_,   \
                         _2_,   \
                         _3_,   \
                         _4_,   \
                         _5_,   \
                         _6,    \
                         _7,    \
                         _8,    \
                         _9,    \
                         _10,   \
                         _11,   \
                         _12,   \
                         _13,   \
                         _14,   \
                         _15,   \
                         _16,   \
                         _17,   \
                         _18,   \
                         _19,   \
                         _20,   \
                         _21,   \
                         _22,   \
                         _23,   \
                         _24,   \
                         _25,   \
                         count, \
                         ...)   \
    count
#define EXPAND_ARGS(args) RETURN_ARG_COUNT args
#define COUNT_ARGS_MAXN(...) \
    EXPAND_ARGS((__VA_ARGS__, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 0))

#define OVERLOAD_MACRO2(name, count) name##count
#define OVERLOAD_MACRO1(name, count) OVERLOAD_MACRO2(name, count)
#define OVERLOAD_MACRO(name, count)  OVERLOAD_MACRO1(name, count)

#define CALL_OVERLOAD(name, exc_class, ctx, ...) \
    GLUE(OVERLOAD_MACRO(name, COUNT_ARGS_MAXN(__VA_ARGS__)), (exc_class, ctx, __VA_ARGS__))
