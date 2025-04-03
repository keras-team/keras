// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <string>

#include "openvino/core/except.hpp"
#include "openvino/frontend/visibility.hpp"

namespace ov {
namespace frontend {
class FRONTEND_API GeneralFailure : public AssertFailure {
public:
    [[noreturn]] static void create(const char* file,
                                    int line,
                                    const char* check_string,
                                    const std::string& context_info,
                                    const std::string& explanation);

protected:
    explicit GeneralFailure(const std::string& what_arg) : ov::AssertFailure(what_arg) {}
};

class FRONTEND_API InitializationFailure : public AssertFailure {
public:
    [[noreturn]] static void create(const char* file,
                                    int line,
                                    const char* check_string,
                                    const std::string& context_info,
                                    const std::string& explanation);

protected:
    explicit InitializationFailure(const std::string& what_arg) : ov::AssertFailure(what_arg) {}
};

class FRONTEND_API OpValidationFailure : public AssertFailure {
public:
    [[noreturn]] static void create(const char* file,
                                    int line,
                                    const char* check_string,
                                    const std::string& context_info,
                                    const std::string& explanation);

protected:
    explicit OpValidationFailure(const std::string& what_arg) : ov::AssertFailure(what_arg) {}
};

class FRONTEND_API OpConversionFailure : public AssertFailure {
public:
    [[noreturn]] static void create(const char* file,
                                    int line,
                                    const char* check_string,
                                    const std::string& context_info,
                                    const std::string& explanation);

protected:
    explicit OpConversionFailure(const std::string& what_arg) : ov::AssertFailure(what_arg) {}
};

class FRONTEND_API NotImplementedFailure : public AssertFailure {
public:
    [[noreturn]] static void create(const char* file,
                                    int line,
                                    const char* check_string,
                                    const std::string& context_info,
                                    const std::string& explanation);

protected:
    explicit NotImplementedFailure(const std::string& what_arg) : ov::AssertFailure(what_arg) {}
};

/// \brief Macro to check whether a boolean condition holds.
/// \param cond Condition to check
/// \param ... Additional error message info to be added to the error message via the `<<`
///            stream-insertion operator. Note that the expressions here will be evaluated lazily,
///            i.e., only if the `cond` evalutes to `false`.
/// \throws ::ov::frontend::GeneralFailure if `cond` is false.
#define FRONT_END_GENERAL_CHECK(...) OPENVINO_ASSERT_HELPER(::ov::frontend::GeneralFailure, "", __VA_ARGS__)

/// \brief Macro to check whether a boolean condition holds.
/// \param cond Condition to check
/// \param ... Additional error message info to be added to the error message via the `<<`
///            stream-insertion operator. Note that the expressions here will be evaluated lazily,
///            i.e., only if the `cond` evalutes to `false`.
/// \throws ::ov::frontend::InitializationFailure if `cond` is false.
#define FRONT_END_INITIALIZATION_CHECK(...) \
    OPENVINO_ASSERT_HELPER(::ov::frontend::InitializationFailure, "", __VA_ARGS__)

/// \brief Macro to check whether a boolean condition holds.
/// \param cond Condition to check
/// \param ... Additional error message info to be added to the error message via the `<<`
///            stream-insertion operator. Note that the expressions here will be evaluated lazily,
///            i.e., only if the `cond` evalutes to `false`.
/// \throws ::ov::frontend::OpConversionFailure if `cond` is false.
#define FRONT_END_OP_CONVERSION_CHECK(...) OPENVINO_ASSERT_HELPER(::ov::frontend::OpConversionFailure, "", __VA_ARGS__)

/// \brief Assert macro.
/// \param NAME Name of the function that is not implemented
/// \throws ::ov::frontend::NotImplementedFailure
#define FRONT_END_NOT_IMPLEMENTED(NAME)                           \
    OPENVINO_ASSERT_HELPER(::ov::frontend::NotImplementedFailure, \
                           "",                                    \
                           false,                                 \
                           #NAME " is not implemented for this FrontEnd class")

/// \brief Assert macro.
/// \param COND Condition. If 'false', throws 'NotImplementedFailure'
/// \param NAME Name of the function that is not implemented
/// \throws ::ov::frontend::NotImplementedFailure
#define FRONT_END_CHECK_IMPLEMENTED(COND, NAME)                   \
    OPENVINO_ASSERT_HELPER(::ov::frontend::NotImplementedFailure, \
                           "",                                    \
                           (COND),                                \
                           #NAME " is not implemented for this FrontEnd class")

/// \brief Assert macro.
/// \param MSG Error message
/// \throws ::ov::frontend::GeneralFailure
#define FRONT_END_THROW(MSG) FRONT_END_GENERAL_CHECK(false, MSG)

}  // namespace frontend
}  // namespace ov
