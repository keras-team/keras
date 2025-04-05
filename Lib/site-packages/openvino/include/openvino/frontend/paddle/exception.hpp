// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/frontend/exception.hpp"

namespace ov {
namespace frontend {
namespace paddle {

class NodeContext;

class OpValidationFailure : public ov::frontend::OpValidationFailure {
public:
    [[noreturn]] static void create(const char* file,
                                    int line,
                                    const char* check_string,
                                    const NodeContext& node,
                                    const std::string& explanation);

protected:
    explicit OpValidationFailure(const std::string& what_arg) : ov::frontend::OpValidationFailure(what_arg) {}

private:
    static std::string get_error_msg_prefix_paddle(const NodeContext& node);
};
}  // namespace paddle
}  // namespace frontend

/// \brief Macro to check whether a boolean condition holds.
/// \param node_context Object of NodeContext class
/// \param cond Condition to check
/// \param ... Additional error message info to be added to the error message via the `<<`
///            stream-insertion operator. Note that the expressions here will be evaluated lazily,
///            i.e., only if the `cond` evaluates to `false`.
/// \throws ::ov::OpValidationFailure if `cond` is false.
#define PADDLE_OP_CHECK(node_context, ...) \
    OPENVINO_ASSERT_HELPER(::ov::frontend::paddle::OpValidationFailure, (node_context), __VA_ARGS__)
}  // namespace ov
