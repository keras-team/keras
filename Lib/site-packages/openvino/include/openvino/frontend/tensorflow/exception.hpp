// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include "openvino/frontend/exception.hpp"

namespace ov {
/// \brief Macro to check whether a boolean condition holds.
/// \param node_context Object of NodeContext class
/// \param cond Condition to check
/// \param ... Additional error message info to be added to the error message via the `<<`
///            stream-insertion operator. Note that the expressions here will be evaluated lazily,
///            i.e., only if the `cond` evalutes to `false`.
/// \throws ::ov::OpValidationFailure if `cond` is false.

#ifndef TENSORFLOW_OP_VALIDATION
#    define TENSORFLOW_OP_VALIDATION(node_context, ...)                                        \
        OPENVINO_ASSERT_HELPER(::ov::frontend::OpValidationFailure,                            \
                               ("While validating node '" + node_context.get_op_type() + "'"), \
                               __VA_ARGS__)
#endif
}  // namespace ov
