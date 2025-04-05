// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/core/except.hpp"
#include "openvino/runtime/common.hpp"

namespace ov {

/**
 * @brief Thrown in case of cancelled asynchronous operation.
 * @ingroup ov_runtime_cpp_api
 */
class OPENVINO_RUNTIME_API Cancelled : public Exception {
public:
    [[noreturn]] static void create(const std::string& explanation);
    ~Cancelled() override;

protected:
    OPENVINO_SUPPRESS_DEPRECATED_START
    explicit Cancelled(const std::string& what_arg) : ov::Exception(what_arg) {}
    OPENVINO_SUPPRESS_DEPRECATED_END
};

/**
 * @brief Thrown in case of calling the InferRequest methods while the request is
 * busy with compute operation.
 * @ingroup ov_runtime_cpp_api
 */
class OPENVINO_RUNTIME_API Busy : public Exception {
public:
    [[noreturn]] static void create(const std::string& explanation);
    ~Busy() override;

protected:
    OPENVINO_SUPPRESS_DEPRECATED_START
    explicit Busy(const std::string& what_arg) : ov::Exception(what_arg) {}
    OPENVINO_SUPPRESS_DEPRECATED_END
};

}  // namespace ov
