// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <fstream>
#include <iostream>
#include <type_traits>

#include "openvino/core/extension.hpp"
#include "openvino/frontend/visibility.hpp"
#include "openvino/pass/graph_rewrite.hpp"
#include "openvino/pass/manager.hpp"
#include "openvino/pass/pass.hpp"

namespace ov {
namespace frontend {

/// \brief Provides callback to report telemetry information back to Python code
class FRONTEND_API TelemetryExtension : public ov::Extension {
public:
    using Ptr = std::shared_ptr<TelemetryExtension>;
    using error_callback = std::function<void(const std::string& category, const std::string& error_message)>;
    using event_callback = std::function<
        void(const std::string& category, const std::string& action, const std::string& label, int value)>;
    TelemetryExtension(const std::string& event_category,
                       const event_callback& send_event,
                       const error_callback& send_error,
                       const error_callback& send_stack_trace);

    void send_event(const std::string& action, const std::string& label, int value = 1);
    void send_error(const std::string& error_message);
    void send_stack_trace(const std::string& error_message);

private:
    std::string m_event_category;
    event_callback m_send_event;
    error_callback m_send_error;
    error_callback m_send_stack_trace;
};

}  // namespace frontend
}  // namespace ov
