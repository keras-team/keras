// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/core/extension.hpp"
#include "openvino/frontend/visibility.hpp"

namespace ov {
namespace frontend {
class FRONTEND_API ProgressReporterExtension : public ov::Extension {
public:
    /// \brief A progress reporting callback signature. A FunctionObject that matches this signature should be passed
    ///        to the constructor of this extension. The extension will then invoke this as a callback each time the
    ///        progress needs to be reported. The callback itself is responsible for consuming the reported values.
    ///
    /// \param progress A float value in the range [0.0, 1.0] indicating the total progress of an operation.
    /// \param total_steps The total number of steps that a given instance of this extension is tracking
    /// \param completed_completed The current number of completed steps (out of the total number of steps to take)
    using progress_notifier_callback = std::function<void(float, unsigned int, unsigned int)>;

    /// \brief The default constructor which creates a reporter that doesn't report progress
    ProgressReporterExtension() : m_callback{[](float, unsigned int, unsigned int) {}} {}
    ProgressReporterExtension(const progress_notifier_callback& callback) : m_callback{callback} {}
    ProgressReporterExtension(progress_notifier_callback&& callback) : m_callback{std::move(callback)} {}

    /// \brief The main method of this extension used to report the progress.
    ///        This method forwards its arguments to the callback stored in this class.
    /// \param progress A float value in the range [0.0, 1.0] indicating the total progress of an operation.
    /// \param total_steps The total number of steps that a given instance of this extension is tracking
    /// \param completed_steps The current number of completed steps (out of the total number of steps to take)
    void report_progress(float progress, unsigned int total_steps, unsigned int completed_steps) const;

private:
    progress_notifier_callback m_callback;
};
}  // namespace frontend
}  // namespace ov
