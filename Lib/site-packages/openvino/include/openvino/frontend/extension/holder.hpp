// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/frontend/extension/conversion.hpp"
#include "openvino/frontend/extension/progress_reporter.hpp"
#include "openvino/frontend/extension/telemetry.hpp"

namespace ov {
namespace frontend {
struct ExtensionHolder {
    std::shared_ptr<ProgressReporterExtension> progress_reporter{std::make_shared<ProgressReporterExtension>()};
    std::shared_ptr<TelemetryExtension> telemetry;
    std::vector<std::shared_ptr<ConversionExtensionBase>> conversions;
};
}  // namespace frontend
}  // namespace ov
