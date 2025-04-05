// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <string>

#include "openvino/core/core_visibility.hpp"
#include "openvino/core/node.hpp"
#include "openvino/core/type.hpp"

namespace ov {
OPENVINO_API
void copy_runtime_info(const std::shared_ptr<ov::Node>& from, const std::shared_ptr<ov::Node>& to);

OPENVINO_API
void copy_runtime_info(const std::shared_ptr<ov::Node>& from, ov::NodeVector to);

OPENVINO_API
void copy_runtime_info(const ov::NodeVector& from, const std::shared_ptr<ov::Node>& to);

OPENVINO_API
void copy_runtime_info(const ov::NodeVector& from, ov::NodeVector to);

OPENVINO_API
void copy_output_runtime_info(const ov::OutputVector& from, ov::OutputVector to);
}  // namespace ov
