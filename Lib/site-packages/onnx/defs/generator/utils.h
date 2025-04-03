/*
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "onnx/defs/schema.h"

namespace ONNX_NAMESPACE {

void ConstantOpInference(InferenceContext& ctx);

} // namespace ONNX_NAMESPACE
