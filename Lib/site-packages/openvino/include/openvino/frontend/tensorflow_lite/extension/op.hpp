// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
#include "openvino/frontend/extension/op.hpp"
#include "openvino/frontend/tensorflow_lite/extension/conversion.hpp"

namespace ov {
namespace frontend {
namespace tensorflow_lite {

template <typename OVOpType = void>
using OpExtension = ov::frontend::OpExtensionBase<ov::frontend::tensorflow_lite::ConversionExtension, OVOpType>;

}  // namespace tensorflow_lite
}  // namespace frontend
}  // namespace ov
