// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
#include "openvino/frontend/extension/op.hpp"
#include "openvino/frontend/paddle/extension/conversion.hpp"

namespace ov {
namespace frontend {
namespace paddle {

template <typename OVOpType = void>
using OpExtension = ov::frontend::OpExtensionBase<ConversionExtension, OVOpType>;

}  // namespace paddle
}  // namespace frontend
}  // namespace ov
