// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
#include "openvino/frontend/extension/op.hpp"
#include "openvino/frontend/pytorch/extension/conversion.hpp"

namespace ov {
namespace frontend {
namespace pytorch {

template <typename OVOpType = void>
using OpExtension = ov::frontend::OpExtensionBase<ov::frontend::pytorch::ConversionExtension, OVOpType>;

}  // namespace pytorch
}  // namespace frontend
}  // namespace ov
