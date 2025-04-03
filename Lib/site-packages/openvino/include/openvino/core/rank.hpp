// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/core/dimension.hpp"

namespace ov {
/// \brief Alias for Dimension, used when the value represents the number of axes in a shape,
///        rather than the size of one dimension in a shape.
///
using Rank = Dimension;
}  // namespace ov
