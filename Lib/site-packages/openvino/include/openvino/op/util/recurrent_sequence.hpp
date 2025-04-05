// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <vector>

#include "openvino/core/partial_shape.hpp"

namespace ov {
namespace op {
namespace util {
///
/// \brief      Validates static rank and dimension for provided input parameters.
///             Additionally input_size dimension is checked for X and W inputs.
///             Applies to LSTM, GRU and RNN Sequences.
///
///
/// \param[in]  input        Vector with RNNSequence-like op inputs in following order:
///                          X, initial_hidden_state, sequence_lengths, W, R and B.
///
OPENVINO_API void validate_seq_input_rank_dimension(const std::vector<ov::PartialShape>& input);
}  // namespace util
}  // namespace op
}  // namespace ov
