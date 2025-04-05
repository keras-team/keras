// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/frontend/hash_table.hpp"

namespace ov {
namespace frontend {
namespace tensorflow {

// this class describes TensorFlow table produced by operations tf.raw_ops.HashTable, tf.raw_ops.HashTableV2,
// tf.raw_ops.MutableHashTable and stores a dictionary of keys mapped to values
// Objects of this class is fed to Lookup* operations for initialization and searching values by keys
// Types of keys and values can be different
using ov::frontend::HashTable;

}  // namespace tensorflow
}  // namespace frontend
}  // namespace ov
