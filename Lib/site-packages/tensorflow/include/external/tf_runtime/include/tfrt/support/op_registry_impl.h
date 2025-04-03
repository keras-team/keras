/*
 * Copyright 2020 The TensorFlow Runtime Authors
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

// This file declares OpRegistryImpl.  OpRegistryImpl aims to reduce boilerplate
// for device specific operation registry implementations.

#ifndef TFRT_SUPPORT_OP_REGISTRY_IMPL_H_
#define TFRT_SUPPORT_OP_REGISTRY_IMPL_H_

#include <string>

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include "tfrt/support/forward_decls.h"

namespace tfrt {
template <typename OpMetadataFnTy, typename DispatchFnTy, typename OpFlagsTy>
class OpRegistryImpl {
 public:
  struct OpEntry {
    OpMetadataFnTy metadata_fn = nullptr;
    DispatchFnTy dispatch_fn = nullptr;
    OpFlagsTy flags;
    string_view op_name;

    llvm::SmallVector<std::string, 4> attr_names;
  };

  void AddOp(string_view op_name, DispatchFnTy dispatch_fn, OpFlagsTy flags,
             ArrayRef<string_view> attr_names) {
    assert(!op_name.empty() && "op names cannot be empty");
    auto& entry = op_mappings_[op_name];
    entry.dispatch_fn = dispatch_fn;
    entry.flags = flags;
    entry.attr_names.reserve(attr_names.size());
    for (auto name : attr_names) entry.attr_names.emplace_back(name);
    entry.op_name = op_mappings_.find(op_name)->first();
  }

  void AddMetadataFn(string_view op_name, OpMetadataFnTy metadata_fn) {
    assert(!op_name.empty() && "op names cannot be empty");
    op_mappings_[op_name].metadata_fn = metadata_fn;
  }

  const OpEntry* LookupOpEntry(string_view op_name) const {
    auto op_it = op_mappings_.find(op_name);
    if (op_it == op_mappings_.end()) return &empty_entry_;

    return &op_it->second;
  }

 private:
  llvm::StringMap<OpEntry> op_mappings_;
  OpEntry empty_entry_;
};
}  // namespace tfrt

#endif
