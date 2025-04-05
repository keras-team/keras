/*
 *
 * Copyright 2017 gRPC authors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 */

#ifndef GRPC_CORE_LIB_SECURITY_TRANSPORT_TARGET_AUTHORITY_TABLE_H
#define GRPC_CORE_LIB_SECURITY_TRANSPORT_TARGET_AUTHORITY_TABLE_H

#include <grpc/support/port_platform.h>

#include "src/core/lib/slice/slice_hash_table.h"

namespace grpc_core {

/// A hash table mapping target addresses to authorities.
typedef SliceHashTable<grpc_core::UniquePtr<char>> TargetAuthorityTable;

/// Returns a channel argument containing \a table.
grpc_arg CreateTargetAuthorityTableChannelArg(TargetAuthorityTable* table);

/// Returns the target authority table from \a args or nullptr.
TargetAuthorityTable* FindTargetAuthorityTableInArgs(
    const grpc_channel_args* args);

}  // namespace grpc_core

#endif /* GRPC_CORE_LIB_SECURITY_TRANSPORT_TARGET_AUTHORITY_TABLE_H */
