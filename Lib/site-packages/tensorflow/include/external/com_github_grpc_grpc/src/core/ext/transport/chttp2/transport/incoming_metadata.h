/*
 *
 * Copyright 2015 gRPC authors.
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

#ifndef GRPC_CORE_EXT_TRANSPORT_CHTTP2_TRANSPORT_INCOMING_METADATA_H
#define GRPC_CORE_EXT_TRANSPORT_CHTTP2_TRANSPORT_INCOMING_METADATA_H

#include <grpc/support/port_platform.h>

#include "src/core/lib/transport/transport.h"

struct grpc_chttp2_incoming_metadata_buffer {
  explicit grpc_chttp2_incoming_metadata_buffer(grpc_core::Arena* arena)
      : arena(arena) {
    grpc_metadata_batch_init(&batch);
    batch.deadline = GRPC_MILLIS_INF_FUTURE;
  }
  ~grpc_chttp2_incoming_metadata_buffer() {
    grpc_metadata_batch_destroy(&batch);
  }

  static constexpr size_t kPreallocatedMDElem = 10;

  grpc_core::Arena* arena;
  size_t size = 0;   // total size of metadata.
  size_t count = 0;  // minimum of count of metadata and kPreallocatedMDElem.
  // These preallocated mdelems are used while count < kPreallocatedMDElem.
  grpc_linked_mdelem preallocated_mdelems[kPreallocatedMDElem];
  grpc_metadata_batch batch;
};

void grpc_chttp2_incoming_metadata_buffer_publish(
    grpc_chttp2_incoming_metadata_buffer* buffer, grpc_metadata_batch* batch);

grpc_error* grpc_chttp2_incoming_metadata_buffer_add(
    grpc_chttp2_incoming_metadata_buffer* buffer,
    grpc_mdelem elem) GRPC_MUST_USE_RESULT;
grpc_error* grpc_chttp2_incoming_metadata_buffer_replace_or_add(
    grpc_chttp2_incoming_metadata_buffer* buffer,
    grpc_mdelem elem) GRPC_MUST_USE_RESULT;
void grpc_chttp2_incoming_metadata_buffer_set_deadline(
    grpc_chttp2_incoming_metadata_buffer* buffer, grpc_millis deadline);

#endif /* GRPC_CORE_EXT_TRANSPORT_CHTTP2_TRANSPORT_INCOMING_METADATA_H */
